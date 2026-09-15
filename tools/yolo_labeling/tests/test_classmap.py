"""Mapping a model's prediction classes onto the dataset's classes."""

from __future__ import annotations

import pytest

from core.classmap import PredictionClassMap, auto_map_by_name, normalize

# A dataset whose ids mean something completely different from COCO's.
DATASET = {0: {"name": "BSD"}, 1: {"name": "BSV"}, 2: {"name": "bus_m"},
           3: {"name": "car"}, 4: {"name": "human"}}
COCO = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


# -- name matching ---------------------------------------------------------


@pytest.mark.parametrize("a,b", [
    ("bus_m", "Bus_M"), ("bus m", "busm"), ("Car", "car"), ("two-wheels", "twowheels"),
])
def test_name_matching_ignores_case_spacing_and_punctuation(a, b):
    assert normalize(a) == normalize(b)


def test_auto_map_matches_by_name_not_by_id():
    """COCO car is id 2; dataset car is id 3. The NAME is what must match."""
    assert auto_map_by_name(COCO, DATASET) == {2: 3}


def test_auto_map_is_empty_when_no_names_coincide():
    assert auto_map_by_name({0: "aardvark"}, DATASET) == {}


def test_auto_map_handles_missing_or_empty_inputs():
    assert auto_map_by_name({}, DATASET) == {}
    assert auto_map_by_name(None, DATASET) == {}
    assert auto_map_by_name(COCO, {}) == {}


def test_auto_map_is_deterministic_with_duplicate_dataset_names():
    dup = {0: {"name": "car"}, 1: {"name": "car"}}
    assert auto_map_by_name({9: "car"}, dup) == {9: 0}, "first definition wins"


# -- translation -----------------------------------------------------------


def test_mapped_class_is_translated():
    m = PredictionClassMap({2: 3})
    assert m.translate(2) == 3


def test_unmapped_class_is_dropped_by_default():
    """A wrong label is worse than a missing one."""
    assert PredictionClassMap({2: 3}).translate(1) is None


def test_unmapped_class_can_be_kept_when_explicitly_asked():
    m = PredictionClassMap({2: 3}, drop_unmapped=False)
    assert m.translate(1) == 1


def test_disabling_the_map_passes_ids_through_untouched():
    m = PredictionClassMap({2: 3}, enabled=False)
    assert m.translate(2) == 2 and m.translate(99) == 99


def test_apply_remaps_and_drops_in_one_pass():
    m = PredictionClassMap({0: 4, 2: 3})
    preds = [(0, 1, 1, 2, 2), (2, 3, 3, 4, 4), (1, 5, 5, 6, 6)]
    assert m.apply(preds) == [(4, 1, 1, 2, 2), (3, 3, 3, 4, 4)]


def test_apply_preserves_box_geometry_exactly():
    m = PredictionClassMap({0: 4})
    assert m.apply([(0, 1.5, 2.5, 3.5, 4.5)]) == [(4, 1.5, 2.5, 3.5, 4.5)]


def test_apply_on_empty_input():
    assert PredictionClassMap({0: 1}).apply([]) == []


def test_numpy_style_class_ids_are_accepted():
    """Ultralytics hands back numpy scalars."""
    np = pytest.importorskip("numpy")
    assert PredictionClassMap({2: 3}).translate(np.int64(2)) == 3


# -- many-to-one -----------------------------------------------------------


def test_several_model_classes_can_collapse_onto_one_dataset_class():
    """COCO car/bus/truck all becoming a single dataset 'vehicle'."""
    m = PredictionClassMap()
    m.map_all_to([2, 5, 7], 2)
    assert [p[0] for p in m.apply([(2, 0, 0, 1, 1), (5, 0, 0, 1, 1), (7, 0, 0, 1, 1)])] == [2, 2, 2]


def test_unmap_removes_entries():
    m = PredictionClassMap({2: 3, 5: 2})
    m.unmap([5])
    assert m.mapping == {2: 3}
    assert m.translate(5) is None


# -- persistence -----------------------------------------------------------


def test_round_trips_through_json():
    m = PredictionClassMap({2: 3, 5: 2}, drop_unmapped=False, enabled=False)
    back = PredictionClassMap.from_json(m.to_json())
    assert back.mapping == m.mapping
    assert back.drop_unmapped is False and back.enabled is False


def test_json_keys_are_strings_for_a_readable_config():
    assert PredictionClassMap({2: 3}).to_json()["mapping"] == {"2": 3}


def test_from_json_tolerates_a_hand_edited_config():
    m = PredictionClassMap.from_json(
        {"mapping": {"2": 3, "oops": 1, "5": "bad"}, "drop_unmapped": True})
    assert m.mapping == {2: 3}, "unparseable entries are skipped, not fatal"


def test_from_json_of_nothing_is_a_usable_default():
    m = PredictionClassMap.from_json(None)
    assert m.mapping == {} and m.drop_unmapped is True and m.enabled is True


def test_summary_reports_coverage():
    m = PredictionClassMap(auto_map_by_name(COCO, DATASET))
    assert "1/6" in m.summary(COCO, DATASET)
    m.enabled = False
    assert "OFF" in m.summary(COCO, DATASET)


# ==========================================================================
# integration: loading a model must not rewrite the dataset's taxonomy
# ==========================================================================


class _FakeModel:
    """Stands in for a YOLO whose class list differs from the dataset's."""

    names = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
             4: "airplane", 5: "bus", 6: "train", 7: "truck"}

    def __call__(self, *a, **k):
        return []


@pytest.fixture
def app_with_model(gui_app, monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "ultralytics",
                        type("M", (), {"YOLO": staticmethod(lambda p: _FakeModel())}))
    return gui_app


@pytest.mark.gui
@pytest.mark.legacy
def test_loading_a_model_does_not_rename_dataset_classes(app_with_model):
    """It used to rename class 2 from 'bus_m' to 'car' and persist that."""
    before = {i: c["name"] for i, c in app_with_model.class_info.items()}
    app_with_model.load_model("coco.pt")
    after = {i: c["name"] for i, c in app_with_model.class_info.items()}
    assert after == before, "the dataset taxonomy must be untouched"


@pytest.mark.gui
@pytest.mark.legacy
def test_loading_a_model_does_not_add_its_classes_to_the_dataset(app_with_model):
    n_before = len(app_with_model.class_info)
    app_with_model.load_model("coco.pt")
    assert len(app_with_model.class_info) == n_before, "80 COCO classes must not appear"


@pytest.mark.gui
@pytest.mark.legacy
def test_loading_a_model_seeds_the_map_by_name(app_with_model):
    """Fixture dataset is person/car/bus_m/bus_l, so person and car match."""
    app_with_model.load_model("coco.pt")
    mapping = app_with_model.prediction_class_map.mapping
    assert mapping[0] == 0, "COCO person -> dataset person"
    assert mapping[2] == 1, "COCO car(2) -> dataset car(1), matched by NAME not id"
    assert 4 not in mapping, "airplane has no dataset counterpart"


@pytest.mark.gui
@pytest.mark.legacy
def test_predictions_are_translated_before_reaching_the_app(app_with_model):
    app_with_model.load_model("coco.pt")
    app_with_model.prediction_class_map.mapping = {2: 1}
    raw = [(2, 0, 0, 10, 10), (7, 0, 0, 10, 10)]
    monkey = app_with_model.prediction_class_map.apply(raw)
    assert monkey == [(1, 0, 0, 10, 10)], "car remapped, truck dropped"


@pytest.mark.gui
@pytest.mark.legacy
def test_the_map_persists_per_model(app_with_model, tmp_path):
    import json

    app_with_model.load_model("coco.pt")
    app_with_model.prediction_class_map.map_all_to([5, 7], 2)
    app_with_model.save_prediction_class_map()

    cfg = json.loads((tmp_path / "class_config.json").read_text())
    saved = cfg["prediction_maps"]["coco.pt"]["mapping"]
    assert saved["5"] == 2 and saved["7"] == 2
    assert {k["name"] for k in cfg["classes"].values()} >= {"bus_m", "bus_l"}, \
        "dataset classes still intact on disk"


# ==========================================================================
# the editor dialog
# ==========================================================================


def _widgets(root, kind):
    """All descendants of `root` of the given widget class."""
    out = []
    stack = list(root.winfo_children())
    while stack:
        w = stack.pop()
        if isinstance(w, kind):
            out.append(w)
        stack += list(w.winfo_children())
    return out


def _button(win, label):
    import tkinter as tk
    from tkinter import ttk

    for b in _widgets(win, (ttk.Button, tk.Button)):
        try:
            if label.lower() in str(b.cget("text")).lower():
                return b
        except Exception:
            continue
    raise AssertionError(f"no button matching {label!r}")


@pytest.fixture
def map_dialog(app_with_model):
    """Open the mapping editor and hand back (app, window, text widget)."""
    import tkinter as tk

    app_with_model.load_model("coco.pt")
    app_with_model.edit_prediction_class_map()
    app_with_model.root.update_idletasks()
    win = [w for w in app_with_model.root.winfo_children() if isinstance(w, tk.Toplevel)][-1]
    text = _widgets(win, tk.Text)[0]
    return app_with_model, win, text


@pytest.mark.gui
@pytest.mark.legacy
def test_dialog_prepopulates_with_the_auto_map(map_dialog):
    _, _, text = map_dialog
    body = text.get("1.0", "end")
    assert "person: person" in body, "name-matched pairs are filled in"
    assert "car: car" in body
    assert "# bicycle" in body, "unmatched model classes are commented out (dropped)"


@pytest.mark.gui
@pytest.mark.legacy
def test_saving_applies_a_many_to_one_mapping(map_dialog):
    import tkinter as tk

    app, win, text = map_dialog
    text.delete("1.0", tk.END)
    text.insert("1.0", "person: person\nbus: bus_m\ntruck: bus_m\n# car:\n")
    _button(win, "Save").invoke()
    app.root.update_idletasks()

    m = app.prediction_class_map.mapping
    ds = {i: c["name"] for i, c in app.class_info.items()}
    assert ds[m[5]] == "bus_m" and ds[m[7]] == "bus_m", "bus AND truck collapse onto bus_m"
    assert 2 not in m, "the commented-out car line means dropped"


@pytest.mark.gui
@pytest.mark.legacy
def test_names_and_numeric_ids_are_both_accepted(map_dialog):
    import tkinter as tk

    app, win, text = map_dialog
    text.delete("1.0", tk.END)
    text.insert("1.0", "2: 1\ncar: 1\n")          # id:id and name:id
    _button(win, "Save").invoke()
    app.root.update_idletasks()
    assert app.prediction_class_map.mapping[2] == 1


@pytest.mark.gui
@pytest.mark.legacy
def test_an_unknown_name_is_reported_and_nothing_is_saved(map_dialog):
    import tkinter as tk

    app, win, text = map_dialog
    before = dict(app.prediction_class_map.mapping)
    text.delete("1.0", tk.END)
    text.insert("1.0", "person: not_a_real_class\n")
    _button(win, "Save").invoke()
    app.root.update_idletasks()
    assert app.prediction_class_map.mapping == before, "a typo must not silently wipe the map"
    assert win.winfo_exists(), "the dialog stays open so the error can be fixed"


@pytest.mark.gui
@pytest.mark.legacy
def test_auto_map_button_reseeds_the_text(map_dialog):
    import tkinter as tk

    _, win, text = map_dialog
    text.delete("1.0", tk.END)
    text.insert("1.0", "person: human\n")
    _button(win, "Auto-map").invoke()
    assert "car: car" in text.get("1.0", "end"), "re-seeded from name matching"


@pytest.mark.gui
@pytest.mark.legacy
def test_dialog_without_a_model_explains_itself(gui_app):
    gui_app.model_names = {}
    gui_app.edit_prediction_class_map()
    assert gui_app.dialogs.infos, "user is told to load a model first"


# ==========================================================================
# prediction confidence threshold
# ==========================================================================


class _T:
    """Minimal stand-in for a torch tensor: supports `.cpu().numpy()`."""

    def __init__(self, v):
        self._v = v

    def __getitem__(self, i):
        import numpy as np

        return _T(np.asarray(self._v)[i])

    def cpu(self):
        return self

    def numpy(self):
        import numpy as np

        return np.asarray(self._v)


class _ScoredModel:
    """Returns boxes at a spread of confidences, honouring the `conf` kwarg."""

    names = {0: "person", 2: "car"}
    ALL = [(0, 0.90), (2, 0.60), (0, 0.30), (2, 0.10)]

    def __init__(self):
        self.calls = 0

    def __call__(self, img, conf=0.25, **k):
        self.calls += 1
        kept = [(c, s) for c, s in self.ALL if s >= conf]

        class _B:
            def __init__(self, c, s):
                self.xyxy = _T([[1.0, 2.0, 3.0, 4.0]])
                self.cls = _T([c])
                self.conf = _T([s])

        return [type("R", (), {"boxes": [_B(c, s) for c, s in kept]})()]


@pytest.fixture
def scored_app(gui_app, monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "ultralytics",
                        type("M", (), {"YOLO": staticmethod(lambda p: _ScoredModel())}))
    gui_app.load_model("scored.pt")
    gui_app.prediction_class_map.mapping = {0: 0, 2: 1}
    return gui_app


@pytest.mark.gui
@pytest.mark.legacy
def test_default_threshold_is_explicit_not_ultralytics_implicit(gui_app):
    assert gui_app.prediction_conf == 0.25


@pytest.mark.gui
@pytest.mark.legacy
def test_threshold_applies_at_read_not_at_inference(scored_app, ds):
    """Inference happens once at the cache floor; the slider filters the result.

    It used to be passed straight to the model, which meant every nudge of the
    threshold discarded the cache and re-inferred the whole dataset.
    """
    import numpy as np

    img = np.zeros((48, 64, 3), np.uint8)
    path = ds.image("camA/0001.jpg")
    # model returns scores 0.90, 0.60, 0.30, 0.10
    scored_app.model_predictions_cache[str(path)] = scored_app._predict_image(img)
    calls_after_inference = scored_app.pretrained_model.calls

    for conf, expected in [(0.50, 2), (0.25, 3), (0.05, 4), (0.80, 1)]:
        scored_app.prediction_conf = conf
        got = scored_app._display_predictions(path)
        assert len(got) == expected, f"at conf={conf}"

    assert scored_app.pretrained_model.calls == calls_after_inference, \
        "re-filtering must not re-run the model"


@pytest.mark.gui
@pytest.mark.legacy
def test_inference_returns_everything_above_the_floor(scored_app):
    """`_predict_image` is threshold-agnostic now -- that is what lets one
    cached result serve the canvas at 0.25 and the scan at 0.40."""
    import numpy as np

    from yolo_labeling.core import predcache

    scored_app.prediction_conf = 0.80
    preds = scored_app._predict_image(np.zeros((48, 64, 3), np.uint8))
    assert len(preds) == 4, "all four scores are at or above the 0.10 floor"
    assert all(p[5] >= predcache.CONF_FLOOR for p in preds)


@pytest.mark.gui
@pytest.mark.legacy
def test_confidence_is_preserved_through_the_class_remap(scored_app, ds):
    import numpy as np

    path = ds.image("camA/0001.jpg")
    scored_app.model_predictions_cache[str(path)] = scored_app._predict_image(
        np.zeros((48, 64, 3), np.uint8))
    scored_app.prediction_conf = 0.50
    preds = scored_app._display_predictions(path)
    assert all(len(p) == 6 for p in preds), "(cls, x1, y1, x2, y2, conf)"
    assert [round(p[5], 2) for p in preds] == [0.90, 0.60]
    assert [p[0] for p in preds] == [0, 1], "class ids still remapped to dataset space"


@pytest.mark.gui
@pytest.mark.legacy
def test_changing_the_threshold_keeps_the_cache(scored_app, ds):
    """The opposite of the old contract, and the point of the whole change.

    Re-inferring 46k images because the user nudged a slider made the threshold
    un-tunable in practice.
    """
    import numpy as np

    path = ds.image("camA/0001.jpg")
    scored_app.model_predictions_cache[str(path)] = scored_app._predict_image(
        np.zeros((48, 64, 3), np.uint8))
    before = dict(scored_app.model_predictions_cache)

    scored_app.prediction_conf_var.set("0.05")
    scored_app._on_prediction_conf_change()

    assert scored_app.prediction_conf == 0.05
    assert scored_app.model_predictions_cache == before, "cache survives the change"


@pytest.mark.gui
@pytest.mark.legacy
def test_threshold_is_clamped_to_a_sane_range(scored_app):
    for typed, expected in (("5", 0.95), ("-1", 0.01), ("0.4", 0.40)):
        scored_app.prediction_conf_var.set(typed)
        scored_app._on_prediction_conf_change()
        assert scored_app.prediction_conf == expected


@pytest.mark.gui
@pytest.mark.legacy
def test_garbage_input_is_reported_and_ignored(scored_app):
    scored_app.prediction_conf = 0.25
    scored_app.prediction_conf_var.set("abc")
    scored_app._on_prediction_conf_change()
    assert scored_app.prediction_conf == 0.25, "unchanged"


@pytest.mark.gui
@pytest.mark.legacy
def test_accepting_a_prediction_writes_only_the_five_label_fields(scored_app):
    """The 6th element must not leak into user_boxes and thence into a label."""
    import numpy as np

    scored_app.current_image = np.zeros((48, 64, 3), np.uint8)
    scored_app.model_boxes = scored_app._predict_image(scored_app.current_image)
    scored_app.user_boxes = []
    scored_app.accept_all_predictions()
    assert scored_app.user_boxes, "predictions were accepted"
    assert all(len(b) == 5 for b in scored_app.user_boxes), "no confidence in labels"
