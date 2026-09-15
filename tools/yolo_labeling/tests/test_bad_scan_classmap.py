"""Class translation inside the 'Find Bad Annotations' scan.

The scan compares a model's predicted class against the ground-truth class. A
model and a dataset almost never share an id space -- a 7-class detector against
a 13-class dataset agrees on nothing -- so without translation the mismatch
signal fires on nearly every object and the feature is worse than useless.

These run headless: the worker touches a queue, a lock and the model, never Tk.
"""

from __future__ import annotations

import queue
import threading

import pytest

from yolo_labeling.core.classmap import PredictionClassMap

#: The real pvn-vehicle taxonomy, which is what makes the arithmetic concrete.
DATASET = [
    "BSD", "BSV", "bus_m", "bus_l", "bus_xl", "car", "human",
    "twowheels_standalone", "bike_with_driver", "motorbike_with_driver",
    "truck_m", "truck_l", "truck_xl",
]
#: The 7 group names, in `data.yaml` insertion order -- the model's class list.
MODEL = ["BSD", "BSV", "bus", "car", "human", "motorbike", "truck"]

#: model id -> dataset id, derived from the groups block (representative member).
MAPPING = {0: 0, 1: 1, 2: 2, 3: 5, 4: 6, 5: 9, 6: 10}

GROUPS = {
    "bus": ["bus_m", "bus_l", "bus_xl"],
    "motorbike": ["motorbike_with_driver", "bike_with_driver", "twowheels_standalone"],
    "truck": ["truck_m", "truck_l", "truck_xl"],
}


# -- a model stand-in ------------------------------------------------------
#
# The worker calls `.cpu().numpy()` on every field, so plain numbers will not
# do; these mimic just enough of the torch surface.


class _Cell:
    def __init__(self, v):
        self.v = v

    def cpu(self):
        return self

    def numpy(self):
        return self.v


class _Field:
    def __init__(self, v):
        self.v = v

    def __getitem__(self, i):
        return _Cell(self.v)


class _Box:
    def __init__(self, cls_id, conf, xyxy):
        self.cls = _Field(cls_id)
        self.conf = _Field(conf)
        self.xyxy = _Field(list(xyxy))


class _Res:
    def __init__(self, boxes):
        self.boxes = boxes


class FakeModel:
    """Returns a fixed set of predictions for every image."""

    def __init__(self, boxes):
        self._boxes = boxes

    def __call__(self, img, verbose=False, conf=0.25):
        return [_Res(self._boxes)]


# -- probe -----------------------------------------------------------------


def _probe(monolith, model, cmap, groups=None):
    """A bare app carrying only what `_bad_scan_worker` reads."""
    app = object.__new__(monolith.YOLOLabelingTool)
    app.class_info = {i: {"name": n, "color": [0, 0, 255]} for i, n in enumerate(DATASET)}
    app.class_groups = dict(groups or {})
    app._rebuild_class_group_lookup()
    app.model_names = dict(enumerate(MODEL))
    app.prediction_class_map = cmap
    app.pretrained_model = model
    app.prediction_conf = 0.25
    app.model_predictions_cache = {}
    app.image_dimensions_cache = {}
    app.pred_cache = None
    app.label_folder = None
    app.per_image_label_folders = {}
    app._model_lock = threading.Lock()
    app._bad_scan_cancelled = threading.Event()
    app._bad_scan_queue = queue.Queue()
    return app


def _scan(app, paths):
    """Run the worker synchronously and return the entries it emitted."""
    app._bad_scan_worker({
        "paths": list(paths),
        "conf": 0.25,
        "iou_match": 0.5,
        "mismatch": True,
        "missing": True,
    })
    out = []
    while True:
        try:
            msg = app._bad_scan_queue.get_nowait()
        except queue.Empty:
            break
        if msg[0] == "entry":
            out.append(msg[1])
        elif msg[0] == "error":                     # pragma: no cover
            pytest.fail(f"scan reported an error: {msg[1]}")
    return out


@pytest.fixture
def one_image(ds):
    """A 64x48 image whose only label we control. Boxes below are in pixels."""
    return ds.image("camA/0001.jpg")


def _gt(ds, cls_id):
    """Write a single centred GT box covering the middle half of the image."""
    ds.label("camA/0001").write_text(f"{cls_id} 0.5 0.5 0.5 0.5\n")


#: The same rectangle as `_gt`, in pixels for a 64x48 image.
SAME_BOX = (16, 12, 48, 36)


# -- the defect ------------------------------------------------------------


def test_matching_car_is_not_flagged(ds, monolith, one_image):
    """Model 'car' (id 3) over GT 'car' (id 5) is agreement, not a mismatch.

    Untranslated, 3 != 5 and every car in the dataset gets flagged.
    """
    _gt(ds, 5)
    app = _probe(monolith, FakeModel([_Box(3, 0.9, SAME_BOX)]),
                 PredictionClassMap(mapping=dict(MAPPING)))
    assert _scan(app, [one_image]) == []


def test_without_a_map_the_same_scene_is_flagged(ds, monolith, one_image):
    """Pins the cost of an empty map: disabled translation reports a mismatch."""
    _gt(ds, 5)
    app = _probe(monolith, FakeModel([_Box(3, 0.9, SAME_BOX)]),
                 PredictionClassMap(mapping={}, enabled=False))
    entries = _scan(app, [one_image])
    assert [e["kind"] for e in entries] == ["class_mismatch"]


def test_genuine_mismatch_still_reported(ds, monolith, one_image):
    """Translation must not suppress real disagreement: 'human' over a GT car."""
    _gt(ds, 5)
    app = _probe(monolith, FakeModel([_Box(4, 0.9, SAME_BOX)]),
                 PredictionClassMap(mapping=dict(MAPPING)))
    entries = _scan(app, [one_image])
    assert len(entries) == 1
    assert entries[0]["kind"] == "class_mismatch"
    assert entries[0]["pred_cls_id"] == 6, "reported in DATASET ids, so the caption names 'human'"


# -- coarse model vs fine dataset -----------------------------------------


@pytest.mark.parametrize("gt_cls", [2, 3, 4])          # bus_m, bus_l, bus_xl
def test_group_absorbs_subtype_the_model_cannot_see(ds, monolith, one_image, gt_cls):
    """A 'bus' prediction cannot distinguish bus_m/bus_l/bus_xl.

    The map alone would flag two of the three; the groups block is what makes
    the coarse-vs-fine comparison honest.
    """
    _gt(ds, gt_cls)
    app = _probe(monolith, FakeModel([_Box(2, 0.9, SAME_BOX)]),
                 PredictionClassMap(mapping=dict(MAPPING)), groups=GROUPS)
    assert _scan(app, [one_image]) == []


def test_without_groups_two_of_three_bus_subtypes_are_flagged(ds, monolith, one_image):
    """The map maps 'bus' to bus_m, so bus_l and bus_xl need the group to pass."""
    flagged = []
    for gt_cls in (2, 3, 4):
        _gt(ds, gt_cls)
        app = _probe(monolith, FakeModel([_Box(2, 0.9, SAME_BOX)]),
                     PredictionClassMap(mapping=dict(MAPPING)))     # no groups
        flagged += [gt_cls for _ in _scan(app, [one_image])]
    assert flagged == [3, 4]


def test_groups_do_not_bridge_unrelated_classes(ds, monolith, one_image):
    """A truck prediction over a bus GT is a real mismatch even with groups."""
    _gt(ds, 3)                                        # bus_l
    app = _probe(monolith, FakeModel([_Box(6, 0.9, SAME_BOX)]),   # truck
                 PredictionClassMap(mapping=dict(MAPPING)), groups=GROUPS)
    assert [e["kind"] for e in _scan(app, [one_image])] == ["class_mismatch"]


# -- unmapped classes ------------------------------------------------------


def test_unmapped_prediction_is_inconclusive_not_a_mismatch(ds, monolith, one_image):
    """An unmapped model class says nothing about the GT class."""
    _gt(ds, 5)
    app = _probe(monolith, FakeModel([_Box(3, 0.9, SAME_BOX)]),
                 PredictionClassMap(mapping={0: 0}))     # 3 is unmapped
    assert _scan(app, [one_image]) == []


def test_unmapped_prediction_does_not_make_the_gt_look_missing(ds, monolith, one_image):
    """Dropping unmapped predictions outright would invent a 'missing' flag.

    The box still proves the model saw an object here, so the GT is covered
    even though we cannot name the class.
    """
    _gt(ds, 5)
    app = _probe(monolith, FakeModel([_Box(3, 0.9, SAME_BOX)]),
                 PredictionClassMap(mapping={0: 0}, drop_unmapped=True))
    assert [e["kind"] for e in _scan(app, [one_image])] == []


def test_genuinely_unseen_gt_is_still_reported_missing(ds, monolith, one_image):
    """The missing signal must survive all of the above."""
    _gt(ds, 5)
    app = _probe(monolith, FakeModel([]), PredictionClassMap(mapping=dict(MAPPING)))
    entries = _scan(app, [one_image])
    assert [e["kind"] for e in entries] == ["missing"]


# -- picking groups up from data.yaml -------------------------------------
#
# A dataset trained with the `groups` feature already states its coarse/fine
# relation in data.yaml. Re-typing it in the GUI is busywork.


def _yaml_with_groups(ds, groups):
    """A sibling data.yaml for the fixture dataset, carrying a groups block."""
    import yaml as _yaml

    src = _yaml.safe_load(ds.yaml.read_text())
    src["groups"] = groups
    out = ds.root / "data_groups.yaml"
    out.write_text(_yaml.safe_dump(src, sort_keys=False), encoding="utf-8")
    return out


@pytest.mark.gui
@pytest.mark.legacy
def test_yaml_groups_are_adopted_when_none_are_configured(gui_app, ds):
    groups = {"bus": ["bus_m", "bus_l"]}
    gui_app.class_groups = {}
    gui_app.load_classes_from_yaml(str(_yaml_with_groups(ds, groups)))

    assert gui_app.class_groups == groups
    # bus_m (2) and bus_l (3) now compare equal; car (1) still does not.
    assert gui_app._classes_equivalent(2, 3)
    assert not gui_app._classes_equivalent(1, 2)


@pytest.mark.gui
@pytest.mark.legacy
def test_hand_edited_groups_are_not_overwritten_by_the_yaml(gui_app, ds):
    """An explicit choice in the GUI outranks the dataset's default."""
    mine = {"everything": ["person", "car"]}
    gui_app.class_groups = dict(mine)
    gui_app.load_classes_from_yaml(str(_yaml_with_groups(ds, {"bus": ["bus_m", "bus_l"]})))

    assert gui_app.class_groups == mine
    assert gui_app._classes_equivalent(0, 1)
    assert not gui_app._classes_equivalent(2, 3)


@pytest.mark.gui
@pytest.mark.legacy
def test_a_yaml_without_groups_changes_nothing(gui_app, ds):
    gui_app.class_groups = {}
    gui_app.load_classes_from_yaml(str(ds.yaml))
    assert gui_app.class_groups == {}


# -- confusion-pair filter -------------------------------------------------
#
# "Show me every GT `car` the model calls a `truck`."


def _pair_scan(app, paths, gt=None, pred=None):
    app._bad_scan_worker({
        "paths": list(paths), "conf": 0.25, "iou_match": 0.5,
        "mismatch": True, "missing": True,
        "gt_class": gt, "pred_class": pred,
    })
    out = []
    while True:
        try:
            msg = app._bad_scan_queue.get_nowait()
        except queue.Empty:
            return out
        if msg[0] == "entry":
            out.append(msg[1])


def test_pair_filter_selects_one_confusion(ds, monolith, one_image):
    """GT car (5) called human (6) matches; asking for a different pair does not."""
    _gt(ds, 5)
    cmap = PredictionClassMap(mapping=dict(MAPPING))

    app = _probe(monolith, FakeModel([_Box(4, 0.9, SAME_BOX)]), cmap)   # -> human
    assert len(_pair_scan(app, [one_image], gt=5, pred=6)) == 1

    app = _probe(monolith, FakeModel([_Box(4, 0.9, SAME_BOX)]), cmap)
    assert _pair_scan(app, [one_image], gt=5, pred=10) == []            # -> truck_m


def test_gt_filter_alone_narrows_to_one_class(ds, monolith, one_image):
    _gt(ds, 5)
    app = _probe(monolith, FakeModel([_Box(4, 0.9, SAME_BOX)]),
                 PredictionClassMap(mapping=dict(MAPPING)))
    assert _pair_scan(app, [one_image], gt=6) == [], "GT is a car, not a human"


def test_naming_a_pair_bypasses_the_group(ds, monolith, one_image):
    """Groups exist to suppress noise, not to censor an explicit question.

    bus_l and bus_m are the same group, so the ordinary scan says nothing --
    but that within-group confusion is precisely what the model cannot see and
    a user might want to inspect.
    """
    _gt(ds, 3)                                             # bus_l
    model = FakeModel([_Box(2, 0.9, SAME_BOX)])            # -> bus_m
    cmap = PredictionClassMap(mapping=dict(MAPPING))

    quiet = _probe(monolith, model, cmap, groups=GROUPS)
    assert _pair_scan(quiet, [one_image]) == []

    asked = _probe(monolith, model, cmap, groups=GROUPS)
    assert len(_pair_scan(asked, [one_image], gt=3, pred=2)) == 1


def test_pred_filter_suppresses_missing_entries(ds, monolith, one_image):
    """A 'missing' entry has no prediction, so it cannot answer 'predicted as B'."""
    _gt(ds, 5)
    app = _probe(monolith, FakeModel([]), PredictionClassMap(mapping=dict(MAPPING)))
    assert _pair_scan(app, [one_image], pred=6) == []

    app = _probe(monolith, FakeModel([]), PredictionClassMap(mapping=dict(MAPPING)))
    assert [e["kind"] for e in _pair_scan(app, [one_image])] == ["missing"]


@pytest.mark.parametrize("seconds,expected", [
    (0, "0s"), (45, "45s"), (60, "1m 00s"), (440, "7m 20s"), (4320, "1h 12m"),
])
def test_eta_formatting(monolith, seconds, expected):
    assert monolith.YOLOLabelingTool._fmt_eta(seconds) == expected


# -- progress reporting ----------------------------------------------------


@pytest.mark.gui
@pytest.mark.legacy
def test_scan_window_shows_a_progress_bar_that_tracks_the_scan(gui_app):
    """Scanning 46k frames at a few per second needs more than a counter."""
    import time

    gui_app._open_crop_review_window(entries=[], target_class=None, mode='bad')
    gui_app.root.update_idletasks()

    bar = getattr(gui_app, '_bad_scan_progress', None)
    assert bar is not None, "bad-annotation mode gets a progress bar"

    gui_app._bad_scan_started_at = time.time() - 10.0
    gui_app._bad_scan_update_status((25, 100), None, None)
    gui_app.root.update_idletasks()

    assert bar['value'] == pytest.approx(25.0)
    status = gui_app._crop_review_status_var.get()
    assert "Scanned 25/100" in status
    assert "img/s" in status and "ETA" in status, f"no rate/ETA in: {status}"

    gui_app._bad_scan_update_status((100, 100), 'complete', None)
    gui_app.root.update_idletasks()
    assert bar['value'] == pytest.approx(100.0)
    assert "✓ done" in gui_app._crop_review_status_var.get()


@pytest.mark.gui
@pytest.mark.legacy
def test_progress_update_survives_a_closed_window(gui_app):
    """The drain loop can fire once more after the user closes the window."""
    gui_app._open_crop_review_window(entries=[], target_class=None, mode='bad')
    gui_app.root.update_idletasks()
    gui_app._crop_review_window.destroy()
    gui_app._crop_review_window = None
    gui_app._bad_scan_update_status((5, 10), None, None)      # must not raise


# -- the scan reuses inference --------------------------------------------
#
# The point of the cache: a 46k-image scan is paid for once.


class CountingModel(FakeModel):
    """FakeModel that records how many images it was asked to infer."""

    def __init__(self, boxes):
        super().__init__(boxes)
        self.calls = 0

    def __call__(self, img, verbose=False, conf=0.25):
        self.calls += 1
        return super().__call__(img, verbose=verbose, conf=conf)


@pytest.fixture
def cached_probe(ds, monolith, tmp_path):
    """A probe wired to a real on-disk prediction cache."""
    from yolo_labeling.core.predcache import PredictionCache

    weights = tmp_path / "best.pt"
    weights.write_bytes(b"weights")

    def _make(model, cmap=None, groups=None):
        app = _probe(monolith, model, cmap or PredictionClassMap(mapping=dict(MAPPING)),
                     groups=groups)
        app.pred_cache = PredictionCache(tmp_path / "preds.sqlite", weights)
        app.update_status = lambda *a, **k: None
        return app

    _make.weights = weights
    _make.db = tmp_path / "preds.sqlite"
    return _make


def test_a_second_scan_runs_no_inference(ds, cached_probe, one_image):
    """Re-scanning with different settings must cost nothing."""
    _gt(ds, 5)
    model = CountingModel([_Box(4, 0.9, SAME_BOX)])

    first = cached_probe(model)
    assert len(_scan(first, [one_image])) == 1
    assert model.calls == 1
    first.pred_cache.commit()
    first.pred_cache.close()

    second = cached_probe(model)          # fresh memory cache, same DB
    assert len(_scan(second, [one_image])) == 1
    assert model.calls == 1, "the second scan re-inferred"
    second.pred_cache.close()


def test_a_re_scan_at_a_higher_threshold_reuses_the_same_inference(ds, cached_probe, one_image):
    """Raising the scan threshold narrows cached data instead of re-running."""
    _gt(ds, 5)
    model = CountingModel([_Box(4, 0.9, SAME_BOX), _Box(6, 0.2, (0, 0, 5, 5))])

    app = cached_probe(model)
    app._bad_scan_worker({"paths": [one_image], "conf": 0.15, "iou_match": 0.5,
                          "mismatch": True, "missing": True})
    app.pred_cache.commit()
    app.pred_cache.close()
    assert model.calls == 1

    strict = cached_probe(model)
    strict._bad_scan_worker({"paths": [one_image], "conf": 0.80, "iou_match": 0.5,
                             "mismatch": True, "missing": True})
    assert model.calls == 1, "a threshold change must not re-infer"
    strict.pred_cache.close()


def test_cancelling_keeps_the_work_already_done(ds, cached_probe):
    """Half a scan is still half a scan's worth of inference banked."""
    _gt(ds, 5)
    paths = [ds.image("camA/0001.jpg"), ds.image("camB/0001.jpg")]
    model = CountingModel([_Box(4, 0.9, SAME_BOX)])

    app = cached_probe(model)
    app._cached_predictions(paths[0], None, blocking=True)
    app.pred_cache.commit()
    app.pred_cache.close()
    assert model.calls == 1

    resumed = cached_probe(model)
    _scan(resumed, paths)
    assert model.calls == 2, "only the un-scanned image should be inferred"
    resumed.pred_cache.close()


def test_editing_the_class_map_does_not_re_infer(ds, cached_probe, one_image):
    """Translation is a read-side view, so remapping re-derives from cache."""
    _gt(ds, 5)
    model = CountingModel([_Box(4, 0.9, SAME_BOX)])

    app = cached_probe(model)
    assert [e['pred_cls_id'] for e in _scan(app, [one_image])] == [6]   # human
    app.pred_cache.commit()
    app.pred_cache.close()

    remapped = cached_probe(model, cmap=PredictionClassMap(mapping={**MAPPING, 4: 10}))
    assert [e['pred_cls_id'] for e in _scan(remapped, [one_image])] == [10]  # truck_m
    assert model.calls == 1, "a class-map edit must not re-infer"
    remapped.pred_cache.close()


# -- a cached scan must not decode ----------------------------------------
#
# Reusing the inference is only half the win. Decoding a 12MP JPEG costs
# ~50-100ms, so decoding every frame to reach cached data is an hour of work
# across 46k images -- indistinguishable, from the outside, from re-inferring.


def test_a_cached_scan_decodes_nothing(ds, cached_probe, one_image, monkeypatch):
    _gt(ds, 5)
    model = CountingModel([_Box(4, 0.9, SAME_BOX)])

    warm = cached_probe(model)
    _scan(warm, [one_image])
    warm.pred_cache.commit()
    warm.pred_cache.close()

    import cv2

    from yolo_labeling.ui import crop_review as _cr

    decodes = []
    real = cv2.imread
    monkeypatch.setattr(_cr.cv2, "imread",
                        lambda *a, **k: (decodes.append(a[0]), real(*a, **k))[1])

    cold = cached_probe(model)                 # fresh memory cache, same DB
    assert len(_scan(cold, [one_image])) == 1
    cold.pred_cache.close()

    assert model.calls == 1, "re-inferred"
    assert decodes == [], f"decoded {len(decodes)} image(s) it already had"


def test_a_cold_scan_still_decodes_what_it_must(ds, cached_probe, one_image, monkeypatch):
    """The fast path must not have broken the slow one."""
    _gt(ds, 5)
    model = CountingModel([_Box(4, 0.9, SAME_BOX)])

    import cv2

    from yolo_labeling.ui import crop_review as _cr

    decodes = []
    real = cv2.imread
    monkeypatch.setattr(_cr.cv2, "imread",
                        lambda *a, **k: (decodes.append(a[0]), real(*a, **k))[1])

    app = cached_probe(model)
    assert len(_scan(app, [one_image])) == 1
    app.pred_cache.close()

    assert model.calls == 1
    assert len(decodes) == 1, "an uncached image must still be decoded"


def test_an_unreadable_image_is_skipped_not_flagged(ds, cached_probe):
    """Header parses, decode fails -> skip. Reporting every GT box as 'missing'
    would invent a screenful of false positives."""
    model = CountingModel([])
    app = cached_probe(model)
    entries = _scan(app, [ds.image("bad/corrupt.jpg")])
    app.pred_cache.close()
    assert entries == []
