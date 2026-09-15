"""Whole-dataset inference: the button, the readout, and the worker.

Lazy inference means the dataset is only ever partly inferred. Without a way to
see how much, there is no way to know whether the next scan takes a minute or an
hour -- and no way to pay that cost deliberately, up front.
"""

from __future__ import annotations

import time

import pytest

pytestmark = [pytest.mark.gui, pytest.mark.legacy]


class CountingModel:
    names = {0: "person"}

    def __init__(self):
        self.calls = 0

    def __call__(self, img, verbose=False, conf=0.25, **k):
        self.calls += 1

        class _T:
            def __init__(self, v):
                self.v = v

            def __getitem__(self, i):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return self.v

        class _B:
            xyxy = _T([1.0, 2.0, 3.0, 4.0])
            cls = _T(0)
            conf = _T(0.9)

        return [type("R", (), {"boxes": [_B()]})()]


@pytest.fixture
def app(gui_app, ds, tmp_path):
    from yolo_labeling.core.predcache import PredictionCache

    weights = tmp_path / "best.pt"
    weights.write_bytes(b"weights")
    gui_app.pretrained_model = CountingModel()
    gui_app.model_names = {0: "person"}
    gui_app.prediction_class_map.enabled = False
    gui_app.pred_cache = PredictionCache(tmp_path / "p.sqlite", weights)
    gui_app.filtered_image_paths = [
        ds.image("camA/0001.jpg"), ds.image("camA/0002.jpg"), ds.image("camB/0001.jpg"),
    ]
    yield gui_app
    gui_app.pred_cache.close()


def _run_to_completion(app, timeout=10.0):
    """Drive the worker + drain loop until it reports done."""
    app.run_dataset_inference()
    deadline = time.time() + timeout
    while app._inference_running and time.time() < deadline:
        app.root.update()
        time.sleep(0.01)
    assert not app._inference_running, "inference did not finish"


# -- the readout -----------------------------------------------------------


def test_coverage_starts_at_zero(app):
    app.update_inference_coverage()
    text = app.inference_coverage_var.get()
    assert "0 of 3" in text, text


def test_coverage_reports_no_model(gui_app):
    gui_app.pretrained_model = None
    gui_app.update_inference_coverage()
    assert gui_app.inference_coverage_var.get() == "No model loaded"


def test_coverage_follows_the_run(app):
    _run_to_completion(app)
    assert "3 of 3" in app.inference_coverage_var.get()
    assert app.inference_progress['value'] == pytest.approx(100.0)


# -- the run ---------------------------------------------------------------


def test_infers_every_image_in_scope(app):
    _run_to_completion(app)
    assert app.pretrained_model.calls == 3
    assert app.pred_cache.count() == 3


def test_a_second_run_skips_everything(app):
    """The whole point: already-cached images are not re-inferred."""
    _run_to_completion(app)
    assert app.pretrained_model.calls == 3

    _run_to_completion(app)
    assert app.pretrained_model.calls == 3, "second pass re-inferred"


def test_declining_the_confirmation_runs_nothing(app):
    app.dialogs.answer = False
    app.run_dataset_inference()
    assert app.pretrained_model.calls == 0
    assert not app._inference_running


def test_scope_is_the_filtered_set(app, ds):
    app.filtered_image_paths = [ds.image("camA/0001.jpg")]
    _run_to_completion(app)
    assert app.pretrained_model.calls == 1


def test_unreadable_images_are_counted_not_fatal(app, ds):
    app.filtered_image_paths = [ds.image("bad/corrupt.jpg"), ds.image("camA/0001.jpg")]
    _run_to_completion(app)
    assert app.pretrained_model.calls == 1, "only the readable one"
    assert "1 unreadable" in app.inference_coverage_var.get()


def test_buttons_toggle_around_a_run(app):
    import tkinter as tk

    assert str(app.inference_cancel_btn.cget('state')) == tk.DISABLED
    _run_to_completion(app)
    assert str(app.inference_run_btn.cget('state')) == tk.NORMAL
    assert str(app.inference_cancel_btn.cget('state')) == tk.DISABLED


def test_no_model_is_refused(gui_app):
    gui_app.pretrained_model = None
    gui_app.run_dataset_inference()
    assert gui_app.dialogs.warnings, "the user is told why nothing happened"


# -- clearing --------------------------------------------------------------


def test_clear_empties_the_cache(app):
    _run_to_completion(app)
    assert app.pred_cache.count() == 3
    app.dialogs.answer = True
    app.clear_prediction_cache()
    assert app.pred_cache.count() == 0
    assert app.model_predictions_cache == {}
    assert "0 of 3" in app.inference_coverage_var.get()


def test_declining_the_clear_keeps_the_cache(app):
    _run_to_completion(app)
    app.dialogs.answer = False
    app.clear_prediction_cache()
    assert app.pred_cache.count() == 3
