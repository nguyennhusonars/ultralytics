"""The GUI thread must never wait on the model lock.

Tk's event loop runs on the GUI thread, and the bad-annotation scan's progress
bar is driven from that loop via `after()`. So a blocking `acquire()` anywhere
on the GUI path does not merely delay one prediction -- it freezes the very
readout that tells the user the scan is alive.

Python locks are not fair, either: the scan worker releases and immediately
re-acquires, so a blocked GUI thread can lose that race many times over.
"""

from __future__ import annotations

import threading
import time

import pytest

pytestmark = [pytest.mark.gui, pytest.mark.legacy]

DWELL = 0.05          # how long the "scan" holds the model per image


class SlowModel:
    def __init__(self, dwell=DWELL):
        self.dwell = dwell
        self.calls = 0

    def __call__(self, img, verbose=False, conf=0.25):
        self.calls += 1
        time.sleep(self.dwell)
        return []


@pytest.fixture
def busy(gui_app, ds):
    """A running background scan that hogs the model lock, as the real one does."""
    app = gui_app
    app.pretrained_model = SlowModel()
    app.model_names = {0: "person"}
    app.filtered_image_paths = [ds.image("camA/0001.jpg"), ds.image("camB/0001.jpg")]

    stop = threading.Event()

    def hog():
        while not stop.is_set():
            with app._model_lock:
                time.sleep(DWELL)          # inference, lock held

    t = threading.Thread(target=hog, daemon=True)
    t.start()
    time.sleep(DWELL / 2)                  # make sure it owns the lock first
    try:
        yield app
    finally:
        stop.set()
        t.join(timeout=2)


def test_lazy_inference_gives_up_instead_of_waiting(busy, ds):
    """This is the freeze: it used to block for as long as the scan ran."""
    busy.current_image_index = 0
    busy.current_image_path = ds.image("camA/0001.jpg")
    import cv2
    busy.current_image = cv2.imread(str(busy.current_image_path))

    t0 = time.perf_counter()
    busy.load_model_predictions()
    elapsed = time.perf_counter() - t0

    assert elapsed < DWELL, f"GUI thread waited {elapsed*1000:.0f}ms on the model"
    assert busy.model_boxes == [], "no predictions yet -- they arrive on retry"


def test_navigating_stays_responsive_while_a_scan_runs(busy, ds):
    """The user's actual action: click an image mid-scan."""
    busy.current_image_index = 0
    t0 = time.perf_counter()
    busy.load_image()
    busy.root.update_idletasks()
    elapsed = time.perf_counter() - t0

    assert elapsed < DWELL * 3, f"load_image took {elapsed*1000:.0f}ms mid-scan"


def test_a_retry_is_scheduled_so_predictions_are_not_lost(busy, ds):
    busy.current_image_path = ds.image("camA/0001.jpg")
    import cv2
    busy.current_image = cv2.imread(str(busy.current_image_path))
    busy.load_model_predictions()
    assert getattr(busy, '_predict_retry_after', None) is not None


def test_predictions_arrive_once_the_model_frees_up(gui_app, ds):
    """With no scan running, the ordinary path is unchanged."""
    app = gui_app
    app.pretrained_model = SlowModel(dwell=0.0)
    app.model_names = {0: "person"}
    app.current_image_path = ds.image("camA/0001.jpg")
    import cv2
    app.current_image = cv2.imread(str(app.current_image_path))

    app.load_model_predictions()
    assert str(app.current_image_path) in app.model_predictions_cache
    assert app.pretrained_model.calls == 1


def test_background_callers_may_still_block(busy):
    """The batch worker runs off the GUI thread, where waiting is correct."""
    import numpy as np
    img = np.zeros((48, 64, 3), np.uint8)

    out = {}

    def worker():
        out['preds'] = busy._predict_image(img, blocking=True)

    t = threading.Thread(target=worker)
    t.start()
    t.join(timeout=5)
    assert out.get('preds') == [], "a blocking caller gets a real result, not None"
