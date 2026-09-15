"""The 'Find Bad Annotations' background scan.

Its defining hazard is that it runs model inference on a worker thread while
the GUI thread infers lazily on every image change. Ultralytics/torch models
are not safe to call concurrently, so both paths must serialise on one lock.
"""

from __future__ import annotations

import threading
import time

import pytest

pytestmark = [pytest.mark.gui, pytest.mark.legacy]


class ReentrancyDetector:
    """A fake model that records whether it was ever called concurrently."""

    names = {0: "person", 2: "car"}

    def __init__(self, dwell=0.02):
        self.dwell = dwell
        self._inside = 0
        self._guard = threading.Lock()
        self.max_concurrent = 0
        self.calls = 0

    def __call__(self, img, conf=0.25, **k):
        with self._guard:
            self._inside += 1
            self.calls += 1
            self.max_concurrent = max(self.max_concurrent, self._inside)
        time.sleep(self.dwell)          # widen the window a real model would have
        with self._guard:
            self._inside -= 1
        return []


def test_model_is_never_called_from_two_threads_at_once(gui_app):
    """Without the lock this reliably reports max_concurrent == 2."""
    import numpy as np

    detector = ReentrancyDetector()
    gui_app.pretrained_model = detector
    gui_app.model_names = dict(detector.names)
    img = np.zeros((48, 64, 3), np.uint8)

    def hammer():
        for _ in range(15):
            with gui_app._model_lock:
                gui_app.pretrained_model(img, conf=0.25)

    threads = [threading.Thread(target=hammer) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert detector.calls == 60
    assert detector.max_concurrent == 1, (
        f"model was entered by {detector.max_concurrent} threads at once")


def test_lazy_gui_inference_takes_the_lock(gui_app):
    """`_predict_image` is the GUI-thread path; it must serialise too."""
    import numpy as np

    detector = ReentrancyDetector(dwell=0.01)
    gui_app.pretrained_model = detector
    gui_app.model_names = dict(detector.names)
    img = np.zeros((48, 64, 3), np.uint8)

    stop = threading.Event()

    def background():
        while not stop.is_set():
            with gui_app._model_lock:
                gui_app.pretrained_model(img, conf=0.25)

    t = threading.Thread(target=background, daemon=True)
    t.start()
    try:
        for _ in range(10):
            gui_app._predict_image(img)          # the real GUI-thread entry point
    finally:
        stop.set()
        t.join(timeout=5)

    assert detector.max_concurrent == 1, "lazy GUI inference raced the worker"


def test_lock_is_shared_between_app_and_crop_review(gui_app):
    """The mixin reaches `self._model_lock`; one object, one lock."""
    assert isinstance(gui_app._model_lock, type(threading.Lock()))
    assert gui_app._model_lock is gui_app._model_lock


def test_a_failing_model_releases_the_lock(gui_app):
    """An exception inside inference must not deadlock every later call."""
    import numpy as np

    class Boom:
        names = {}

        def __call__(self, *a, **k):
            raise RuntimeError("cuda oom")

    gui_app.pretrained_model = Boom()
    gui_app._predict_image(np.zeros((48, 64, 3), np.uint8))

    assert gui_app._model_lock.acquire(timeout=1), "lock was left held after a failure"
    gui_app._model_lock.release()
