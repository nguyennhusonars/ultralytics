"""The durable prediction cache.

Scanning 46k images is a coffee-break operation. Every property here exists to
make sure that work is spent once: reused across scans, across threshold
changes, across class-map edits, and across restarts -- while still being thrown
away when it is genuinely stale.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from yolo_labeling.core.predcache import (
    CONF_FLOOR,
    MAX_BOXES,
    PredictionCache,
    clip_boxes,
    filter_conf,
    model_key,
)

BOXES = [
    (3, 10.0, 20.0, 30.0, 40.0, 0.90),
    (5, 11.0, 21.0, 31.0, 41.0, 0.55),
    (2, 12.0, 22.0, 32.0, 42.0, 0.15),
]


@pytest.fixture
def cache(tmp_path):
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"weights")
    c = PredictionCache(tmp_path / "c.sqlite", weights)
    c.weights = weights
    yield c
    c.close()


@pytest.fixture
def img(tmp_path):
    p = tmp_path / "frame.jpg"
    p.write_bytes(b"x" * 512)
    return p


# -- store and narrow ------------------------------------------------------


def test_round_trips_boxes(cache, img):
    cache.put(img, BOXES)
    got = cache.get(img)
    assert [b[0] for b in got] == [3, 5, 2]
    assert got[0][1:5] == (10.0, 20.0, 30.0, 40.0)


@pytest.mark.parametrize("conf,expected", [
    (None, 3), (0.10, 3), (0.50, 2), (0.80, 1), (0.95, 0),
])
def test_one_entry_serves_every_threshold(cache, img, conf, expected):
    """The scan at 0.40 and the canvas at 0.25 share this row."""
    cache.put(img, BOXES)
    assert len(cache.get(img, conf)) == expected


def test_a_threshold_below_the_floor_is_a_miss(cache, img):
    """We never inferred below the floor, so we must not pretend to know."""
    cache.put(img, BOXES)
    assert cache.get(img, CONF_FLOOR - 0.01) is None


def test_below_floor_boxes_are_not_stored(cache, img):
    cache.put(img, BOXES + [(1, 0.0, 0.0, 1.0, 1.0, 0.01)])
    assert len(cache.get(img)) == 3


def test_box_count_is_capped_keeping_the_strongest(cache, img):
    """A low floor on a busy frame can return the detector's NMS maximum."""
    many = [(0, 1.0, 2.0, 3.0, 4.0, 0.10 + i * 0.001) for i in range(MAX_BOXES + 50)]
    cache.put(img, many)
    got = cache.get(img)
    assert len(got) == MAX_BOXES
    assert min(b[5] for b in got) > min(b[5] for b in many), "weakest dropped first"


# -- invalidation ----------------------------------------------------------


def test_a_changed_image_invalidates_its_entry(cache, img):
    cache.put(img, BOXES)
    assert cache.get(img) is not None
    time.sleep(1.1)                      # mtime has 1s resolution here
    img.write_bytes(b"y" * 999)
    assert cache.get(img) is None, "re-exported frame must be re-inferred"


def test_a_retrained_checkpoint_invalidates_everything(tmp_path, img):
    """Same filename, new weights -- the old predictions are worthless."""
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"v1")
    c1 = PredictionCache(tmp_path / "c.sqlite", weights)
    c1.put(img, BOXES)
    c1.close()

    time.sleep(1.1)
    weights.write_bytes(b"v2-retrained-longer")
    c2 = PredictionCache(tmp_path / "c.sqlite", weights)
    try:
        assert c2.get(img) is None
    finally:
        c2.close()


def test_two_models_do_not_share_entries(tmp_path, img):
    a = tmp_path / "a.pt"
    a.write_bytes(b"a")
    b = tmp_path / "b.pt"
    b.write_bytes(b"b")
    ca, cb = PredictionCache(tmp_path / "c.sqlite", a), PredictionCache(tmp_path / "c.sqlite", b)
    try:
        ca.put(img, BOXES)
        assert ca.get(img) is not None
        assert cb.get(img) is None
        assert ca.count() == 1 and cb.count() == 0
    finally:
        ca.close()
        cb.close()


def test_a_missing_image_is_a_miss_not_a_crash(cache, tmp_path):
    assert cache.get(tmp_path / "nope.jpg") is None
    cache.put(tmp_path / "nope.jpg", BOXES)      # must not raise
    assert cache.count() == 0


def test_a_corrupt_row_is_a_miss_not_a_crash(cache, img):
    cache.put(img, BOXES)
    with cache._lock:
        cache._db.execute("UPDATE preds SET boxes='{{{not json'")
        cache._db.commit()
    assert cache.get(img) is None


# -- durability ------------------------------------------------------------


def test_survives_a_restart(tmp_path, img):
    """The headline: reopening must not mean re-inferring."""
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"weights")
    db = tmp_path / "c.sqlite"

    first = PredictionCache(db, weights)
    first.put(img, BOXES)
    first.close()

    second = PredictionCache(db, weights)
    try:
        assert len(second.get(img)) == 3
    finally:
        second.close()


def test_uncommitted_writes_survive_a_reopen_after_commit(tmp_path, img):
    """The scan writes with commit=False and flushes periodically."""
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"w")
    db = tmp_path / "c.sqlite"

    c = PredictionCache(db, weights)
    c.put(img, BOXES, commit=False)
    c.commit()
    c.close()

    c2 = PredictionCache(db, weights)
    try:
        assert c2.get(img) is not None
    finally:
        c2.close()


def test_clear_drops_only_this_model(tmp_path, img):
    a = tmp_path / "a.pt"
    a.write_bytes(b"a")
    b = tmp_path / "b.pt"
    b.write_bytes(b"bb")
    ca, cb = PredictionCache(tmp_path / "c.sqlite", a), PredictionCache(tmp_path / "c.sqlite", b)
    try:
        ca.put(img, BOXES)
        cb.put(img, BOXES)
        ca.clear()
        assert ca.count() == 0 and cb.count() == 1
    finally:
        ca.close()
        cb.close()


# -- pure helpers ----------------------------------------------------------


def test_filter_conf_leaves_unscored_boxes_alone():
    """Legacy 5-tuples carry no score; dropping them would lose real boxes."""
    assert len(filter_conf([(0, 1, 2, 3, 4)], 0.9)) == 1


def test_clip_boxes_is_stable_when_nothing_is_dropped():
    assert clip_boxes(BOXES) == BOXES


def test_model_key_handles_a_missing_file():
    assert model_key(Path("/nonexistent/x.pt")) == "x.pt"
    assert model_key(None) == "<none>"


def test_model_key_tracks_content(tmp_path):
    w = tmp_path / "m.pt"
    w.write_bytes(b"one")
    first = model_key(w)
    time.sleep(1.1)
    w.write_bytes(b"two-different-length")
    assert model_key(w) != first


def test_db_is_created_under_a_missing_directory(tmp_path):
    w = tmp_path / "m.pt"
    w.write_bytes(b"m")
    nested = tmp_path / "a" / "b" / "c.sqlite"
    c = PredictionCache(nested, w)
    try:
        assert nested.exists()
    finally:
        c.close()
    assert os.path.getsize(nested) > 0


# -- dimension reads must not decode ---------------------------------------


def test_image_size_reads_the_header_not_the_pixels(tmp_path, monkeypatch):
    """`ImageOps.exif_transpose()` returns a *transformed copy*, so calling it
    for dimensions decodes and rotates the whole frame -- measured slower than
    `cv2.imread`. Everything that only wants (w, h) must stay header-only.
    """
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from fixtures import build_dataset

    ds = build_dataset(tmp_path / "ds")

    from yolo_labeling.ui import filters as _filters

    decoded = []
    real_imread = _filters.cv2.imread
    monkeypatch.setattr(_filters.cv2, "imread",
                        lambda *a, **k: (decoded.append(a[0]), real_imread(*a, **k))[1])

    probe = type("P", (_filters.FiltersMixin,), {})()
    probe.image_dimensions_cache = {}

    assert probe._image_size(str(ds.image("camA/0001.jpg"))) == (64, 48)
    assert decoded == [], "a dimension read must not decode the image"


def test_image_size_still_honours_exif_rotation(tmp_path):
    """The fast path must give the same answer as the slow one it replaced."""
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from fixtures import build_dataset

    ds = build_dataset(tmp_path / "ds")

    from yolo_labeling.ui import filters as _filters

    probe = type("P", (_filters.FiltersMixin,), {})()
    probe.image_dimensions_cache = {}

    rotated = str(ds.image("camB/rot.jpg"))          # EXIF orientation 6
    import cv2 as _cv2
    h, w = _cv2.imread(rotated).shape[:2]
    assert probe._image_size(rotated) == (w, h), "must agree with the decoder"
