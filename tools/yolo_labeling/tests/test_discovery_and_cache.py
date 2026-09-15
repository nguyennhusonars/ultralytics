"""Phase 5 fixes: image discovery, cache lifecycle, overlap detection."""

from __future__ import annotations

import pytest

from fixtures import snapshot, assert_only_changed

pytestmark = pytest.mark.legacy


# -- case-insensitive image discovery --------------------------------------


def test_uppercase_extensions_are_discovered(ds, monolith):
    """`0004.JPG` used to be invisible to the lowercase-only glob."""
    found = {p.name for p in monolith.YOLOLabelingTool._iter_images(ds.root / "camA" / "images")}
    assert "0004.JPG" in found
    assert "0001.jpg" in found


def test_discovery_ignores_non_images(ds, monolith):
    (ds.root / "camA" / "images" / "notes.txt").write_text("x")
    found = {p.name for p in monolith.YOLOLabelingTool._iter_images(ds.root / "camA" / "images")}
    assert "notes.txt" not in found


def test_discovery_is_not_recursive(ds, monolith):
    """Matches the previous glob semantics -- one directory only."""
    found = {p.name for p in monolith.YOLOLabelingTool._iter_images(ds.root / "camA")}
    assert found == set(), "images live in camA/images, not camA"


# -- get_label_boxes now populates the cache -------------------------------


def test_get_label_boxes_populates_labels_cache(ds, legacy):
    """It used to READ the cache but never write it, so every filter pass
    re-parsed every label file in the dataset."""
    p = legacy()
    key = str(ds.image("camA/0001.jpg"))
    assert key not in p.app.labels_cache

    boxes = p.app.get_label_boxes(key, str(ds.label("camA/0001")))
    assert len(boxes) == 2
    assert key in p.app.labels_cache, "second call must hit the cache"


def test_second_call_returns_the_cached_value_without_touching_disk(ds, legacy):
    p = legacy()
    key = str(ds.image("camA/0001.jpg"))
    first = p.app.get_label_boxes(key, str(ds.label("camA/0001")))
    ds.label("camA/0001").unlink()                 # cache must survive this
    assert p.app.get_label_boxes(key, str(ds.label("camA/0001"))) == first


def test_get_label_boxes_is_read_only(ds, legacy):
    before = snapshot(ds.root)
    p = legacy()
    for rel, lab in (("camA/0001.jpg", "camA/0001"), ("seg/poly1.jpg", "seg/poly1"),
                     ("bad/ok.jpg", "bad/ok")):
        p.app.get_label_boxes(str(ds.image(rel)), str(ds.label(lab)))
    assert_only_changed(before, snapshot(ds.root), set())


def test_malformed_file_yields_the_valid_boxes_not_an_empty_list(ds, legacy):
    """The bare `except: pass` used to make a partly-bad file look unlabelled."""
    p = legacy()
    boxes = p.app.get_label_boxes(str(ds.image("bad/ok.jpg")), str(ds.label("bad/ok")))
    assert len(boxes) >= 6, "valid lines survive a malformed neighbour"


# -- EXIF consistency ------------------------------------------------------


def test_image_size_honours_exif_orientation(ds, legacy):
    """`_image_size` must agree with cv2 (which applies EXIF), not raw PIL."""
    import cv2

    p = legacy()
    path = str(ds.image("camB/rot.jpg"))
    h, w = cv2.imread(path).shape[:2]
    assert p.app._image_size(path) == (w, h)


def test_image_size_is_cached(ds, legacy):
    p = legacy()
    path = str(ds.image("camA/0001.jpg"))
    assert p.app._image_size(path) == (64, 48)
    assert path in p.app.image_dimensions_cache


def test_image_size_of_unreadable_image_is_none(ds, legacy):
    p = legacy()
    assert p.app._image_size(str(ds.image("bad/corrupt.jpg"))) is None


# -- overlap detection -----------------------------------------------------


def test_overlaps_are_detected_between_different_classes(legacy):
    p = legacy()
    p.app.filter_overlap_threshold = 0.3
    p.app.label_boxes = [(0, 0, 0, 10, 10), (1, 2, 2, 12, 12)]
    p.app.detect_current_overlaps()
    assert p.app.overlapping_boxes == [0, 1]


def test_same_class_boxes_are_not_flagged(legacy):
    p = legacy()
    p.app.filter_overlap_threshold = 0.3
    p.app.label_boxes = [(0, 0, 0, 10, 10), (0, 1, 1, 11, 11)]
    p.app.detect_current_overlaps()
    assert p.app.overlapping_boxes == []


def test_indices_are_unique_and_sorted(legacy):
    """The old list-membership version could be O(b^3); this pins the contract."""
    p = legacy()
    p.app.filter_overlap_threshold = 0.1
    p.app.label_boxes = [(i % 2, 0, 0, 10, 10) for i in range(6)]
    p.app.detect_current_overlaps()
    got = p.app.overlapping_boxes
    assert got == sorted(set(got)) == [0, 1, 2, 3, 4, 5]


def test_no_boxes_is_not_an_error(legacy):
    p = legacy()
    p.app.label_boxes = []
    p.app.detect_current_overlaps()
    assert p.app.overlapping_boxes == []


def test_many_overlapping_boxes_stay_fast(legacy):
    """200 mutually-overlapping boxes: O(b^2)=40k is instant, O(b^3)=8M is not."""
    import time

    p = legacy()
    p.app.filter_overlap_threshold = 0.1
    p.app.label_boxes = [(i % 2, 0, 0, 10, 10) for i in range(200)]
    t0 = time.perf_counter()
    p.app.detect_current_overlaps()
    assert time.perf_counter() - t0 < 1.0
    assert len(p.app.overlapping_boxes) == 200
