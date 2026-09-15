"""Unit tests for core.geometry -- pure, fast, no fixtures needed."""

from __future__ import annotations

import pytest

from core.geometry import (
    box_area,
    iou_xyxy,
    normalize_box,
    pixel_to_yolo,
    polygon_bounds,
    yolo_to_pixel,
)


# -- IoU -------------------------------------------------------------------


@pytest.mark.parametrize(
    "a,b,expected",
    [
        ((0, 0, 10, 10), (0, 0, 10, 10), 1.0),        # identical
        ((0, 0, 10, 10), (20, 20, 30, 30), 0.0),      # disjoint
        ((0, 0, 10, 10), (10, 10, 20, 20), 0.0),      # touching corners only
        ((0, 0, 10, 10), (5, 0, 15, 10), 1 / 3),      # half overlap
        ((0, 0, 10, 10), (0, 0, 5, 10), 0.5),         # contained
    ],
)
def test_iou(a, b, expected):
    assert iou_xyxy(a, b) == pytest.approx(expected)


@pytest.mark.parametrize(
    "a,b",
    [
        ((0, 0, 0, 0), (0, 0, 0, 0)),      # both degenerate
        ((5, 5, 5, 5), (0, 0, 10, 10)),    # zero-area inside a real box
        ((10, 10, 0, 0), (0, 0, 10, 10)),  # inverted corners
    ],
)
def test_iou_degenerate_returns_zero_not_error(a, b):
    assert iou_xyxy(a, b) == 0.0


def test_iou_is_symmetric():
    a, b = (0, 0, 10, 10), (3, 4, 14, 12)
    assert iou_xyxy(a, b) == pytest.approx(iou_xyxy(b, a))


# -- yolo -> pixel ---------------------------------------------------------


def test_full_frame_box_maps_to_the_whole_image():
    assert yolo_to_pixel(0.5, 0.5, 1.0, 1.0, 100, 100) == (0, 0, 100, 100)


def test_degenerate_box_collapses_to_a_point():
    assert yolo_to_pixel(0.5, 0.5, 0.0, 0.0, 100, 100) == (50, 50, 50, 50)


def test_zero_sized_image_does_not_divide_by_zero():
    assert yolo_to_pixel(0.5, 0.5, 0.2, 0.2, 0, 0) == (0, 0, 0, 0)
    assert pixel_to_yolo(0, 0, 10, 10, 0, 0) == (0.0, 0.0, 0.0, 0.0)


def test_out_of_range_input_is_clipped_when_asked():
    """cx=1.4 pushes x past the right edge; y stays legitimately at 40..60."""
    assert yolo_to_pixel(1.4, 0.5, 0.2, 0.2, 100, 100) == (100, 40, 100, 60)
    assert yolo_to_pixel(1.4, 0.5, 0.2, 0.2, 100, 100, clip=False) == (130, 40, 150, 60)


def test_trunc_and_round_differ_by_at_most_one_pixel():
    """The legacy int() floors both edges; round() is centred."""
    t = yolo_to_pixel(0.5, 0.5, 0.2, 0.3, 64, 48, rounding="trunc", clip=False)
    r = yolo_to_pixel(0.5, 0.5, 0.2, 0.3, 64, 48, rounding="round", clip=False)
    assert t == (25, 16, 38, 31)
    assert r == (26, 17, 38, 31)
    assert all(abs(x - y) <= 1 for x, y in zip(t, r))


def test_rounding_mode_is_validated():
    with pytest.raises(ValueError, match="rounding"):
        yolo_to_pixel(0.5, 0.5, 0.2, 0.2, 10, 10, rounding="floor")


# -- pixel -> yolo ---------------------------------------------------------


def test_pixel_to_yolo_clamps_to_the_unit_interval():
    for v in pixel_to_yolo(-50, -50, 5000, 5000, 64, 48):
        assert 0.0 <= v <= 1.0


def test_pixel_to_yolo_without_clip_preserves_overflow():
    """Reproduces the paste-across-sizes bug when clipping is off."""
    cx, _, w, _ = pixel_to_yolo(0, 0, 3500, 100, 640, 480, clip=False)
    assert w > 1.0 and cx > 1.0


def test_inverted_corners_are_ordered():
    assert pixel_to_yolo(30, 30, 10, 10, 100, 100) == pixel_to_yolo(10, 10, 30, 30, 100, 100)


def test_round_trip_is_exact_with_round_and_aligned_input():
    """Pixel -> yolo -> pixel returns the original box."""
    original = (10, 20, 50, 40)
    cx, cy, w, h = pixel_to_yolo(*original, 100, 100)
    assert yolo_to_pixel(cx, cy, w, h, 100, 100) == original


@pytest.mark.parametrize("box", [(0, 0, 64, 48), (1, 1, 63, 47), (25, 16, 38, 31)])
def test_round_trip_is_stable_over_many_cycles(box):
    """With round(), repeated save/load cycles must not drift at all."""
    W, H = 64, 48
    cur = box
    for _ in range(25):
        cur = yolo_to_pixel(*pixel_to_yolo(*cur, W, H), W, H)
    assert cur == box


# -- helpers ---------------------------------------------------------------


def test_normalize_box_orders_and_clamps():
    assert normalize_box(30, 40, 10, 5, 64, 48) == (10, 5, 30, 40)
    assert normalize_box(-10, -10, 500, 500, 64, 48) == (0, 0, 64, 48)


def test_box_area():
    assert box_area((0, 0, 10, 5)) == 50
    assert box_area((10, 10, 0, 0)) == 0      # inverted -> no negative area


def test_polygon_bounds():
    assert polygon_bounds([0.1, 0.1, 0.2, 0.1, 0.2, 0.2, 0.1, 0.2]) == (0.1, 0.1, 0.2, 0.2)
    assert polygon_bounds([]) == (0.0, 0.0, 0.0, 0.0)
