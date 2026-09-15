"""Caption plates and per-layer anchoring.

The bug these guard against is visual, which usually means "not worth a test".
But the *geometry* underneath it is arithmetic: a prediction caption and its
ground-truth caption must not occupy the same pixels. That is checkable.
"""

from __future__ import annotations

import numpy as np
import pytest

from yolo_labeling.core.draw import draw_tag


def _blank(h=200, w=400):
    return np.zeros((h, w, 3), np.uint8)


def _overlap(a, b):
    """Area shared by two (l, t, r, b) rectangles."""
    lo_x, lo_y = max(a[0], b[0]), max(a[1], b[1])
    hi_x, hi_y = min(a[2], b[2]), min(a[3], b[3])
    return max(0, hi_x - lo_x) * max(0, hi_y - lo_y)


def test_the_three_layers_do_not_collide_on_one_box():
    """A prediction, its label and a user box on the SAME rectangle.

    This is the everyday case in review: all three captions used to be drawn
    at (x1, y1-5) in the same colour.
    """
    img = _blank()
    x1, y1, y2 = 50, 80, 140

    pred = draw_tag(img, "car 0.87", x1, y1, (200, 60, 60), anchor="above")
    label = draw_tag(img, "L car", x1, y1 + 2, (200, 60, 60), anchor="below")
    user = draw_tag(img, "U car", x1, y2 - 2, (200, 60, 60), anchor="above")

    assert _overlap(pred, label) == 0
    assert _overlap(pred, user) == 0
    assert _overlap(label, user) == 0


def test_above_and_below_meet_but_never_cross():
    img = _blank()
    above = draw_tag(img, "x", 10, 100, (255, 255, 255), anchor="above")
    below = draw_tag(img, "x", 10, 100, (255, 255, 255), anchor="below")
    assert above[3] == 100 and below[1] == 100, "they abut at the anchor line"
    assert _overlap(above, below) == 0


def test_plate_is_filled_so_text_never_sits_on_raw_pixels():
    """Every pixel under the caption is plate or glyph, never the photo."""
    img = _blank()
    img[:] = (17, 17, 17)                       # stand-in for image content
    x0, y0, x1, y1 = draw_tag(img, "bus_l", 20, 60, (240, 30, 30), anchor="below")
    region = img[y0:y1, x0:x1]
    assert not (region == 17).all(axis=2).any(), "no original pixel shows through"


@pytest.mark.parametrize("plate,darker", [
    ((255, 255, 255), True),    # light plate -> black text
    ((10, 10, 10), False),      # dark plate  -> white text
])
def test_text_contrasts_with_its_plate(plate, darker):
    """Glyphs are anti-aliased, so assert the DIRECTION of contrast rather
    than an exact pixel value."""
    img = _blank()
    x0, y0, x1, y1 = draw_tag(img, "WWWW", 20, 60, plate, anchor="below")
    region = img[y0:y1, x0:x1].astype(int).sum(axis=2)
    level = sum(plate)
    assert (region < level).any() if darker else (region > level).any()


@pytest.mark.parametrize("x,y,anchor", [
    (0, 0, "above"),           # top-left corner, caption would go off-buffer
    (395, 5, "above"),         # past the right edge
    (10, 199, "below"),        # past the bottom edge
    (-40, 100, "below"),       # negative origin
])
def test_captions_at_the_image_edge_stay_inside(x, y, anchor):
    """A box against the frame edge is exactly where an unclamped caption
    vanishes -- and exactly where you most need to read it."""
    img = _blank()
    x0, y0, x1, y1 = draw_tag(img, "truck_xl 0.42", x, y, (200, 200, 60), anchor=anchor)
    assert x0 >= 0 and y0 >= 0
    assert y0 < img.shape[0] and x0 < img.shape[1]
    assert (img != 0).any(), "something was actually drawn"


def test_returns_the_plate_rect():
    img = _blank()
    rect = draw_tag(img, "ab", 30, 90, (100, 100, 100), anchor="below")
    assert rect[0] == 30 and rect[1] == 90
    assert rect[2] > rect[0] and rect[3] > rect[1]
