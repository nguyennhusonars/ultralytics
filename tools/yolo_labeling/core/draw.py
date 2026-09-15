"""Caption drawing shared by the single-image canvas and the batch grid.

Captions are the whole reason this module exists. A prediction and the
ground-truth box it belongs to sit on the *same object*, so their captions land
within a few pixels of each other -- and being the same class, they are drawn in
the same colour. As bare text on top of photographic pixels that overprints into
an unreadable smear.

Two things fix it, and both live here: a filled plate behind every caption so it
never competes with the image, and an `anchor` so the three layers claim three
different spots on the box instead of all piling onto the top-left corner.
"""

from __future__ import annotations

import cv2

FONT = cv2.FONT_HERSHEY_SIMPLEX
_PAD = 2


def _contrasting(color):
    """Black on light plates, white on dark ones."""
    r, g, b = color
    return (0, 0, 0) if (0.299 * r + 0.587 * g + 0.114 * b) > 140 else (255, 255, 255)


def draw_tag(img, text, x, y, color, scale=0.5, thickness=1, anchor="above"):
    """Draw `text` at (x, y) on a filled plate, clamped inside `img`.

    Args:
        img: image buffer, modified in place.
        color: plate colour, in the buffer's own channel order.
        anchor: ``"above"`` puts the plate's BOTTOM edge at `y`, ``"below"``
            puts its TOP edge there. This is how overlapping layers are kept
            apart -- predictions sit above the box, labels just inside its top
            edge, user boxes just inside its bottom edge.

    Returns:
        The plate's ``(left, top, right, bottom)``.
    """
    text = str(text)
    scale = max(0.25, float(scale))
    thickness = max(1, int(thickness))
    (tw, th), baseline = cv2.getTextSize(text, FONT, scale, thickness)
    pw = tw + 2 * _PAD
    ph = th + baseline + 2 * _PAD

    top = int(y) - ph if anchor == "above" else int(y)
    ih, iw = img.shape[:2]
    # Clamp, so a box against the image edge still gets a readable caption
    # rather than one drawn off-buffer or half-cut.
    left = max(0, min(int(x), iw - pw))
    top = max(0, min(top, ih - ph))

    color = tuple(int(c) for c in color[:3])
    cv2.rectangle(img, (left, top), (left + pw, top + ph), color, -1)
    cv2.putText(img, text, (left + _PAD, top + _PAD + th), FONT, scale,
                _contrasting(color), thickness, cv2.LINE_AA)
    return (left, top, left + pw, top + ph)
