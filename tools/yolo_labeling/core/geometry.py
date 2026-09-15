"""Pure geometry helpers. No I/O, no tkinter, no global state.

Consolidates implementations that were duplicated across the monolith:

* IoU -- `calculate_iou` (L7077) and `_iou_xyxy` (L3894), numerically identical.
* YOLO -> pixel -- four copies (L1735, L4322, L7022, L7186).
* pixel -> YOLO -- two copies (L2027, L7500).

The conversions take explicit `rounding` and `clip` arguments rather than baking
in a policy. During the extraction phase every call site passes the *legacy*
values (``rounding="trunc", clip=False``) so behaviour is bit-identical; the
fixes are then a visible one-line change at each call site instead of being
smuggled inside a code move.
"""

from __future__ import annotations

Box = tuple[float, float, float, float]


def iou_xyxy(a: Box, b: Box) -> float:
    """Intersection-over-union of two ``(x1, y1, x2, y2)`` boxes.

    Degenerate and non-overlapping boxes give ``0.0`` rather than raising.
    """
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    a_area = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    b_area = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = a_area + b_area - inter
    return inter / union if union > 0 else 0.0


def _quantize(v: float, rounding: str) -> int:
    if rounding == "trunc":
        return int(v)      # truncates toward zero -- the legacy behaviour
    if rounding == "round":
        return int(round(v))
    raise ValueError(f"rounding must be 'trunc' or 'round', got {rounding!r}")


def yolo_to_pixel(
    cx: float,
    cy: float,
    w: float,
    h: float,
    img_w: int,
    img_h: int,
    *,
    rounding: str = "round",
    clip: bool = True,
) -> tuple[int, int, int, int]:
    """Normalized centre/size -> integer pixel ``(x1, y1, x2, y2)``.

    Args:
        rounding: ``"round"`` (correct) or ``"trunc"`` (legacy ``int()``, which
            floors both edges and so shrinks every box by up to a pixel).
        clip: clamp the result to the image rectangle.

    A zero-sized image yields an all-zero box instead of dividing by zero.
    """
    if img_w <= 0 or img_h <= 0:
        return (0, 0, 0, 0)
    x1 = _quantize((cx - w / 2) * img_w, rounding)
    y1 = _quantize((cy - h / 2) * img_h, rounding)
    x2 = _quantize((cx + w / 2) * img_w, rounding)
    y2 = _quantize((cy + h / 2) * img_h, rounding)
    if clip:
        x1, x2 = max(0, min(x1, img_w)), max(0, min(x2, img_w))
        y1, y2 = max(0, min(y1, img_h)), max(0, min(y2, img_h))
    return (x1, y1, x2, y2)


def pixel_to_yolo(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    img_w: int,
    img_h: int,
    *,
    clip: bool = True,
) -> tuple[float, float, float, float]:
    """Pixel ``(x1, y1, x2, y2)`` -> normalized ``(cx, cy, w, h)``.

    Args:
        clip: clamp every component into ``[0, 1]``. The legacy save path did
            not, so a paste across differently-sized images could write values
            like ``2.73`` straight to disk.
    """
    if img_w <= 0 or img_h <= 0:
        return (0.0, 0.0, 0.0, 0.0)
    lo_x, hi_x = min(x1, x2), max(x1, x2)
    lo_y, hi_y = min(y1, y2), max(y1, y2)
    cx = (lo_x + hi_x) / (2 * img_w)
    cy = (lo_y + hi_y) / (2 * img_h)
    w = (hi_x - lo_x) / img_w
    h = (hi_y - lo_y) / img_h
    if clip:
        cx, cy = min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0)
        w, h = min(max(w, 0.0), 1.0), min(max(h, 0.0), 1.0)
    return (cx, cy, w, h)


def normalize_box(
    x1: float, y1: float, x2: float, y2: float, img_w: int, img_h: int
) -> tuple[int, int, int, int]:
    """Order the corners and clamp them to the image (monolith L1902-1927)."""
    lo_x, hi_x = min(x1, x2), max(x1, x2)
    lo_y, hi_y = min(y1, y2), max(y1, y2)
    return (
        int(max(0, min(lo_x, img_w))),
        int(max(0, min(lo_y, img_h))),
        int(max(0, min(hi_x, img_w))),
        int(max(0, min(hi_y, img_h))),
    )


def box_area(box: Box) -> float:
    """Non-negative area of an ``(x1, y1, x2, y2)`` box."""
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def polygon_bounds(coords) -> Box:
    """Axis-aligned extent of a flat ``[x1, y1, x2, y2, ...]`` polygon.

    Used to render a segmentation label as a box without discarding the
    underlying polygon.
    """
    xs, ys = coords[0::2], coords[1::2]
    if not xs or not ys:
        return (0.0, 0.0, 0.0, 0.0)
    return (min(xs), min(ys), max(xs), max(ys))
