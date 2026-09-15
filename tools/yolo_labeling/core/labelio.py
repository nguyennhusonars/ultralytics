"""Reading and writing YOLO label files.

Two properties drive the whole design:

1. **Untouched lines round-trip byte-for-byte.** Every ``LabelLine`` keeps the
   exact source text it came from, terminator included, and only lines that were
   actually edited are re-serialized. That is what preserves segmentation
   polygons, confidence columns, CRLF terminators and a missing final newline --
   all of which the previous 5-field parse/write pair destroyed.

2. **Writes are atomic.** Content goes to a temporary file in the same directory
   and is then moved into place with ``os.replace``, which is atomic on POSIX.
   A crash can leave the old file or the new file, never a truncated one.

Replaces seven inlined parse/serialize copies in the monolith (L1730, L2026,
L3798, L4119, L4310, L7015, L7183).
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

#: When true, `write_label_file` / `delete_label_file` log instead of touching
#: disk. Lets the tool be driven over a real dataset with zero risk.
DRY_RUN = False

#: Populated in DRY_RUN mode: (operation, path, content-or-None).
DRY_RUN_LOG: list[tuple[str, str, str | None]] = []


@dataclass
class LabelLine:
    """One line of a YOLO label file.

    Attributes:
        cls_id: Class index, or ``None`` when the line could not be parsed.
        coords: Normalized numbers after the class id -- 4 for a bounding box
            (``cx, cy, w, h``), 2N for a polygon, and possibly a trailing
            confidence value which stays in ``coords``.
        raw: The original text of the line *including* its terminator. Emitted
            verbatim unless ``dirty`` is set.
        dirty: Set by any mutation; only dirty lines get re-serialized.
    """

    cls_id: int | None
    coords: tuple[float, ...] = ()
    raw: str = ""
    dirty: bool = field(default=False, compare=False)

    @property
    def parsed(self) -> bool:
        """Whether this line yielded a usable class id and >= 4 coordinates."""
        return self.cls_id is not None and len(self.coords) >= 4

    @property
    def is_polygon(self) -> bool:
        """A polygon carries more than the 4 numbers of a bounding box."""
        return len(self.coords) > 5

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """Normalized ``(cx, cy, w, h)``.

        For a polygon this is the axis-aligned extent of its points, so a
        segmentation label can be displayed as a box without losing the polygon.
        """
        if not self.parsed:
            return (0.0, 0.0, 0.0, 0.0)
        if self.is_polygon:
            xs, ys = self.coords[0::2], self.coords[1::2]
            x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
            return ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)
        return tuple(self.coords[:4])  # type: ignore[return-value]

    def with_class(self, new_cls_id: int) -> "LabelLine":
        """Return a copy with a different class id, preserving every coordinate.

        This is how a reassignment keeps polygons and extra columns intact.
        """
        return LabelLine(int(new_cls_id), self.coords, self.raw, dirty=True)

    def with_bbox(self, cx: float, cy: float, w: float, h: float) -> "LabelLine":
        """Return a copy with new box geometry. Discards polygon points."""
        return LabelLine(self.cls_id, (cx, cy, w, h), self.raw, dirty=True)

    def render(self, precision: int = 6) -> str:
        """Serialize. Clean lines return their original text unchanged."""
        if not self.dirty:
            return self.raw
        nums = " ".join(f"{c:.{precision}f}" for c in self.coords)
        return f"{int(self.cls_id)} {nums}\n"


def parse_label_text(text: str) -> list[LabelLine]:
    """Parse label file text. Never raises on malformed input.

    A line that cannot be parsed is preserved verbatim with ``cls_id=None``, so
    it survives a rewrite instead of aborting the parse and taking the rest of
    the file with it.
    """
    out: list[LabelLine] = []
    for raw in text.splitlines(keepends=True):
        parts = raw.split()
        if len(parts) < 5:
            out.append(LabelLine(None, (), raw))          # blank/short/comment
            continue
        try:
            cls_id = int(float(parts[0]))
            coords = tuple(float(p) for p in parts[1:])
        except ValueError:
            out.append(LabelLine(None, (), raw))          # non-numeric field
            continue
        out.append(LabelLine(cls_id, coords, raw))
    return out


def render_label_text(lines: list[LabelLine], precision: int = 6) -> str:
    """Serialize lines back to file text."""
    return "".join(ln.render(precision) for ln in lines)


def read_label_file(path: str | Path) -> list[LabelLine]:
    """Parse a label file. A missing file is an empty list, not an error."""
    p = Path(path)
    if not p.exists():
        return []
    return parse_label_text(p.read_bytes().decode("utf-8", errors="replace"))


def write_label_file(
    path: str | Path, lines: list[LabelLine] | str, *, precision: int = 6
) -> None:
    """Atomically write label content.

    Writes to a temporary file alongside the target and then ``os.replace``s it
    into position, so an interrupted write can never truncate the original.
    Accepts either parsed lines or raw text.
    """
    p = Path(path)
    text = lines if isinstance(lines, str) else render_label_text(lines, precision)

    if DRY_RUN:
        DRY_RUN_LOG.append(("write", str(p), text))
        return

    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=f".{p.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)          # atomic on POSIX
    except BaseException:
        # Leave the original untouched and take the temp file with us.
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def delete_label_file(path: str | Path) -> bool:
    """Delete a label file. Returns whether anything was removed."""
    p = Path(path)
    if DRY_RUN:
        DRY_RUN_LOG.append(("delete", str(p), None))
        return p.exists()
    if not p.exists():
        return False
    p.unlink()
    return True
