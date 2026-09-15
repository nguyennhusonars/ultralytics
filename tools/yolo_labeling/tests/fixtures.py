"""Synthetic dataset builder and change-detection helpers.

Everything is generated programmatically -- no binary fixtures are committed.
The whole tree builds in well under a second.

The dataset deliberately encodes every edge case the audits turned up, so a
single `build_dataset()` covers: multi-folder layouts, same-stem collisions
across folders, `startswith` prefix collisions, segmentation polygons, extra
columns, malformed lines, unreadable images, uppercase extensions, EXIF
orientation, unicode names, and a label with no image.
"""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image

# --------------------------------------------------------------------------
# label text constants -- referenced directly by tests, so keep them exact
# --------------------------------------------------------------------------

CAMA_0001 = "0 0.500000 0.500000 0.200000 0.300000\n1 0.250000 0.250000 0.100000 0.100000\n"
CAMB_0001 = "2 0.750000 0.750000 0.150000 0.150000\n"
WIDE = "0 0.500000 0.500000 1.000000 1.000000\n"

#: Two polygons: 4-point (9 fields) and 5-point (11 fields).
POLY1 = (
    "0 0.10 0.10 0.20 0.10 0.20 0.20 0.10 0.20\n"
    "1 0.30 0.30 0.45 0.30 0.45 0.50 0.38 0.55 0.30 0.50\n"
)
#: One ordinary bbox line followed by one polygon line.
MIXED = "0 0.50 0.50 0.20 0.20\n1 0.60 0.60 0.70 0.60 0.70 0.70 0.60 0.70\n"

#: Two perfectly valid boxes sitting next to an UNREADABLE image. If the tool
#: deletes this file, that is the headline data-loss bug.
BAD_CORRUPT = "0 0.400000 0.400000 0.200000 0.200000\n1 0.600000 0.600000 0.100000 0.100000\n"
BAD_EMPTY_IMG = "0 0.500000 0.500000 0.300000 0.300000\n"

#: Every malformed-line shape we must survive. Note: no trailing newline on the
#: last line -- that is itself a case.
GAUNTLET = (
    "1 0.5 0.5 0.2 0.2\n"           # valid
    "99 0.5 0.5 0.1 0.1\n"          # class id past nc
    "-1 0.5 0.5 0.1 0.1\n"          # negative class id
    "0 1.4 0.5 0.2 0.2\n"           # cx > 1
    "0 0.5 0.5\n"                   # too few fields
    "0 0.5 0.5 0.2 0.2 0.95\n"      # 6 fields (confidence column)
    "0 0.5 0.5 0.2 0.2\r\n"         # CRLF
    "abc 0.5 0.5 0.2 0.2\n"         # non-numeric class
    "\n"                            # blank
    "   \n"                         # whitespace only
    "0 0.5 0.5 0.2 0.2"             # NO trailing newline
)

CLASS_NAMES = ["person", "car", "bus_m", "bus_l"]


# --------------------------------------------------------------------------
# builders
# --------------------------------------------------------------------------


def _img(path: Path, w: int = 64, h: int = 48) -> None:
    """Write a small JPEG with two coloured rectangles (visually distinct)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    a = np.full((h, w, 3), 40, np.uint8)
    cv2.rectangle(a, (w // 8, h // 8), (w // 2, h // 2), (0, 180, 0), -1)
    cv2.rectangle(a, (w // 2, h // 2), (w - w // 8, h - h // 8), (0, 0, 180), -1)
    cv2.imwrite(str(path), a)


def _exif_rotated(path: Path, w: int = 64, h: int = 48) -> None:
    """JPEG carrying EXIF orientation=6, so PIL and cv2 disagree on dimensions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.new("RGB", (w, h), (90, 90, 90))
    exif = im.getexif()
    exif[274] = 6  # Orientation tag
    buf = io.BytesIO()
    im.save(buf, format="JPEG", exif=exif)
    path.write_bytes(buf.getvalue())


def _txt(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))


def read_exact(path: Path) -> str:
    """Read text WITHOUT universal-newline translation.

    `Path.read_text()` silently rewrites `\\r\\n` to `\\n`, which would make a
    byte-identity round-trip assertion pass against a file the tool had in fact
    rewritten. Always use this when comparing label content exactly.
    """
    return Path(path).read_bytes().decode("utf-8")


@dataclass
class DatasetFixture:
    """Handle onto a generated dataset tree."""

    root: Path

    @property
    def yaml(self) -> Path:
        return self.root / "data.yaml"

    @property
    def config(self) -> Path:
        return self.root / "class_config.json"

    def image(self, rel: str) -> Path:
        """`image("camA/0001.jpg")` -> `<root>/camA/images/0001.jpg`."""
        folder, name = rel.rsplit("/", 1)
        return self.root / folder / "images" / name

    def label(self, rel: str) -> Path:
        """`label("camA/0001")` -> `<root>/camA/labels/0001.txt`."""
        folder, name = rel.rsplit("/", 1)
        return self.root / folder / "labels" / f"{Path(name).stem}.txt"

    def all_labels(self) -> list[Path]:
        return sorted(self.root.rglob("*.txt"))

    def snapshot(self) -> dict[str, str]:
        return snapshot(self.root)


def build_dataset(root: Path) -> DatasetFixture:
    """Materialise the full edge-case dataset under `root`."""
    root.mkdir(parents=True, exist_ok=True)

    # -- camA: the ordinary case, plus an empty label and a missing label ----
    _img(root / "camA/images/0001.jpg")
    _txt(root / "camA/labels/0001.txt", CAMA_0001)
    _img(root / "camA/images/0002.jpg")
    _txt(root / "camA/labels/0002.txt", "")           # valid negative sample
    _img(root / "camA/images/0003.jpg")               # no label file at all
    _img(root / "camA/images/0004.JPG")               # UPPERCASE extension

    # -- camB: same stem as camA (cross-contamination trap) + odd geometry ---
    _img(root / "camB/images/0001.jpg")
    _txt(root / "camB/labels/0001.txt", CAMB_0001)
    _img(root / "camB/images/wide.jpg", w=100, h=10)  # extreme aspect
    _txt(root / "camB/labels/wide.txt", WIDE)
    _exif_rotated(root / "camB/images/rot.jpg")
    _txt(root / "camB/labels/rot.txt", CAMB_0001)

    # -- frames: parent NOT named 'images', and no labels dir anywhere -------
    _img(root / "frames/x0001.jpg")
    _img(root / "frames/x0002.jpg")

    # -- seg: polygons that must survive byte-identically -------------------
    _img(root / "seg/images/poly1.jpg")
    _txt(root / "seg/labels/poly1.txt", POLY1)
    _img(root / "seg/images/mixed.jpg")
    _txt(root / "seg/labels/mixed.txt", MIXED)

    # -- bad: unreadable images guarding VALID labels ------------------------
    (root / "bad/images").mkdir(parents=True, exist_ok=True)
    (root / "bad/images/corrupt.jpg").write_bytes(b"this is definitely not an image")
    _txt(root / "bad/labels/corrupt.txt", BAD_CORRUPT)
    (root / "bad/images/empty.jpg").write_bytes(b"")
    _txt(root / "bad/labels/empty.txt", BAD_EMPTY_IMG)
    _img(root / "bad/images/ok.jpg")
    _txt(root / "bad/labels/ok.txt", GAUNTLET)

    # -- startswith prefix collision: set1 is a prefix of set10 --------------
    _img(root / "ds_set1/images/a.jpg")
    _txt(root / "ds_set1/labels/a.txt", "0 0.100000 0.100000 0.050000 0.050000\n")
    _img(root / "ds_set10/images/a.jpg")
    _txt(root / "ds_set10/labels/a.txt", "1 0.900000 0.900000 0.050000 0.050000\n")

    # -- unicode + space in the stem ----------------------------------------
    _img(root / "weird/images/имя пробел.jpg")
    _txt(root / "weird/labels/имя пробел.txt", "0 0.500000 0.500000 0.100000 0.100000\n")

    # -- a label with no corresponding image --------------------------------
    _txt(root / "labels_only/labels/ghost.txt", "0 0.500000 0.500000 0.100000 0.100000\n")

    # -- dataset descriptors -------------------------------------------------
    (root / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(root),
                "train": ["camA/images", "camB/images", "frames"],
                "val": ["seg/images"],
                "nc": len(CLASS_NAMES),
                "names": CLASS_NAMES,
            },
            sort_keys=False,   # group/name order is semantically meaningful
        ),
        encoding="utf-8",
    )
    (root / "class_config.json").write_text(
        json.dumps(
            {
                str(i): {"name": n, "color": [0, 255, 0], "hotkey": str(i + 1)}
                for i, n in enumerate(CLASS_NAMES)
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return DatasetFixture(root)


# --------------------------------------------------------------------------
# change detection -- the single most valuable assertion in the suite
# --------------------------------------------------------------------------


def snapshot(root: Path) -> dict[str, str]:
    """Map every label file under `root` to the sha256 of its bytes.

    Missing files are simply absent from the mapping, so creation and deletion
    both show up as a difference.
    """
    out: dict[str, str] = {}
    for p in sorted(Path(root).rglob("*.txt")):
        out[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def diff_snapshots(before: dict[str, str], after: dict[str, str]) -> dict[str, str]:
    """Return `{relpath: 'added'|'removed'|'modified'}` for everything that moved."""
    changed: dict[str, str] = {}
    for k in before.keys() | after.keys():
        if k not in after:
            changed[k] = "removed"
        elif k not in before:
            changed[k] = "added"
        elif before[k] != after[k]:
            changed[k] = "modified"
    return changed


def assert_only_changed(
    before: dict[str, str], after: dict[str, str], expected: set[str]
) -> None:
    """Assert exactly `expected` label files changed.

    Pass `set()` -- the common case -- to assert nothing was written at all.
    Nearly every data-loss bug in this tool is "a file changed that shouldn't
    have", so this is the default assertion for navigation, filtering, batch
    browsing and crop review.
    """
    changed = diff_snapshots(before, after)
    unexpected = {k: v for k, v in changed.items() if k not in expected}
    missing = expected - changed.keys()
    msg = []
    if unexpected:
        msg.append(f"unexpectedly changed: {unexpected}")
    if missing:
        msg.append(f"expected to change but did not: {sorted(missing)}")
    assert not msg, "; ".join(msg)
