#!/usr/bin/env python3
"""Convert CrowdHuman .odgt annotations to YOLO-format labels.

CrowdHuman annotates a single category (person), carrying three boxes each: hbox (head), vbox (visible body) and
fbox (full/amodal body). Separate `mask` entries mark regions deliberately left unannotated. This emits a rich
class set so the head/body/ignore decisions can be deferred to training time via the `groups` argument:

    0 head            clean head
    1 body            clean visible body (vbox)
    2 head_flagged    head with ignore/unsure set, or head of an ignored person
    3 body_flagged    visible body of an ignored person
    4 fullbody        full amodal body (fbox) of a clean person
    5 ignore_region   `mask` region -- real people, deliberately unlabeled

Typical use:
    groups={head: [0], body: [1]}          # the standard head + visible-body setup
    groups={head: [0,2], body: [1,3]}      # include uncertain annotations
    groups={head: [0], body: [4]}          # standard CrowdHuman full-body protocol

Never group 1 and 4 together: that emits two boxes per person.
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from pathlib import Path

from PIL import Image

NAMES = ["head", "body", "head_flagged", "body_flagged", "fullbody", "ignore_region"]


def to_yolo(box, w_img, h_img):
    """Clip an [x, y, w, h] box to the image and return normalized (cx, cy, w, h), or None if degenerate."""
    x, y, w, h = box
    x1, y1 = min(max(x, 0), w_img), min(max(y, 0), h_img)
    x2, y2 = min(max(x + w, 0), w_img), min(max(y + h, 0), h_img)
    bw, bh = x2 - x1, y2 - y1
    if bw <= 1 or bh <= 1:  # fully outside the image, or collapsed to a line
        return None
    vals = ((x1 + x2) / 2 / w_img, (y1 + y2) / 2 / h_img, bw / w_img, bh / h_img)
    return tuple(min(max(v, 0.0), 1.0) for v in vals)


def convert(record, w_img, h_img, stats):
    """Turn one .odgt record into a list of YOLO label lines."""
    lines = []
    for b in record["gtboxes"]:
        head_attr, extra = b.get("head_attr") or {}, b.get("extra") or {}
        if b.get("tag") == "mask":
            boxes = [(5, b["hbox"])]  # hbox == fbox for every mask entry
        else:
            person_ignored = bool(extra.get("ignore"))
            head_flagged = person_ignored or bool(head_attr.get("ignore")) or bool(head_attr.get("unsure"))
            boxes = [(2 if head_flagged else 0, b["hbox"]), (3 if person_ignored else 1, b["vbox"])]
            if not person_ignored:
                boxes.append((4, b["fbox"]))
        for cls, box in boxes:
            if (v := to_yolo(box, w_img, h_img)) is None:
                stats["dropped_degenerate"] += 1
                continue
            stats[f"class_{cls}_{NAMES[cls]}"] += 1
            lines.append(f"{cls} {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {v[3]:.6f}")
    return lines


def run_split(root, split, workers):
    """Convert one split, writing <split>/labels/<ID>.txt for every record."""
    img_dir, lbl_dir = root / split / "images", root / split / "labels"
    lbl_dir.mkdir(parents=True, exist_ok=True)
    records = [json.loads(line) for line in (root / f"annotation_{split}.odgt").open()]

    def size_of(rec):
        try:
            return rec, Image.open(img_dir / f"{rec['ID']}.jpg").size
        except Exception:
            return rec, None

    stats = Counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for rec, size in pool.map(size_of, records):
            if size is None:
                stats["missing_image"] += 1
                continue
            lines = convert(rec, *size, stats)
            (lbl_dir / f"{rec['ID']}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            stats["images"] += 1
            if not lines:
                stats["empty_background_images"] += 1
    return stats


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root", type=Path, help="CrowdHuman root containing annotation_*.odgt and train/ val/")
    p.add_argument("--workers", type=int, default=16, help="threads for reading image dimensions")
    args = p.parse_args()

    for split in ("train", "val"):
        stats = run_split(args.root, split, args.workers)
        print(f"\n{split}:")
        for k in sorted(stats):
            print(f"  {k:28s} {stats[k]}")

    yaml = args.root / "data.yaml"
    yaml.write_text(
        f"path: {args.root.resolve()}\ntrain: train/images\nval: val/images\n\nnames:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES))
    )
    print(f"\nWrote {yaml}")


if __name__ == "__main__":
    main()
