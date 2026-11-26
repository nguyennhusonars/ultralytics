#!/usr/bin/env python3
"""Adapt bounding box aspect ratios by adjusting the lower edge.

Strategy:
- Keep original boxes for classes (hat, helmet, hood, no_helmet) whose aspect ratio (w/h)
  is within [min_ratio, max_ratio] (default: 0.8 to 1.2).
- For boxes outside this range:
  * ratio > max_ratio (wide/flat): lower edge expands downward to reach max_ratio (held at top edge).
  * ratio < min_ratio (tall/narrow): lower edge contracts upward to reach min_ratio (held at top edge).
- When a box expands downward, it is clipped if it hits the bottom of the image frame.

Workflow & Safety:
- Dry-run by default (no files modified).
- When `--apply` is specified:
  * Creates `labels_backup/` if it does not already exist so original labels are preserved.
  * Writes modified labels to `labels/`.
  * Removes stale YOLO `.cache` files.
- When `--revert` is specified:
  * Restores `labels/` from `labels_backup/` and clears `.cache` files.
- Reads pristine labels from `labels_backup/` whenever present so repeated runs are idempotent.
"""

import argparse
import shutil
import statistics
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

NAMES = ["hat", "helmet", "hood", "no_helmet"]


def adapt_lower_edge(cy, h, ratio_in, min_ratio=0.8, max_ratio=1.2, allow_shrink=True):
    """Adapt lower edge holding top edge constant so ratio falls in [min_ratio, max_ratio].

    Returns (cy_new, h_new, modified, clipped).
    """
    top = max(cy - h / 2.0, 0.0)

    if ratio_in > max_ratio:
        target_ratio = max_ratio
        want = h * (ratio_in / target_ratio)
        room = 1.0 - top
        h2 = min(want, room)
        clipped = want > room + 1e-9
        return top + h2 / 2.0, h2, True, clipped
    elif ratio_in < min_ratio and allow_shrink:
        target_ratio = min_ratio
        want = h * (ratio_in / target_ratio)
        h2 = max(want, 1e-4)
        return top + h2 / 2.0, h2, True, False
    else:
        return cy, h, False, False


def read_split(root, split, prefer_backup=True):
    """Return [(label_path, img_w, img_h, [(cls, cx, cy, w, h, ratio), ...]), ...]."""
    backup_dir = root / split / "labels_backup"
    lbl_dir = backup_dir if (prefer_backup and backup_dir.exists()) else (root / split / "labels")
    img_dir = root / split / "images"

    def load(f):
        for ext in (".jpg", ".jpeg", ".png"):
            p = img_dir / (f.stem + ext)
            if p.exists():
                try:
                    img_w, img_h = Image.open(p).size
                except Exception:
                    return None
                break
        else:
            return None
        rows = []
        for line in f.read_text().splitlines():
            q = line.split()
            if len(q) != 5:
                continue
            c = int(float(q[0]))
            cx, cy, w, h = (float(v) for v in q[1:])
            ratio = (w * img_w) / (h * img_h + 1e-12)
            rows.append((c, cx, cy, w, h, ratio))
        return f, img_w, img_h, rows

    if not lbl_dir.exists():
        return []

    with ThreadPoolExecutor(max_workers=16) as pool:
        return [r for r in pool.map(load, sorted(lbl_dir.glob("*.txt"))) if r]


def revert_split(root, split):
    """Revert labels/ from labels_backup/ if backup exists."""
    lbl_dir = root / split / "labels"
    backup = root / split / "labels_backup"
    if not backup.exists():
        print(f"{split}: no {backup.name}/ found, cannot revert")
        return False

    if lbl_dir.exists():
        shutil.rmtree(lbl_dir)
    shutil.copytree(backup, lbl_dir)
    print(f"{split}: reverted labels/ from pristine {backup.name}/")

    for cache in (lbl_dir.with_suffix(".cache"), lbl_dir.parent / "labels.cache"):
        if cache.exists():
            cache.unlink()
            print(f"{split}: removed stale {cache.name}")
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root", type=Path, help="dataset root containing train/ and valid/")
    p.add_argument("--splits", nargs="+", default=["train", "valid"], help="splits to process")
    p.add_argument("--classes", nargs="+", type=int, default=[0, 1, 2, 3], help="class ids to adapt")
    p.add_argument("--min-ratio", type=float, default=0.8, help="minimum acceptable aspect ratio w/h")
    p.add_argument("--max-ratio", type=float, default=1.2, help="maximum acceptable aspect ratio w/h")
    p.add_argument("--no-shrink", action="store_true", help="do not shrink tall boxes (ratio < min_ratio)")
    p.add_argument("--apply", action="store_true", help="write changes in place (default: dry run)")
    p.add_argument("--revert", action="store_true", help="revert labels from labels_backup/ and exit")
    args = p.parse_args()

    def name(c):
        return NAMES[c] if c < len(NAMES) else str(c)

    if args.revert:
        print("REVERTING labels to pristine backup\n")
        for split in args.splits:
            revert_split(args.root, split)
        return

    allow_shrink = not args.no_shrink
    targets = set(args.classes)

    print("=" * 70)
    print(f"Mode: {'APPLYING CHANGES' if args.apply else 'DRY RUN (pass --apply to write)'}")
    print(f"Target ratio range: [{args.min_ratio:.2f}, {args.max_ratio:.2f}]")
    print(f"Shrink tall boxes (< {args.min_ratio}): {'YES' if allow_shrink else 'NO (expand only)'}")
    print(f"Classes to check: {', '.join(name(c) for c in sorted(targets))}")
    print("=" * 70 + "\n")

    for split in args.splits:
        data = read_split(args.root, split, prefer_backup=True)
        if not data:
            print(f"{split}: no data found, skipping\n")
            continue

        backup_dir = args.root / split / "labels_backup"
        target_dir = args.root / split / "labels"

        if args.apply:
            if not backup_dir.exists():
                shutil.copytree(target_dir, backup_dir)
                print(f"{split}: backed up pristine labels to {backup_dir.name}/")
            else:
                print(f"{split}: using existing pristine {backup_dir.name}/")

        before_ratios = defaultdict(list)
        after_ratios = defaultdict(list)
        kept_count = defaultdict(int)
        modified_count = defaultdict(int)
        clipped_count = defaultdict(int)

        out_data = []

        for f, img_w, img_h, rows in data:
            out_rows = []
            for c, cx, cy, w, h, r in rows:
                before_ratios[c].append(r)
                if c in targets:
                    cy2, h2, mod, cut = adapt_lower_edge(
                        cy, h, r, min_ratio=args.min_ratio, max_ratio=args.max_ratio, allow_shrink=allow_shrink
                    )
                    r2 = (w * img_w) / (h2 * img_h + 1e-12)
                    if mod:
                        modified_count[c] += 1
                        if cut:
                            clipped_count[c] += 1
                    else:
                        kept_count[c] += 1
                else:
                    cy2, h2 = cy, h
                    r2 = r
                    kept_count[c] += 1

                after_ratios[c].append(r2)
                out_rows.append(f"{c} {cx:.6f} {cy2:.6f} {w:.6f} {h2:.6f}")

            dest_file = target_dir / f.name
            out_data.append((dest_file, out_rows))

        if args.apply:
            target_dir.mkdir(parents=True, exist_ok=True)
            for dest_file, out_rows in out_data:
                dest_file.write_text("\n".join(out_rows) + ("\n" if out_rows else ""))
            for cache in (target_dir.with_suffix(".cache"), target_dir.parent / "labels.cache"):
                if cache.exists():
                    cache.unlink()
                    print(f"{split}: removed stale {cache.name}")

        print(f"[{split.upper()}] ({len(data)} images)")
        print(
            f"  {'class':12s}{'total':>7s}{'kept':>7s}{'modified':>10s}{'clipped':>9s} | "
            f"{'med_in':>7s} -> {'med_out':>7s} | {'min_in':>7s} -> {'min_out':>7s} | {'max_in':>7s} -> {'max_out':>7s}"
        )
        print("  " + "-" * 88)

        for c in sorted(before_ratios):
            b = before_ratios[c]
            a = after_ratios[c]
            print(
                f"  {name(c):12s}{len(b):7d}{kept_count[c]:7d}{modified_count[c]:10d}{clipped_count[c]:9d} | "
                f"{statistics.median(b):7.2f} -> {statistics.median(a):7.2f} | "
                f"{min(b):7.2f} -> {min(a):7.2f} | "
                f"{max(b):7.2f} -> {max(a):7.2f}"
            )
        print()


if __name__ == "__main__":
    main()
