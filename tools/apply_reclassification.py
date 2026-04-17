"""
Apply Reclassification Results to YOLO Dataset
===============================================
Reads the checker JSON output and creates NEW label files with reclassified
class IDs.  Old labels are preserved untouched in a separate directory.

**Only lines that appear in the results JSON are modified.**
Unchecked classes (e.g. car, motorbike, human) keep their original class IDs
exactly as they were — no remapping, no reordering.

Workflow:
  1. Reads results JSON  (_checker_autosave.json or verification_results.json)
  2. Reads the original data.yaml
  3. Renames  labels/  →  labels_original/   (backup, done once)
  4. Creates new  labels/  — checked lines get new IDs, unchecked lines stay as-is
  5. Writes  data_reclassified.yaml  with original names + appended new names

Usage:
    python apply_reclassification.py \
        --results /path/to/_checker_autosave.json \
        --yaml    /path/to/data.yaml \
        [--split val] \
        [--drop-skip]          # remove SKIP (pedestrian) lines (default: keep original) \
        [--min-confidence LOW] # LOW = accept all, MEDIUM = skip LOW, HIGH = only HIGH \
        [--dry-run]            # print what would happen, don't write
"""

import argparse
import json
import os
import shutil
import sys
from collections import defaultdict

import yaml

# Must match dataset_checker_qwen.py exactly
RECLASS_NAMES = [
    "bike", "motorbike", "sedan", "suv", "pickup", "hatchback",
    "truck_m", "bus_m", "truck_l", "bus_l", "truck_xl", "bus_xl",
]

CONF_ORDER = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}


def load_results(path: str) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} results from {path}")
    return data


def build_lookup(results: list[dict]) -> dict:
    """Build  (label_file, line_idx) → result  lookup."""
    lookup = {}
    # Handle both autosave format (label_file) and export format (label_file)
    for r in results:
        lbl = r.get("label_file") or r.get("label_path", "")
        idx = r.get("line_idx", -1)
        if lbl and idx >= 0:
            lookup[(lbl, idx)] = r
    return lookup


def apply(args):
    # ── Load data.yaml ──
    with open(args.yaml) as f:
        data_cfg = yaml.safe_load(f)

    base_dir = os.path.dirname(os.path.abspath(args.yaml))
    old_names = list(data_cfg.get("names", []))
    print(f"Original classes ({len(old_names)}): {old_names}")

    split_data = data_cfg.get(args.split, [])
    if isinstance(split_data, str):
        split_data = [split_data]

    # ── Load results ──
    results = load_results(args.results)
    lookup = build_lookup(results)
    min_conf = CONF_ORDER.get(args.min_confidence, 0)

    # ── Scan results to find which NEW class names are actually used ──
    used_new_names = set()
    for r in results:
        qwen_cls = r.get("qwen_class") or r.get("new_class")
        is_valid = r.get("is_correct", r.get("valid"))
        if is_valid and qwen_cls and qwen_cls != "SKIP" and qwen_cls in RECLASS_NAMES:
            used_new_names.add(qwen_cls)

    # ── Build new class list: original names first (preserve IDs), then append new ones ──
    new_names = list(old_names)  # keep all original names + IDs
    for name in RECLASS_NAMES:
        if name in used_new_names and name not in new_names:
            new_names.append(name)

    new_name_to_id = {n: i for i, n in enumerate(new_names)}
    print(f"New classes ({len(new_names)}): {new_names}")
    print(f"  Appended: {[n for n in new_names[len(old_names):]]}")

    # ── Stats ──
    stats = defaultdict(int)

    # ── Process each split directory ──
    for img_dir_rel in split_data:
        img_dir = os.path.join(base_dir, img_dir_rel)
        lbl_dir = img_dir.replace("/images", "/labels")
        lbl_orig_dir = img_dir.replace("/images", "/labels_original")

        if not os.path.isdir(lbl_dir):
            print(f"  ⚠ Labels dir not found: {lbl_dir}, skipping")
            continue

        # Step 1: backup  labels → labels_original
        if os.path.isdir(lbl_orig_dir):
            print(f"  ⚠ Backup dir already exists: {lbl_orig_dir}")
            print(f"    Skipping backup (using existing labels_original as source)")
            src_lbl_dir = lbl_orig_dir
        else:
            if args.dry_run:
                print(f"  [DRY] Would rename {lbl_dir} → {lbl_orig_dir}")
                src_lbl_dir = lbl_dir
            else:
                shutil.move(lbl_dir, lbl_orig_dir)
                print(f"  ✅ Backed up: {lbl_dir} → {lbl_orig_dir}")
                src_lbl_dir = lbl_orig_dir

        # Step 2: create new labels dir
        if not args.dry_run:
            os.makedirs(lbl_dir, exist_ok=True)

        # Step 3: process each label file
        label_files = sorted(f for f in os.listdir(src_lbl_dir) if f.endswith(".txt"))
        print(f"  Processing {len(label_files)} label files in {src_lbl_dir} ...")

        for lbl_fname in label_files:
            src_path = os.path.join(src_lbl_dir, lbl_fname)
            dst_path = os.path.join(lbl_dir, lbl_fname)

            # Figure out original label path (as stored in results JSON)
            # Results use the original lbl_dir path
            orig_lbl_path = os.path.join(lbl_dir.replace("/labels_original", "/labels")
                                         if src_lbl_dir == lbl_orig_dir
                                         else src_lbl_dir, lbl_fname)
            # Also try the backup path variant
            alt_lbl_path = os.path.join(lbl_orig_dir, lbl_fname) if src_lbl_dir != lbl_orig_dir else ""

            with open(src_path) as f:
                lines = f.readlines()

            new_lines = []
            for line_idx, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                old_cls_id = int(parts[0])
                old_cls_name = old_names[old_cls_id] if old_cls_id < len(old_names) else f"class_{old_cls_id}"

                # Try to find this entry in results
                r = lookup.get((orig_lbl_path, line_idx))
                if r is None and alt_lbl_path:
                    r = lookup.get((alt_lbl_path, line_idx))

                if r is None:
                    # ── NOT in results → keep line exactly as-is (untouched class) ──
                    new_lines.append(line if line.endswith("\n") else line + "\n")
                    stats["kept_untouched"] += 1
                    continue

                # ── We have a reclassification result ──
                qwen_cls = r.get("qwen_class") or r.get("new_class")
                conf = r.get("qwen_confidence") or r.get("confidence", "")
                is_valid = r.get("is_correct", r.get("valid"))

                # Handle SKIP (standalone pedestrian)
                if qwen_cls == "SKIP":
                    if args.drop_skip:
                        stats["dropped_skip"] += 1
                        continue
                    else:
                        # Keep with original class ID (don't remap)
                        new_lines.append(line if line.endswith("\n") else line + "\n")
                        stats["kept_skip_original"] += 1
                    continue

                # Check confidence threshold
                if CONF_ORDER.get(conf, 0) < min_conf:
                    # Below threshold → keep original line unchanged
                    new_lines.append(line if line.endswith("\n") else line + "\n")
                    stats["kept_low_conf"] += 1
                    continue

                # Valid reclassification → apply new class ID
                if is_valid and qwen_cls in new_name_to_id:
                    new_id = new_name_to_id[qwen_cls]
                    parts[0] = str(new_id)
                    new_lines.append(" ".join(parts) + "\n")
                    if qwen_cls != old_cls_name:
                        stats["reclassified"] += 1
                    else:
                        stats["kept_same_class"] += 1
                else:
                    # Error / invalid result → keep original line unchanged
                    new_lines.append(line if line.endswith("\n") else line + "\n")
                    stats["kept_error"] += 1

            # Write new label file
            if args.dry_run:
                if new_lines:
                    stats["files_written"] += 1
            else:
                if new_lines:
                    with open(dst_path, "w") as f:
                        f.writelines(new_lines)
                    stats["files_written"] += 1
                # If no lines left, write empty file (or skip)
                elif os.path.isfile(dst_path):
                    os.remove(dst_path)

    # ── Write new data.yaml ──
    new_yaml = dict(data_cfg)
    new_yaml["names"] = new_names
    new_yaml["nc"] = len(new_names)

    yaml_out = os.path.join(base_dir, "data_reclassified.yaml")
    if args.dry_run:
        print(f"\n[DRY] Would write new data.yaml to {yaml_out}")
    else:
        with open(yaml_out, "w") as f:
            yaml.dump(new_yaml, f, default_flow_style=False, allow_unicode=True)
        print(f"\n✅ New data.yaml: {yaml_out}")

    # ── Summary ──
    print("\n" + "=" * 55)
    print("  APPLY RECLASSIFICATION SUMMARY")
    print("=" * 55)
    print(f"New class list ({len(new_names)}):")
    for i, n in enumerate(new_names):
        print(f"  {i:>2}: {n}")
    print()
    for k, v in sorted(stats.items()):
        print(f"  {k:<25} {v:>8}")
    total_out = sum(v for k, v in stats.items()
                    if k.startswith("reclassified") or k.startswith("kept"))
    total_dropped = sum(v for k, v in stats.items() if k.startswith("dropped"))
    print(f"\n  Total labels written:  {total_out}")
    print(f"  Total labels dropped:  {total_dropped}")
    print(f"  Label files created:   {stats.get('files_written', 0)}")
    print(f"\n  Old labels preserved in: labels_original/")
    if not args.dry_run:
        print(f"  New data.yaml:          {yaml_out}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Apply reclassification results to YOLO dataset labels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Apply reclassification (only checked classes are remapped, rest untouched)
  python apply_reclassification.py \\
      --results /workspace/dataset/_checker_autosave.json \\
      --yaml /workspace/dataset/data.yaml

  # Drop pedestrian SKIP lines, only accept HIGH confidence
  python apply_reclassification.py \\
      --results /workspace/dataset/_checker_autosave.json \\
      --yaml /workspace/dataset/data.yaml \\
      --drop-skip --min-confidence HIGH

  # Preview changes without writing
  python apply_reclassification.py \\
      --results /workspace/dataset/_checker_autosave.json \\
      --yaml /workspace/dataset/data.yaml --dry-run
        """,
    )
    parser.add_argument("--results", required=True,
                        help="Path to checker JSON (autosave or export)")
    parser.add_argument("--yaml", required=True,
                        help="Path to original data.yaml")
    parser.add_argument("--split", default="val",
                        help="Dataset split to process (default: val)")
    parser.add_argument("--drop-skip", action="store_true",
                        help="Remove SKIP (pedestrian) entries instead of keeping with original class ID")
    parser.add_argument("--min-confidence", default="LOW",
                        choices=["HIGH", "MEDIUM", "LOW"],
                        help="Minimum confidence to accept reclassification (default: LOW = accept all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would happen without writing files")
    args = parser.parse_args()

    if not os.path.isfile(args.results):
        sys.exit(f"❌ Results file not found: {args.results}")
    if not os.path.isfile(args.yaml):
        sys.exit(f"❌ data.yaml not found: {args.yaml}")

    apply(args)


if __name__ == "__main__":
    main()
