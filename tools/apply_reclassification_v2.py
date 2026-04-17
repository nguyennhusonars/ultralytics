"""
Apply Reclassification Results to YOLO Dataset  (v2 — safe, basename-based)
============================================================================
Reads the checker JSON and creates NEW label files with reclassified class IDs.
**Original labels/ are NEVER moved or renamed.**

Workflow:
  1. Read results JSON  (_checker_autosave.json)
  2. Read data.yaml
  3. For each split:
       - Read from  labels/  (originals)
       - Write to   labels_new/  (with reclassified IDs)
  4. Rename:  labels → labels_original ,  labels_new → labels
  5. Write  data_reclassified.yaml  (same paths, new class list)

Key differences from v1:
  • Basename-based lookup — works across machines (RunPod → local)
  • Writes to labels_new/ first, only renames at the end
  • Skips splits with missing labels/ (prints warning, excludes from yaml)
  • Spot-check verification after writing

Usage:
    # Apply both train and val at once:
    python apply_reclassification_v2.py \\
        --yaml /path/to/data.yaml \\
        --apply train:/path/_checker_autosave_train.json \\
        --apply val:/path/_checker_autosave_val.json

    # Single split (backward compat):
    python apply_reclassification_v2.py \\
        --results /path/to/_checker_autosave.json \\
        --yaml /path/to/data.yaml --split val \\
        [--drop-skip] [--min-confidence LOW] [--dry-run]
"""

import argparse
import json
import os
import shutil
import sys
from collections import Counter, defaultdict

import yaml

CONF_ORDER = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}


def load_results(path: str) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} results from {path}")
    return data


def build_lookup(results: list[dict]) -> dict:
    """Build  (basename, line_idx) → result  dict.

    Uses basename of the label file so paths from RunPod/local
    both match the same key.
    """
    lookup: dict[tuple[str, int], dict] = {}
    for r in results:
        lbl = r.get("label_file") or r.get("label_path", "")
        idx = r.get("line_idx", -1)
        if lbl and idx >= 0:
            key = (os.path.basename(lbl), idx)
            lookup[key] = r
    return lookup


def apply(args):
    with open(args.yaml) as f:
        data_cfg = yaml.safe_load(f)

    base_dir = os.path.dirname(os.path.abspath(args.yaml))
    old_names = list(data_cfg.get("names", []))
    print(f"Original classes ({len(old_names)}): {old_names}")

    # ── Build list of (split_name, results_path) pairs ──
    split_jobs = []
    if args.apply_pairs:
        for pair in args.apply_pairs:
            if ":" not in pair:
                sys.exit(f"❌ --apply must be SPLIT:PATH, got: {pair}")
            sp, path = pair.split(":", 1)
            if not os.path.isfile(path):
                sys.exit(f"❌ Results not found: {path}")
            split_jobs.append((sp, path))
    else:
        split_jobs.append((args.split, args.results))

    # ── Load ALL results and merge into one lookup + scan new names ──
    all_results = []
    for sp, path in split_jobs:
        all_results.extend(load_results(path))

    lookup = build_lookup(all_results)
    min_conf = CONF_ORDER.get(args.min_confidence, 0)

    sample_keys = list(lookup.keys())[:3]
    print(f"Sample lookup keys: {sample_keys}")

    # Auto-discover new class names from JSON results
    used_new_names = []
    seen = set(old_names)
    for r in all_results:
        qwen_cls = r.get("qwen_class") or r.get("new_class")
        is_valid = r.get("is_correct", r.get("valid"))
        if is_valid and qwen_cls and qwen_cls != "SKIP" and qwen_cls not in seen:
            used_new_names.append(qwen_cls)
            seen.add(qwen_cls)

    new_names = list(old_names)
    for name in used_new_names:
        new_names.append(name)

    new_name_to_id = {n: i for i, n in enumerate(new_names)}
    print(f"New classes ({len(new_names)}): {new_names}")
    print(f"  Appended: {new_names[len(old_names):]}")

    stats = defaultdict(int)
    valid_splits = {}  # split_name → [valid_paths]

    # ── Process each split ──
    for split_name, _ in split_jobs:
        split_data = data_cfg.get(split_name, [])
        if isinstance(split_data, str):
            split_data = [split_data]
        if not split_data:
            print(f"\n⚠ No paths found for split '{split_name}' in data.yaml")
            continue

        print(f"\n{'='*55}")
        print(f"  Processing split: {split_name} ({len(split_data)} dirs)")
        print(f"{'='*55}")

        valid_paths = []

        for img_dir_rel in split_data:
            img_dir = os.path.join(base_dir, img_dir_rel)
            lbl_dir = img_dir.replace("/images", "/labels")
            lbl_new_dir = img_dir.replace("/images", "/labels_new")
            lbl_orig_dir = img_dir.replace("/images", "/labels_original")

            # Determine source: --from-original reads labels_original/, default reads labels/
            if args.from_original and os.path.isdir(lbl_orig_dir):
                src_dir = lbl_orig_dir
                print(f"\n  [{img_dir_rel}] Using labels_original/ as source")
            elif os.path.isdir(lbl_dir):
                src_dir = lbl_dir
                print(f"\n  [{img_dir_rel}] Using labels/ as source")
            elif os.path.isdir(lbl_orig_dir):
                src_dir = lbl_orig_dir
                print(f"\n  [{img_dir_rel}] Using labels_original/ (labels/ missing)")
            else:
                print(f"\n  ⚠ [{img_dir_rel}] No labels found — SKIPPING")
                stats["splits_skipped"] += 1
                continue

            valid_paths.append(img_dir_rel)

            if not args.dry_run:
                os.makedirs(lbl_new_dir, exist_ok=True)

            label_files = sorted(f for f in os.listdir(src_dir) if f.endswith(".txt"))
            print(f"    {len(label_files)} label files")

            for lbl_fname in label_files:
                src_path = os.path.join(src_dir, lbl_fname)
                dst_path = os.path.join(lbl_new_dir, lbl_fname)

                with open(src_path) as f:
                    lines = f.readlines()

                new_lines = []
                for line_idx, line in enumerate(lines):
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue

                    old_cls_id = int(parts[0])
                    old_cls_name = old_names[old_cls_id] if old_cls_id < len(old_names) else f"class_{old_cls_id}"

                    # Lookup by basename + line_idx
                    r = lookup.get((lbl_fname, line_idx))

                    if r is None:
                        new_lines.append(line if line.endswith("\n") else line + "\n")
                        stats["kept_untouched"] += 1
                        continue

                    qwen_cls = r.get("qwen_class") or r.get("new_class")
                    conf = r.get("qwen_confidence") or r.get("confidence", "")
                    is_valid = r.get("is_correct", r.get("valid"))

                    if qwen_cls == "SKIP":
                        if args.drop_skip:
                            stats["dropped_skip"] += 1
                            continue
                        new_lines.append(line if line.endswith("\n") else line + "\n")
                        stats["kept_skip_original"] += 1
                        continue

                    if CONF_ORDER.get(conf, 0) < min_conf:
                        new_lines.append(line if line.endswith("\n") else line + "\n")
                        stats["kept_low_conf"] += 1
                        continue

                    if is_valid and qwen_cls in new_name_to_id:
                        new_id = new_name_to_id[qwen_cls]
                        parts[0] = str(new_id)
                        new_lines.append(" ".join(parts) + "\n")
                        if qwen_cls != old_cls_name:
                            stats["reclassified"] += 1
                        else:
                            stats["kept_same_class"] += 1
                    else:
                        new_lines.append(line if line.endswith("\n") else line + "\n")
                        stats["kept_error"] += 1

                if args.dry_run:
                    if new_lines:
                        stats["files_written"] += 1
                else:
                    if new_lines:
                        with open(dst_path, "w") as f:
                            f.writelines(new_lines)
                        stats["files_written"] += 1

        valid_splits[split_name] = valid_paths

    # ── Activate: labels_new → labels (all splits) ──
    all_valid_paths = [p for paths in valid_splits.values() for p in paths]
    if not args.dry_run:
        print("\n  Activating labels_new/ → labels/ ...")
        for img_dir_rel in all_valid_paths:
            img_dir = os.path.join(base_dir, img_dir_rel)
            lbl_dir = img_dir.replace("/images", "/labels")
            lbl_new_dir = img_dir.replace("/images", "/labels_new")
            lbl_orig_dir = img_dir.replace("/images", "/labels_original")
            dir_name = os.path.basename(os.path.dirname(lbl_dir))

            if not os.path.isdir(lbl_new_dir):
                continue

            if os.path.isdir(lbl_dir) and not os.path.isdir(lbl_orig_dir):
                os.rename(lbl_dir, lbl_orig_dir)
                print(f"    {dir_name}: labels/ → labels_original/")
            elif os.path.isdir(lbl_dir):
                shutil.rmtree(lbl_dir)

            os.rename(lbl_new_dir, lbl_dir)
            print(f"    {dir_name}: labels_new/ → labels/")

    # ── Write data_reclassified.yaml (all splits, only valid paths) ──
    new_yaml = dict(data_cfg)
    new_yaml["names"] = new_names
    new_yaml["nc"] = len(new_names)
    for sp, paths in valid_splits.items():
        new_yaml[sp] = paths

    yaml_out = os.path.join(base_dir, "data_reclassified.yaml")
    if args.dry_run:
        for sp, paths in valid_splits.items():
            print(f"\n[DRY] Would write {sp}: {len(paths)} dirs")
    else:
        with open(yaml_out, "w") as f:
            yaml.dump(new_yaml, f, default_flow_style=False, allow_unicode=True)
        print(f"\n✅ New data.yaml: {yaml_out}")

    # ── Spot-check ──
    print("\n  Spot-check (looking for reclassified class IDs >= 7):")
    found = 0
    for img_dir_rel in all_valid_paths[:5]:
        if found >= 5:
            break
        img_dir = os.path.join(base_dir, img_dir_rel)
        lbl_dir = img_dir.replace("/images", "/labels")
        if not os.path.isdir(lbl_dir):
            continue
        for fname in sorted(os.listdir(lbl_dir))[:100]:
            if found >= 5:
                break
            if not fname.endswith(".txt"):
                continue
            with open(os.path.join(lbl_dir, fname)) as f:
                for line in f:
                    cid = int(line.strip().split()[0])
                    if cid >= len(old_names):
                        cname = new_names[cid] if cid < len(new_names) else f"?{cid}"
                        print(f"    ✓ {fname}: class {cid} = {cname}")
                        found += 1
                        break
    if found == 0:
        print("    ⚠️  No reclassified IDs found! Something may be wrong.")

    # ── Summary ──
    print("\n" + "=" * 55)
    print("  APPLY RECLASSIFICATION SUMMARY")
    print("=" * 55)
    print(f"New class list ({len(new_names)}):")
    for i, n in enumerate(new_names):
        marker = " *" if i >= len(old_names) else ""
        print(f"  {i:>2}: {n}{marker}")
    print()
    for k, v in sorted(stats.items()):
        print(f"  {k:<25} {v:>8}")
    total_out = sum(v for k, v in stats.items()
                    if k.startswith("reclassified") or k.startswith("kept"))
    print(f"\n  Total labels written:  {total_out}")
    print(f"  Label files created:   {stats.get('files_written', 0)}")
    for sp, paths in valid_splits.items():
        print(f"  {sp} dirs processed:   {len(paths)}")
    print(f"  Splits skipped:        {stats.get('splits_skipped', 0)}")
    if not args.dry_run:
        print(f"\n  Original labels in:  labels_original/")
        print(f"  New data.yaml:       {yaml_out}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Apply reclassification results to YOLO labels (v2 — safe)",
    )
    parser.add_argument("--apply", dest="apply_pairs", action="append",
                        metavar="SPLIT:PATH",
                        help="Apply results: e.g. --apply train:autosave_train.json --apply val:autosave_val.json")
    parser.add_argument("--results", default=None,
                        help="Path to checker JSON (single-split mode)")
    parser.add_argument("--yaml", required=True,
                        help="Path to data.yaml")
    parser.add_argument("--split", default="val",
                        help="Split name for single-split mode (default: val)")
    parser.add_argument("--from-original", action="store_true",
                        help="Read from labels_original/ instead of labels/ (for first-pass apply)")
    parser.add_argument("--drop-skip", action="store_true",
                        help="Remove SKIP entries")
    parser.add_argument("--min-confidence", default="LOW",
                        choices=["HIGH", "MEDIUM", "LOW"])
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview without writing")
    args = parser.parse_args()

    if not args.apply_pairs and not args.results:
        sys.exit("❌ Provide --apply SPLIT:PATH or --results PATH")
    if args.results and not os.path.isfile(args.results):
        sys.exit(f"❌ Results not found: {args.results}")
    if not os.path.isfile(args.yaml):
        sys.exit(f"❌ data.yaml not found: {args.yaml}")

    apply(args)


if __name__ == "__main__":
    main()
