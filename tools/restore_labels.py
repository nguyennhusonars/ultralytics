"""
Restore / Clean Up YOLO Dataset Labels
=======================================
Scans for labels/ and labels_original/ dirs, shows status,
and can restore originals or clean up the mess.

Usage:
    # Just scan and show status
    python restore_labels.py --yaml /path/to/data.yaml

    # Restore: labels_original → labels (delete current labels/)
    python restore_labels.py --yaml /path/to/data.yaml --restore

    # Clean up: remove labels_original/ (keep current labels/)
    python restore_labels.py --yaml /path/to/data.yaml --cleanup

    # Process all splits (train + val + test)
    python restore_labels.py --yaml /path/to/data.yaml --restore --split all
"""

import argparse
import os
import shutil
import sys

import yaml


def scan_dataset(yaml_path: str, splits: list[str]):
    """Scan and report the state of labels dirs."""
    with open(yaml_path) as f:
        data_cfg = yaml.safe_load(f)

    base_dir = os.path.dirname(os.path.abspath(yaml_path))
    names = data_cfg.get("names", [])
    print(f"Dataset: {yaml_path}")
    print(f"Classes ({len(names)}): {names}")
    print()

    dirs_info = []

    for split in splits:
        split_data = data_cfg.get(split, [])
        if not split_data:
            continue
        if isinstance(split_data, str):
            split_data = [split_data]

        for img_dir_rel in split_data:
            img_dir = os.path.join(base_dir, img_dir_rel)
            lbl_dir = img_dir.replace("/images", "/labels")
            lbl_orig_dir = img_dir.replace("/images", "/labels_original")

            has_labels = os.path.isdir(lbl_dir)
            has_orig = os.path.isdir(lbl_orig_dir)
            has_images = os.path.isdir(img_dir)

            n_images = len([f for f in os.listdir(img_dir) if f.lower().endswith(
                (".jpg", ".jpeg", ".png", ".bmp", ".webp"))]) if has_images else 0
            n_labels = len([f for f in os.listdir(lbl_dir) if f.endswith(".txt")]) if has_labels else 0
            n_orig = len([f for f in os.listdir(lbl_orig_dir) if f.endswith(".txt")]) if has_orig else 0

            info = {
                "split": split,
                "img_dir": img_dir,
                "lbl_dir": lbl_dir,
                "lbl_orig_dir": lbl_orig_dir,
                "has_images": has_images,
                "has_labels": has_labels,
                "has_orig": has_orig,
                "n_images": n_images,
                "n_labels": n_labels,
                "n_orig": n_orig,
            }
            dirs_info.append(info)

            # Status display
            status = "✅ Clean" if has_labels and not has_orig else ""
            if has_labels and has_orig:
                status = "⚠️  BOTH labels/ and labels_original/ exist"
            elif not has_labels and has_orig:
                status = "❌ labels/ MISSING, only labels_original/ exists"
            elif not has_labels and not has_orig:
                status = "❌ NO labels found"

            print(f"[{split}] {os.path.basename(os.path.dirname(lbl_dir))}/")
            print(f"  images/          : {'✅' if has_images else '❌'} ({n_images} files)")
            print(f"  labels/          : {'✅' if has_labels else '❌'} ({n_labels} files)")
            print(f"  labels_original/ : {'✅' if has_orig else '—'} ({n_orig} files)")
            print(f"  Status: {status}")

            # Check if labels look reclassified (class IDs > original nc)
            if has_labels and n_labels > 0:
                nc = len(names)
                sample_file = os.path.join(lbl_dir, sorted(os.listdir(lbl_dir))[0])
                max_id = -1
                with open(sample_file) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cid = int(parts[0])
                            max_id = max(max_id, cid)
                if max_id >= nc:
                    print(f"  ⚠️  labels/ has class IDs up to {max_id} (original nc={nc}) → likely reclassified")
                else:
                    print(f"  Class IDs in range 0-{max_id} (original nc={nc}) → looks original")

            if has_orig and n_orig > 0:
                nc = len(names)
                sample_file = os.path.join(lbl_orig_dir, sorted(os.listdir(lbl_orig_dir))[0])
                max_id = -1
                with open(sample_file) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cid = int(parts[0])
                            max_id = max(max_id, cid)
                if max_id >= nc:
                    print(f"  ⚠️  labels_original/ has class IDs up to {max_id} (nc={nc}) → NOT truly original!")
                else:
                    print(f"  labels_original/ IDs in range 0-{max_id} → looks like true originals")
            print()

    return dirs_info


def restore(dirs_info: list[dict], dry_run: bool):
    """Restore labels_original → labels."""
    print("=" * 55)
    print("  RESTORING ORIGINAL LABELS")
    print("=" * 55)

    for info in dirs_info:
        lbl_dir = info["lbl_dir"]
        lbl_orig_dir = info["lbl_orig_dir"]

        if not info["has_orig"]:
            print(f"\n[{info['split']}] No labels_original/ found — nothing to restore")
            continue

        print(f"\n[{info['split']}] {os.path.basename(os.path.dirname(lbl_dir))}/")

        # Remove current labels/
        if info["has_labels"]:
            if dry_run:
                print(f"  [DRY] Would remove {lbl_dir}/ ({info['n_labels']} files)")
            else:
                shutil.rmtree(lbl_dir)
                print(f"  🗑 Removed {lbl_dir}/ ({info['n_labels']} files)")

        # Move labels_original → labels
        if dry_run:
            print(f"  [DRY] Would rename {lbl_orig_dir}/ → {lbl_dir}/")
        else:
            shutil.move(lbl_orig_dir, lbl_dir)
            print(f"  ✅ Restored: {lbl_orig_dir}/ → {lbl_dir}/ ({info['n_orig']} files)")

    if not dry_run:
        # Also remove data_reclassified.yaml if it exists
        for info in dirs_info:
            reclass_yaml = os.path.join(os.path.dirname(info["lbl_dir"]).replace("/val", "").replace("/train", "").replace("/test", ""),
                                        "..", "data_reclassified.yaml")
            # Try common location
            base = os.path.dirname(os.path.dirname(info["lbl_dir"]))
            for candidate in [os.path.join(base, "data_reclassified.yaml"),
                              os.path.join(base, "..", "data_reclassified.yaml")]:
                if os.path.isfile(candidate):
                    os.remove(candidate)
                    print(f"  🗑 Removed {candidate}")
                    break

    print("\n✅ Restore complete. Your dataset is back to the original state.")


def cleanup(dirs_info: list[dict], dry_run: bool):
    """Remove labels_original/ dirs (keep current labels/)."""
    print("=" * 55)
    print("  CLEANING UP labels_original/")
    print("=" * 55)

    for info in dirs_info:
        lbl_orig_dir = info["lbl_orig_dir"]

        if not info["has_orig"]:
            print(f"\n[{info['split']}] No labels_original/ — already clean")
            continue

        print(f"\n[{info['split']}] {os.path.basename(os.path.dirname(lbl_orig_dir))}/")

        if dry_run:
            print(f"  [DRY] Would remove {lbl_orig_dir}/ ({info['n_orig']} files)")
        else:
            shutil.rmtree(lbl_orig_dir)
            print(f"  🗑 Removed {lbl_orig_dir}/ ({info['n_orig']} files)")

    print("\n✅ Cleanup complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Scan, restore, or clean up YOLO dataset labels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--yaml", required=True, help="Path to data.yaml")
    parser.add_argument("--split", default="val",
                        help="Split to process (train/val/test/all). Default: val")
    parser.add_argument("--restore", action="store_true",
                        help="Restore labels_original → labels (deletes current labels/)")
    parser.add_argument("--cleanup", action="store_true",
                        help="Remove labels_original/ (keeps current labels/)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would happen without changing files")
    args = parser.parse_args()

    if not os.path.isfile(args.yaml):
        sys.exit(f"❌ Not found: {args.yaml}")

    if args.split == "all":
        splits = ["train", "val", "test"]
    else:
        splits = [args.split]

    dirs_info = scan_dataset(args.yaml, splits)

    if not dirs_info:
        print("No split directories found.")
        return

    if args.restore:
        if not args.dry_run:
            confirm = input("\n⚠️  This will DELETE current labels/ and restore from labels_original/.\n"
                          "Type 'yes' to confirm: ")
            if confirm.strip().lower() != "yes":
                print("Cancelled.")
                return
        restore(dirs_info, args.dry_run)
    elif args.cleanup:
        if not args.dry_run:
            confirm = input("\n⚠️  This will DELETE labels_original/ (keeping current labels/).\n"
                          "Type 'yes' to confirm: ")
            if confirm.strip().lower() != "yes":
                print("Cancelled.")
                return
        cleanup(dirs_info, args.dry_run)
    else:
        print("\nUse --restore to restore originals, or --cleanup to remove labels_original/")


if __name__ == "__main__":
    main()
