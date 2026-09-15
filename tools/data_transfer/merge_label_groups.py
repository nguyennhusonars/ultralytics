#!/usr/bin/env python3
"""Build a merged-class label set from a data.yaml `groups:` block.

The vehicle dataset carries two taxonomies at once: 13 fine classes in the
label files, and a `groups:` block that collapses them into 7 coarse ones. The
grouped view is what the deployed detector predicts; the fine view is what the
labels record. Training either way should not mean hand-editing 450k label
lines.

This writes a second copy of every label file with the class id rewritten to its
group index, leaving everything else on the line byte-identical -- polygons,
extra columns, spacing and line endings included. The output can be used in
place, or zipped in the same format `label_transfer.py` produces so it swaps in
through `dataset_transfer.sh import`.

Group ORDER defines the merged ids, because that is the convention the training
code already follows: model class i is the i-th group. For pvn-vehicle that is
BSD=0, BSV=1, bus=2, car=3, human=4, motorbike=5, truck=6.

Only directories named by data.yaml are touched. That matters here: the dataset
also contains `data_UK/`, which uses a DIFFERENT 15-class taxonomy (13=LP_UK,
14=pickup) and is deliberately absent from the yaml. A blind filesystem walk
would silently mis-map 11k boxes.

Usage:
    ./merge_label_groups.py DATA_YAML [--out DIR] [--zip PATH] [--dry-run]
    ./merge_label_groups.py DATA_YAML --report-only
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path

try:
    import yaml
except ImportError:                                     # pragma: no cover
    sys.exit("PyYAML is required: pip install pyyaml")


# --------------------------------------------------------------------------
# reading the dataset description
# --------------------------------------------------------------------------


def load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise SystemExit(f"{path}: not a YAML mapping")
    return data


def dataset_root(yaml_path: Path, data: dict) -> Path:
    """Base the relative entries resolve against."""
    raw = data.get("path")
    if raw:
        base = Path(raw)
        if not base.is_absolute():
            base = yaml_path.parent / base
        return base.resolve()
    return yaml_path.parent.resolve()


def class_names(data: dict) -> dict[int, str]:
    names = data.get("names")
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, list):
        return dict(enumerate(str(n) for n in names))
    raise SystemExit("data.yaml has no usable 'names'")


def build_mapping(data: dict) -> tuple[dict[int, int], list[str]]:
    """Return ({fine_id: group_id}, [group names in order]).

    Refuses anything ambiguous rather than guessing: a class in two groups, or
    a class in none, would silently relabel part of the dataset.
    """
    groups = data.get("groups")
    if not isinstance(groups, dict) or not groups:
        raise SystemExit(
            "data.yaml has no 'groups:' block, so there is nothing to merge.\n"
            "Expected e.g.  groups:\n                 bus: [bus_m, bus_l, bus_xl]")

    names = class_names(data)
    by_name = {n: i for i, n in names.items()}
    group_names = list(groups)                 # insertion order defines merged ids

    mapping: dict[int, int] = {}
    problems: list[str] = []
    for gi, gname in enumerate(group_names):
        members = groups[gname]
        if not isinstance(members, (list, tuple)):
            problems.append(f"group '{gname}' is not a list")
            continue
        for m in members:
            fid = by_name.get(str(m))
            if fid is None:
                problems.append(f"group '{gname}' names unknown class '{m}'")
                continue
            if fid in mapping:
                problems.append(
                    f"class '{m}' is in both '{group_names[mapping[fid]]}' and '{gname}'")
                continue
            mapping[fid] = gi

    missing = sorted(set(names) - set(mapping))
    if missing:
        problems.append("classes in no group: "
                        + ", ".join(f"{i} '{names[i]}'" for i in missing))
    if problems:
        raise SystemExit("Cannot build a merge mapping:\n  - "
                         + "\n  - ".join(problems))
    return mapping, group_names


def label_dirs(root: Path, data: dict) -> list[Path]:
    """Label directories for every split entry, in yaml order, deduplicated.

    Follows the ultralytics convention: a `.../images` directory pairs with the
    sibling `.../labels`.
    """
    seen: dict[Path, None] = {}
    for split in ("train", "val", "test"):
        entries = data.get(split)
        if not entries:
            continue
        if isinstance(entries, str):
            entries = [entries]
        for entry in entries:
            p = Path(entry)
            if not p.is_absolute():
                p = root / p
            lbl = p.parent / "labels" if p.name == "images" else p / "labels"
            seen.setdefault(lbl.resolve(), None)
    return list(seen)


# --------------------------------------------------------------------------
# rewriting
# --------------------------------------------------------------------------


def remap_text(text: str, mapping: dict[int, int]) -> tuple[str, Counter, Counter]:
    """Rewrite field 0 of every label line; leave the rest of the line alone.

    Returns (new_text, seen_counts, unmapped_counts). Splitting on the first
    run of whitespace and re-joining only that boundary is what keeps polygons,
    confidence columns and odd spacing byte-identical -- reformatting the
    numbers would rewrite 450k files for no reason and lose precision.
    """
    seen: Counter = Counter()
    unmapped: Counter = Counter()
    out: list[str] = []

    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if not stripped:
            out.append(line)                    # blank lines survive untouched
            continue
        head, sep, tail = stripped.partition(" ")
        try:
            fine = int(float(head))
        except ValueError:
            out.append(line)                    # not a label line; leave as-is
            continue
        seen[fine] += 1
        if fine not in mapping:
            unmapped[fine] += 1
            out.append(line)
            continue
        # Preserve the original line ending exactly.
        eol = line[len(line.rstrip("\r\n")):]
        leading = line[:len(line) - len(line.lstrip())]
        out.append(f"{leading}{mapping[fine]}{sep}{tail}{eol}")

    return "".join(out), seen, unmapped


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Build merged-class labels from a data.yaml groups: block.")
    ap.add_argument("data_yaml", type=Path)
    ap.add_argument("--out", type=Path, default=None,
                    help="staging directory for the merged labels "
                         "(default: <dataset>/labels_merged_<N>g)")
    ap.add_argument("--zip", type=Path, default=None,
                    help="also package the result as a transfer archive that "
                         "dataset_transfer.sh can import")
    ap.add_argument("--report-only", action="store_true",
                    help="show the mapping and class distribution, write nothing")
    ap.add_argument("--dry-run", action="store_true",
                    help="do all the work but write no files")
    ap.add_argument("--force", action="store_true",
                    help="merge even if the labels look already merged")
    ap.add_argument("--allow-unmapped", action="store_true",
                    help="copy lines whose class is outside the mapping instead "
                         "of refusing (they keep their original id)")
    args = ap.parse_args(argv)

    yaml_path = args.data_yaml.resolve()
    if not yaml_path.exists():
        raise SystemExit(f"not found: {yaml_path}")
    data = load_yaml(yaml_path)
    root = dataset_root(yaml_path, data)
    names = class_names(data)
    mapping, group_names = build_mapping(data)

    print(f"dataset root : {root}")
    print(f"classes      : {len(names)} -> {len(group_names)} groups\n")
    print("  merged id  group                 from")
    for gi, gname in enumerate(group_names):
        members = [f"{f} {names[f]}" for f, g in sorted(mapping.items()) if g == gi]
        print(f"    {gi:<8}  {gname:<20}  {', '.join(members)}")
    print()

    dirs = label_dirs(root, data)
    present = [d for d in dirs if d.is_dir()]
    absent = [d for d in dirs if not d.is_dir()]
    print(f"label dirs   : {len(present)} found"
          + (f", {len(absent)} MISSING" if absent else ""))
    for d in absent[:5]:
        print(f"  missing: {d}")
    if len(absent) > 5:
        print(f"  ... and {len(absent) - 5} more")
    if not present:
        raise SystemExit("No label directories found — is this the right data.yaml?")

    out_dir = args.out or (root / f"labels_merged_{len(group_names)}g")
    write = not (args.report_only or args.dry_run)

    total_seen: Counter = Counter()
    total_unmapped: Counter = Counter()
    n_files = 0
    for d in present:
        for src in sorted(d.rglob("*.txt")):
            n_files += 1
            text = src.read_text(encoding="utf-8", errors="replace")
            new_text, seen, unmapped = remap_text(text, mapping)
            total_seen.update(seen)
            total_unmapped.update(unmapped)
            if write:
                dst = out_dir / src.resolve().relative_to(root)
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_text(new_text, encoding="utf-8")

    print(f"label files  : {n_files:,}\n")

    print("  fine class                       boxes     -> merged")
    for fid in sorted(total_seen):
        gid = mapping.get(fid)
        tgt = f"{gid} {group_names[gid]}" if gid is not None else "UNMAPPED"
        print(f"    {fid:>3} {names.get(fid, '?'):<28} {total_seen[fid]:>8,}  -> {tgt}")

    merged_counts: Counter = Counter()
    for fid, n in total_seen.items():
        if fid in mapping:
            merged_counts[mapping[fid]] += n
    print("\n  merged class                     boxes")
    for gi, gname in enumerate(group_names):
        print(f"    {gi:>3} {gname:<28} {merged_counts[gi]:>8,}")

    if total_unmapped:
        detail = ", ".join(f"{k} x{v:,}" for k, v in sorted(total_unmapped.items()))
        msg = (f"\n{sum(total_unmapped.values()):,} box(es) have a class id outside "
               f"the mapping: {detail}")
        if not args.allow_unmapped:
            raise SystemExit(msg + "\n\nRefusing: these would train as the wrong "
                                   "class.\nFix the labels, or pass --allow-unmapped "
                                   "to copy them through unchanged.")
        print(msg + "\n(kept with their original ids, as requested)")

    # Refuse to merge an already-merged set. Running this twice is silently
    # catastrophic: with the 7-group mapping, a merged `3 car` would be read as
    # fine class 3 (bus_l) and rewritten to group 2 (bus). Fine labels always
    # contain at least one id >= the group count; merged ones never can.
    if total_seen and max(total_seen) < len(group_names):
        msg = (f"\nEvery class id present is below {len(group_names)}, so these "
               "labels look ALREADY MERGED.\nMerging again would relabel them "
               "wrongly (a merged 'car' would be read as a fine class and "
               "become 'bus').")
        if not args.force:
            raise SystemExit(msg + "\n\nRefusing. Restore the fine-grained "
                                   "labels first, or pass --force if you are "
                                   "certain.")
        print(msg + "\n(proceeding anyway, as requested)")

    if args.report_only:
        print("\n--report-only: nothing written.")
        return 0

    # The merged dataset's classes ARE the groups, so the yaml describes seven
    # flat classes with no groups: block -- leaving one in would invite a second
    # round of collapsing.
    merged = dict(data)
    merged["nc"] = len(group_names)
    merged["names"] = list(group_names)
    merged.pop("groups", None)
    # Pin the root. The split entries are relative, so a yaml that does not
    # say what they are relative to only works when it sits at the dataset
    # root -- and this one is written into a staging directory.
    merged["path"] = str(root)
    merged_yaml_name = f"data_{len(group_names)}g.yaml"

    if args.dry_run:
        print(f"\n--dry-run: would write {n_files:,} files to {out_dir}")
        print(f"--dry-run: would write {merged_yaml_name}")
        return 0

    (out_dir / merged_yaml_name).write_text(
        yaml.safe_dump(merged, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"\nwrote {n_files:,} label files to {out_dir}")
    print(f"wrote {out_dir / merged_yaml_name}")

    if args.zip:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # The archive must match what label_transfer.py's importer reads:
        # labels under `labels/`, yamls under `yamls/`, and metadata carrying
        # `label_files`/`yaml_files` -- it sizes the run from those keys, so
        # omitting them reports "Importing 0 labels" while importing 45k.
        label_arcs = []
        for f in sorted(out_dir.rglob("*.txt")):
            label_arcs.append({
                "archive_path": f"labels/{f.relative_to(out_dir).as_posix()}",
                "source_path": str(f),
                "size": f.stat().st_size,
            })
        # Ship the 7-class yaml AS data.yaml, so importing this archive leaves
        # the dataset self-consistent. Importing it alongside 13-class names
        # would train class 3 as 'bus_l' when it now means 'car'.
        yaml_src = out_dir / merged_yaml_name
        yaml_arcs = [{
            "archive_path": "yamls/data.yaml",
            "source_path": str(yaml_src),
            "size": yaml_src.stat().st_size,
        }]
        meta = {
            "export_date": datetime.now().isoformat(),
            "source_root": str(root),
            "source_yaml": str(yaml_path),
            "total_labels": n_files,
            "label_files": label_arcs,
            "yaml_files": yaml_arcs,
            "merged_from": len(names),
            "merged_to": len(group_names),
            "group_names": group_names,
            "mapping": {str(k): v for k, v in sorted(mapping.items())},
            "created_by": "merge_label_groups.py",
        }
        args.zip.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(args.zip, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("metadata.json", json.dumps(meta, indent=2))
            for rec in label_arcs:
                zf.write(rec["source_path"], rec["archive_path"])
            zf.write(yaml_src, "yamls/data.yaml")
        size_mb = args.zip.stat().st_size / 1e6
        print(f"wrote {args.zip} ({size_mb:.1f} MB, stamp {stamp})")

    print("\nTo train on the merged set:")
    print(f"  yolo detect train data={out_dir / merged_yaml_name} ...")
    print("To swap the merged labels into the dataset in place:")
    print(f"  ./dataset_transfer.sh import {args.zip or '<zip>'} {root}")
    print("To go back to the 13-class labels, import your backup zip the same way.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
