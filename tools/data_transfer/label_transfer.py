#!/usr/bin/env python3
"""
Label Export/Import Utility for YOLO Datasets
Exports all labels while preserving relative directory structure.
Allows transferring labels to different machines/paths with different prefix.

Usage:
    # Export labels
    python label_transfer.py export /path/to/dataset --output labels_export.zip
    
    # Import labels
    python label_transfer.py import labels_export.zip /new/dataset/path
"""

import json
import os
import shutil
import zipfile
import tarfile
from pathlib import Path
from typing import Dict, List, Tuple
import argparse
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class LabelExporter:
    """Export labels from YOLO dataset while preserving relative structure."""

    def __init__(self, dataset_root: str, extra_yamls: List[str] | None = None,
                 include_yaml: bool = True):
        """
        Args:
            dataset_root: Top of the dataset tree containing label files.
            extra_yamls: Additional explicit YAML files to bundle (paths
                outside ``dataset_root`` are allowed).
            include_yaml: If True (default), also bundle any *.yaml / *.yml
                files found at the top level of ``dataset_root``. Set to
                False if you want labels only.
        """
        self.dataset_root = Path(dataset_root).resolve()
        self.include_yaml = include_yaml
        self.extra_yamls = [Path(p).resolve() for p in (extra_yamls or [])]
        self.labels_found = []
        self.structure_info = {
            "export_date": datetime.now().isoformat(),
            "source_root": str(self.dataset_root),
            "total_labels": 0,
            "subdirectories": set(),
            "files": {}
        }

    def find_labels(self) -> List[Path]:
        """Recursively find all .txt label files."""
        labels = []
        for txt_file in self.dataset_root.rglob("*.txt"):
            # Skip YAML files and other configs
            if txt_file.name.endswith('.yaml') or txt_file.name.endswith('.py'):
                continue
            labels.append(txt_file)
        return labels

    def find_yamls(self) -> List[Path]:
        """Find dataset YAML configs to ship alongside the labels.

        Sources:
        - any ``*.yaml`` / ``*.yml`` at the top level of ``dataset_root``
          (typically ``data.yaml``, ``dataset.yaml``, etc.) — disabled when
          ``include_yaml=False``,
        - any explicit paths passed via ``extra_yamls`` — always included
          regardless of ``include_yaml``.

        The data YAML is what gives meaning to class IDs (``names:``,
        ``groups:``, ``min_rel_wh:`` etc.). Including it in the archive
        makes the export self-contained: a receiver can re-train without
        guessing what id=5 means.
        """
        seen: List[Path] = []
        seen_resolved = set()
        # Auto-detection on by default; --no-yaml turns this off.
        if self.include_yaml and self.dataset_root.is_dir():
            for p in sorted(self.dataset_root.iterdir()):
                if not p.is_file():
                    continue
                if p.suffix.lower() not in {'.yaml', '.yml'}:
                    continue
                resolved = p.resolve()
                if resolved in seen_resolved:
                    continue
                seen.append(p)
                seen_resolved.add(resolved)
        # Explicit --yaml paths are always included (intent is "include this
        # specific file", not subject to the auto-detect toggle).
        for p in self.extra_yamls:
            if not p.exists() or not p.is_file():
                logger.warning(f"  --yaml: {p} not found, skipping")
                continue
            if p.resolve() in seen_resolved:
                continue
            seen.append(p)
            seen_resolved.add(p.resolve())
        return seen

    def _yaml_arcname(self, yaml_path: Path) -> str:
        """Pick a stable archive path for a YAML file.

        YAMLs that live inside ``dataset_root`` keep their relative path
        under ``yamls/``. Externally-supplied YAMLs land at
        ``yamls/<basename>``."""
        try:
            rel = yaml_path.resolve().relative_to(self.dataset_root)
            return f"yamls/{rel.as_posix()}"
        except ValueError:
            return f"yamls/{yaml_path.name}"

    def get_relative_path(self, label_file: Path) -> str:
        """Get relative path from dataset root."""
        return str(label_file.relative_to(self.dataset_root))
    
    def export_to_zip(self, output_file: str, compress: bool = True) -> Dict:
        """Export labels (and dataset YAMLs) to ZIP archive."""
        output_path = Path(output_file)

        logger.info(f"Scanning dataset: {self.dataset_root}")
        labels = self.find_labels()
        yamls = self.find_yamls()
        self.structure_info["total_labels"] = len(labels)

        logger.info(f"Found {len(labels)} label files and {len(yamls)} YAML file(s)")

        with zipfile.ZipFile(output_path, 'w',
                            compression=zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED) as zf:

            # Add metadata
            metadata = {
                "export_date": self.structure_info["export_date"],
                "source_root": str(self.dataset_root),
                "total_labels": len(labels),
                "label_files": [],
                "yaml_files": [],
            }

            for i, label_file in enumerate(labels):
                rel_path = self.get_relative_path(label_file)

                # Read label content
                with open(label_file, 'r') as f:
                    content = f.read()

                # Store in archive
                arcname = f"labels/{rel_path}"
                zf.writestr(arcname, content)

                metadata["label_files"].append({
                    "path": rel_path,
                    "size": label_file.stat().st_size,
                    "lines": len(content.strip().split('\n')) if content.strip() else 0
                })

                if (i + 1) % 10000 == 0:
                    logger.info(f"  Processed {i + 1}/{len(labels)} files...")

            # Add YAMLs (typically tiny — read them as text and write as-is)
            for y in yamls:
                arcname = self._yaml_arcname(y)
                with open(y, 'rb') as f:
                    zf.writestr(arcname, f.read())
                metadata["yaml_files"].append({
                    "archive_path": arcname,
                    "source_path": str(y.resolve()),
                    "size": y.stat().st_size,
                })
                logger.info(f"  + yaml: {arcname} ({y.stat().st_size} bytes)")

            # Write metadata.json
            zf.writestr("metadata.json", json.dumps(metadata, indent=2))

        logger.info(f"✓ Exported {len(labels)} labels + {len(yamls)} yaml(s) to: {output_path}")
        logger.info(f"  Archive size: {output_path.stat().st_size / (1024**3):.2f} GB")

        return metadata
    
    def export_to_tar(self, output_file: str, compress: str = 'gz') -> Dict:
        """Export labels (and dataset YAMLs) to TAR archive (gz, bz2, xz)."""
        output_path = Path(output_file)
        mode = f'w:{compress}' if compress else 'w'

        logger.info(f"Scanning dataset: {self.dataset_root}")
        labels = self.find_labels()
        yamls = self.find_yamls()
        self.structure_info["total_labels"] = len(labels)

        logger.info(f"Found {len(labels)} label files and {len(yamls)} YAML file(s)")

        with tarfile.open(output_path, mode) as tf:

            # Create metadata
            metadata = {
                "export_date": self.structure_info["export_date"],
                "source_root": str(self.dataset_root),
                "total_labels": len(labels),
                "label_files": [],
                "yaml_files": [],
            }

            for i, label_file in enumerate(labels):
                rel_path = self.get_relative_path(label_file)

                # Add file to archive with new path
                arcname = f"labels/{rel_path}"
                tf.add(str(label_file), arcname=arcname, recursive=False)

                metadata["label_files"].append({
                    "path": rel_path,
                    "size": label_file.stat().st_size,
                    "lines": len(open(label_file).read().strip().split('\n'))
                })

                if (i + 1) % 10000 == 0:
                    logger.info(f"  Processed {i + 1}/{len(labels)} files...")

            # Add YAMLs
            for y in yamls:
                arcname = self._yaml_arcname(y)
                tf.add(str(y), arcname=arcname, recursive=False)
                metadata["yaml_files"].append({
                    "archive_path": arcname,
                    "source_path": str(y.resolve()),
                    "size": y.stat().st_size,
                })
                logger.info(f"  + yaml: {arcname} ({y.stat().st_size} bytes)")

            # Write metadata
            metadata_json = json.dumps(metadata, indent=2)
            info = tarfile.TarInfo(name="metadata.json")
            info.size = len(metadata_json)
            tf.addfile(tarinfo=info, fileobj=__import__('io').BytesIO(metadata_json.encode()))

        logger.info(f"✓ Exported {len(labels)} labels + {len(yamls)} yaml(s) to: {output_path}")
        logger.info(f"  Archive size: {output_path.stat().st_size / (1024**3):.2f} GB")

        return metadata


class LabelImporter:
    """Import labels from exported archive to a new dataset location."""
    
    def __init__(self, archive_file: str):
        self.archive_path = Path(archive_file).resolve()
        self.metadata = None
        self._load_metadata()
    
    def _load_metadata(self):
        """Load metadata from archive."""
        if self.archive_path.suffix == '.zip':
            with zipfile.ZipFile(self.archive_path, 'r') as zf:
                self.metadata = json.loads(zf.read("metadata.json"))
        else:  # tar
            with tarfile.open(self.archive_path, 'r:*') as tf:
                self.metadata = json.loads(tf.extractfile("metadata.json").read())
    
    def import_to_path(self, target_root: str, skip_existing: bool = False,
                       dry_run: bool = False, include_yaml: bool = True) -> Dict:
        """Import labels (and YAMLs if present in the archive) to target path.

        Args:
            target_root: Destination dataset root.
            skip_existing: If True, don't overwrite existing label files OR
                YAMLs at the destination.
            dry_run: Report what would happen; don't write anything.
            include_yaml: If False, skip the bundled YAMLs even if they
                exist in the archive (useful when the target already has
                its own data.yaml that you want to keep).
        """
        target_path = Path(target_root).resolve()

        if not dry_run:
            target_path.mkdir(parents=True, exist_ok=True)

        n_labels = len(self.metadata.get('label_files', []))
        n_yamls = len(self.metadata.get('yaml_files', [])) if include_yaml else 0
        logger.info(f"Importing {n_labels} labels and {n_yamls} yaml(s) to: {target_path}")

        stats = {
            "imported": 0,
            "skipped": 0,
            "failed": 0,
            "yaml_imported": 0,
            "yaml_skipped": 0,
            "yaml_failed": 0,
            "errors": []
        }

        try:
            if self.archive_path.suffix == '.zip':
                self._import_from_zip(target_path, skip_existing, dry_run, stats,
                                      include_yaml)
            else:  # tar
                self._import_from_tar(target_path, skip_existing, dry_run, stats,
                                      include_yaml)
        except Exception as e:
            logger.error(f"✗ Import failed: {e}")
            stats["errors"].append(str(e))
            return stats

        logger.info(f"✓ Import complete:")
        logger.info(f"  Labels: imported={stats['imported']}  "
                    f"skipped={stats['skipped']}  failed={stats['failed']}")
        if n_yamls or stats['yaml_imported'] or stats['yaml_skipped']:
            logger.info(f"  YAMLs:  imported={stats['yaml_imported']}  "
                        f"skipped={stats['yaml_skipped']}  failed={stats['yaml_failed']}")

        if stats['errors']:
            logger.warning(f"  Errors: {len(stats['errors'])}")
            for err in stats['errors'][:5]:  # Show first 5 errors
                logger.warning(f"    - {err}")

        return stats
    
    def _import_from_zip(self, target_path: Path, skip_existing: bool,
                        dry_run: bool, stats: Dict, include_yaml: bool = True):
        """Import from ZIP archive."""
        with zipfile.ZipFile(self.archive_path, 'r') as zf:
            label_files = [f for f in zf.namelist() if f.startswith('labels/')]
            yaml_files = [f for f in zf.namelist()
                          if f.startswith('yamls/') and not f.endswith('/')]

            for i, arcname in enumerate(label_files):
                try:
                    rel_path = arcname.replace('labels/', '', 1)
                    target_file = target_path / rel_path

                    if skip_existing and target_file.exists():
                        stats["skipped"] += 1
                        if (i + 1) % 10000 == 0:
                            logger.info(f"  Processed {i + 1}/{len(label_files)} files...")
                        continue

                    if not dry_run:
                        target_file.parent.mkdir(parents=True, exist_ok=True)
                        with open(target_file, 'wb') as f:
                            f.write(zf.read(arcname))

                    stats["imported"] += 1

                    if (i + 1) % 10000 == 0:
                        logger.info(f"  Processed {i + 1}/{len(label_files)} files...")

                except Exception as e:
                    stats["failed"] += 1
                    stats["errors"].append(f"{arcname}: {str(e)}")

            # YAMLs go to <target_root>/<relative_path> (same scheme as labels).
            if include_yaml:
                for arcname in yaml_files:
                    try:
                        rel_path = arcname.replace('yamls/', '', 1)
                        target_file = target_path / rel_path
                        if skip_existing and target_file.exists():
                            stats["yaml_skipped"] += 1
                            logger.info(f"  yaml exists, skipping: {rel_path}")
                            continue
                        if not dry_run:
                            target_file.parent.mkdir(parents=True, exist_ok=True)
                            with open(target_file, 'wb') as f:
                                f.write(zf.read(arcname))
                        stats["yaml_imported"] += 1
                        logger.info(f"  yaml: wrote {rel_path}")
                    except Exception as e:
                        stats["yaml_failed"] += 1
                        stats["errors"].append(f"{arcname}: {str(e)}")
    
    def _import_from_tar(self, target_path: Path, skip_existing: bool,
                        dry_run: bool, stats: Dict, include_yaml: bool = True):
        """Import from TAR archive."""
        with tarfile.open(self.archive_path, 'r:*') as tf:
            members = tf.getmembers()
            label_members = [m for m in members if m.name.startswith('labels/')]
            yaml_members = [m for m in members
                            if m.name.startswith('yamls/') and m.isfile()]

            for i, member in enumerate(label_members):
                try:
                    rel_path = member.name.replace('labels/', '', 1)
                    target_file = target_path / rel_path

                    if skip_existing and target_file.exists():
                        stats["skipped"] += 1
                        if (i + 1) % 10000 == 0:
                            logger.info(f"  Processed {i + 1}/{len(label_members)} files...")
                        continue

                    if not dry_run:
                        target_file.parent.mkdir(parents=True, exist_ok=True)
                        f = tf.extractfile(member)
                        with open(target_file, 'wb') as out:
                            out.write(f.read())

                    stats["imported"] += 1

                    if (i + 1) % 10000 == 0:
                        logger.info(f"  Processed {i + 1}/{len(label_members)} files...")

                except Exception as e:
                    stats["failed"] += 1
                    stats["errors"].append(f"{member.name}: {str(e)}")

            if include_yaml:
                for member in yaml_members:
                    try:
                        rel_path = member.name.replace('yamls/', '', 1)
                        target_file = target_path / rel_path
                        if skip_existing and target_file.exists():
                            stats["yaml_skipped"] += 1
                            logger.info(f"  yaml exists, skipping: {rel_path}")
                            continue
                        if not dry_run:
                            target_file.parent.mkdir(parents=True, exist_ok=True)
                            f = tf.extractfile(member)
                            with open(target_file, 'wb') as out:
                                out.write(f.read())
                        stats["yaml_imported"] += 1
                        logger.info(f"  yaml: wrote {rel_path}")
                    except Exception as e:
                        stats["yaml_failed"] += 1
                        stats["errors"].append(f"{member.name}: {str(e)}")
    
    def list_contents(self) -> List[str]:
        """List all label files in archive."""
        if self.metadata:
            return [f["path"] for f in self.metadata["label_files"]]
        return []
    
    def verify_archive(self) -> bool:
        """Verify archive integrity."""
        try:
            if self.archive_path.suffix == '.zip':
                with zipfile.ZipFile(self.archive_path, 'r') as zf:
                    return zf.testzip() is None
            else:
                with tarfile.open(self.archive_path, 'r:*') as tf:
                    return tf.getmembers() is not None
        except Exception as e:
            logger.error(f"Archive verification failed: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Export/Import YOLO dataset labels with path prefix handling"
    )
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export labels from dataset')
    export_parser.add_argument('dataset', help='Path to dataset root directory')
    export_parser.add_argument('-o', '--output', default='labels_export.zip',
                             help='Output archive file (default: labels_export.zip)')
    export_parser.add_argument('--format', choices=['zip', 'tar', 'tar.gz', 'tar.bz2', 'tar.xz'],
                             default='zip', help='Archive format (default: zip)')
    export_parser.add_argument('--no-compress', action='store_true',
                             help='Disable compression (zip only)')
    export_parser.add_argument('--no-yaml', action='store_true',
                             help='Do NOT include dataset YAML files in the archive '
                                  '(by default any *.yaml/*.yml at the dataset root '
                                  'is bundled so class names / groups / filter '
                                  'settings travel with the labels).')
    export_parser.add_argument('--yaml', action='append', default=None,
                             metavar='PATH',
                             help='Additional explicit YAML file(s) to bundle '
                                  '(repeatable). Use when the data.yaml lives '
                                  'outside the dataset root.')

    # Import command
    import_parser = subparsers.add_parser('import', help='Import labels to new location')
    import_parser.add_argument('archive', help='Archive file to import from')
    import_parser.add_argument('target', help='Target dataset root directory')
    import_parser.add_argument('--skip-existing', action='store_true',
                             help='Skip labels and YAMLs that already exist at target')
    import_parser.add_argument('--dry-run', action='store_true',
                             help='Show what would be imported without doing it')
    import_parser.add_argument('--no-yaml', action='store_true',
                             help='Skip bundled YAMLs even if present in the '
                                  'archive (useful when the target already has '
                                  'its own data.yaml you want to keep).')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show archive information')
    info_parser.add_argument('archive', help='Archive file to inspect')
    info_parser.add_argument('--list-files', action='store_true',
                           help='List all files in archive')
    info_parser.add_argument('--verify', action='store_true',
                           help='Verify archive integrity')
    
    args = parser.parse_args()
    
    if args.command == 'export':
        exporter = LabelExporter(
            args.dataset,
            extra_yamls=args.yaml,
            include_yaml=not args.no_yaml,
        )

        if args.format == 'zip':
            exporter.export_to_zip(args.output, compress=not args.no_compress)
        else:
            compress_format = args.format.split('.')[-1] if '.' in args.format else ''
            exporter.export_to_tar(args.output, compress=compress_format)

    elif args.command == 'import':
        importer = LabelImporter(args.archive)
        importer.import_to_path(
            args.target,
            skip_existing=args.skip_existing,
            dry_run=args.dry_run,
            include_yaml=not args.no_yaml,
        )

    elif args.command == 'info':
        importer = LabelImporter(args.archive)
        logger.info(f"Archive: {args.archive}")
        logger.info(f"Source: {importer.metadata.get('source_root', 'N/A')}")
        logger.info(f"Export Date: {importer.metadata.get('export_date', 'N/A')}")
        logger.info(f"Total Labels: {importer.metadata.get('total_labels', 0)}")
        yamls = importer.metadata.get('yaml_files', []) or []
        logger.info(f"Total YAMLs: {len(yamls)}")
        for y in yamls:
            logger.info(f"  - {y.get('archive_path', '?')} "
                        f"(src: {y.get('source_path', '?')}, "
                        f"{y.get('size', 0)} bytes)")

        if args.verify:
            is_valid = importer.verify_archive()
            logger.info(f"Archive Valid: {'✓ Yes' if is_valid else '✗ No'}")

        if args.list_files:
            files = importer.list_contents()
            logger.info(f"\nLabel files ({len(files)} total):")
            for f in files[:20]:
                logger.info(f"  {f}")
            if len(files) > 20:
                logger.info(f"  ... and {len(files) - 20} more")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
