#!/usr/bin/env python3
"""
Dataset Path Remapper
Updates YAML config files with new dataset path prefix.
Useful when transferring datasets to different machines/locations.

Usage:
    python dataset_path_remapper.py config.yaml /old/path /new/path --output config_updated.yaml
"""

import yaml
from pathlib import Path
import argparse
import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class DatasetPathRemapper:
    """Remap dataset paths in YAML configuration files."""
    
    def __init__(self, yaml_file: str):
        self.yaml_path = Path(yaml_file)
        self.config = self._load_yaml()
    
    def _load_yaml(self) -> Dict:
        """Load YAML file."""
        with open(self.yaml_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _remap_path(self, path_str: str, old_prefix: str, new_prefix: str) -> str:
        """Remap a single path string."""
        if isinstance(path_str, str) and old_prefix in path_str:
            return path_str.replace(old_prefix, new_prefix, 1)
        return path_str
    
    def _remap_paths_recursive(self, obj: Any, old_prefix: str, new_prefix: str) -> Any:
        """Recursively remap paths in nested structures."""
        if isinstance(obj, dict):
            return {k: self._remap_paths_recursive(v, old_prefix, new_prefix) 
                    for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._remap_paths_recursive(item, old_prefix, new_prefix) 
                    for item in obj]
        elif isinstance(obj, str):
            return self._remap_path(obj, old_prefix, new_prefix)
        else:
            return obj
    
    def remap(self, old_prefix: str, new_prefix: str, dry_run: bool = False) -> Dict:
        """Remap all paths in config."""
        logger.info(f"Remapping paths:")
        logger.info(f"  From: {old_prefix}")
        logger.info(f"  To:   {new_prefix}")
        
        remapped_config = self._remap_paths_recursive(self.config, old_prefix, new_prefix)
        
        # Count changes
        changes = self._count_changes(self.config, remapped_config)
        logger.info(f"  Changes detected: {changes}")
        
        if not dry_run:
            self.config = remapped_config
        
        return remapped_config
    
    def _count_changes(self, orig: Any, remapped: Any) -> int:
        """Count how many path changes were made."""
        count = 0
        if isinstance(orig, dict) and isinstance(remapped, dict):
            for k in orig:
                count += self._count_changes(orig.get(k), remapped.get(k))
        elif isinstance(orig, list) and isinstance(remapped, list):
            for o, r in zip(orig, remapped):
                count += self._count_changes(o, r)
        elif isinstance(orig, str) and isinstance(remapped, str) and orig != remapped:
            count += 1
        return count
    
    def save(self, output_file: str):
        """Save remapped config to file."""
        output_path = Path(output_file)
        with open(output_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"✓ Saved to: {output_path}")
    
    def show_config(self, key: str = None):
        """Display config or specific key."""
        if key:
            if key in self.config:
                logger.info(f"{key}: {self.config[key]}")
            else:
                logger.warning(f"Key '{key}' not found")
        else:
            logger.info(yaml.dump(self.config, default_flow_style=False, sort_keys=False))


def main():
    parser = argparse.ArgumentParser(
        description="Remap dataset paths in YAML configuration files"
    )
    
    parser.add_argument('yaml_file', help='Path to YAML config file')
    parser.add_argument('old_prefix', help='Old path prefix to replace')
    parser.add_argument('new_prefix', help='New path prefix')
    parser.add_argument('-o', '--output', help='Output file (if not specified, updates original)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would change without modifying')
    parser.add_argument('--show', action='store_true',
                       help='Display the config after remapping')
    
    args = parser.parse_args()
    
    remapper = DatasetPathRemapper(args.yaml_file)
    
    remapped = remapper.remap(args.old_prefix, args.new_prefix, dry_run=args.dry_run)
    
    if args.show:
        logger.info("\nRemapped config:")
        remapper.show_config()
    
    if not args.dry_run:
        output = args.output or args.yaml_file
        remapper.save(output)


if __name__ == "__main__":
    main()
