# Dataset Label Transfer Guide

Complete solution for exporting and importing YOLO labels across different machines and paths.

## Problem Solved

When working with large datasets on different machines:
- **Images remain unchanged** (no need to copy 15GB)
- **Labels can be exported/imported easily** (only ~100MB for 85K files)
- **Paths automatically adjusted** when dataset location changes
- **Relative structure preserved** so any prefix works

## Quick Start

### 1. Export Labels from Source Machine

```bash
cd /path/to/ultralytics_sonnn/tools/data_transfer

# Export to ZIP (recommended, faster)
python label_transfer.py export /home/sonnn/Videos/tmp_dataset/pvn-dataset/pvn-vehicle/251222 \
    --output labels_20250514.zip

# Or export to compressed TAR
python label_transfer.py export /home/sonnn/Videos/tmp_dataset/pvn-dataset/pvn-vehicle/251222 \
    --output labels_20250514.tar.gz --format tar.gz
```

**Time**: ~2-5 minutes for 85K files depending on disk speed

### 2. Transfer Archive to New Machine

```bash
# Copy the small archive file
scp labels_20250514.zip user@newmachine:/path/to/

# Or use cloud storage, USB drive, etc.
```

### 3. Import Labels on Target Machine

```bash
# Dry-run first to see what will happen
python label_transfer.py import labels_20250514.zip \
    /path/to/new/dataset/location --dry-run

# Actually import
python label_transfer.py import labels_20250514.zip \
    /path/to/new/dataset/location

# Import and skip existing labels (for updates)
python label_transfer.py import labels_20250514.zip \
    /path/to/new/dataset/location --skip-existing
```

### 4. Update YAML Config Files (Optional)

If your data.yaml or other config files reference full paths:

```bash
# Update paths in YAML
python dataset_path_remapper.py data.yaml \
    "/home/sonnn/Videos/tmp_dataset" \
    "/mnt/data/dataset" \
    --output data_newpath.yaml

# Or update in-place (with backup)
cp data.yaml data.yaml.bak
python dataset_path_remapper.py data.yaml \
    "/home/sonnn/Videos/tmp_dataset" \
    "/mnt/data/dataset"
```

## Complete Workflow Example

### Scenario: Transfer dataset from Windows to Linux machine

**Machine A (Windows) - Export:**
```bash
python label_transfer.py export "C:\Users\Data\Dataset" --output labels.zip
# File size: ~100-200MB (depending on dataset)
```

**Transfer:**
```bash
# Upload to Google Drive, Dropbox, or SCP
```

**Machine B (Linux) - Import:**
```bash
python label_transfer.py import labels.zip /home/user/data/pvn_dataset
# Creates: /home/user/data/pvn_dataset/*/train/labels/, /valid/labels/, etc.
```

**Update configs:**
```bash
python dataset_path_remapper.py data.yaml \
    "C:\Users\Data\Dataset" \
    "/home/user/data/pvn_dataset"
```

## Advanced Usage

### Check Archive Contents

```bash
# List files in archive
python label_transfer.py info labels.zip --list-files

# Verify archive integrity
python label_transfer.py info labels.zip --verify
```

### Export with Different Compression

```bash
# No compression (faster, larger file)
python label_transfer.py export /dataset --output labels.zip --no-compress

# TAR with different compression
python label_transfer.py export /dataset --output labels.tar.gz --format tar.gz
python label_transfer.py export /dataset --output labels.tar.bz2 --format tar.bz2
python label_transfer.py export /dataset --output labels.tar.xz --format tar.xz
```

### Skip Existing Files (Update Only)

```bash
# If labels directory already exists, only import new/changed ones
python label_transfer.py import labels_v2.zip /dataset --skip-existing
```

## How It Works

### Export Process

1. **Scan** all `.txt` label files recursively
2. **Preserve** relative paths (removes absolute prefix)
3. **Compress** into single archive with metadata
4. **Metadata includes**: export date, total count, original source path

Example archive structure:
```
labels_export.zip
├── metadata.json          (export info)
├── labels/
│   ├── 8_data6/train/8_data6_43_jpg.rf.xxx.txt
│   ├── 8_data6/valid/8_data6_210_jpg.rf.xxx.txt
│   ├── 1_data5/train/1_data5_100_jpg.rf.xxx.txt
│   └── ...
```

### Import Process

1. **Extract** archive with preserved relative paths
2. **Create** directory structure at target location
3. **Copy** each label file to correct position
4. **Validate** file counts and integrity

The path structure is **independent of the prefix**, so:
- Original: `/home/sonnn/Videos/tmp_dataset/pvn-vehicle/251222/8_data6/train/labels/file.txt`
- New location: `/mnt/data/dataset/8_data6/train/labels/file.txt`

The relative path `8_data6/train/labels/file.txt` is preserved!

## Performance Notes

### Expected Times (85K label files, 15GB dataset)

| Operation | Time | Archive Size |
|-----------|------|--------------|
| Export (ZIP) | 2-5 min | 150-250 MB |
| Export (TAR.GZ) | 3-7 min | 100-150 MB |
| Import | 2-3 min | - |
| Path Remap | < 1 sec | - |

### Optimization Tips

1. **Use ZIP for speed** (faster than TAR)
2. **Disable compression** if bandwidth not limited: `--no-compress`
3. **On SSD drives**, everything is 2-3x faster
4. **Network transfer**: Use `tar.xz` for smallest size, `zip` for speed

## Troubleshooting

### Import seems slow?

Check if disk I/O is bottleneck:
```bash
# Linux
iostat -x 1

# Windows
Get-Counter "\PhysicalDisk(_Total)\% Disk Time" -Continuous
```

### Archive corruption?

Verify after transfer:
```bash
python label_transfer.py info labels.zip --verify
```

### Path remap didn't work?

Check the exact path strings:
```bash
# Show config before editing
python dataset_path_remapper.py data.yaml /old /new --dry-run --show
```

## Script Details

### `label_transfer.py`

**Commands:**
- `export` - Export labels from dataset
- `import` - Import labels to new location  
- `info` - Show archive information

**Key Features:**
- Preserves relative directory structure
- Progress reporting (every 10K files)
- Metadata tracking
- Archive integrity verification
- Graceful error handling

### `dataset_path_remapper.py`

**Features:**
- Remaps all paths in YAML files recursively
- Handles nested structures (lists, dicts)
- Shows change count
- Dry-run mode to preview changes
- Preserves YAML formatting

## File Format (YOLO Labels)

For reference, labels are stored in YOLO format:
```
<class_id> <x_center> <y_center> <width> <height>
<class_id> <x_center> <y_center> <width> <height>
...
```

Where coordinates are **normalized** (0-1) relative to image size.

Example (from your dataset):
```
8 0.740625 0.43646080760095013 0.3859375 0.498812351543943
19 0.13828125 0.5320665083135392 0.115625 0.26009501187648454
4 0.36328125 0.5118764845605701 0.0703125 0.2339667458432304
```

## Recommended Workflow for Your Dataset

Given your dataset structure (85K labels across multiple subdirectories):

```bash
# 1. Export once
python label_transfer.py export /home/sonnn/Videos/tmp_dataset/pvn-dataset/pvn-vehicle/251222 \
    --output labels_pvn_20250514.zip

# 2. Keep backup & version control
cp labels_pvn_20250514.zip /path/to/backup/
git add labels_pvn_20250514.zip  # if using git

# 3. When transferring to new machine, just:
python label_transfer.py import labels_pvn_20250514.zip /new/dataset/path

# 4. Update configs
python dataset_path_remapper.py data.yaml "/old/path" "/new/path"

# Done! No need to copy images or manually organize folders
```

## Integration with Training

After importing labels, update your training config:

```yaml
# data.yaml
path: /new/dataset/path
train: train/images
val: valid/images
test: test/images

nc: 21
names: [BSD, BSV, bus, ...]  # Your classes
```

Or use the remapper automatically!

## Questions?

For large datasets, consider:
- **Disk space**: Archives are ~1-2% of total dataset size
- **Network**: TAR.XZ smallest, ZIP fastest
- **Compatibility**: ZIP works on all systems, TAR on Linux/Mac
