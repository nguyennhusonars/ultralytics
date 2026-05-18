# PVN Vehicle Dataset Transfer - Quick Reference

**Your Dataset:**
- 📊 85,017 label files
- 💾 15 GB total size
- 📁 Multiple subdirectories (8_data6, 1_data5, crop_data, data_check_vh, etc.)
- 🏷️ YOLO format labels (class_id, x, y, w, h normalized)

## ⚡ Quick Start (3 Steps)

### Machine A: Export Labels

```bash
cd ~/Documents/ultralytics_sonnn/tools/data_transfer

# Option 1: Using shell script (interactive)
./dataset_transfer.sh export /home/sonnn/Videos/tmp_dataset/pvn-dataset/pvn-vehicle/251222

# Option 2: Direct Python (non-interactive)
python label_transfer.py export /home/sonnn/Videos/tmp_dataset/pvn-dataset/pvn-vehicle/251222 \
    --output labels_pvn_20250514.zip
```

**Result:** 
- Creates `labels_pvn_20250514.zip` (~40-50 MB)
- Contains all 85,017 labels with relative paths
- Metadata with export info

### Transfer Archive

```bash
# Copy to new machine (choose one method)

# SCP
scp labels_pvn_20250514.zip user@newmachine:/path/to/

# Or upload to cloud storage, USB drive, etc.
# File is small (~40MB), easy to transfer
```

### Machine B: Import Labels

```bash
cd ~/Documents/ultralytics_sonnn/tools/data_transfer  # or /path/to/tools/data_transfer

# Option 1: Using shell script (interactive with dry-run)
./dataset_transfer.sh import labels_pvn_20250514.zip /path/to/new/dataset

# Option 2: Direct Python
python label_transfer.py import labels_pvn_20250514.zip /path/to/new/dataset
```

**Result:**
- Recreates full directory structure at new location
- All 85,017 labels imported in correct places
- Ready to use without copying images!

## 🔄 Complete Workflow (Path Prefix Handling)

If your YAML config files contain full paths, update them:

```bash
# Before updating data.yaml:
cat data.yaml  # Shows: /home/sonnn/Videos/tmp_dataset/...

# Update paths
python dataset_path_remapper.py data.yaml \
    "/home/sonnn/Videos/tmp_dataset/pvn-dataset/pvn-vehicle/251222" \
    "/mnt/new/dataset/location"

# After update: YAML now has /mnt/new/dataset/location/...
```

Or all in one command:

```bash
./dataset_transfer.sh workflow \
    /home/sonnn/Videos/tmp_dataset/pvn-dataset/pvn-vehicle/251222 \
    data.yaml
```

## 📋 Common Scenarios

### Scenario 1: Transfer to Different Linux Machine

```bash
# Machine A (Ubuntu)
cd ~/Documents/ultralytics_sonnn/tools/data_transfer
python label_transfer.py export \
    /home/sonnn/Videos/tmp_dataset/pvn-dataset/pvn-vehicle/251222 \
    --output labels.zip

scp labels.zip user@server2.com:/tmp/

# Machine B (CentOS)
cd /path/to/tools/data_transfer
python label_transfer.py import labels.zip /data/pvn_dataset
python dataset_path_remapper.py data.yaml "/home/sonnn/Videos/tmp_dataset/pvn-dataset/pvn-vehicle/251222" "/data/pvn_dataset"
```

### Scenario 2: Transfer to Windows Machine

```bash
# Linux - Export
./dataset_transfer.sh export /home/sonnn/Videos/tmp_dataset/pvn-dataset/pvn-vehicle/251222

# Upload to Google Drive or email labels.zip to Windows machine

# Windows - Import
python label_transfer.py import labels.zip C:\Users\Data\dataset
python dataset_path_remapper.py data.yaml "^C:\\Users.*" "C:\Users\Data\dataset" 
```

### Scenario 3: Update Labels Only (Keep Images, Update Labels)

```bash
# Export new labels from updated dataset
python label_transfer.py export /path/to/updated/dataset --output labels_v2.zip

# Import to existing dataset (skip existing images)
python label_transfer.py import labels_v2.zip /path/to/existing/dataset --skip-existing
```

## 🛠️ Advanced Commands

### Verify Archive Before Transfer

```bash
python label_transfer.py info labels.zip --verify --list-files
```

### Export with Different Compression

```bash
# Fastest (ZIP, best speed)
python label_transfer.py export /dataset --output labels.zip

# Smallest (TAR.XZ, best compression)
python label_transfer.py export /dataset --output labels.tar.xz --format tar.xz

# Balance (TAR.GZ, good compression + compatibility)
python label_transfer.py export /dataset --output labels.tar.gz --format tar.gz
```

### Dry-run Before Import

```bash
# Preview without copying
python label_transfer.py import labels.zip /dataset --dry-run
```

### Handle Different Path Structures

```bash
# If original path was: /a/b/c/dataset/8_data6/train/labels/file.txt
# Your dataset structure preserved: dataset/8_data6/train/labels/file.txt

# You can import to ANY location:
python label_transfer.py import labels.zip /completely/different/path
# Creates: /completely/different/path/8_data6/train/labels/file.txt

# Path prefix doesn't matter - relative structure is preserved!
```

## 📊 Performance Metrics (Your Dataset)

| Operation | Time | Notes |
|-----------|------|-------|
| Export | 3-5 min | 85K files, SSD faster |
| Archive Size | 40-50 MB | ~0.3% of dataset |
| Transfer | < 1 min | On good network |
| Import | 2-3 min | Parallel I/O possible |
| Path Remap | < 1 sec | YAML file only |

## ✅ Verification Checklist

After import, verify everything worked:

```bash
# Count files
find /new/path -name "*.txt" | wc -l  # Should be 85017

# Sample content
head /new/path/8_data6/train/labels/8_data6_43_jpg.rf.*.txt

# Directory structure matches
tree /new/path -L 3  # Compare with original
```

## 🆘 Troubleshooting

### "Archive not found"
```bash
# Check file exists
ls -lh labels_*.zip
```

### "Import seems stuck"
```bash
# It's actually working, just processing files
# Progress shown every 10K files
# Total: 85017 files / 10000 = 8-9 progress lines

# You can check in another terminal
find /target/path -name "*.txt" | wc -l  # Count current imports
```

### "Path remap not working"
```bash
# Preview first without applying
python dataset_path_remapper.py data.yaml /old /new --dry-run --show

# Check exact strings match
grep "/old" data.yaml  # Verify path is there
```

### "Some files failed to import"
```bash
# Re-run import - it will skip successfully imported files
python label_transfer.py import labels.zip /path --skip-existing
```

## 📁 File Structure Example

Your dataset structure (preserved in export/import):

```
labels_pvn_20250514.zip
├── metadata.json    (export info, dates, counts)
└── labels/
    ├── 8_data6/
    │   ├── train/
    │   │   └── labels/*.txt
    │   └── valid/
    │       └── labels/*.txt
    ├── 1_data5/
    │   ├── train/labels/*.txt
    │   └── valid/labels/*.txt
    ├── crop_data/
    │   ├── crop_cam14_1/
    │   │   ├── train/labels/*.txt
    │   │   └── valid/labels/*.txt
    ├── data_check_vh/
    │   ├── train/*/train/labels/*.txt
    │   └── valid/*/valid/labels/*.txt
    └── ...
```

After import to `/mnt/data/dataset`, becomes:

```
/mnt/data/dataset/
├── 8_data6/train/labels/*.txt
├── 1_data5/train/labels/*.txt
├── crop_data/crop_cam14_1/train/labels/*.txt
└── ...
```

## 🚀 Next Steps

1. **Export today:**
   ```bash
   ./dataset_transfer.sh export /home/sonnn/Videos/tmp_dataset/pvn-dataset/pvn-vehicle/251222
   ```

2. **Keep archive as backup:**
   ```bash
   cp labels_*.zip /backup/location/
   ```

3. **When ready to transfer:**
   ```bash
   scp labels_*.zip newmachine:/path/
   ```

4. **On new machine:**
   ```bash
   ./dataset_transfer.sh import labels_*.zip /new/dataset/path
   ./dataset_transfer.sh remap data.yaml "/old" "/new"
   ```

5. **Start training immediately** (images stay on different machine, labels now imported)

## 📞 Need Help?

Review full documentation:
- `LABEL_TRANSFER_GUIDE.md` - Detailed guide
- `label_transfer.py --help` - Export/import options
- `dataset_path_remapper.py --help` - Path remapping options
- `./dataset_transfer.sh help` - Shell script usage
