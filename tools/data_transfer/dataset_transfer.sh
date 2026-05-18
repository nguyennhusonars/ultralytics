#!/bin/bash
# Dataset Transfer Helper
# Complete workflow for exporting, transferring, and importing labels
# 
# Usage:
#   ./dataset_transfer.sh export /path/to/dataset
#   ./dataset_transfer.sh import ./labels_export.zip /path/to/new/location
#   ./dataset_transfer.sh info ./labels_export.zip

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

show_help() {
    echo -e "${BLUE}Dataset Label Transfer Helper${NC}

${YELLOW}Usage:${NC}
    ./dataset_transfer.sh <command> <args>

${YELLOW}Commands:${NC}
    export <dataset_path>                Export labels from dataset
    import <archive> <target_path>       Import labels to new location
    remap <yaml_file> <old> <new>        Update paths in YAML config
    info <archive>                       Show archive information
    workflow <dataset> <config>          Complete workflow (export + info)

${YELLOW}Examples:${NC}
    # Export your dataset
    ./dataset_transfer.sh export /home/data/dataset

    # Import to new location
    ./dataset_transfer.sh import labels_export.zip /mnt/data/dataset

    # Update YAML config
    ./dataset_transfer.sh remap data.yaml \"/old/path\" \"/new/path\"

    # Complete workflow
    ./dataset_transfer.sh workflow /home/data/dataset data.yaml
"
}

cmd_export() {
    local dataset_path="$1"
    
    if [ -z "$dataset_path" ]; then
        print_error "Dataset path required"
        echo "Usage: ./dataset_transfer.sh export /path/to/dataset"
        exit 1
    fi
    
    if [ ! -d "$dataset_path" ]; then
        print_error "Directory not found: $dataset_path"
        exit 1
    fi
    
    local archive_name="labels_$(date +%Y%m%d_%H%M%S).zip"
    
    print_header "Exporting Labels"
    echo "Dataset: $dataset_path"
    echo "Output: $archive_name"
    echo ""
    
    python "$SCRIPT_DIR/label_transfer.py" export "$dataset_path" \
        --output "$archive_name"
    
    print_success "Export complete"
    echo ""
    echo "Next steps:"
    echo "1. Transfer the archive to your new machine"
    echo "   scp $archive_name user@newmachine:/path/to/"
    echo ""
    echo "2. Import on new machine:"
    echo "   ./dataset_transfer.sh import $archive_name /path/to/dataset"
    echo ""
    echo "3. Update YAML config (if needed):"
    echo "   ./dataset_transfer.sh remap data.yaml \"/old/path\" \"/new/path\""
}

cmd_import() {
    local archive="$1"
    local target="$2"
    
    if [ -z "$archive" ] || [ -z "$target" ]; then
        print_error "Archive and target path required"
        echo "Usage: ./dataset_transfer.sh import <archive> <target_path>"
        exit 1
    fi
    
    if [ ! -f "$archive" ]; then
        print_error "Archive not found: $archive"
        exit 1
    fi
    
    print_header "Importing Labels"
    echo "Archive: $archive"
    echo "Target: $target"
    echo ""
    
    # Show what will be done
    echo "Performing dry-run first..."
    python "$SCRIPT_DIR/label_transfer.py" import "$archive" "$target" --dry-run
    
    echo ""
    read -p "Proceed with import? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python "$SCRIPT_DIR/label_transfer.py" import "$archive" "$target"
        print_success "Import complete"
        echo ""
        echo "Labels are now at: $target"
    else
        print_warning "Import cancelled"
    fi
}

cmd_remap() {
    local yaml_file="$1"
    local old_prefix="$2"
    local new_prefix="$3"
    
    if [ -z "$yaml_file" ] || [ -z "$old_prefix" ] || [ -z "$new_prefix" ]; then
        print_error "YAML file, old path, and new path required"
        echo "Usage: ./dataset_transfer.sh remap <yaml_file> <old_path> <new_path>"
        exit 1
    fi
    
    if [ ! -f "$yaml_file" ]; then
        print_error "File not found: $yaml_file"
        exit 1
    fi
    
    print_header "Remapping Dataset Paths"
    echo "Config: $yaml_file"
    echo "From: $old_prefix"
    echo "To: $new_prefix"
    echo ""
    
    # Show preview
    echo "Preview of changes:"
    python "$SCRIPT_DIR/dataset_path_remapper.py" "$yaml_file" "$old_prefix" "$new_prefix" \
        --dry-run --show
    
    echo ""
    read -p "Apply changes? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python "$SCRIPT_DIR/dataset_path_remapper.py" "$yaml_file" "$old_prefix" "$new_prefix"
        print_success "Paths updated in: $yaml_file"
    else
        print_warning "Update cancelled"
    fi
}

cmd_info() {
    local archive="$1"
    
    if [ -z "$archive" ]; then
        print_error "Archive path required"
        echo "Usage: ./dataset_transfer.sh info <archive>"
        exit 1
    fi
    
    if [ ! -f "$archive" ]; then
        print_error "Archive not found: $archive"
        exit 1
    fi
    
    print_header "Archive Information"
    python "$SCRIPT_DIR/label_transfer.py" info "$archive" --verify
    
    echo ""
    read -p "List all files? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python "$SCRIPT_DIR/label_transfer.py" info "$archive" --list-files
    fi
}

cmd_workflow() {
    local dataset="$1"
    local yaml_file="$2"
    
    if [ -z "$dataset" ]; then
        print_error "Dataset path required"
        echo "Usage: ./dataset_transfer.sh workflow <dataset_path> [yaml_config]"
        exit 1
    fi
    
    print_header "Complete Dataset Transfer Workflow"
    
    # Step 1: Export
    echo ""
    echo "Step 1/3: Exporting labels..."
    cmd_export "$dataset"
    
    # Find the latest export
    local latest_export=$(ls -t labels_*.zip 2>/dev/null | head -1)
    
    if [ -z "$latest_export" ]; then
        print_error "Export failed"
        exit 1
    fi
    
    # Step 2: Archive info
    echo ""
    echo "Step 2/3: Archive information..."
    python "$SCRIPT_DIR/label_transfer.py" info "$latest_export" --verify
    
    # Step 3: Instructions for YAML (if provided)
    if [ -n "$yaml_file" ] && [ -f "$yaml_file" ]; then
        echo ""
        echo "Step 3/3: Path remapping..."
        echo ""
        print_warning "After transferring to new machine, run:"
        echo "  ./dataset_transfer.sh remap $yaml_file \"/old/path\" \"/new/path\""
    fi
    
    echo ""
    print_success "Workflow preparation complete!"
    echo ""
    echo "📦 Archive ready for transfer: $latest_export"
    echo "📋 Save these commands for the new machine:"
    echo "   ./dataset_transfer.sh import $latest_export /path/to/new/location"
    if [ -n "$yaml_file" ]; then
        echo "   ./dataset_transfer.sh remap $yaml_file \"/old/path\" \"/new/path\""
    fi
}

# Main
if [ "$#" -eq 0 ]; then
    show_help
    exit 0
fi

case "$1" in
    export)
        cmd_export "$2"
        ;;
    import)
        cmd_import "$2" "$3"
        ;;
    remap)
        cmd_remap "$2" "$3" "$4"
        ;;
    info)
        cmd_info "$2"
        ;;
    workflow)
        cmd_workflow "$2" "$3"
        ;;
    help|-h|--help)
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        echo "Run './dataset_transfer.sh help' for usage"
        exit 1
        ;;
esac
