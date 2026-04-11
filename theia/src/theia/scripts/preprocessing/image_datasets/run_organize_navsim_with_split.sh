#!/bin/bash
# Script to organize NavSim dataset with automatic train/val split

# Configuration
IMAGE_DIR="/data/shengzhenli/navtrain/sensor_blobs/trainval"
OUTPUT_PATH="/data/shengzhenli/theia_navsim_datasets"
SPLIT_CONFIG="/data/shengzhenli/DiffusionDrive/navsim/planning/script/config/training/default_train_val_test_log_split.yaml"

echo "=========================================="
echo "NavSim Dataset Organization with Train/Val Split"
echo "=========================================="
echo ""
echo "Image Directory: $IMAGE_DIR"
echo "Output Path: $OUTPUT_PATH"
echo "Split Config: $SPLIT_CONFIG"
echo ""

# Check if split config exists
if [ ! -f "$SPLIT_CONFIG" ]; then
    echo "ERROR: Split config file not found: $SPLIT_CONFIG"
    exit 1
fi

# Check if image directory exists
if [ ! -d "$IMAGE_DIR" ]; then
    echo "ERROR: Image directory not found: $IMAGE_DIR"
    exit 1
fi

# Run the organization script with auto-split
python organize_navsim_webdataset_front_new.py \
    --dataset navsim \
    --output-path "$OUTPUT_PATH" \
    --image-dir "$IMAGE_DIR" \
    --split-config "$SPLIT_CONFIG" \
    --samples-per-shard 1000

echo ""
echo "=========================================="
echo "Organization Complete!"
echo "=========================================="
echo ""
echo "Output structure:"
echo "  $OUTPUT_PATH/navsim/images/"
echo "    ├── navsim_train-*.tar  (train shards)"
echo "    ├── navsim_val-*.tar    (val shards)"
echo "    └── splits.json                   (metadata)"
echo ""
echo "Check splits.json:"
cat "$OUTPUT_PATH/navsim/images/splits.json"
