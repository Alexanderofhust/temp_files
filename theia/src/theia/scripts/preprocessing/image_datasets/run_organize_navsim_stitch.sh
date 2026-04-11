#!/bin/bash
# Script to organize NavSim stitched camera dataset (CAM_L0 + CAM_F0 + CAM_R0)
# with automatic train/val split
# Target size: 1008x252 (4:1 aspect ratio, compatible with patch size 14)

# Configuration
IMAGE_DIR="/data/shengzhenli/navtrain/sensor_blobs/trainval"
NAVSIM_LOGS_DIR="/data/shengzhenli/navtrain/navsim_logs/trainval"
OUTPUT_PATH="/data/shengzhenli/theia_navsim_stitch_datasets"
SPLIT_CONFIG="/data/shengzhenli/DiffusionDrive/navsim/planning/script/config/training/default_train_val_test_log_split.yaml"
DATASET_NAME="navsim_stitch"

echo "=========================================="
echo "NavSim Stitched Camera Dataset Organization"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Dataset Name: $DATASET_NAME"
echo "  Image Directory: $IMAGE_DIR"
echo "  NavSim Logs: $NAVSIM_LOGS_DIR"
echo "  Output Path: $OUTPUT_PATH"
echo "  Split Config: $SPLIT_CONFIG"
echo "  Target Size: 1008x252 (4:1 aspect ratio)"
echo "  Patch Compatible: 1008/14=72, 252/14=18 ✓"
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

# Check if navsim logs directory exists
if [ ! -d "$NAVSIM_LOGS_DIR" ]; then
    echo "ERROR: NavSim logs directory not found: $NAVSIM_LOGS_DIR"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_PATH"

echo "=========================================="
echo "Starting dataset processing..."
echo "=========================================="
echo ""
echo "Note: This will process ~152,495 images"
echo "Estimated time: 14-21 hours"
echo "Estimated output size: ~115GB"
echo ""

# Run the organization script with auto-split
python organize_navsim_webdataset_stitch.py \
    --dataset "$DATASET_NAME" \
    --output-path "$OUTPUT_PATH" \
    --image-dir "$IMAGE_DIR" \
    --navsim-logs-dir "$NAVSIM_LOGS_DIR" \
    --split-config "$SPLIT_CONFIG" \
    --samples-per-shard 1000

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Organization Complete!"
else
    echo "✗ Organization Failed (exit code: $EXIT_CODE)"
fi
echo "=========================================="
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "Output structure:"
    echo "  $OUTPUT_PATH/$DATASET_NAME/images/"
    echo "    ├── ${DATASET_NAME}_train-*.tar  (train shards)"
    echo "    ├── ${DATASET_NAME}_val-*.tar    (val shards)"
    echo "    └── splits.json                  (metadata)"
    echo ""

    if [ -f "$OUTPUT_PATH/$DATASET_NAME/images/splits.json" ]; then
        echo "Dataset statistics:"
        cat "$OUTPUT_PATH/$DATASET_NAME/images/splits.json"
        echo ""
    fi

    echo "Next steps:"
    echo "  1. Verify the generated dataset"
    echo "  2. Modify feature extraction code for 1008x252 input"
    echo "  3. Extract features with interpolate_pos_encoding=True"
    echo "  4. Train theia model"
fi

echo ""
