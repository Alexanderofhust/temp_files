#!/bin/bash
# Organize NavSim images into DrivoR-aligned four-camera WebDataset.
# Cameras: CAM_F0, CAM_B0, CAM_L0, CAM_R0
# Image size: 1148x672, matching DrivoR config.image_size [1148, 672].

IMAGE_DIR="/data/shengzhenli/navtrain/sensor_blobs/trainval"
NAVSIM_LOGS_DIR="/data/shengzhenli/navtrain/navsim_logs/trainval"
OUTPUT_PATH="/data/shengzhenli/theia_navsim_drivor_datasets"
SPLIT_CONFIG="/data/shengzhenli/DiffusionDrive/navsim/planning/script/config/training/default_train_val_test_log_split.yaml"
DATASET_NAME="navsim_drivor_cameras"

echo "=========================================="
echo "NavSim DrivoR Camera Dataset Organization"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Dataset Name: $DATASET_NAME"
echo "  Cameras: CAM_F0 CAM_B0 CAM_L0 CAM_R0"
echo "  Target Size: 1148x672"
echo "  Image Directory: $IMAGE_DIR"
echo "  NavSim Logs: $NAVSIM_LOGS_DIR"
echo "  Output Path: $OUTPUT_PATH"
echo "  Split Config: $SPLIT_CONFIG"
echo ""

if [ ! -f "$SPLIT_CONFIG" ]; then
    echo "ERROR: Split config file not found: $SPLIT_CONFIG"
    exit 1
fi

if [ ! -d "$IMAGE_DIR" ]; then
    echo "ERROR: Image directory not found: $IMAGE_DIR"
    exit 1
fi

if [ ! -d "$NAVSIM_LOGS_DIR" ]; then
    echo "ERROR: NavSim logs directory not found: $NAVSIM_LOGS_DIR"
    exit 1
fi

mkdir -p "$OUTPUT_PATH"

python organize_navsim_webdataset_drivor_cameras.py \
    --dataset "$DATASET_NAME" \
    --output-path "$OUTPUT_PATH" \
    --image-dir "$IMAGE_DIR" \
    --navsim-logs-dir "$NAVSIM_LOGS_DIR" \
    --split-config "$SPLIT_CONFIG" \
    --target-width 1148 \
    --target-height 672 \
    --samples-per-shard 1000

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Organization Complete"
    echo "Output:"
    echo "  $OUTPUT_PATH/$DATASET_NAME/images/"
    echo "Splits:"
    cat "$OUTPUT_PATH/$DATASET_NAME/splits.json"
else
    echo "Organization Failed (exit code: $EXIT_CODE)"
fi
echo "=========================================="

exit $EXIT_CODE
