"""Organize NavSim raw images (Front View CAM_F0 only) to webdataset format.
Matched to ImageNet/Feature Extraction standards.
Supports automatic train/val split based on NavSim log names.
"""

import argparse
import glob
import os
import shutil
import tarfile
import json
from io import BytesIO
from pathlib import Path

import numpy as np
import webdataset as wds
from numpy.typing import NDArray
from PIL import Image
from torchvision.transforms.v2 import Compose, Resize


def load_split_config(split_config_path: str) -> tuple[set[str], set[str]]:
    """Load train/val split configuration from YAML file.

    Args:
        split_config_path: Path to the YAML config file with train_logs and val_logs

    Returns:
        Tuple of (train_log_names, val_log_names) as sets
    """
    try:
        import yaml
    except ImportError:
        print("PyYAML not installed. Install with: pip install pyyaml")
        raise

    with open(split_config_path, 'r') as f:
        config = yaml.safe_load(f)

    train_logs = set(config.get('train_logs', []))
    val_logs = set(config.get('val_logs', []))

    print(f"Loaded split config: {len(train_logs)} train logs, {len(val_logs)} val logs")
    return train_logs, val_logs


def get_log_name_from_path(image_path: str) -> str:
    """Extract log name from image path.

    Example path: .../sensor_blobs/trainval/2021.05.12.19.36.12_veh-35_00005_00204/CAM_F0/xxx.jpg
    Returns: 2021.05.12.19.36.12_veh-35_00005_00204
    """
    parts = image_path.split(os.sep)
    if "CAM_F0" in parts:
        cam_idx = parts.index("CAM_F0")
        if cam_idx > 0:
            return parts[cam_idx - 1]
    return None


def check_existing_shard(path: str) -> bool:
    """Check the integrity of the existing webdataset shard."""
    if not os.path.exists(path):
        return False
    try:
        tarf = tarfile.open(path)
        for _ in tarf.getmembers():
            pass
    except (ValueError, tarfile.ReadError, tarfile.CompressionError) as e:
        print(e)
        return False
    return True


def create_shard(
    args: argparse.Namespace,
    shard_idx: int,
    shard_path: str | None,
    remote_shard_path: str,
    frames: list[tuple[NDArray, str]],
) -> None:
    """Create a webdataset shard."""
    if check_existing_shard(remote_shard_path):
        print(f"creating {args.dataset} shard {shard_idx:06d} - check pass, skip\r", end="")
        return
    print(f"creating {args.dataset} shard {shard_idx:06d} ({len(frames)} samples)\r", end="")

    if shard_path is None:
        shard_path = remote_shard_path

    with wds.TarWriter(shard_path) as tar_writer:
        for i, (image, basename) in enumerate(frames):
            image_out = BytesIO()
            np.save(image_out, image)
            # Standard WebDataset format
            sample = {"__key__": basename, "image": image_out.getvalue()}
            tar_writer.write(sample)
            if (i + 1) % 20 == 0:
                print(f"creating {args.dataset} shard {shard_idx:06d} - {(i+1) * 100 // len(frames):02d}%\r", end="")

    if shard_path != remote_shard_path:
        shutil.move(shard_path, remote_shard_path)
    print("")


def process_split(
    args: argparse.Namespace,
    split: str,
    train_logs: set[str] | None,
    val_logs: set[str] | None
) -> None:
    """Process a single split (train or val)."""

    # Setup paths
    if args.tmp_shard_path == "None":
        TMP_SHARD_PATH = None
    else:
        TMP_SHARD_PATH = os.path.join(args.tmp_shard_path, args.dataset, "images")
        if not os.path.exists(TMP_SHARD_PATH):
            os.makedirs(TMP_SHARD_PATH)

    OUTPUT_SHARD_PATH = os.path.join(args.output_path, args.dataset, "images")
    if not os.path.exists(OUTPUT_SHARD_PATH):
        os.makedirs(OUTPUT_SHARD_PATH, exist_ok=True)

    # Search images - CAM_F0 ONLY
    print(f"Scanning {args.image_dir} for CAM_F0 (Front View) images...")
    search_pattern = os.path.join(args.image_dir, "*", "CAM_F0", "*.jpg")
    image_paths = sorted(glob.glob(search_pattern))

    if not image_paths:
        print(f"No images found in pattern: {search_pattern}")
        print("Trying recursive search for any CAM_F0 folder...")
        image_paths = sorted(glob.glob(os.path.join(args.image_dir, "**", "CAM_F0", "*.jpg"), recursive=True))

    print(f"Found total {len(image_paths)} images (Front View Only).")

    # Filter images based on split if split config is provided
    if train_logs is not None or val_logs is not None:
        filtered_paths = []
        target_logs = train_logs if split == 'train' else val_logs

        for img_path in image_paths:
            log_name = get_log_name_from_path(img_path)
            if log_name and log_name in target_logs:
                filtered_paths.append(img_path)

        print(f"Filtered to {len(filtered_paths)} images for {split} split (from {len(target_logs)} logs)")
        image_paths = filtered_paths
    else:
        print(f"No split filtering applied. Processing all {len(image_paths)} images as '{split}'.")

    total_images_found = len(image_paths)

    if args.max_images is not None and args.max_images < len(image_paths):
        image_paths = image_paths[:args.max_images]
        print(f"--- TEST MODE: Limiting to first {len(image_paths)} images ---")
        total_images_processed = len(image_paths)
    else:
        total_images_processed = total_images_found

    if len(image_paths) == 0:
        print(f"No images to process for {split} split. Skipping.")
        return

    transform = Compose([Resize((224, 224), antialias=True)])

    shard_idx = 0
    shard_buffer: list[tuple[NDArray, str]] = []

    for i, image_path in enumerate(image_paths):
        try:
            # Extract basename from image path, similar to ImageNet
            # IMPORTANT: Remove dots from basename to avoid webdataset key conflicts
            # WebDataset uses dots to separate __key__ from extensions
            parts = image_path.split(os.sep)
            if "CAM_F0" in parts:
                cam_idx = parts.index("CAM_F0")
                if cam_idx > 0:
                    scene_name = parts[cam_idx-1].replace('.', '_')  # Replace dots with underscores
                    file_hash = os.path.splitext(parts[-1])[0]
                    # Use underscore to join, creating unique key like ImageNet
                    basename = f"{scene_name}_cam_f0_{file_hash}"
                else:
                    basename = os.path.splitext(os.path.basename(image_path))[0].replace('.', '_')
            else:
                basename = os.path.splitext(os.path.basename(image_path))[0].replace('.', '_')

            img_pil = Image.open(image_path)
            if img_pil.mode != 'RGB':
                img_pil = img_pil.convert('RGB')

            image = np.array(transform(img_pil))
            shard_buffer.append((image, basename))

            if len(shard_buffer) % 20 == 0:
                print(f"Processing image {i+1}/{len(image_paths)} (Buffer: {len(shard_buffer)})\r", end="")

            if len(shard_buffer) == args.samples_per_shard:
                print(f"\nBuffer full ({len(shard_buffer)}). Writing shard {shard_idx}...")
                shard_fn = f"{args.dataset}_{split}-{shard_idx:06d}-{split}.tar"
                local_shard_path = os.path.join(TMP_SHARD_PATH, shard_fn) if TMP_SHARD_PATH else None
                remote_shard_path = os.path.join(OUTPUT_SHARD_PATH, shard_fn)
                create_shard(args, shard_idx, local_shard_path, remote_shard_path, shard_buffer)
                shard_buffer = []
                shard_idx += 1
        except Exception as e:
            print(f"\nError processing {image_path}: {e}")

    if len(shard_buffer) > 0:
        print(f"\nProcessing remaining {len(shard_buffer)} images. Writing shard {shard_idx}...")
        shard_fn = f"{args.dataset}_{split}-{shard_idx:06d}-{split}.tar"
        local_shard_path = os.path.join(TMP_SHARD_PATH, shard_fn) if TMP_SHARD_PATH else None
        remote_shard_path = os.path.join(OUTPUT_SHARD_PATH, shard_fn)
        create_shard(args, shard_idx, local_shard_path, remote_shard_path, shard_buffer)

    print(f"\n\nConversion complete for {split}! Processed {total_images_processed} images.")

    # Update splits.json
    splits_path = os.path.join(OUTPUT_SHARD_PATH, "splits.json")
    print(f"Updating splits.json at {splits_path}...")

    splits_data = {}
    if os.path.exists(splits_path):
        try:
            with open(splits_path, 'r') as f:
                splits_data = json.load(f)
        except:
            pass

    splits_data[split] = total_images_processed

    with open(splits_path, 'w') as f:
        json.dump(splits_data, f, indent=4)
    print(f"Metadata saved: {splits_data}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="navsim")
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--image-dir", type=str, required=True,
                       help="Path to the root folder (e.g., .../sensor_blobs/trainval)")
    parser.add_argument("--tmp-shard-path", type=str, default="None")
    parser.add_argument("--samples-per-shard", type=int, default=1000)
    parser.add_argument("--split", type=str, default=None,
                       help="Specific split to process (train/val). If None, process both based on split-config")
    parser.add_argument("--split-config", type=str, default=None,
                       help="Path to YAML file with train_logs and val_logs (e.g., default_train_val_test_log_split.yaml)")
    parser.add_argument("--max-images", type=int, default=None,
                       help="Maximum number of images to process (for testing)")
    args = parser.parse_args()

    # Determine processing mode
    if args.split_config:
        # Auto-split mode: process both train and val based on config
        print(f"Auto-split mode enabled with config: {args.split_config}")
        train_logs, val_logs = load_split_config(args.split_config)
        process_splits = ['train', 'val']
    elif args.split:
        # Manual split mode: process only specified split
        print(f"Manual split mode: processing {args.split} only")
        train_logs, val_logs = None, None
        process_splits = [args.split]
    else:
        # Legacy mode: process as 'train' by default
        print("Legacy mode: processing all images as 'train' split")
        train_logs, val_logs = None, None
        process_splits = ['train']

    # Process each split
    for current_split in process_splits:
        print(f"\n{'='*60}")
        print(f"Processing split: {current_split}")
        print(f"{'='*60}\n")

        process_split(args, current_split, train_logs, val_logs)


if __name__ == "__main__":
    main()
