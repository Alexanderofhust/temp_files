"""Organize NavSim raw images (Stitched CAM_L0 + CAM_F0 + CAM_R0) to webdataset format.
Stitches three camera views following DiffusionDrive's logic and resizes to 1024x256.
Matched to ImageNet/Feature Extraction standards.
Supports automatic train/val split based on NavSim log names.
Note: Changed to 1024x256 to match patch16 requirements (1024÷16=64, 256÷16=16).
"""

import argparse
import glob
import os
import shutil
import tarfile
import json
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import webdataset as wds
from numpy.typing import NDArray
from PIL import Image


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
    for cam in ["CAM_F0", "CAM_L0", "CAM_R0"]:
        if cam in parts:
            cam_idx = parts.index(cam)
            if cam_idx > 0:
                return parts[cam_idx - 1]
    return None


def find_camera_triplets(image_dir: str, navsim_logs_dir: str = None) -> list[tuple[str, str, str, str]]:
    """Find matching triplets of CAM_L0, CAM_F0, CAM_R0 images using NavSim pkl files.

    Args:
        image_dir: Root directory containing camera images (sensor_blobs/trainval)
        navsim_logs_dir: Directory containing NavSim pkl files (navsim_logs/trainval)
                        If None, will try to infer from image_dir

    Returns:
        List of tuples (log_name, cam_l0_path, cam_f0_path, cam_r0_path)
    """
    import pickle

    # Infer navsim_logs_dir if not provided
    if navsim_logs_dir is None:
        # Assume structure: .../sensor_blobs/trainval and .../navsim_logs/trainval
        parent_dir = os.path.dirname(os.path.dirname(image_dir))
        navsim_logs_dir = os.path.join(parent_dir, "navsim_logs", "trainval")

    print(f"Scanning NavSim logs from: {navsim_logs_dir}")
    print(f"Image directory: {image_dir}")

    if not os.path.exists(navsim_logs_dir):
        print(f"ERROR: NavSim logs directory not found: {navsim_logs_dir}")
        print("Please provide the correct path to navsim_logs/trainval")
        return []

    # Find all pkl files
    pkl_files = sorted(glob.glob(os.path.join(navsim_logs_dir, "*.pkl")))
    print(f"Found {len(pkl_files)} pkl files")
    print(f"Scanning for camera triplets (this may take 10-30 minutes)...")
    print(f"Progress will be shown every 100 pkl files.")
    print()

    triplets = []
    missing_count = 0
    total_frames = 0

    for pkl_idx, pkl_file in enumerate(pkl_files):
        try:
            with open(pkl_file, 'rb') as f:
                frames = pickle.load(f)

            log_name = os.path.splitext(os.path.basename(pkl_file))[0]

            for frame in frames:
                total_frames += 1

                if 'cams' not in frame:
                    continue

                cams = frame['cams']

                # Check if all three cameras exist
                if 'CAM_L0' in cams and 'CAM_F0' in cams and 'CAM_R0' in cams:
                    cam_l0_rel = cams['CAM_L0']['data_path']
                    cam_f0_rel = cams['CAM_F0']['data_path']
                    cam_r0_rel = cams['CAM_R0']['data_path']

                    # Construct absolute paths
                    cam_l0_path = os.path.join(image_dir, cam_l0_rel)
                    cam_f0_path = os.path.join(image_dir, cam_f0_rel)
                    cam_r0_path = os.path.join(image_dir, cam_r0_rel)

                    # Verify files exist
                    if os.path.exists(cam_l0_path) and os.path.exists(cam_f0_path) and os.path.exists(cam_r0_path):
                        triplets.append((log_name, cam_l0_path, cam_f0_path, cam_r0_path))
                    else:
                        missing_count += 1
                        if missing_count <= 5:
                            print(f"Missing files for frame {frame.get('token', 'unknown')}")
                            print(f"  L0: {os.path.exists(cam_l0_path)}")
                            print(f"  F0: {os.path.exists(cam_f0_path)}")
                            print(f"  R0: {os.path.exists(cam_r0_path)}")

            # Progress update every 100 pkl files
            if (pkl_idx + 1) % 100 == 0 or (pkl_idx + 1) == len(pkl_files):
                progress_pct = (pkl_idx + 1) * 100 // len(pkl_files)
                print(f"Progress: {pkl_idx + 1}/{len(pkl_files)} pkl files ({progress_pct}%) - "
                      f"Found {len(triplets)} triplets, {missing_count} missing")

        except Exception as e:
            print(f"Error processing {pkl_file}: {e}")
            continue

    print()
    print(f"Scan complete!")
    print(f"  Total frames scanned: {total_frames}")
    if missing_count > 0:
        print(f"  Warning: {missing_count} frames missing one or more camera files")
    print(f"  Found {len(triplets)} complete camera triplets")
    return triplets


def stitch_cameras(cam_l0_path: str, cam_f0_path: str, cam_r0_path: str, target_size: tuple[int, int] = (1024, 256)) -> NDArray:
    """Stitch three camera images following DiffusionDrive's logic.

    Args:
        cam_l0_path: Path to left camera image
        cam_f0_path: Path to front camera image
        cam_r0_path: Path to right camera image
        target_size: Target size (width, height) for final stitched image
                    Default (1024, 256) matches patch16 requirements (1024÷16=64, 256÷16=16)

    Returns:
        Stitched and resized image as numpy array
    """
    # Load images
    l0 = cv2.imread(cam_l0_path)
    f0 = cv2.imread(cam_f0_path)
    r0 = cv2.imread(cam_r0_path)

    # Convert BGR to RGB
    l0 = cv2.cvtColor(l0, cv2.COLOR_BGR2RGB)
    f0 = cv2.cvtColor(f0, cv2.COLOR_BGR2RGB)
    r0 = cv2.cvtColor(r0, cv2.COLOR_BGR2RGB)

    # Crop to ensure 4:1 aspect ratio (following DiffusionDrive logic)
    # Original images are typically 1600x900
    # Crop: [28:-28, ...] removes 28 pixels from top and bottom
    # Crop: [..., 416:-416] removes 416 pixels from left and right
    l0_cropped = l0[28:-28, 416:-416]  # Left camera, crop edges
    f0_cropped = f0[28:-28]             # Front camera, only crop top/bottom
    r0_cropped = r0[28:-28, 416:-416]  # Right camera, crop edges

    # Stitch horizontally: left + front + right
    stitched_image = np.concatenate([l0_cropped, f0_cropped, r0_cropped], axis=1)

    # Resize to target size (default 1024x256)
    resized_image = cv2.resize(stitched_image, target_size, interpolation=cv2.INTER_LINEAR)

    return resized_image


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

    # Find camera triplets
    navsim_logs_dir = getattr(args, 'navsim_logs_dir', None)
    triplets = find_camera_triplets(args.image_dir, navsim_logs_dir)

    if not triplets:
        print(f"No camera triplets found in {args.image_dir}")
        return

    # Filter triplets based on split if split config is provided
    if train_logs is not None or val_logs is not None:
        filtered_triplets = []
        target_logs = train_logs if split == 'train' else val_logs

        for log_name, cam_l0, cam_f0, cam_r0 in triplets:
            if log_name and log_name in target_logs:
                filtered_triplets.append((log_name, cam_l0, cam_f0, cam_r0))

        print(f"Filtered to {len(filtered_triplets)} triplets for {split} split (from {len(target_logs)} logs)")
        triplets = filtered_triplets
    else:
        print(f"No split filtering applied. Processing all {len(triplets)} triplets as '{split}'.")

    total_triplets_found = len(triplets)

    if args.max_images is not None and args.max_images < len(triplets):
        triplets = triplets[:args.max_images]
        print(f"--- TEST MODE: Limiting to first {len(triplets)} triplets ---")
        total_triplets_processed = len(triplets)
    else:
        total_triplets_processed = total_triplets_found

    if len(triplets) == 0:
        print(f"No triplets to process for {split} split. Skipping.")
        return

    shard_idx = 0
    shard_buffer: list[tuple[NDArray, str]] = []

    for i, (log_name, cam_l0_path, cam_f0_path, cam_r0_path) in enumerate(triplets):
        try:
            # Create unique basename
            # Extract filename from cam_f0_path
            filename = os.path.splitext(os.path.basename(cam_f0_path))[0]
            scene_name = log_name.replace('.', '_')  # Replace dots with underscores
            basename = f"{scene_name}_stitched_{filename}"

            # Stitch cameras
            stitched_image = stitch_cameras(cam_l0_path, cam_f0_path, cam_r0_path, target_size=(1024, 256))
            shard_buffer.append((stitched_image, basename))

            if len(shard_buffer) % 20 == 0:
                print(f"Processing triplet {i+1}/{len(triplets)} (Buffer: {len(shard_buffer)})\r", end="")

            if len(shard_buffer) == args.samples_per_shard:
                print(f"\nBuffer full ({len(shard_buffer)}). Writing shard {shard_idx}...")
                shard_fn = f"{args.dataset}_{split}-{shard_idx:06d}-{split}.tar"
                local_shard_path = os.path.join(TMP_SHARD_PATH, shard_fn) if TMP_SHARD_PATH else None
                remote_shard_path = os.path.join(OUTPUT_SHARD_PATH, shard_fn)
                create_shard(args, shard_idx, local_shard_path, remote_shard_path, shard_buffer)
                shard_buffer = []
                shard_idx += 1
        except Exception as e:
            print(f"\nError processing triplet {log_name}/{filename}: {e}")
            import traceback
            traceback.print_exc()

    if len(shard_buffer) > 0:
        print(f"\nProcessing remaining {len(shard_buffer)} triplets. Writing shard {shard_idx}...")
        shard_fn = f"{args.dataset}_{split}-{shard_idx:06d}-{split}.tar"
        local_shard_path = os.path.join(TMP_SHARD_PATH, shard_fn) if TMP_SHARD_PATH else None
        remote_shard_path = os.path.join(OUTPUT_SHARD_PATH, shard_fn)
        create_shard(args, shard_idx, local_shard_path, remote_shard_path, shard_buffer)

    print(f"\n\nConversion complete for {split}! Processed {total_triplets_processed} stitched images.")

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

    splits_data[split] = total_triplets_processed

    with open(splits_path, 'w') as f:
        json.dump(splits_data, f, indent=4)
    print(f"Metadata saved: {splits_data}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Organize NavSim camera triplets (L0+F0+R0) into stitched webdataset format (1024x256)"
    )
    parser.add_argument("--dataset", type=str, default="navsim_stitch",
                       help="Dataset name (default: navsim_stitch)")
    parser.add_argument("--output-path", type=str, required=True,
                       help="Output path for webdataset shards")
    parser.add_argument("--image-dir", type=str, required=True,
                       help="Path to the root folder (e.g., .../sensor_blobs/trainval)")
    parser.add_argument("--navsim-logs-dir", type=str, default=None,
                       help="Path to NavSim logs directory (e.g., .../navsim_logs/trainval). "
                            "If not provided, will try to infer from image-dir")
    parser.add_argument("--tmp-shard-path", type=str, default="None",
                       help="Temporary path for shard creation (default: None)")
    parser.add_argument("--samples-per-shard", type=int, default=1000,
                       help="Number of samples per webdataset shard (default: 1000)")
    parser.add_argument("--split", type=str, default=None,
                       help="Specific split to process (train/val). If None, process both based on split-config")
    parser.add_argument("--split-config", type=str, default=None,
                       help="Path to YAML file with train_logs and val_logs")
    parser.add_argument("--max-images", type=int, default=None,
                       help="Maximum number of images to process (for testing)")
    args = parser.parse_args()

    print("="*80)
    print("NavSim Stitched Camera Dataset Preparation")
    print("="*80)
    print(f"Configuration:")
    print(f"  - Dataset name: {args.dataset}")
    print(f"  - Output path: {args.output_path}")
    print(f"  - Image directory: {args.image_dir}")
    print(f"  - Target size: 1024x256 (4:1 aspect ratio)")
    print("="*80 + "\n")

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

    print("\n" + "="*80)
    print("All processing complete!")
    print("="*80)


if __name__ == "__main__":
    main()
