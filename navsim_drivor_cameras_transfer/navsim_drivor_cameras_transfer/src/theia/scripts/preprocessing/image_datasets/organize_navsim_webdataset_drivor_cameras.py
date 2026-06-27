"""Organize NavSim CAM_F0/CAM_B0/CAM_L0/CAM_R0 images for DrivoR-aligned Theia distillation.

Each NavSim frame is expanded into four independent image samples in WebDataset:
front, back, left, and right. Images are resized exactly like DrivoR's image
branch input, but are kept as raw RGB uint8 arrays for Theia teacher extraction.
"""

import argparse
import glob
import json
import os
import pickle
import shutil
import tarfile
from io import BytesIO
from typing import Optional

import numpy as np
import webdataset as wds
from numpy.typing import NDArray
from PIL import Image


DRIVOR_CAMERAS = ("CAM_F0", "CAM_B0", "CAM_L0", "CAM_R0")
CAMERA_KEY_NAMES = {
    "CAM_F0": "cam_f0",
    "CAM_B0": "cam_b0",
    "CAM_L0": "cam_l0",
    "CAM_R0": "cam_r0",
}


def load_split_config(split_config_path: str) -> tuple[set[str], set[str]]:
    try:
        import yaml
    except ImportError:
        print("PyYAML not installed. Install with: pip install pyyaml")
        raise

    with open(split_config_path, "r") as f:
        config = yaml.safe_load(f)

    train_logs = set(config.get("train_logs", []))
    val_logs = set(config.get("val_logs", []))
    print(f"Loaded split config: {len(train_logs)} train logs, {len(val_logs)} val logs")
    return train_logs, val_logs


def sanitize_key(value: str) -> str:
    return value.replace(".", "_").replace("/", "_")


def check_existing_shard(path: str) -> bool:
    if not os.path.exists(path):
        return False
    try:
        with tarfile.open(path) as tarf:
            for _ in tarf.getmembers():
                pass
    except (ValueError, tarfile.ReadError, tarfile.CompressionError) as e:
        print(e)
        return False
    return True


def create_shard(
    args: argparse.Namespace,
    shard_idx: int,
    shard_path: Optional[str],
    remote_shard_path: str,
    samples: list[tuple[NDArray, str]],
) -> None:
    if check_existing_shard(remote_shard_path):
        print(f"creating {args.dataset} shard {shard_idx:06d} - check pass, skip")
        return

    print(f"creating {args.dataset} shard {shard_idx:06d} ({len(samples)} samples)")
    if shard_path is None:
        shard_path = remote_shard_path

    with wds.TarWriter(shard_path) as tar_writer:
        for image, basename in samples:
            image_out = BytesIO()
            np.save(image_out, image)
            tar_writer.write({"__key__": basename, "image": image_out.getvalue()})

    if shard_path != remote_shard_path:
        shutil.move(shard_path, remote_shard_path)


def load_resized_rgb(image_path: str, target_size: tuple[int, int]) -> NDArray:
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img = img.resize(target_size, Image.BILINEAR)
        return np.asarray(img)


def iter_drivor_camera_frames(
    image_dir: str,
    navsim_logs_dir: str,
    target_logs: Optional[set[str]],
):
    pkl_files = sorted(glob.glob(os.path.join(navsim_logs_dir, "*.pkl")))
    print(f"Found {len(pkl_files)} pkl files in {navsim_logs_dir}")

    total_frames = 0
    usable_frames = 0
    missing_frames = 0

    for pkl_idx, pkl_file in enumerate(pkl_files):
        log_name = os.path.splitext(os.path.basename(pkl_file))[0]
        if target_logs is not None and log_name not in target_logs:
            continue

        try:
            with open(pkl_file, "rb") as f:
                frames = pickle.load(f)
        except Exception as e:
            print(f"Error loading {pkl_file}: {e}")
            continue

        for frame_idx, frame in enumerate(frames):
            total_frames += 1
            cams = frame.get("cams", {})
            if not all(camera in cams for camera in DRIVOR_CAMERAS):
                missing_frames += 1
                continue

            camera_paths = {}
            missing = False
            for camera in DRIVOR_CAMERAS:
                camera_path = os.path.join(image_dir, cams[camera]["data_path"])
                if not os.path.exists(camera_path):
                    missing = True
                    break
                camera_paths[camera] = camera_path
            if missing:
                missing_frames += 1
                continue

            usable_frames += 1
            frame_token = frame.get("token", f"{frame_idx:06d}")
            yield log_name, frame_token, camera_paths

        if (pkl_idx + 1) % 100 == 0 or (pkl_idx + 1) == len(pkl_files):
            print(
                f"Progress: {pkl_idx + 1}/{len(pkl_files)} pkl files - "
                f"{usable_frames} usable frames, {missing_frames} missing"
            )

    print(f"Scan complete: {usable_frames} usable frames from {total_frames} scanned frames")


def update_splits(output_dataset_path: str, split: str, count: int) -> None:
    for splits_path in [
        os.path.join(output_dataset_path, "splits.json"),
        os.path.join(output_dataset_path, "images", "splits.json"),
    ]:
        splits_data = {}
        if os.path.exists(splits_path):
            try:
                with open(splits_path, "r") as f:
                    splits_data = json.load(f)
            except Exception:
                splits_data = {}
        splits_data[split] = count
        with open(splits_path, "w") as f:
            json.dump(splits_data, f, indent=4)
        print(f"Metadata saved: {splits_path} -> {splits_data}")


def process_split(
    args: argparse.Namespace,
    split: str,
    train_logs: Optional[set[str]],
    val_logs: Optional[set[str]],
) -> None:
    if args.tmp_shard_path == "None":
        tmp_shard_path = None
    else:
        tmp_shard_path = os.path.join(args.tmp_shard_path, args.dataset, "images")
        os.makedirs(tmp_shard_path, exist_ok=True)

    output_dataset_path = os.path.join(args.output_path, args.dataset)
    output_shard_path = os.path.join(output_dataset_path, "images")
    os.makedirs(output_shard_path, exist_ok=True)

    target_logs = train_logs if split == "train" else val_logs
    target_size = (args.target_width, args.target_height)
    print(f"Processing {split}: cameras={DRIVOR_CAMERAS}, target_size={target_size}")

    shard_idx = 0
    sample_count = 0
    frame_count = 0
    shard_buffer: list[tuple[NDArray, str]] = []

    for log_name, frame_token, camera_paths in iter_drivor_camera_frames(args.image_dir, args.navsim_logs_dir, target_logs):
        if args.max_frames is not None and frame_count >= args.max_frames:
            break
        frame_count += 1
        scene_name = sanitize_key(log_name)
        frame_key = sanitize_key(str(frame_token))

        for camera in DRIVOR_CAMERAS:
            image = load_resized_rgb(camera_paths[camera], target_size)
            camera_key = CAMERA_KEY_NAMES[camera]
            basename = f"{scene_name}_{frame_key}_{camera_key}_{args.target_width}x{args.target_height}"
            shard_buffer.append((image, basename))
            sample_count += 1

            if len(shard_buffer) == args.samples_per_shard:
                shard_fn = f"{args.dataset}_{split}-{shard_idx:06d}-{split}.tar"
                local_shard_path = os.path.join(tmp_shard_path, shard_fn) if tmp_shard_path else None
                remote_shard_path = os.path.join(output_shard_path, shard_fn)
                create_shard(args, shard_idx, local_shard_path, remote_shard_path, shard_buffer)
                shard_buffer = []
                shard_idx += 1

        if frame_count % 1000 == 0:
            print(f"{split}: processed {frame_count} frames / {sample_count} camera images")

    if shard_buffer:
        shard_fn = f"{args.dataset}_{split}-{shard_idx:06d}-{split}.tar"
        local_shard_path = os.path.join(tmp_shard_path, shard_fn) if tmp_shard_path else None
        remote_shard_path = os.path.join(output_shard_path, shard_fn)
        create_shard(args, shard_idx, local_shard_path, remote_shard_path, shard_buffer)

    print(f"Conversion complete for {split}: {frame_count} frames, {sample_count} camera images")
    update_splits(output_dataset_path, split, sample_count)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Organize NavSim CAM_F0/CAM_B0/CAM_L0/CAM_R0 into DrivoR-aligned WebDataset"
    )
    parser.add_argument("--dataset", type=str, default="navsim_drivor_cameras")
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--image-dir", type=str, required=True)
    parser.add_argument("--navsim-logs-dir", type=str, required=True)
    parser.add_argument("--split-config", type=str, required=True)
    parser.add_argument("--tmp-shard-path", type=str, default="None")
    parser.add_argument("--samples-per-shard", type=int, default=1000)
    parser.add_argument("--target-width", type=int, default=1148)
    parser.add_argument("--target-height", type=int, default=672)
    parser.add_argument("--split", type=str, default=None, choices=["train", "val"])
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    print("=" * 80)
    print("NavSim DrivoR Camera Dataset Preparation")
    print("=" * 80)
    print(f"Dataset: {args.dataset}")
    print(f"Cameras: {DRIVOR_CAMERAS}")
    print(f"Output path: {args.output_path}")
    print(f"Target image size: {args.target_width}x{args.target_height}")
    print("=" * 80)

    train_logs, val_logs = load_split_config(args.split_config)
    splits = [args.split] if args.split else ["train", "val"]
    for split in splits:
        print(f"\n{'=' * 60}\nProcessing split: {split}\n{'=' * 60}")
        process_split(args, split, train_logs, val_logs)


if __name__ == "__main__":
    main()
