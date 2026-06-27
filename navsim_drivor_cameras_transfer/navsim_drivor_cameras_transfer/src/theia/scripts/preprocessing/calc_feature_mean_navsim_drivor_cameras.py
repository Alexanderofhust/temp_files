# Copyright (c) 2024 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""Calculate channel-wise mean/std for navsim_drivor_cameras feature shards."""

import argparse
import glob
import os
from io import BytesIO

import numpy as np
import torch
import webdataset as wds
from einops import rearrange
from safetensors.torch import load as sft_load
from torch.utils.data import default_collate


def decode_dataset_sample(key: str, data: bytes) -> bytes | torch.Tensor:
    if ".safetensors" in key:
        sft = sft_load(data)
        return rearrange(sft["embedding"], "c h w -> (h w) c")
    if key == ".image":
        return torch.from_numpy(np.load(BytesIO(data)))
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str)
    parser.add_argument("--output-path", type=str)
    parser.add_argument("--stats-prefix", type=str, default="navsim_drivor_cameras")
    args = parser.parse_args()

    models = [m for m in os.listdir(args.dataset_path) if os.path.isdir(os.path.join(args.dataset_path, m))]
    for model in models:
        if model in {"images", "image", "images_val"}:
            continue

        mean_path = f"{args.output_path}/{args.stats_prefix}_mean_{model}.npy"
        var_path = f"{args.output_path}/{args.stats_prefix}_var_{model}.npy"
        if os.path.exists(mean_path) and os.path.exists(var_path):
            continue

        print(model)
        model_mean: torch.Tensor = None
        model_var_sum: torch.Tensor = None
        n = 0
        ds = (
            wds.WebDataset(sorted(glob.glob(f"{args.dataset_path}/{model}/*.tar")), shardshuffle=False)
            .decode(decode_dataset_sample)
            .batched(256, collation_fn=default_collate)
        )

        key = f"{model}.safetensors".lower()
        for batch_idx, batch in enumerate(ds):
            if model_mean is None:
                model_mean = torch.zeros((batch[key].size(-1)))
            new_n = np.prod(batch[key].size()[:2])
            batch_mean = batch[key].float().mean((0, 1))
            model_mean = (model_mean * n + batch_mean * new_n) / (n + new_n)
            n += new_n
            print(f"calc {model} mean {batch_idx * 256:07d}\r", end="")

        np.save(mean_path, model_mean.numpy())

        for batch_idx, batch in enumerate(ds):
            if model_var_sum is None:
                model_var_sum = torch.zeros((batch[key].size(-1)))
            model_var_sum += ((batch[key].float() - model_mean) ** 2).sum((0, 1))
            print(f"calc {model} var {batch_idx * 256:07d}\r", end="")

        model_var = torch.sqrt(model_var_sum / (n - 1))
        np.save(var_path, model_var.numpy())


if __name__ == "__main__":
    main()
