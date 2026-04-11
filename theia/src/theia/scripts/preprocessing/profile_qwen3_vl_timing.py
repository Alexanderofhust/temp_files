# Copyright (c) 2024 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import argparse
import time
from statistics import mean

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save as safe_torch_save

from theia.foundation_models.vision_language_models.qwen3_vl import (
    _qwen3_vl_patch_embed,
    _reshape_qwen3_vl_tokens,
    get_qwen3_vl_model,
)


def synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure_section(device: torch.device, fn):
    synchronize_if_needed(device)
    start = time.perf_counter()
    result = fn()
    synchronize_if_needed(device)
    return result, time.perf_counter() - start


def build_dummy_images(batch_size: int, height: int, width: int) -> list[np.ndarray]:
    return [np.random.randint(0, 256, size=(height, width, 3), dtype=np.uint8) for _ in range(batch_size)]


def profile_once(
    model: torch.nn.Module,
    processor,
    images: list[np.ndarray],
    output_dtype: torch.dtype,
) -> dict[str, float]:
    timings: dict[str, float] = {}
    device = model.device

    inputs, timings["processor"] = measure_section(
        torch.device("cpu"),
        lambda: processor.image_processor(images=images, return_tensors="pt"),
    )
    inputs, timings["to_device"] = measure_section(
        device,
        lambda: {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()},
    )

    with torch.no_grad():
        hidden_states, timings["patch_embed"] = measure_section(
            device,
            lambda: _qwen3_vl_patch_embed(model, inputs["pixel_values"]),
        )
        pos_embeds, timings["pos_embed_interp"] = measure_section(
            device,
            lambda: model.fast_pos_embed_interpolate(inputs["image_grid_thw"]),
        )
        hidden_states, timings["add_pos_embed"] = measure_section(device, lambda: hidden_states + pos_embeds)
        rotary_pos_emb, timings["rotary_pos_emb"] = measure_section(
            device,
            lambda: model.rot_pos_emb(inputs["image_grid_thw"]),
        )

        def build_attention_inputs():
            seq_len, _ = hidden_states.size()
            seq_hidden_states = hidden_states.reshape(seq_len, -1)
            seq_rotary = rotary_pos_emb.reshape(seq_len, -1)
            emb = torch.cat((seq_rotary, seq_rotary), dim=-1)
            position_embeddings = (emb.cos(), emb.sin())
            cu_seqlens = torch.repeat_interleave(
                inputs["image_grid_thw"][:, 1] * inputs["image_grid_thw"][:, 2],
                inputs["image_grid_thw"][:, 0],
            ).cumsum(dim=0, dtype=torch.int32)
            cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
            return seq_hidden_states, position_embeddings, cu_seqlens

        (hidden_states, position_embeddings, cu_seqlens), timings["attention_inputs"] = measure_section(
            device,
            build_attention_inputs,
        )

        block_timings = []
        for blk in model.blocks:
            hidden_states, block_time = measure_section(
                device,
                lambda blk=blk, hidden_states=hidden_states: blk(
                    hidden_states,
                    cu_seqlens=cu_seqlens,
                    position_embeddings=position_embeddings,
                ),
            )
            block_timings.append(block_time)

        timings["blocks_total"] = sum(block_timings)
        timings["blocks_mean"] = mean(block_timings)
        timings["blocks_min"] = min(block_timings)
        timings["blocks_max"] = max(block_timings)

        visual_tokens, timings["reshape_tokens"] = measure_section(
            device,
            lambda: _reshape_qwen3_vl_tokens(hidden_states, inputs["image_grid_thw"]),
        )

        def pool_outputs():
            pooled_output = visual_tokens.flatten(start_dim=2).mean(dim=2)
            cls_token = pooled_output.unsqueeze(1)
            pooled_cls_token = cls_token
            return cls_token, pooled_cls_token

        (cls_token, pooled_cls_token), timings["pool_outputs"] = measure_section(device, pool_outputs)

    def copy_to_cpu():
        return {
            "embedding": visual_tokens.detach().cpu().to(output_dtype).contiguous(),
            "cls_token": cls_token.detach().cpu().to(output_dtype).contiguous(),
            "pooled_cls_token": pooled_cls_token.detach().cpu().to(output_dtype).contiguous(),
        }

    cpu_features, timings["detach_cpu"] = measure_section(torch.device("cpu"), copy_to_cpu)

    def serialize_batch():
        serialized = []
        batch_size = cpu_features["embedding"].shape[0]
        for batch_index in range(batch_size):
            sample = {
                feature_key: feature_value[batch_index]
                for feature_key, feature_value in cpu_features.items()
            }
            serialized.append(safe_torch_save(sample))
        return serialized

    _, timings["safe_torch_save"] = measure_section(torch.device("cpu"), serialize_batch)

    timings["forward_total"] = (
        timings["patch_embed"]
        + timings["pos_embed_interp"]
        + timings["add_pos_embed"]
        + timings["rotary_pos_emb"]
        + timings["attention_inputs"]
        + timings["blocks_total"]
        + timings["reshape_tokens"]
        + timings["pool_outputs"]
    )
    timings["feature_pipeline_total"] = (
        timings["processor"]
        + timings["to_device"]
        + timings["forward_total"]
        + timings["detach_cpu"]
        + timings["safe_torch_save"]
    )
    return timings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--output-dtype", type=str, default="bfloat16", choices=["float32", "bfloat16", "float16"])
    args = parser.parse_args()

    output_dtype = getattr(torch, args.output_dtype)
    device = torch.device(args.device)

    load_start = time.perf_counter()
    model, processor = get_qwen3_vl_model(device=device)
    synchronize_if_needed(device)
    load_seconds = time.perf_counter() - load_start

    images = build_dummy_images(args.batch_size, args.height, args.width)
    print(f"model_load_seconds {load_seconds:.4f}")
    print(f"device {device}")
    print(f"batch_size {args.batch_size}")
    print(f"image_size {args.height}x{args.width}")
    print(f"model_attn_impl {getattr(model.config, '_attn_implementation', None)}")

    for _ in range(args.warmup):
        _ = profile_once(model, processor, images, output_dtype)

    all_timings = [profile_once(model, processor, images, output_dtype) for _ in range(args.repeat)]

    keys = [
        "processor",
        "to_device",
        "patch_embed",
        "pos_embed_interp",
        "add_pos_embed",
        "rotary_pos_emb",
        "attention_inputs",
        "blocks_total",
        "blocks_mean",
        "blocks_min",
        "blocks_max",
        "reshape_tokens",
        "pool_outputs",
        "forward_total",
        "detach_cpu",
        "safe_torch_save",
        "feature_pipeline_total",
    ]

    for key in keys:
        values = [timing[key] for timing in all_timings]
        print(f"{key} mean={mean(values):.4f}s")


if __name__ == "__main__":
    main()
