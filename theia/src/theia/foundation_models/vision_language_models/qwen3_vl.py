# Copyright (c) 2024 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import gc
from typing import Any, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor


QWEN3_VL_8B_MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"


def _select_qwen3_vl_dtype(device: Union[str, int, torch.device]) -> torch.dtype:
    """Prefer bfloat16 on accelerators and float32 on CPU."""
    if isinstance(device, int):
        return torch.bfloat16 if torch.cuda.is_available() else torch.float32

    if isinstance(device, torch.device):
        return torch.bfloat16 if device.type != "cpu" else torch.float32

    return torch.bfloat16 if str(device).lower() != "cpu" else torch.float32


def _get_qwen3_vl_device(model: nn.Module) -> torch.device:
    """Resolve the current device from the live module parameters."""
    try:
        return next(model.parameters()).device
    except StopIteration as exc:
        raise ValueError("Qwen3-VL visual model has no parameters to infer device from.") from exc


class _Qwen3VLVisualWrapper(nn.Module):
    """Thin wrapper that exposes the extracted `.visual` module with the repo's expected interface."""

    def __init__(self, visual_model: nn.Module, config: Any):
        super().__init__()
        self.visual_model = visual_model
        self.config = getattr(visual_model, "config", config)

    @property
    def device(self) -> torch.device:
        return _get_qwen3_vl_device(self.visual_model)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.visual_model(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.visual_model, name)


def _reshape_qwen3_vl_tokens(
    visual_tokens: torch.Tensor,
    image_grid_thw: torch.LongTensor,
) -> torch.Tensor:
    """Convert pre-merge Qwen3-VL vision tokens into BCHW.

    In the local `transformers==4.57.3` implementation, `grid_thw` produced by the official image processor
    already describes the pre-merge vision grid. For a 224x224 image we observed `image_grid_thw = [1, 16, 16]`.
    """
    if visual_tokens.dim() == 2:
        token_sizes = [int(torch.prod(grid).item()) for grid in image_grid_thw]
        token_chunks = torch.split(visual_tokens, token_sizes, dim=0)
    elif visual_tokens.dim() == 3:
        token_chunks = tuple(visual_tokens[i] for i in range(visual_tokens.size(0)))
    else:
        raise ValueError(f"Unexpected Qwen3-VL feature rank: {visual_tokens.dim()}")

    spatial_features = []
    for feature, grid in zip(token_chunks, image_grid_thw):
        t, h, w = [int(v) for v in grid.tolist()]
        if feature.size(0) != t * h * w:
            raise ValueError(
                "Qwen3-VL feature shape does not match image_grid_thw. "
                f"Got {feature.size(0)} tokens but grid {tuple(grid.tolist())} implies {t * h * w}."
            )

        feature = feature.reshape(t, h, w, feature.size(-1))
        feature = feature.permute(3, 0, 1, 2).reshape(feature.size(-1), t * h, w)
        spatial_features.append(feature)

    return torch.stack(spatial_features, dim=0)


def _qwen3_vl_patch_embed(
    model: nn.Module,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    """Apply the vision patch embedding using an equivalent linear projection.

    The upstream implementation uses a Conv3d whose kernel and stride both cover exactly one patch.
    Because processor output is already flattened per patch, this is mathematically identical to a
    single linear projection and avoids an extremely slow Conv3d kernel path on RTX 3090.
    """
    if hidden_states.dim() != 2:
        return model.patch_embed(hidden_states)

    proj = model.patch_embed.proj
    target_dtype = proj.weight.dtype
    weight = proj.weight.reshape(proj.out_channels, -1)
    return F.linear(hidden_states.to(dtype=target_dtype), weight, proj.bias)


def _forward_qwen3_vl_premerge(
    model: nn.Module,
    hidden_states: torch.Tensor,
    grid_thw: torch.LongTensor,
) -> torch.Tensor:
    """Run the Qwen3-VL vision tower up to the layer right before the final patch merger."""
    hidden_states = _qwen3_vl_patch_embed(model, hidden_states)

    pos_embeds = model.fast_pos_embed_interpolate(grid_thw)
    hidden_states = hidden_states + pos_embeds

    rotary_pos_emb = model.rot_pos_emb(grid_thw)

    seq_len, _ = hidden_states.size()
    hidden_states = hidden_states.reshape(seq_len, -1)
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())

    cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
        dim=0, dtype=torch.int32
    )
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    for blk in model.blocks:
        hidden_states = blk(
            hidden_states,
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
        )

    return hidden_states


def get_qwen3_vl_feature(
    model: nn.Module, processor: AutoProcessor, images: list[np.ndarray], requires_grad: bool = False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Get final-layer visual features from the Qwen3-VL vision encoder only.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            cls_token: pseudo cls token from global average pooling, [B, 1, 1152]
            visual_tokens: final raw vision-layer feature map in BCHW, [B, 1152, H, W]
            pooled_cls_token: same as cls_token for compatibility with other teachers
    """
    device = _get_qwen3_vl_device(model)
    inputs = processor.image_processor(images=images, return_tensors="pt")
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    if "pixel_values" not in inputs or "image_grid_thw" not in inputs:
        raise ValueError(
            "Qwen3-VL processor output must contain `pixel_values` and `image_grid_thw` to extract image features."
        )

    if requires_grad:
        last_hidden_state = _forward_qwen3_vl_premerge(model, inputs["pixel_values"], inputs["image_grid_thw"])
    else:
        with torch.no_grad():
            last_hidden_state = _forward_qwen3_vl_premerge(model, inputs["pixel_values"], inputs["image_grid_thw"])

    visual_tokens = _reshape_qwen3_vl_tokens(last_hidden_state, inputs["image_grid_thw"])
    pooled_output = visual_tokens.flatten(start_dim=2).mean(dim=2)
    cls_token = pooled_output.unsqueeze(1)
    pooled_cls_token = cls_token
    return cls_token, visual_tokens, pooled_cls_token


def get_qwen3_vl_model(
    model_name: str = QWEN3_VL_8B_MODEL_NAME,
    device: Union[str, int, torch.device] = "cuda",
) -> tuple[nn.Module, AutoProcessor]:
    """Load Qwen3-VL via `AutoModel.from_pretrained(...).visual` while preserving the repo interface.

    This matches the loading style used in the external reference implementation while returning a visual-only
    module compatible with the rest of this repo.
    """
    if model_name != QWEN3_VL_8B_MODEL_NAME:
        raise NotImplementedError(f"Only {QWEN3_VL_8B_MODEL_NAME} is supported in this repo.")

    selected_dtype = _select_qwen3_vl_dtype(device)
    processor = AutoProcessor.from_pretrained(model_name)
    full_model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        dtype=selected_dtype,
        low_cpu_mem_usage=True,
    )

    visual_config = getattr(full_model.visual, "config", None)
    if visual_config is None:
        visual_config = getattr(getattr(full_model, "config", None), "vision_config", None)

    model = _Qwen3VLVisualWrapper(full_model.visual, config=visual_config)
    model = model.to(device=device, dtype=selected_dtype)
    model.eval()

    del full_model
    gc.collect()

    return model, processor
