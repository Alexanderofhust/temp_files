import math

import timm
import torch
import torch.nn.functional as F
from torch import nn


class _LoRAQKV(nn.Module):
    def __init__(self, qkv: nn.Module, rank: int):
        super().__init__()
        self.qkv = qkv
        self.dim = qkv.in_features
        self.linear_a_q = nn.Linear(self.dim, rank, bias=False)
        self.linear_b_q = nn.Linear(rank, self.dim, bias=False)
        self.linear_a_v = nn.Linear(self.dim, rank, bias=False)
        self.linear_b_v = nn.Linear(rank, self.dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.linear_a_q.weight, a=math.sqrt(5))
        nn.init.zeros_(self.linear_b_q.weight)
        nn.init.kaiming_uniform_(self.linear_a_v.weight, a=math.sqrt(5))
        nn.init.zeros_(self.linear_b_v.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv(x)
        qkv[:, :, : self.dim] += self.linear_b_q(self.linear_a_q(x))
        qkv[:, :, -self.dim :] += self.linear_b_v(self.linear_a_v(x))
        return qkv


class DinoV3ImageEncoder(nn.Module):
    """DINOv3 image encoder with optional LoRA adaptation."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.intermediate_indices = tuple(config.image_encoder_out_indices)
        pretrained = config.image_encoder_pretrained
        try:
            self.model = timm.create_model(
                config.image_architecture,
                pretrained=pretrained,
                num_classes=0,
                img_size=(config.image_encoder_size[1], config.image_encoder_size[0]),
            )
        except Exception as exc:
            if not pretrained:
                raise
            print(f"Falling back to non-pretrained DINOv3 encoder: {exc}")
            self.model = timm.create_model(
                config.image_architecture,
                pretrained=False,
                num_classes=0,
                img_size=(config.image_encoder_size[1], config.image_encoder_size[0]),
            )

        self.num_features = self.model.num_features
        self.patch_size = self.model.patch_embed.patch_size[0]
        self.base_num_prefix_tokens = getattr(self.model, "num_prefix_tokens", 0)
        self.total_num_prefix_tokens = self.base_num_prefix_tokens + config.num_scene_tokens
        self.scene_tokens = nn.Parameter(
            torch.randn(1, config.num_scene_tokens, self.num_features) * 1e-6
        )
        for blk in self.model.blocks:
            if hasattr(blk, "attn") and hasattr(blk.attn, "num_prefix_tokens"):
                blk.attn.num_prefix_tokens = self.total_num_prefix_tokens
        if config.image_encoder_use_lora:
            self._inject_lora(config.image_encoder_lora_rank)
        else:
            for param in self.model.parameters():
                param.requires_grad = True
            self.model.train()

    def _inject_lora(self, rank: int) -> None:
        for param in self.model.parameters():
            param.requires_grad = False
        for blk in self.model.blocks:
            blk.attn.qkv = _LoRAQKV(blk.attn.qkv, rank)
        self.model.train()

    def forward(self, x: torch.Tensor):
        target_width, target_height = self.model.patch_embed.img_size[1], self.model.patch_embed.img_size[0]
        pad_h = max(0, target_height - x.shape[-2])
        pad_w = max(0, target_width - x.shape[-1])
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        return self._forward_scene_intermediates(x)

    def _forward_scene_intermediates(self, x: torch.Tensor):
        x = self.model.patch_embed(x)
        if x.ndim != 4:
            raise ValueError(f"Expected patch embeddings in NHWC format, got shape {tuple(x.shape)}")
        _, grid_h, grid_w, _ = x.shape
        x, rope = self.model._pos_embed(x)
        scene_tokens = self.scene_tokens.expand(x.shape[0], -1, -1)
        if self.base_num_prefix_tokens > 0:
            x = torch.cat(
                [
                    x[:, : self.base_num_prefix_tokens],
                    scene_tokens,
                    x[:, self.base_num_prefix_tokens :],
                ],
                dim=1,
            )
        else:
            x = torch.cat([scene_tokens, x], dim=1)
        x = self.model.norm_pre(x)

        outputs = []
        expected_patch_tokens = grid_h * grid_w
        for idx, blk in enumerate(self.model.blocks):
            x = blk(x, rope=rope)
            if idx in self.intermediate_indices:
                patch_tokens = x[:, self.total_num_prefix_tokens :]
                if patch_tokens.shape[1] != expected_patch_tokens:
                    raise ValueError(
                        f"Unexpected patch token count {patch_tokens.shape[1]} at block {idx}, "
                        f"expected {expected_patch_tokens}"
                    )
                scene_token_features = x[:, self.base_num_prefix_tokens : self.total_num_prefix_tokens].contiguous()
                outputs.append(scene_token_features)
        return outputs
