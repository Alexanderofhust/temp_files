import math

import timm
import torch
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
            x = nn.functional.pad(x, (0, pad_w, 0, pad_h))
        return self.model.forward_intermediates(
            x,
            indices=list(self.intermediate_indices),
            output_fmt="NCHW",
            intermediates_only=True,
        )
