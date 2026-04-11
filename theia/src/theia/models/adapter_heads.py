# Copyright (c) 2024 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.


from itertools import chain
from typing import Union

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from torch.nn.functional import interpolate


class Interpolation(nn.Module):
    """Interpolation nn.Module wrap for nn.functional.interpolate.

    Attributes:
        target_size (Union[tuple[int, int], torch.Size]): target spatial size of this interpolation.
    """

    def __init__(self, target_size: Union[tuple[int, int], torch.Size]) -> None:
        super().__init__()
        self.target_size = target_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Very simple forward pass to call interpolate()."""
        return interpolate(x, self.target_size)
    

class LinearAdapterHead(nn.Module):
    """Adapter head contains a single linear layer."""
    def __init__(
        self, source_size: Union[tuple[int, ...], torch.Size], target_size: Union[tuple[int, ...], torch.Size]
    ):
        """Initialization function for LinearAdapterHead.
        Args:
            source_size (Union[tuple[int, ...], torch.Size]): the size of the source feature.
            target_size (Union[tuple[int, ...], torch.Size]): the size of the target feature.
            num_layer (int): number of MLP layers (One linear layer if num_layer = 1).
        """
        super().__init__()

        self.source_size = source_size
        self.target_size = target_size

        source_channel_size = self.source_size[0]
        target_channel_size = self.target_size[0]

        self.adapter = nn.Sequential(
            nn.Linear(source_channel_size, target_channel_size),
        )

    def forward(self, x: torch.Tensor, backbone_no_cls: bool = False) -> torch.Tensor:
        """Forward pass for the adapter. """
        assert backbone_no_cls == False
        # x: [B, (1+H*W), C]
        # LinearAdapterHead is used only when there is cls token in the backbone.
        x = x[:, 0]
        x = self.adapter(x)
        return x  # [B, (H*W), C]


class MLPAdapterHead(nn.Module):
    """MLP Adapter module.

    Transforms features in shape source size [B, (H_s*W_s), C_s] to target size [B, (H_t*W_t), C_t].
    Will first do interpolation to match the spatial size [H_t, W_t],
    followed by MLP to project to the target channel dimension [C_t].

    Attributes:
        source_size (Union[tuple[int, ...], torch.Size]): the size of the source feature. [C, H, W]
        target_size (Union[tuple[int, ...], torch.Size]): the size of the target feature. [C, H, W]
        adapter     (nn.Module):                    the adapter module.
        interpolation (nn.Module):                  interpolation to adjust sizes before MLP.
    """

    def __init__(
        self, source_size: Union[tuple[int, ...], torch.Size], target_size: Union[tuple[int, ...], torch.Size], num_layer: int
    ):
        """Initialization function for MLPAdapter.

        Args:
            source_size (Union[tuple[int, ...], torch.Size]): the size of the source feature.
            target_size (Union[tuple[int, ...], torch.Size]): the size of the target feature.
            num_layer (int): number of MLP layers (One linear layer if num_layer = 1).
        """
        super().__init__()
        assert num_layer >= 1, f"`num_layer` in {self._get_name()} should >= 1. Got {num_layer}"

        self.source_size = source_size
        self.target_size = target_size

        source_channel_size = self.source_size[0]
        target_channel_size = self.target_size[0]

        self.interpolation = nn.Sequential(
            nn.Identity(),
        )
        if self.source_size[1] != self.target_size[1]:
            self.interpolation = nn.Sequential(
                Rearrange("b (h w) c-> b c h w", h=self.source_size[1], w=self.source_size[2]),
                Interpolation(self.target_size[1:]),
                Rearrange("b c h w-> b (h w) c"),
            )

        if num_layer == 1:
            self.adapter = nn.Sequential(
                nn.Linear(source_channel_size, target_channel_size),
            )
        elif num_layer >= 2:
            hidden_dim = source_channel_size * 2
            self.adapter = nn.Sequential(
                nn.Linear(source_channel_size, hidden_dim),
                *list(
                    chain.from_iterable([[nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)] for _ in range(num_layer - 2)])
                ),
                nn.ReLU(),
                nn.Linear(hidden_dim, target_channel_size),
            )

    def forward(self, x: torch.Tensor, backbone_no_cls: bool = False) -> torch.Tensor:
        """Forward pass for the adapter. First interpolation then MLP."""
        # x: [B, (1)+H*W, C]
        if not backbone_no_cls:
            x = x[:, 1:]
        # x: [B, (H*W), C]
        x = self.interpolation(x)
        x = self.adapter(x)
        return x  # [B, (H*W), C]


class ConvAdapterHead(nn.Module):
    """Convolutional Adapter module.

    Transforms features in shape source size [B, (H_s*W_s), C_s] to target size [B, (H_t*W_t), C_t].
    Uses CNN to map channel and spatial sizes jointly.
    Note: only work for (16, 16), (any, any), any <= 14, and (64, 64) spatial sizes for now.

    Attributes:
        source_size (Union[tuple[int, ...], torch.Size]): the size of the source feature.
        target_size (Union[tuple[int, ...], torch.Size]): the size of the target feature.
        adapter     (nn.Module):                    the adapter module.
        interpolation (nn.Module):                  interpolation to adjust sizes before MLP.
    """

    def __init__(
        self,
        source_size: Union[tuple[int, ...], torch.Size],
        target_size: Union[tuple[int, ...], torch.Size],
    ):
        """Initialization function for ConvAdapter.

        Args:
            source_size (Union[tuple[int, ...], torch.Size]): the size of the source feature.
            target_size (Union[tuple[int, ...], torch.Size]): the size of the target feature.
        """
        super().__init__()
        self.source_size = source_size
        self.target_size = target_size

        hidden_dim = self.source_size[0] * 2
        source_channel_size = self.source_size[0]
        target_channel_size = self.target_size[0]

        if self.source_size[1] < 12:
            raise NotImplementedError("feature spatial size smaller than 12x12 is not supported.")
        elif self.source_size[1] < 16:  # pad (any, any), any <= 14 to (16, 16)
            self.pad = nn.Sequential(
                Rearrange("b (h w) c-> b c h w", h=self.source_size[1], w=self.source_size[2]),
                nn.ConvTranspose2d(
                    source_channel_size,
                    source_channel_size,
                    kernel_size=3,
                    stride=1,
                    output_padding=14 - self.source_size[1],
                ),
            )
            self.source_size = (self.source_size[0], 16, 16)
        elif self.source_size[1] == 16 or self.source_size[1] == 64:  # do nothing for (16, 16) and (64, 64)
            self.pad = nn.Sequential(
                Rearrange("b (h w) c-> b c h w", h=self.source_size[1], w=self.source_size[2]),
            )
        else:
            raise NotImplementedError("feature spatial size (>=16x16) other than 16x16 and 64x64 is not supported.")

        if self.source_size[1] < self.target_size[1]:  # (16, 16) / (14, 14) to (64, 64)
            self.adapter = nn.Sequential(
                nn.LayerNorm(self.source_size),
                nn.ConvTranspose2d(source_channel_size, hidden_dim, kernel_size=3, stride=2, padding=1),  # 31
                nn.ReLU(),
                nn.LayerNorm([hidden_dim, 31, 31]),
                nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, output_padding=1),  # 64
                nn.ReLU(),
                nn.LayerNorm([hidden_dim, 64, 64]),
                nn.ConvTranspose2d(hidden_dim, target_channel_size, kernel_size=3, stride=1, padding=1),  # 64
                Rearrange("b c h w-> b (h w) c"),
            )
        elif self.source_size[1] == self.target_size[1]:  # (16, 16) to (16, 16)
            self.adapter = nn.Sequential(
                nn.LayerNorm(self.source_size),
                nn.Conv2d(source_channel_size, hidden_dim, kernel_size=3, padding=1),  # 16
                nn.ReLU(),
                nn.LayerNorm([hidden_dim, *self.source_size[1:]]),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),  # 16
                nn.ReLU(),
                nn.LayerNorm([hidden_dim, *self.source_size[1:]]),
                nn.Conv2d(hidden_dim, target_channel_size, kernel_size=3, padding=1),  # 16
                Rearrange("b c h w-> b (h w) c"),
            )
        else:  # (64, 64) to (16, 16)
            self.adapter = nn.Sequential(
                nn.LayerNorm(self.source_size),
                nn.Conv2d(source_channel_size, hidden_dim, kernel_size=3, stride=2, padding=1),  # 32
                nn.ReLU(),
                nn.LayerNorm([hidden_dim, 32, 32]),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1),  # 16
                nn.ReLU(),
                nn.LayerNorm([hidden_dim, 16, 16]),
                nn.Conv2d(hidden_dim, target_channel_size, kernel_size=3, padding=1),  # 16
                Rearrange("b c h w-> b (h w) c"),
            )

    def forward(self, x: torch.Tensor, backbone_no_cls: bool = False) -> torch.Tensor:
        """Forward pass for ConvAdapter"""
        # x: [B, (1)+H*W, C]
        if not backbone_no_cls:
            x = x[:, 1:]
        # x: [B, H*W, C]
        x = self.pad(x)
        x = self.adapter(x)
        return x  # B, (H*W), C


class LightConvAdapterHead(nn.Module):
    """Light Convolutional Adapter module.

    Transforms features from source size in [B, (H_s*W_s), C_s] to target size [B, (H_t*W_t), C_t].
    Uses CNN to map channel and spatial sizes jointly.
    Supports both square and rectangular feature maps:
    - Square: (14, 14), (16, 16), (64, 64)
    - Rectangular: (16, 64) for 1024x256 input with patch16
    - Target: (18, 73) for patch14 teachers, (16, 64) for patch16 teachers

    Attributes:
        source_size (Union[tuple[int, ...], torch.Size]): the size of the source feature,
            channel first (C, H, W).
        target_size (Union[tuple[int, ...], torch.Size]): the size of the target feature,
            channel first (C, H, W).
        adapter     (nn.Module):                    the adapter module.
    """

    def __init__(
        self,
        source_size: Union[tuple[int, ...], torch.Size],
        target_size: Union[tuple[int, ...], torch.Size],
        hidden_size_factor: Union[int, float] = 1.0,
    ):
        """Initialization function for LightConvAdapter.

        Args:
            source_size (Union[tuple[int, ...], torch.Size]): the size of the source feature (C, H, W).
            target_size (Union[tuple[int, ...], torch.Size]): the size of the target feature (C, H, W).
            hidden_size_factor (Union[int, float]): the size of hidden dim of feature translator
                as a factor of input feature hidden dim.
        """
        super().__init__()
        # Remove square-only restriction to support rectangular feature maps
        self.source_size = source_size
        self.target_size = target_size
        self.hidden_size_factor = hidden_size_factor

        hidden_dim = int(self.source_size[0] * hidden_size_factor)
        source_channel_size = self.source_size[0]
        target_channel_size = self.target_size[0]

        # Padding logic - handle both square and rectangular inputs
        if self.source_size[1] < 12 or self.source_size[2] < 12:
            raise NotImplementedError("feature spatial size smaller than 12 is not supported.")

        # For small square inputs, pad to 16x16
        if self.source_size[1] < 16 and self.source_size[2] < 16 and \
           self.source_size[1] == self.source_size[2] and \
           (self.target_size[1] >= 16 or self.target_size[2] >= 16):
            self.pad = nn.Sequential(
                Rearrange("b (h w) c-> b c h w", h=self.source_size[1], w=self.source_size[2]),
                nn.ConvTranspose2d(
                    source_channel_size,
                    source_channel_size,
                    kernel_size=3,
                    stride=1,
                    output_padding=14 - self.source_size[1],
                ),
            )
            self.source_size = (self.source_size[0], 16, 16)
        else:
            # No padding for other cases - just rearrange
            self.pad = nn.Sequential(
                Rearrange("b (h w) c-> b c h w", h=self.source_size[1], w=self.source_size[2]),
            )

        # Adapter logic - handle different spatial transformations
        # Case 1: (16, 64) → (18, 73) - student to patch14 teachers (ViT/CLIP/DINOv2)
        if self.source_size[1] == 16 and self.source_size[2] == 64 and \
           self.target_size[1] == 18 and self.target_size[2] == 73:
            self.adapter = nn.Sequential(
                nn.LayerNorm(self.source_size),
                # Slight upsample: 16x64 → 18x73
                # Use interpolation + conv for fine-grained control
                nn.Conv2d(source_channel_size, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.LayerNorm([hidden_dim, 16, 64]),
                # Upsample to target size using ConvTranspose2d
                nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0)),  # 18x66
                nn.ReLU(),
                Interpolation((18, 73)),  # Final adjustment to exact size
                nn.LayerNorm([hidden_dim, 18, 73]),
                Rearrange("b c h w-> b (h w) c"),
                nn.Linear(hidden_dim, target_channel_size),
            )
        # Case 2: (16, 64) → (16, 64) - student to patch16 teachers (SAM/Depth-Anything)
        elif self.source_size[1] == 16 and self.source_size[2] == 64 and \
             self.target_size[1] == 16 and self.target_size[2] == 64:
            self.adapter = nn.Sequential(
                nn.LayerNorm(self.source_size),
                nn.Conv2d(source_channel_size, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.LayerNorm([hidden_dim, 16, 64]),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.LayerNorm([hidden_dim, 16, 64]),
                Rearrange("b c h w-> b (h w) c"),
                nn.Linear(hidden_dim, target_channel_size),
            )
        # Case 3: (16, 16) → (64, 64) - original square logic
        elif self.source_size[1] == 16 and self.source_size[2] == 16 and \
             self.target_size[1] == 64 and self.target_size[2] == 64:
            self.adapter = nn.Sequential(
                nn.LayerNorm(self.source_size),
                nn.ConvTranspose2d(source_channel_size, hidden_dim, kernel_size=3, stride=2, padding=1),  # 31
                nn.ReLU(),
                nn.LayerNorm([hidden_dim, 31, 31]),
                nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, output_padding=1),  # 64
                nn.ReLU(),
                nn.LayerNorm([hidden_dim, 64, 64]),
                Rearrange("b c h w-> b (h w) c"),
                nn.Linear(hidden_dim, target_channel_size),
            )
        # Case 4: Same spatial size - works for both square and rectangular
        elif self.source_size[1] == self.target_size[1] and self.source_size[2] == self.target_size[2]:
            self.adapter = nn.Sequential(
                nn.LayerNorm(self.source_size),
                nn.Conv2d(source_channel_size, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.LayerNorm([hidden_dim, *self.source_size[1:]]),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.LayerNorm([hidden_dim, *self.source_size[1:]]),
                Rearrange("b c h w-> b (h w) c"),
                nn.Linear(hidden_dim, target_channel_size),
            )
        # Case 5: (64, 64) → (16, 16) - original square logic
        elif self.source_size[1] == 64 and self.source_size[2] == 64 and \
             self.target_size[1] == 16 and self.target_size[2] == 16:
            self.adapter = nn.Sequential(
                nn.LayerNorm(self.source_size),
                nn.Conv2d(source_channel_size, hidden_dim, kernel_size=3, stride=2, padding=1),  # 32
                nn.ReLU(),
                nn.LayerNorm([hidden_dim, 32, 32]),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1),  # 16
                nn.ReLU(),
                nn.LayerNorm([hidden_dim, 16, 16]),
                Rearrange("b c h w-> b (h w) c"),
                nn.Linear(hidden_dim, target_channel_size),
            )
        # Case 6: (16, 64) → (8, 32) - student to SigLIP2 for 1024x256 input
        elif self.source_size[1] == 16 and self.source_size[2] == 64 and \
             self.target_size[1] == 8 and self.target_size[2] == 32:
            self.adapter = nn.Sequential(
                nn.LayerNorm(self.source_size),
                # Downsample: 16x64 → 8x32 (stride 2 in both dimensions)
                nn.Conv2d(source_channel_size, hidden_dim, kernel_size=3, stride=2, padding=1),  # 8x32
                nn.ReLU(),
                nn.LayerNorm([hidden_dim, 8, 32]),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),  # 8x32
                nn.ReLU(),
                nn.LayerNorm([hidden_dim, 8, 32]),
                Rearrange("b c h w-> b (h w) c"),
                nn.Linear(hidden_dim, target_channel_size),
            )
        # Case 7: (16, 64) → (16, 64) - student to DINOv3/Siglip2 for 1024x256 input (same spatial size)
        # This is already covered by Case 4 (same spatial size)
        # Case 8: Target is 7x7 - original logic
        elif self.target_size[1] == 7 and self.target_size[2] == 7:
            self.adapter = nn.Sequential(
                nn.LayerNorm(self.source_size),
                nn.Conv2d(source_channel_size, hidden_dim, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.LayerNorm([hidden_dim, 7, 7]),
                Rearrange("b c h w-> b (h w) c"),
                nn.Linear(hidden_dim, target_channel_size)
            )
        else:
            raise NotImplementedError(
                f"Spatial transformation from {self.source_size[1:]} to {self.target_size[1:]} is not supported."
            )

    def forward(self, x: torch.Tensor, backbone_no_cls: bool = False) -> torch.Tensor:
        """Forward pass for ConvAdapter"""
        # x: [B, (1)+H*W, C]
        if not backbone_no_cls:
            x = x[:, 1:]
        x = self.pad(x)
        x = self.adapter(x)
        return x  # [B, H*W, C]
