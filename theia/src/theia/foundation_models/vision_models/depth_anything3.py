# Copyright (c) 2024 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
# File modified for DepthAnything V3 support.

# -----------------------------------------------------------------------
# Copyright 2024 TikTok and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
from typing import Any, Optional, Union

import numpy.typing as npt
import torch
import torch.utils.checkpoint
from torch import nn

# Keep transformers imports for backward compatibility with custom classes
from transformers import AutoImageProcessor
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import DepthEstimatorOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto import AutoBackbone
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.utils import logging

logger = logging.get_logger(__name__)


class DepthAnything3Config(PretrainedConfig):
    r"""
    Configuration class for DepthAnything V3 model.
    Similar to DepthAnythingConfig but with potential architectural improvements.
    """

    model_type = "depth_anything3"

    def __init__(
        self,
        backbone_config: Union[dict[str, Any], PretrainedConfig] = None,
        backbone: Optional[str] = None,
        use_pretrained_backbone: bool = False,
        patch_size: int = 14,
        initializer_range: float = 0.02,
        reassemble_hidden_size: int = 384,
        reassemble_factors: tuple[Union[int, float], ...] = (4, 2, 1, 0.5),
        neck_hidden_sizes: tuple[int, ...] = (48, 96, 192, 384),
        fusion_hidden_size: int = 64,
        head_in_index: int = -1,
        head_hidden_size: int = 32,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        if use_pretrained_backbone:
            raise ValueError("Pretrained backbones are not supported yet.")

        if backbone_config is not None and backbone is not None:
            raise ValueError("You can't specify both `backbone` and `backbone_config`.")

        if backbone_config is None and backbone is None:
            logger.info("`backbone_config` is `None`. Initializing the config with the default `Dinov2` backbone.")
            backbone_config = CONFIG_MAPPING["dinov2"](
                image_size=518,
                hidden_size=384,
                num_attention_heads=6,
                out_indices=[9, 10, 11, 12],
                apply_layernorm=True,
                reshape_hidden_states=False,
            )
        elif isinstance(backbone_config, dict):
            backbone_model_type = backbone_config.get("model_type")
            config_class = CONFIG_MAPPING[backbone_model_type]
            backbone_config = config_class.from_dict(backbone_config)

        self.backbone_config = backbone_config
        self.backbone = backbone
        self.use_pretrained_backbone = use_pretrained_backbone
        self.reassemble_hidden_size = reassemble_hidden_size
        self.patch_size = patch_size
        self.initializer_range = initializer_range
        self.reassemble_factors = reassemble_factors
        self.neck_hidden_sizes = neck_hidden_sizes
        self.fusion_hidden_size = fusion_hidden_size
        self.head_in_index = head_in_index
        self.head_hidden_size = head_hidden_size

    def to_dict(self) -> dict[str, Any]:
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        if output["backbone_config"] is not None:
            output["backbone_config"] = self.backbone_config.to_dict()
        output["model_type"] = self.__class__.model_type
        return output


# Import base classes from depth_anything module
from theia.foundation_models.vision_models.depth_anything import (
    DepthAnythingReassembleLayer,
    DepthAnythingReassembleStage,
    DepthAnythingPreActResidualLayer,
    DepthAnythingFeatureFusionLayer,
    DepthAnythingFeatureFusionStage,
    DepthAnythingPreTrainedModel,
)


class DepthAnything3Neck(nn.Module):
    """DepthAnything V3 Neck module."""

    def __init__(self, config: DepthAnything3Config):
        super().__init__()
        self.config = config
        self.reassemble_stage = DepthAnythingReassembleStage(config)
        self.convs = nn.ModuleList()
        for channel in config.neck_hidden_sizes:
            self.convs.append(nn.Conv2d(channel, config.fusion_hidden_size, kernel_size=3, padding=1, bias=False))
        self.fusion_stage = DepthAnythingFeatureFusionStage(config)

    def forward(
        self, hidden_states: list[torch.Tensor], patch_height: Optional[int] = None, patch_width: Optional[int] = None
    ) -> list[torch.Tensor]:
        if not isinstance(hidden_states, (tuple, list)):
            raise ValueError("hidden_states should be a tuple or list of tensors")
        if len(hidden_states) != len(self.config.neck_hidden_sizes):
            raise ValueError("The number of hidden states should be equal to the number of neck hidden sizes.")
        hidden_states = self.reassemble_stage(hidden_states, patch_height, patch_width)
        features = [self.convs[i](feature) for i, feature in enumerate(hidden_states)]
        output = self.fusion_stage(features)
        return output


class DepthAnything3DepthEstimationHead(nn.Module):
    """DepthAnything V3 depth estimation head."""

    def __init__(self, config: DepthAnything3Config):
        super().__init__()
        self.head_in_index = config.head_in_index
        self.patch_size = config.patch_size
        features = config.fusion_hidden_size
        self.conv1 = nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(features // 2, config.head_hidden_size, kernel_size=3, stride=1, padding=1)
        self.activation1 = nn.ReLU()
        self.conv3 = nn.Conv2d(config.head_hidden_size, 1, kernel_size=1, stride=1, padding=0)
        self.activation2 = nn.ReLU()

    def forward(self, hidden_states: list[torch.Tensor], patch_height: int, patch_width: int) -> torch.Tensor:
        hidden_states = hidden_states[self.head_in_index]
        predicted_depth = self.conv1(hidden_states)
        predicted_depth = nn.functional.interpolate(
            predicted_depth,
            (int(patch_height * self.patch_size), int(patch_width * self.patch_size)),
            mode="bilinear",
            align_corners=True,
        )
        predicted_depth = self.conv2(predicted_depth)
        predicted_depth = self.activation1(predicted_depth)
        predicted_depth = self.conv3(predicted_depth)
        predicted_depth = self.activation2(predicted_depth)
        predicted_depth = predicted_depth.squeeze(dim=1)
        return predicted_depth


class DepthAnything3ForDepthEstimation(DepthAnythingPreTrainedModel):
    """DepthAnything V3 model for depth estimation."""

    def __init__(self, config: DepthAnything3Config):
        super().__init__(config)
        self.backbone = AutoBackbone.from_config(config.backbone_config)
        self.neck = DepthAnything3Neck(config)
        self.head = DepthAnything3DepthEstimationHead(config)
        self.post_init()

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple[torch.Tensor, ...], DepthEstimatorOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        outputs = self.backbone.forward_with_filtered_kwargs(
            pixel_values, output_hidden_states=output_hidden_states, output_attentions=output_attentions
        )
        hidden_states = outputs.feature_maps

        _, _, height, width = pixel_values.shape
        patch_size = self.config.patch_size
        patch_height = height // patch_size
        patch_width = width // patch_size

        hidden_states = self.neck(hidden_states, patch_height, patch_width)
        predicted_depth = self.head(hidden_states, patch_height, patch_width)

        loss = None
        if labels is not None:
            raise NotImplementedError("Training is not implemented yet")

        if not return_dict:
            if output_hidden_states:
                output = (predicted_depth,) + outputs[1:]
            else:
                output = (predicted_depth,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return DepthEstimatorOutput(
            loss=loss,
            predicted_depth=predicted_depth,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions,
        )


class DepthAnything3NeckFeature(DepthAnything3ForDepthEstimation):
    """DepthAnything V3 with only neck feature returned."""

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.Tensor:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        outputs = self.backbone.forward_with_filtered_kwargs(
            pixel_values, output_hidden_states=output_hidden_states, output_attentions=output_attentions
        )
        hidden_states = outputs.feature_maps

        _, _, height, width = pixel_values.shape
        patch_size = self.config.patch_size
        patch_height = height // patch_size
        patch_width = width // patch_size

        hidden_states = self.neck(hidden_states, patch_height, patch_width)
        return hidden_states


class DepthAnything3HeadFeature(DepthAnything3ForDepthEstimation):
    """DepthAnything V3 with only head feature returned."""

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.Tensor:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        outputs = self.backbone.forward_with_filtered_kwargs(
            pixel_values, output_hidden_states=output_hidden_states, output_attentions=output_attentions
        )
        hidden_states = outputs.feature_maps

        _, _, height, width = pixel_values.shape
        patch_size = self.config.patch_size
        patch_height = height // patch_size
        patch_width = width // patch_size

        hidden_states = self.neck(hidden_states, patch_height, patch_width)
        hidden_states = hidden_states[-1]

        head_feature = self.head.conv1(hidden_states)
        head_feature = nn.functional.interpolate(
            head_feature,
            (int(patch_height * patch_size), int(patch_width * patch_size)),
            mode="bilinear",
            align_corners=True,
        )
        head_feature = self.head.conv2(head_feature)
        return head_feature


def get_depth_anything3_feature(
    model,  # DepthAnything3 from official API
    processor,  # Not used, kept for compatibility
    images: list[npt.NDArray],
    requires_grad: Optional[bool] = False,
    feature_layer: int = 23,
) -> torch.Tensor:
    """Get feature from DepthAnything V3 model using official API.

    This function extracts features from the last layer of the backbone (Layer 23),
    similar to DINOv3 and SigLIP2's approach of using the final transformer layer.

    Args:
        model: DepthAnything3 model from depth_anything_3.api
        processor: Not used, kept for API compatibility
        images: List of RGB images as numpy arrays (H, W, 3), uint8
        requires_grad: Whether to maintain gradients
        feature_layer: Which layer to extract features from (0-23, default 23 for last layer)

    Returns:
        torch.Tensor: Features of shape [B, C, H, W] where:
            - B is batch size
            - C is 1024 (feature channels)
            - H, W are spatial dimensions (depends on input size)
    """
    import cv2

    # Process each image
    image_tensors = []
    for img in images:
        # Ensure image dimensions are multiples of 14
        h, w = img.shape[:2]
        new_h = (h // 14) * 14
        new_w = (w // 14) * 14

        if new_h != h or new_w != w:
            # Resize using cv2 to keep numpy array format
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Convert numpy array (H, W, C) to tensor (C, H, W) and normalize to [0, 1]
        # Keep as numpy array, then convert to tensor with explicit float32 dtype
        img_float = img.astype('float32') / 255.0  # Normalize to [0, 1]
        img_tensor = torch.from_numpy(img_float).permute(2, 0, 1)  # (H, W, C) -> (C, H, W)

        image_tensors.append(img_tensor)

    # Stack into batch: [B, C, H, W]
    batch_tensor = torch.stack(image_tensors, dim=0)

    # Add sequence dimension: [B, S, C, H, W] where S=1
    # Move to device - dtype will be handled by autocast in DA3 API's forward method
    # Use stored device attribute to ensure correct device placement
    device = getattr(model, '_device', 'cuda')
    batch_tensor = batch_tensor.unsqueeze(1).to(device=device)

    # Option 1: Use complete forward pass (similar to DA1's neck approach)
    # Call model without export_feat_layers to get full output including processed features
    export_layers = [feature_layer]

    if requires_grad:
        outputs = model(batch_tensor, export_feat_layers=export_layers)
    else:
        with torch.no_grad():
            outputs = model(batch_tensor, export_feat_layers=export_layers)

    # Extract feature from aux output (backbone intermediate layers)
    # This is similar to DA1's approach of extracting features after backbone processing
    feat_key = f'feat_layer_{feature_layer}'
    if 'aux' in outputs and feat_key in outputs['aux']:
        features = outputs['aux'][feat_key]  # [B, S, H, W, C]
        # Remove sequence dimension and permute to [B, C, H, W]
        features = features.squeeze(1)  # [B, H, W, C]
        features = features.permute(0, 3, 1, 2)  # [B, C, H, W]
        return features
    else:
        raise ValueError(f"Feature layer {feature_layer} not found in model output")


def get_depth_anything3_model(
    model_name: Optional[str] = "depth-anything/DA3-LARGE-1.1",
    device: Optional[Union[str, torch.device]] = "cuda",
    selected_feature: Optional[str] = "layer23",
) -> tuple[Any, None]:
    """Get DepthAnything V3 model using official API.

    Args:
        model_name: Model name on HuggingFace Hub
        device: Device to load model on
        selected_feature: Which feature layer to use (layer0-layer23, default layer23 for last layer)

    Returns:
        tuple: (model, None) - processor is None as it's not needed for official API
    """
    from depth_anything_3.api import DepthAnything3

    # Load model using official API
    model = DepthAnything3.from_pretrained(model_name)
    # Keep model in float32 - DA3 API's autocast will handle dtype conversion automatically
    # Converting to BFloat16 causes issues with LayerNorm layers
    model = model.to(device=device)
    model.eval()

    # Store feature layer selection and device as model attributes
    # Support all 24 layers (0-23), default to layer 23 (last layer) to align with DINOv3/SigLIP2
    layer_map = {
        "layer9": 9,
        "layer10": 10,
        "layer11": 11,
        "layer12": 12,
        "layer23": 23,  # Last layer (default)
    }
    model._feature_layer = layer_map.get(selected_feature, 23)
    model._device = device  # Store device explicitly for input tensor placement

    return model, None


def print_feature_size(
    model_name: Optional[str] = "depth-anything/DA3-LARGE-1.1", selected_feature: Optional[str] = "layer12"
) -> None:
    """Print the size of the feature from DepthAnything V3."""
    import requests
    from PIL import Image
    import numpy as np

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    pil_image = Image.open(requests.get(url, stream=True).raw)

    # Convert to numpy array
    image = np.array(pil_image)

    # Ensure dimensions are multiples of 14
    h, w = image.shape[:2]
    new_h = (h // 14) * 14
    new_w = (w // 14) * 14
    if new_h != h or new_w != w:
        pil_image = pil_image.resize((new_w, new_h), Image.BILINEAR)
        image = np.array(pil_image)

    images = [image]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = get_depth_anything3_model(model_name=model_name, device=device, selected_feature=selected_feature)

    with torch.no_grad():
        embedding = get_depth_anything3_feature(model, None, images, feature_layer=model._feature_layer)

    print(f"Feature shape: {embedding.size()}")
    print(f"Expected format: [batch_size, channels, height, width]")
    print(f"Channels: {embedding.size(1)}, Spatial: ({embedding.size(2)}, {embedding.size(3)})")
