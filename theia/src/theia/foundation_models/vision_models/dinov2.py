# Copyright (c) 2024 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import numpy as np
import torch
from transformers import AutoImageProcessor, Dinov2Model


def get_dinov2_feature(
    model: Dinov2Model, processor: AutoImageProcessor, images: list[np.ndarray], requires_grad: bool = False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Get DINOv2 features.

    Args:
        model (Dinov2Model): DINOv2 model.
        processor (AutoImageProcessor): DINOv2 input processor.
        images (list[np.ndarray]): images to be encoded, in RGB, uint8. Supports arbitrary resolution.
        requires_grad (bool): maintains gradient. Defaults to False.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (
            cls_token:      last layer embedding from cls token # (B, 1, 1024) if dinov2-large,
            visual_tokens:  last layer embeddings from image # (B, 1024, H, W) BCHW if dinov2-large,
            pooled_cls_token: last layer embedding from cls + layernorm # (B, 1, 1024) if dinov2-large
        )
    """
    # Don't resize, keep original resolution; enable position encoding interpolation
    inputs = processor(images, return_tensors="pt", do_resize=False).to(model.device)
    if requires_grad:
        outputs = model(**inputs, interpolate_pos_encoding=True)
    else:
        with torch.no_grad():
            outputs = model(**inputs, interpolate_pos_encoding=True)
    cls_token = outputs.last_hidden_state[:, :1]  # (B, 1, 1024) if dinov2-large
    visual_tokens = outputs.last_hidden_state[:, 1:]  # (B, num_patches, 1024) if dinov2-large
    pooled_cls_token = outputs.pooler_output.unsqueeze(1)  # (B, 1, 1024) if dinov2-large
    batch_size, num_patches, num_channels = visual_tokens.size()
    visual_tokens = visual_tokens.transpose(1, 2)

    # Calculate spatial dimensions based on patch size
    # For patch14: 1024÷14≈73, 256÷14≈18 → (18, 73) patches
    h_patches = int(np.sqrt(num_patches))
    w_patches = num_patches // h_patches
    if h_patches * w_patches != num_patches:
        # For rectangular inputs, calculate actual dimensions
        h_patches = int(np.sqrt(num_patches / 4))  # Approximate for 4:1 aspect ratio
        w_patches = num_patches // h_patches

    visual_tokens = visual_tokens.reshape(
        batch_size, num_channels, h_patches, w_patches
    )  # (B, 1024, H, W) BCHW
    return cls_token, visual_tokens, pooled_cls_token


def get_dinov2_model(
    model_name: str = "facebook/dinov2-large", device: str | torch.device = "cuda"
) -> tuple[Dinov2Model, AutoImageProcessor]:
    """Get DINOv2 model and its input processor.

    Args:
        model_name (str, optional): name of DINOv2 model. Defaults to "facebook/dinov2-large".
        device (str | torch.device, optional): device to put the model on. Defaults to "cuda".

    Returns:
        tuple[Dinov2Model, AutoImageProcessor]: DINOv2 model and the corresponding input processor
    """
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = Dinov2Model.from_pretrained(model_name).to(device)
    return model, processor


def print_feature_size(model_name: str = "facebook/dinov2-large") -> None:
    """Print the sizes of features from DINOv2.

    Args:
        model_name (str, optional): the name of DINOv2. Defaults to "facebook/dinov2-large".
    """
    from datasets import load_dataset

    dataset = load_dataset("huggingface/cats-image")
    image = dataset["test"]["image"][0]
    image = [np.array(image)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor = get_dinov2_model(model_name=model_name, device=device)
    cls_token, visual_tokens, pooled_cls_token = get_dinov2_feature(model, processor, image)
    print(cls_token.size(), visual_tokens.size(), pooled_cls_token.size())
    # (1, 1, 1024), (1, 1024, 16, 16), (1, 1, 1024) for dinov2-large
