# Copyright (c) 2024 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import numpy as np
import torch
from transformers import AutoImageProcessor, ViTModel


def get_vit_feature(
    model: ViTModel, processor: AutoImageProcessor, images: list[np.ndarray], requires_grad: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get feature from ViT model.

    Args:
        model (ViTModel): ViT model.
        processor (AutoImageProcessor): ViT input processor.
        images (list[np.ndarray]): images to be encoded. Supports arbitrary resolution (e.g., 1024x256).
        requires_grad (bool): maintains gradient. Defaults to False.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: (cls_token, feature)
            - For 224x224 input with patch14: (B, 1280), (B, 1280, 16, 16)
            - For 1024x256 input with patch14: (B, 1280), (B, 1280, 18, 73)
    """
    # Don't resize, keep original resolution; enable position encoding interpolation
    inputs = processor(images, return_tensors="pt", do_resize=False).to(model.device)
    if requires_grad:
        outputs = model(**inputs, interpolate_pos_encoding=True)
    else:
        with torch.no_grad():
            outputs = model(**inputs, interpolate_pos_encoding=True)
    cls_token, last_hidden_state = outputs.last_hidden_state[:, 0], outputs.last_hidden_state[:, 1:]
    batch_size, num_patches, num_channels = last_hidden_state.size()
    last_hidden_state = last_hidden_state.transpose(1, 2)

    # Calculate spatial dimensions based on patch size
    # For patch14: 1024÷14≈73, 256÷14≈18 → (18, 73) patches
    # Note: num_patches may not be a perfect square for rectangular inputs
    h_patches = int(np.sqrt(num_patches))
    w_patches = num_patches // h_patches
    if h_patches * w_patches != num_patches:
        # For rectangular inputs, calculate actual dimensions
        # Assuming width > height for NavSim (1024x256)
        h_patches = int(np.sqrt(num_patches / 4))  # Approximate for 4:1 aspect ratio
        w_patches = num_patches // h_patches

    last_hidden_state = last_hidden_state.reshape(
        batch_size, num_channels, h_patches, w_patches
    )
    return cls_token, last_hidden_state


def get_vit_model(
    model_name: str = "google/vit-huge-patch14-224-in21k", device: str | torch.device = "cuda"
) -> tuple[ViTModel, AutoImageProcessor]:
    """Get ViT model and its corresponding input processor.

    Args:
        model_name (str, optional): the name of vit model. Defaults to "google/vit-huge-patch14-224-in21k".
        device (str | torch.device, optional): device to put model on. Defaults to "cuda".

    Returns:
        tuple[ViTModel, AutoImageProcessor]: _description_
    """
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = ViTModel.from_pretrained(model_name).to(device)
    return model, processor


def print_feature_size(model_name: str = "google/vit-huge-patch14-224-in21k") -> None:
    """Print the size of the feature from ViT.

    Args:
        model_name (str, optional): the name of ViT model. Defaults to "google/vit-huge-patch14-224-in21k".
    """
    from datasets import load_dataset

    dataset = load_dataset("huggingface/cats-image")
    image = dataset["test"]["image"][0]
    image = np.array(image)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor = get_vit_model(model_name=model_name, device=device)
    cls_token, feature = get_vit_feature(model, processor, image)
    print(cls_token.size(), feature.size())
    # cls (1, 1280)
    # feature (1, 1280, 16, 16) BCHW for vit-huge
