# Copyright (c) 2024 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel


def get_dinov3_feature(
    model: torch.nn.Module, processor: AutoImageProcessor, images: list[np.ndarray], requires_grad: bool = False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Get DINOv3 features.

    Args:
        model (torch.nn.Module): DINOv3 model.
        processor (AutoImageProcessor): DINOv3 input processor.
        images (list[np.ndarray]): images to be encoded, in RGB, uint8. Supports arbitrary resolution.
        requires_grad (bool): maintains gradient. Defaults to False.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (
            cls_token:      last layer embedding from cls token # (B, 1, 1024) for dinov3-vitl16,
            visual_tokens:  last layer embeddings from image # (B, 1024, H, W) BCHW for dinov3-vitl16,
            pooled_cls_token: last layer embedding from cls + layernorm # (B, 1, 1024) for dinov3-vitl16
        )

    Note:
        DINOv3 supports dynamic image sizes with interpolate_pos_encoding=True.
        For 1024×256 input with patch16: outputs 64×16 patches (1024 patches total).
        DINOv3 has 4 register tokens between CLS and patch tokens.
    """
    # Don't resize, keep original resolution; enable position encoding interpolation
    inputs = processor(images, return_tensors="pt", do_resize=False).to(model.device)
    if requires_grad:
        outputs = model(**inputs, interpolate_pos_encoding=True)
    else:
        with torch.no_grad():
            outputs = model(**inputs, interpolate_pos_encoding=True)

    # DINOv3 token structure: [CLS (1)] + [register (4)] + [patches (N)]
    cls_token = outputs.last_hidden_state[:, :1]  # (B, 1, 1024)

    # Skip CLS token and 4 register tokens to get patch tokens
    num_register_tokens = 4
    visual_tokens = outputs.last_hidden_state[:, 1 + num_register_tokens:]  # (B, num_patches, 1024)

    pooled_cls_token = outputs.pooler_output.unsqueeze(1)  # (B, 1, 1024)

    batch_size, num_patches, num_channels = visual_tokens.size()
    visual_tokens = visual_tokens.transpose(1, 2)  # (B, 1024, num_patches)

    # Calculate spatial dimensions based on input image size and patch size
    input_h, input_w = inputs["pixel_values"].shape[2], inputs["pixel_values"].shape[3]
    patch_size = 16  # DINOv3-vitl16 uses patch size 16

    h_patches = input_h // patch_size
    w_patches = input_w // patch_size

    visual_tokens = visual_tokens.reshape(
        batch_size, num_channels, h_patches, w_patches
    )  # (B, 1024, H, W) BCHW

    return cls_token, visual_tokens, pooled_cls_token


def get_dinov3_model(
    model_name: str = "facebook/dinov3-vitl16-pretrain-lvd1689m", device: str | torch.device = "cuda"
) -> tuple[torch.nn.Module, AutoImageProcessor]:
    """Get DINOv3 model and its input processor.

    Args:
        model_name (str, optional): name of DINOv3 model. Defaults to "facebook/dinov3-vitl16-pretrain-lvd1689m".
        device (str | torch.device, optional): device to put the model on. Defaults to "cuda".

    Returns:
        tuple[torch.nn.Module, AutoImageProcessor]: DINOv3 model and the corresponding input processor
    """
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    return model, processor


def print_feature_size(model_name: str = "facebook/dinov3-vitl16-pretrain-lvd1689m") -> None:
    """Print the sizes of features from DINOv3.

    Args:
        model_name (str, optional): the name of DINOv3. Defaults to "facebook/dinov3-vitl16-pretrain-lvd1689m".
    """
    from PIL import Image
    test_image = Image.new('RGB', (1024, 256), color='blue')
    image = [np.array(test_image)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor = get_dinov3_model(model_name=model_name, device=device)
    cls_token, visual_tokens, pooled_cls_token = get_dinov3_feature(model, processor, image)
    
    print(f"Input: 1024x256")
    print(f"cls_token: {cls_token.size()}")
    print(f"visual_tokens: {visual_tokens.size()}")  # Expected: (1, 1024, 16, 64)
    print(f"pooled_cls_token: {pooled_cls_token.size()}")
    # (1, 1, 1024), (1, 1024, H, W), (1, 1, 1024) for dinov3-vitl16


if __name__ == "__main__":
    print_feature_size()