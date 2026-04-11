# Copyright (c) 2024 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import numpy as np
import torch
from transformers import AutoProcessor, Siglip2VisionModel


def get_siglip2_feature(
    model: Siglip2VisionModel, processor: AutoProcessor, images: list[np.ndarray], requires_grad: bool = False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Get features from the visual encoder of SigLIP2.

    Args:
        model (Siglip2VisionModel): SigLIP2 vision model.
        processor (AutoProcessor): SigLIP2 input processor.
        images (list[np.ndarray]): images to be encoded, in RGB, uint8.
        requires_grad (bool): maintains gradient. Defaults to False.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: features from SigLIP2 (
            cls_token:      pooled global feature (used as pseudo cls_token)  # (B, 1, 1152),
            visual_tokens:  last layer embeddings from image patches          # (B, 1152, H, W) BCHW,
            pooled_cls_token: same as cls_token (pooler_output)               # (B, 1, 1152)
        )

    Note:
        SigLIP2 does not have a CLS token like CLIP. It only outputs patch tokens.
        We use the pooler_output (global average pooling) as a pseudo cls_token for compatibility.

        SigLIP2-naflex supports flexible aspect ratios with max 1024 patches:
        - For 224x224 input: processor resizes to 256x256 -> 16x16 patches
        - For 1024x256 input: outputs 64x16 patches (preserves 4:1 aspect ratio)
        - spatial_shapes in inputs tells us the actual (H, W) layout
    """
    # Let processor handle resizing automatically
    # SigLIP2-naflex will automatically adjust to input aspect ratio with max 1024 patches
    inputs = processor(images=images, max_num_patches=1024, return_tensors="pt").to(model.device)
    if requires_grad:
        outputs = model(**inputs)
    else:
        with torch.no_grad():
            outputs = model(**inputs)

    # SigLIP2 architecture specifics:
    # - last_hidden_state: (B, num_patches, 1152) - only patch tokens, no CLS token
    # - pooler_output: (B, 1152) - global pooled feature
    # - spatial_shapes: (B, 2) - actual (H, W) layout of patches

    # Use pooler_output as pseudo cls_token for compatibility with other models
    pooler_output = outputs.pooler_output.unsqueeze(1)  # (B, 1, 1152)
    cls_token = pooler_output  # (B, 1, 1152)

    # Extract patch tokens
    visual_tokens = outputs.last_hidden_state  # (B, num_patches, 1152)

    # Get spatial dimensions from inputs
    # spatial_shapes format: (B, 2) where each row is [height, width]
    if "spatial_shapes" in inputs:
        h, w = inputs["spatial_shapes"][0].tolist()  # Get first batch's spatial shape
    else:
        # Fallback: assume square layout
        batch_size, num_patches, num_channels = visual_tokens.size()
        h = w = int(np.sqrt(num_patches))

    # Reshape to spatial format (B, num_patches, 1152) -> (B, 1152, H, W)
    batch_size, num_patches, num_channels = visual_tokens.size()
    visual_tokens = visual_tokens.transpose(1, 2)  # (B, 1152, num_patches)
    visual_tokens = visual_tokens.reshape(batch_size, num_channels, h, w)  # (B, 1152, H, W) BCHW

    pooled_cls_token = pooler_output  # (B, 1, 1152)

    return cls_token, visual_tokens, pooled_cls_token


def get_siglip2_model(
    model_name: str = "google/siglip2-so400m-patch16-naflex",
    device: str | torch.device = "cuda",
    max_num_patches: int = 1024,
) -> tuple[Siglip2VisionModel, AutoProcessor]:
    """Get SigLIP2 vision model and its input processor.

    Args:
        model_name (str, optional): name of SigLIP2 model. Defaults to "google/siglip2-so400m-patch16-naflex".
        device (str | torch.device, optional): device to put the model on. Defaults to "cuda".
        max_num_patches (int, optional): maximum number of patches for naflex. Defaults to 1024.

    Returns:
        tuple[Siglip2VisionModel, AutoProcessor]: SigLIP2 vision model and the corresponding input processor.
    """
    processor = AutoProcessor.from_pretrained(model_name)
    # Update processor's max_num_patches for naflex to support larger images
    if hasattr(processor, "image_processor") and hasattr(processor.image_processor, "max_num_patches"):
        processor.image_processor.max_num_patches = max_num_patches
    model = Siglip2VisionModel.from_pretrained(model_name).to(device)
    model.eval()
    return model, processor


def print_feature_size(model_name: str = "google/siglip2-so400m-patch16-naflex") -> None:
    """Print the sizes of features from SigLIP2.

    Args:
        model_name (str, optional): the name of SigLIP2 model. Defaults to "google/siglip2-so400m-patch16-naflex".
    """
    import requests
    from PIL import Image

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = [np.array(Image.open(requests.get(url, stream=True, timeout=10).raw))]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor = get_siglip2_model(model_name, device=device)
    cls_token, visual_tokens, pooled_cls_token = get_siglip2_feature(model, processor, image)

    print(model_name, cls_token.size(), visual_tokens.size(), pooled_cls_token.size())


if __name__ == "__main__":
    print_feature_size()