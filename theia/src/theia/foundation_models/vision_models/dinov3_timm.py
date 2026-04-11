# Copyright (c) 2024 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import os
import numpy as np
import torch
import timm
from torchvision import transforms


def get_dinov3_feature(
    model: torch.nn.Module, processor: transforms.Compose, images: list[np.ndarray], requires_grad: bool = False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Get DINOv3 features using timm model.

    Args:
        model (torch.nn.Module): DINOv3 model from timm.
        processor (transforms.Compose): Image preprocessing pipeline.
        images (list[np.ndarray]): images to be encoded, in RGB, uint8.
        requires_grad (bool): maintains gradient. Defaults to False.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (
            cls_token:      last layer embedding from cls token # (B, 1, 1024) for dinov3-vitl16,
            visual_tokens:  last layer embeddings from image # (B, 1024, H, W) BCHW for dinov3-vitl16,
            pooled_cls_token: last layer embedding from cls (same as cls_token for DINOv3)
        )

    Note:
        DINOv3 from timm supports dynamic image sizes with dynamic_img_size=True.
        For 1024×256 input with patch16: outputs 16×64 patches (1024 patches total).
        For 224×224 input with patch16: outputs 14×14 patches (196 patches total).
    """
    # Preprocess images - normalize only, no resize
    processed_images = []
    for img in images:
        # Convert numpy array to PIL Image if needed
        if isinstance(img, np.ndarray):
            from PIL import Image
            if img.dtype == np.uint8:
                img = Image.fromarray(img)
            else:
                img = Image.fromarray((img * 255).astype(np.uint8))
        processed_images.append(processor(img))

    # Stack into batch
    inputs = torch.stack(processed_images).to(model.device if hasattr(model, 'device') else 'cpu')

    # Forward pass
    if requires_grad:
        features = model.forward_features(inputs)
    else:
        with torch.no_grad():
            features = model.forward_features(inputs)

    # Extract tokens
    # DINOv3 with register tokens: [CLS (1) + register (4) + patches (N)] tokens
    cls_token = features[:, 0:1]  # (B, 1, 1024)
    patch_tokens = features[:, 5:]  # Skip register tokens, take patch tokens (B, N, 1024)

    # Reshape patch tokens to spatial format
    batch_size, num_patches, num_channels = patch_tokens.size()

    # Calculate spatial dimensions based on input size and patch size
    # For 1024×256 with patch16: 64×16 = 1024 patches
    # For 224×224 with patch16: 14×14 = 196 patches
    input_h, input_w = inputs.shape[2], inputs.shape[3]
    patch_size = 16
    h = input_h // patch_size
    w = input_w // patch_size

    visual_tokens = patch_tokens.transpose(1, 2).reshape(
        batch_size, num_channels, h, w
    )  # (B, 1024, H, W) BCHW

    # For DINOv3, pooled_cls_token is the same as cls_token
    pooled_cls_token = cls_token

    return cls_token, visual_tokens, pooled_cls_token


def get_dinov3_model(
    model_name: str = "facebook/dinov3-vitl16-pretrain-lvd1689m",
    device: str | torch.device = "cuda",
    local_weights_path: str | None = None
) -> tuple[torch.nn.Module, transforms.Compose]:
    """Get DINOv3 model and its input processor using timm.

    Args:
        model_name (str, optional): name of DINOv3 model. Defaults to "facebook/dinov3-vitl16-pretrain-lvd1689m".
        device (str | torch.device, optional): device to put the model on. Defaults to "cuda".
        local_weights_path (str | None, optional): path to local weights file (.pth).
            If provided, will load from local file. Defaults to None (use timm pretrained weights).

    Returns:
        tuple[torch.nn.Module, transforms.Compose]: DINOv3 model and the corresponding input processor
    """
    print(f"Loading DINOv3 model using timm architecture")

    # Map model name to timm model name
    timm_model_name = 'vit_large_patch16_dinov3'

    # Create model with timm
    if local_weights_path is not None and os.path.exists(local_weights_path):
        print(f"  Creating model architecture: {timm_model_name}")
        model = timm.create_model(timm_model_name, pretrained=False, num_classes=0)

        print(f"  Loading local weights from: {local_weights_path}")
        state_dict = torch.load(local_weights_path, map_location='cpu')
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        print(f"  Loaded local weights")
        if len(missing_keys) > 0:
            print(f"    Missing keys: {len(missing_keys)}")
        if len(unexpected_keys) > 0:
            print(f"    Unexpected keys: {len(unexpected_keys)}")
    else:
        print(f"  Loading pretrained weights from timm: {timm_model_name}")
        model = timm.create_model(timm_model_name, pretrained=True, num_classes=0)

    # Move model to device
    if isinstance(device, str):
        model = model.to(device)
    else:
        model = model.to(device)

    model.eval()

    # Create image preprocessing pipeline
    # DINOv3 uses standard ImageNet normalization
    # Remove Resize and CenterCrop to support arbitrary resolutions
    processor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print(f"✓ DINOv3 model loaded successfully")
    print(f"  Architecture: {timm_model_name}")
    print(f"  Embed dim: {model.embed_dim}")
    print(f"  Patch size: {model.patch_embed.patch_size}")
    print(f"  Supports dynamic resolution: {hasattr(model, 'dynamic_img_size') and model.dynamic_img_size}")
    print(f"  Expected output: (B, 1024, H, W) where H=input_h/16, W=input_w/16")

    return model, processor


def print_feature_size(model_name: str = "facebook/dinov3-vitl16-pretrain-lvd1689m") -> None:
    """Print the sizes of features from DINOv3.

    Args:
        model_name (str, optional): the name of DINOv3. Defaults to "facebook/dinov3-vitl16-pretrain-lvd1689m".
    """
    from PIL import Image

    # Create a dummy image
    image = Image.new('RGB', (224, 224), color='red')
    image = [np.array(image)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor = get_dinov3_model(model_name=model_name, device=device)
    cls_token, visual_tokens, pooled_cls_token = get_dinov3_feature(model, processor, image)
    print(cls_token.size(), visual_tokens.size(), pooled_cls_token.size())
    # (1, 1, 1024), (1, 1024, 14, 14), (1, 1, 1024) for dinov3-vitl16
