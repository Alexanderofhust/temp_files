# Copyright (c) 2024 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import math

import torch

MODELS = [
    "facebook/dinov2-large",
    "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "facebook/sam-vit-huge",
    "facebook/sam3",
    "google/vit-huge-patch14-224-in21k",
    "google/siglip2-so400m-patch16-naflex",
    "Qwen/Qwen3-VL-8B-Instruct",
    "llava-hf/llava-1.5-7b-hf",
    "openai/clip-vit-large-patch14",
    "LiheYoung/depth-anything-large-hf",
    "depth-anything/DA3-LARGE",
]

# handy model feature size constants
# in the format of (latent_dim, height, width)
# For 1024x256 input resolution with respective patch sizes
# MODEL_FEATURE_SIZES = {
#     "facebook/dinov2-large": (1024, 18, 73),  # patch14: 256÷14=18, 1024÷14=73
#     "facebook/dinov3-vitl16-pretrain-lvd1689m": (1024, 16, 64),  # patch16: 256÷16=16, 1024÷16=64
#     "facebook/sam-vit-huge": (256, 16, 64),    # patch16: 256÷16=16, 1024÷16=64
#     "facebook/sam3": (256, 16, 64),   # patch16: 256÷16=16, 1024÷16=64 (SAM3 similar to SAM2)
#     "google/vit-huge-patch14-224-in21k": (1280, 18, 73),  # patch14: 256÷14=18, 1024÷14=73
#     "google/siglip2-so400m-patch16-naflex": (1152, 16, 64),  # naflex: set max_num_patches=1024 , 16×64 for 1024×256
#     "Qwen/Qwen3-VL-8B-Instruct": (1152, 16, 64),  # pre-merge final vision-layer tokens at 1024x256
#     "llava-hf/llava-1.5-7b-hf": (1024, 24, 24),  # Note: LLaVA may need different handling
#     "openai/clip-vit-large-patch14": (1024, 18, 73),  # patch14: 256÷14=18, 1024÷14=73
#     "LiheYoung/depth-anything-large-hf": (32, 16, 64),  # patch16: 256÷16=16, 1024÷16=64
#     "depth-anything/DA3-LARGE": (1024, 18, 73),  # patch14: 252÷14=18, 1022÷14=73 (DA3 uses patch14, 1024x256 input auto-resizes to 1022x252)
# }
# For 224x224 input resolution with respective patch sizes
MODEL_FEATURE_SIZES = {
    "facebook/dinov3-vitl16-pretrain-lvd1689m": (1024, 14, 14),  # patch16
    "google/siglip2-so400m-patch16-naflex": (1152, 14, 14),  # naflex: set max_num_patches=1024 
    "Qwen/Qwen3-VL-8B-Instruct": (1152, 16, 16),  # pre-merge final vision-layer tokens measured at 224x224 input
    "depth-anything/DA3-LARGE": (1024, 16, 16),  # patch14
}
def get_model_feature_size(
    model_name: str, keep_spatial: bool = False, return_torch_size: bool = False
) -> tuple[int, ...] | torch.Size:
    """
    Get the size of queried model feature.

    Args:
        model_name (str): name of the model.
        keep_spatial (bool): whether to preserve spatial dim. Defaults to False.
        return_torch_size (bool): return torch.Size instead of python tuple. Defaults to False.

    Returns:
        tuple[int, ...] | torch.Size: the size of the feature.
    """
    size: tuple[int, ...] = MODEL_FEATURE_SIZES[model_name]

    if not keep_spatial:
        size = (size[0], math.prod(size[1:]))

    if return_torch_size:
        size = torch.Size(size)

    return size


def get_max_model_spatial_size(
    keep_spatial: bool = True,
    return_torch_size: bool = False,
    return_model_name: bool = False,
) -> tuple[int, ...] | tuple[tuple[int, ...], str]:
    """Get the maximal spatial dimensions from available models

    Args:
        keep_spatial (bool): whether to preserve spatial dim. Defaults to True.
        return_torch_size (bool): return torch.Size instead of python tuple. Defaults to False.
        return_model_name (bool): the name of the model with maximal size. Defaults to False.

    Returns:
        tuple[int, ...] | tuple[tuple[int, ...], str]: the maximal size and optional model name.
    """
    max_flatten_size = -1
    max_size: tuple[int, ...] = ()
    max_size_model_name: str = ""
    for model, size in MODEL_FEATURE_SIZES.items():
        flatten_size = math.prod(size[1:])
        if flatten_size > max_flatten_size:
            max_flatten_size = flatten_size
            max_size = size[1:]
            max_size_model_name = model

    if not keep_spatial:
        max_size = (max_flatten_size,)

    if return_torch_size:
        max_size = torch.Size(max_size)

    if return_model_name:
        return max_size, max_size_model_name
    else:
        return max_size
