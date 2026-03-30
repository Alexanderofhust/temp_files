#!/usr/bin/env python3
import os
import json
import time
import base64
import math
from io import BytesIO
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Literal

import torch
import torch.distributed as dist
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset as TorchDataset
from tqdm import tqdm
from transformers import AutoTokenizer

# 导入InternVL核心模块（对齐预训练代码路径）
from internvl.model.internvl_chat import InternVLChatModel, InternVLChatConfig
from internvl.train.dataset import build_transform, dynamic_preprocess
from internvl.train.constants import IMG_START_TOKEN, IMG_END_TOKEN, IMG_CONTEXT_TOKEN

# 全局配置（对齐预训练默认参数）
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
IGNORE_INDEX = -100


# ------------------------
# 工具函数（对齐预训练逻辑）
# ------------------------
def build_internvl_transform(
    image_size: int = 448,
    pad2square: bool = False,
    normalize_type: Literal['imagenet', 'clip', 'siglip'] = 'imagenet',
    is_train: bool = False
):
    """构建InternVL风格的图像变换（对齐预训练）"""
    return build_transform(
        is_train=is_train,
        input_size=image_size,
        pad2square=pad2square,
        normalize_type=normalize_type
    )


def process_multiple_images(
    img_list: List,
    transform,
    dynamic_image_size: bool = False,
    min_dynamic_patch: int = 1,
    max_dynamic_patch: int = 12,
    use_thumbnail: bool = False,
    image_size: int = 448,
    max_pixels: int = 262144
):
    """处理多张图片，返回堆叠后的张量 [N_images * N_patches_per_img, 3, H, W]"""
    all_pixel_values = []
    total_num_patches = 0
    
    # 获取单张图片对应的 patch 数量 (InternVL 内部计算)
    # 假设 transform 后的单张图，经过 InternVL 内部处理后，产生 1 个 num_patches
    
    for img_path_or_b64 in img_list:
        pil_img = None
        try:
            # 1. 加载图片
            if isinstance(img_path_or_b64, str) and os.path.exists(img_path_or_b64):
                pil_img = Image.open(img_path_or_b64).convert("RGB")
            elif isinstance(img_path_or_b64, str):
                img_bytes = base64.b64decode(img_path_or_b64)
                pil_img = Image.open(BytesIO(img_bytes)).convert("RGB")
            
            if pil_img is None:
                raise ValueError("Invalid image path or format.")
            
            # 2. 分辨率限制
            w, h = pil_img.size
            if w * h > max_pixels:
                ratio = (max_pixels / (w * h)) ** 0.5
                pil_img = pil_img.resize((max(1, int(w * ratio)), max(1, int(h * ratio))))
            
            # 3. 图像变换 (默认单 patch 模式)
            x = transform(pil_img)  # [3, H, W]
            if x.dim() == 4 and x.shape[0] == 1:
                x = x.squeeze(0)  # 保证 [3, H, W]
                
            all_pixel_values.append(x.unsqueeze(0)) # [1, 3, H, W]
            total_num_patches += 1 # 假设每张图对应 1 个 num_patches
            
        except Exception as e:
            # 异常时返回空白图像的 patch
            blank_img = Image.new("RGB", (image_size, image_size), (255, 255, 255))
            x = transform(blank_img).squeeze(0).unsqueeze(0) # [1, 3, H, W]
            all_pixel_values.append(x)
            total_num_patches += 1
            
    if not all_pixel_values:
        # 如果 img_list 为空，添加一个空白图
        blank_img = Image.new("RGB", (image_size, image_size), (255, 255, 255))
        x = transform(blank_img).squeeze(0).unsqueeze(0)
        all_pixel_values.append(x)
        total_num_patches = 1
        
    # 堆叠所有图像张量：形状变为 [N_images, 3, H, W]
    pixel_values = torch.cat(all_pixel_values, dim=0)
    
    return pixel_values, total_num_patches


def add_image_placeholder(text: str, num_patches: int, tokenizer) -> str:
    """为文本添加图像token占位符（对齐预训练）"""
    if num_patches == 0 or '<image>' not in text:
        return text
    
    # 替换<image>为InternVL的特殊token序列
    img_token_seq = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * tokenizer.num_image_token * num_patches + IMG_END_TOKEN
    return text.replace('<image>', img_token_seq)


class SampleDataset(TorchDataset):
    """自定义 Dataset（保留原有逻辑，新增图像张量存储）"""
    def __init__(self, samples: List[Dict]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            "id": sample["id"],
            "prompt": sample["prompt"],
            "pixel_values": sample["pixel_values"],
            "num_patches": sample["num_patches"]
        }


def collate_fn(batch: List[Dict]):
    ids = [b["id"] for b in batch]
    texts = [b["prompt"] for b in batch]
    
    # 核心修改：使用 torch.cat 拼接所有图像，形状：[N_total, 3, H, W]
    pixel_values = torch.cat([b["pixel_values"] for b in batch], dim=0) 

    num_patches_list = [b["num_patches"] for b in batch]

    return {
        "ids": ids,
        "texts": texts,
        "pixel_values": pixel_values,
        "num_patches_list": num_patches_list
    }


# ------------------------
# 主函数
# ------------------------
def main(
    model_name_or_path: str,
    dataset_jsonl: str,
    dataset_info_json: str,
    save_name: str = "predictions.jsonl",
    batch_size: int = 4,
    num_workers: int = 8,
    image_max_pixels: int = 262144,
    # InternVL专属参数（对齐预训练默认值）
    image_size: int = 448,
    pad2square: bool = False,
    normalize_type: Literal['imagenet', 'clip', 'siglip'] = 'imagenet',
    dynamic_image_size: bool = False,
    min_dynamic_patch: int = 1,
    max_dynamic_patch: int = 12,
    use_thumbnail: bool = False,
    down_sample_ratio: float = 0.5,
    max_new_tokens: int = 512,

):
    
    with open(dataset_info_json, "r", encoding="utf-8") as f:
        dataset_info = json.load(f)

    # 取 VAL_QA 配置
    val_info = dataset_info["VAL_QA"]
    dataset_file = val_info.get("file_name", dataset_jsonl)  # 如果 dataset_jsonl 给了路径就用它
    messages_field = val_info["columns"]["messages"]
    images_field = val_info["columns"]["images"]

    role_tag = val_info["tags"]["role_tag"]
    content_tag = val_info["tags"]["content_tag"]
    user_tag = val_info["tags"]["user_tag"]
    assistant_tag = val_info["tags"]["assistant_tag"]
    # ------------------------
    # 1. 初始化分布式
    # ------------------------
    world_size = int(os.getenv("WORLD_SIZE", 1))
    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    distributed = world_size > 1

    if distributed:
        dist.init_process_group(backend="nccl", world_size=world_size, rank=rank)
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        # 新增：限制每个GPU显存使用（避免OOM）
        torch.cuda.set_per_process_memory_fraction(0.9, device=local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ------------------------
    # 2. 加载模型和Tokenizer（对齐预训练逻辑）
    # ------------------------
    # 2.1 加载Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        add_eos_token=False,
        trust_remote_code=True,
        local_files_only=True
    )
    # 添加InternVL特殊token（对齐预训练）
    token_list = [IMG_START_TOKEN, IMG_END_TOKEN, IMG_CONTEXT_TOKEN]
    tokenizer.add_tokens(token_list, special_tokens=True)
    tokenizer.model_max_length = 8192  # 对齐预训练max_seq_length

    # 2.2 加载InternVL模型配置
    config = InternVLChatConfig.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        local_files_only=True
    )
    config.vision_config.image_size = image_size
    config.downsample_ratio = down_sample_ratio
    config.pad2square = pad2square
    config.dynamic_image_size = dynamic_image_size
    config.use_thumbnail = use_thumbnail
    config.min_dynamic_patch = min_dynamic_patch
    config.max_dynamic_patch = max_dynamic_patch

    # 2.3 计算image token数（对齐预训练）
    patch_size = config.vision_config.patch_size
    tokenizer.num_image_token = int(
        (image_size // patch_size) ** 2 * (down_sample_ratio ** 2)
    )

    # 2.4 加载模型
    if distributed:
        # 分布式推理：仅加载模型到本地GPU，不包装DDP！
        model = InternVLChatModel.from_pretrained(
            model_name_or_path,
            config=config,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            local_files_only=True
        ).to(device)  # 仅移到本地GPU，删除DDP包装
        # 【关键】推理阶段禁用梯度，提升速度+稳定性
        model.eval()
    else:
        # 非分布式环境：逻辑不变（device_map="auto" 自动分配多卡）
        model = InternVLChatModel.from_pretrained(
            model_name_or_path,
            config=config,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            local_files_only=True
        )
        model.eval()

    # 获取新增的图像特殊token的ID
    model.img_start_token_id = tokenizer.convert_tokens_to_ids(IMG_START_TOKEN)
    model.img_end_token_id = tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
    model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    # 新增：关闭所有梯度
    for param in model.parameters():
        param.requires_grad = False
    # 新增：启用混合精度推理（加速且省显存）
    scaler = torch.cuda.amp.GradScaler(enabled=False)  # 推理时禁用梯度缩放，但保留AMP上下文

    # ------------------------
    # 3. 构建图像变换器
    # ------------------------
    transform = build_internvl_transform(
        image_size=image_size,
        pad2square=pad2square,
        normalize_type=normalize_type,
        is_train=False  # 推理模式
    )

    # ------------------------
    # 4. 读取数据并处理图片（保留原有逻辑，适配新处理函数）
    # ------------------------
    lines = open(dataset_file, "r", encoding="utf-8").readlines()
    samples = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {}
        for idx, line in enumerate(lines):
            data = json.loads(line)
            # 根据 dataset_info.json 提取 messages 和 images
            messages = data.get(messages_field, [])
            user_msgs = [m[content_tag] for m in messages if m.get(role_tag) == user_tag]
            prompt_text = user_msgs[-1] if user_msgs else ""
            
            images = data.get(images_field, [])
            N_images = len(images)
            
            # 构造 prompt_text：确保 <image> 占位符数量与图像数量 N_images 严格匹配
            if N_images > 0:
                # 移除所有原有的 <image> 占位符，以防数量不匹配
                prompt_no_images = prompt_text.replace("<image>", "").strip()
                
                # 根据实际图像数量 N_images 插入 N 个 <image> 占位符
                image_placeholders = "\n".join(["<image>"] * N_images)
                
                # 重新构造 prompt：[N 个 <image>] + [原始问题文本]
                prompt_text = f"{image_placeholders}\n{prompt_no_images}"
                
            # 提交多图任务
            future = executor.submit(
                process_multiple_images, # 调用多图处理函数
                images,                  # 传入整个图片列表
                transform,
                dynamic_image_size,
                min_dynamic_patch,
                max_dynamic_patch,
                use_thumbnail,
                image_size,
                image_max_pixels
            )
            futures[idx] = (prompt_text, future, data.get("id", str(idx)))

        # 收集处理结果
        for idx in tqdm(range(len(futures)), desc="Processing images"):
            prompt_text, future, sample_id = futures[idx]
            pixel_values, num_patches = future.result() 
            
            # num_patches 现在是 N_images (每张图一个 num_patches)
            prompt_with_placeholder = add_image_placeholder(prompt_text, num_patches, tokenizer)
            samples.append({
                "id": sample_id,
                "prompt": prompt_with_placeholder,
                "pixel_values": pixel_values,
                "num_patches": num_patches
            })
    # ------------------------
    # 5. 构建 DataLoader
    # ------------------------
    dataset = SampleDataset(samples)
    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            shuffle=False,  # 推理必须False，避免顺序打乱
            drop_last=False  # 保留最后一个不完整批次
        )
    else:
        sampler = None
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # 推理绝对不能shuffle（sampler已控制分片）
        num_workers=0,  # 分布式推理强制设为0，避免多进程CUDA冲突
        collate_fn=collate_fn,
        sampler=sampler,
        pin_memory=True,  # 加速CPU→GPU传输
        drop_last=False   # 保留最后一批次
    )

    # ------------------------
    # 6. 推理（适配InternVL输入格式）
    # ------------------------
    ret_all = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Inference rank {rank}"):
            # 6.1 文本tokenize
            text_inputs = tokenizer(
                batch["texts"],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=tokenizer.model_max_length
            ).to(device)
            
            # 6.2 图像张量处理
            pixel_values = batch["pixel_values"].to(device, dtype=torch.float16)
            
            # 6.3 构建模型输入（对齐InternVL forward格式）
            inputs = {
                "input_ids": text_inputs["input_ids"],
                "attention_mask": text_inputs["attention_mask"],
                "pixel_values": pixel_values
            }
            
            # 6.4 生成预测
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # 推理默认贪心解码
                temperature=0.0
            )
            
            # 6.5 解码结果
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for i, sample_id in enumerate(batch["ids"]):
                ret_all.append({
                    "id": sample_id,
                    "predict": decoded[i],
                    "meta_info": {"num_patches": batch["num_patches_list"][i]}
                })

    # ------------------------
    # 7. 收集多 GPU 结果（保留原有逻辑）
    # ------------------------
    if distributed:
        # 将 JSON 序列化为 bytes
        serialized = [json.dumps(r, ensure_ascii=False).encode("utf-8") for r in ret_all]
        joined = b"\n".join(serialized)
        tensor = torch.ByteTensor(list(joined)).to(device)

        # 收集所有 GPU 的 tensor
        sizes = [torch.tensor(0, device=device) for _ in range(world_size)]
        local_size = torch.tensor(len(tensor), device=device)
        dist.all_gather(sizes, local_size)

        max_size = max([s.item() for s in sizes])
        if len(tensor) < max_size:
            tensor = torch.cat([tensor, torch.zeros(max_size - len(tensor), dtype=torch.uint8, device=device)])
        gathered = [torch.empty_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered, tensor)

        if rank == 0:
            final_results = []
            for i, g in enumerate(gathered):
                size = sizes[i].item()
                data_bytes = g[:size].cpu().numpy().tobytes()
                for line in data_bytes.decode("utf-8").splitlines():
                    final_results.append(json.loads(line))
            ret_all = final_results

    # ------------------------
    # 8. 保存结果（保留原有逻辑）
    # ------------------------
    if rank == 0:
        save_dir = os.path.dirname(save_name)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        with open(save_name, "w", encoding="utf-8") as f:
            for r in ret_all:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Saved predictions to {save_name}")


if __name__ == "__main__":
    import fire
    fire.Fire(main)