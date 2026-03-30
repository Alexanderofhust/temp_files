import os
import re
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, ConcatDataset
from transformers import AutoTokenizer, HfArgumentParser, TrainingArguments
from dataclasses import dataclass, field
from typing import Optional
from functools import partial

# 假设你的类定义都在当前文件或已导入
# from your_module import ImplicitCoTDataset, implicit_cot_collator, LazySupervisedDataset

# ==============================================================================
# 1. 简化的参数定义（仅用于测试）
# ==============================================================================
@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="OpenGVLab/InternVL2-2B") # 请确保路径正确
    ps_version: str = field(default='v2')
    M: int = field(default=4)

@dataclass
class DataTrainingArguments:
    meta_path: str = field(default="path/to/your/nuscenes_meta.json") # 修改为你的实际 meta path
    conv_style: str = field(default='internvl2_5')
    force_image_size: int = field(default=448)
    max_dynamic_patch: int = field(default=12)
    down_sample_ratio: float = field(default=0.5)
    dynamic_image_size: bool = field(default=True)
    use_thumbnail: bool = field(default=True)
    max_seq_length: int = field(default=8192)

# ==============================================================================
# 2. 测试函数
# ==============================================================================
def test_dataloader():
    # --- A. 初始化参数 ---
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    # 模拟命令行输入
    model_args, data_args, training_args = parser.parse_args_into_dataclasses(args=[
        "--output_dir", "./test_out",
        "--model_name_or_path", "/data1/chenxiwu/ReCogDrive/model/internVL3_2B", # 替换为你的模型路径
        "--meta_path", "/data1/chenxiwu/Drive/recogdrive/internvl_chat/shell/data_info/nuscenes_val.json" # 替换为你的meta路径
    ])

    # --- B. 初始化 Tokenizer ---
    print("Initializing Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True, use_fast=False)
    
    # 添加 InternVL 特有 Token
    from internvl.train.constants import (IMG_START_TOKEN, IMG_END_TOKEN, IMG_CONTEXT_TOKEN)
    tokenizer.add_tokens([IMG_START_TOKEN, IMG_END_TOKEN, IMG_CONTEXT_TOKEN], special_tokens=True)

    # --- C. 初始化数据集 ---
    print(f"Loading datasets from {data_args.meta_path}...")
    with open(data_args.meta_path, 'r') as f:
        ds_collections = json.load(f)

    datasets = []
    for ds_name, meta in ds_collections.items():
        print(f"Creating dataset: {ds_name}")
        ds = ImplicitCoTDataset(
            template_name=data_args.conv_style,
            meta=meta,
            tokenizer=tokenizer,
            tcs_loader=None,
            ds_name=ds_name,
            num_image_token=256, # 预估值，测试流程用
            image_size=data_args.force_image_size,
            dynamic_image_size=data_args.dynamic_image_size,
            use_thumbnail=data_args.use_thumbnail,
            min_dynamic_patch=1,
            max_dynamic_patch=data_args.max_dynamic_patch,
            hidden_state_dir=None
        )
        datasets.append(ds)
    
    train_dataset = ConcatDataset(datasets)
    print(f"Total samples: {len(train_dataset)}")

    # --- D. 初始化 DataLoader ---
    data_collator = partial(implicit_cot_collator, tokenizer=tokenizer)
    dataloader = DataLoader(
        train_dataset,
        batch_size=2, # 测试 batch_size 为 2，检查对齐
        shuffle=False,
        num_workers=0, # 测试时设为 0 方便 debug
        collate_fn=data_collator
    )

    # --- E. 迭代并检查 ---
    print("\n" + "="*50)
    print("STARTING DATALOADER TEST")
    print("="*50)

    for i, batch in enumerate(dataloader):
        if i >= 1: break # 只看第一个 batch

        print(f"\nBatch {i} Keys: {batch.keys()}")
        
        # 1. 检查基础 VLM 字段
        print(f"Input IDs shape: {batch['input_ids'].shape}")
        print(f"Pixel Values shape: {batch['pixel_values'].shape}")
        print(f"Image Flags: {batch['image_flags']}")

        # 2. 检查自定义字段 - Trajectory
        if 'gt_trajectory' in batch:
            print(f"GT Trajectory shape: {batch['gt_trajectory'].shape}")
            print(f"GT Trajectory sample (first 2 points of item 0):\n{batch['gt_trajectory'][0][:2]}")
        else:
            print("❌ ERROR: 'gt_trajectory' missing from batch!")

        # 3. 检查自定义字段 - Teacher Hidden States
        if 'teacher_hidden_states' in batch:
            print(f"Teacher HS shape: {batch['teacher_hidden_states'].shape} (Expect [B, L, D])")
        else:
            print("❌ ERROR: 'teacher_hidden_states' missing from batch!")

        # 4. 检查自定义字段 - Teacher Text IDs
        if 'teacher_text_ids' in batch:
            print(f"Teacher Text IDs shape: {batch['teacher_text_ids'].shape}")
            print(f"Teacher Text IDs sample: {batch['teacher_text_ids'][0][:5]}")
        else:
            print("❌ ERROR: 'teacher_text_ids' missing from batch!")

        # 5. 验证 Padding 对齐
        if batch['input_ids'].shape[0] > 1:
            # 检查 labels 是否被正确处理 (InternVL 的 labels 通常在 prompt 部分是 -100)
            valid_label_counts = (batch['labels'] != -100).sum(dim=1)
            print(f"Valid label counts per sample: {valid_label_counts.tolist()}")

    print("\n" + "="*50)
    print("DATALOADER TEST FINISHED")
    print("="*50)

if __name__ == "__main__":
    # 执行测试
    try:
        test_dataloader()
    except Exception as e:
        print(f"\n❌ TEST FAILED with error:")
        traceback.print_exc()