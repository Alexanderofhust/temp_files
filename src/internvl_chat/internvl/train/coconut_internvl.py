# 原版coconut是LM，base model改为internvl,需要做一下改动
# 1. 适应多模态输入
# 2. 修改tokenizer为internvl的tokenizer
import logging
import math
import os
import random
import sys
import re
# 获取当前文件（coconut_internvl.py）的目录路径
current_file_dir = os.path.dirname(os.path.abspath(__file__))
# 回溯到internvl的上级根目录（/data/chenxiwu/ICoT-Drive/src/internvl_chat/）
internvl_root_dir = os.path.abspath(os.path.join(current_file_dir, "../../"))
# 将根目录添加到Python的模块搜索路径
if internvl_root_dir not in sys.path:
    sys.path.insert(0, internvl_root_dir)
print(current_file_dir)
print(internvl_root_dir)
import traceback
import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, Literal, Optional

import numpy as np

try:
    import orjson as json
except:
    import json

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.distributed as dist
import transformers
from collections import namedtuple
#多卡训练用
from internvl.dist_utils import init_dist
#internVL LLM模块，替换原gpt2
from internvl.model.internlm2.modeling_internlm2 import InternLM2ForCausalLM
#internVL 多模态部分
from internvl.model.internvl_chat import (InternVisionConfig,
                                          InternVisionModel,
                                          InternVLChatConfig,
                                          InternVLChatModel)
#训练优化
from internvl.patch import (concat_pad_data_collator,
                            replace_internlm2_attention_class,
                            replace_llama_attention_class,
                            replace_llama_rmsnorm_with_fused_rmsnorm,
                            replace_phi3_attention_class,
                            replace_qwen2_attention_class,
                            replace_train_dataloader, replace_train_sampler)
#推理标记
from internvl.train.constants import (BOX_END_TOKEN, BOX_START_TOKEN,
                                      IMG_CONTEXT_TOKEN, IMG_END_TOKEN,
                                      IMG_START_TOKEN, QUAD_END_TOKEN,
                                      QUAD_START_TOKEN, REF_END_TOKEN,
                                      REF_START_TOKEN, LOC_START_TOKEN, LOC_END_TOKEN,
                                      FRONT_VIEW_TOKEN, FRONT_LEFT_VIEW_TOKEN, FRONT_RIGHT_VIEW_TOKEN,
                                      BACK_LEFT_VIEW_TOKEN, BACK_RIGHT_VIEW_TOKEN, BACK_VIEW_TOKEN)
#数据输入
from internvl.train.dataset import (ConcatDataset, TCSLoader,
                                    WeightedConcatDataset, build_transform,
                                    check_conversations_repetition,
                                    dynamic_preprocess, preprocess,
                                    preprocess_internlm,
                                    preprocess_internvl2_5, preprocess_mpt,read_frames_decord,read_frames_gif,
                                    preprocess_phi3)
#打包数据集
from internvl.train.dataset_packed import PackedDataset, packed_collate_fn
Outputs = namedtuple("Outputs", ["loss", "inputs_embeds", "logits"])
MAX_N_LATENT = 8



class Coconut_internVL(nn.Module):
    def __init__(
        self,
        base_causallm,
        latent_token_id,
        start_latent_id,
        end_latent_id,
        eos_token_id,
    ):
        super(Coconut_internVL, self).__init__()
        self.base_causallm = base_causallm
        self.latent_token_id = latent_token_id
        self.eos_token_id = eos_token_id
        self.start_latent_id = start_latent_id
        self.end_latent_id = end_latent_id
        
        # 适配InternVL（区分纯文本和多模态模型）
        if isinstance(base_causallm, InternLM2ForCausalLM):
            # 纯文本：直接提取InternLM2的词嵌入
            self.embedding = base_causallm.get_input_embeddings()
            self.is_multimodal = False
        elif isinstance(base_causallm, InternVLChatModel):
            # 多模态：提取language_model的词嵌入+记录多模态属性
            self.embedding = base_causallm.language_model.get_input_embeddings()
            self.is_multimodal = True
            # 记录图像上下文标记ID（InternVL多模态必需，若外部传入可替换）
            self.img_context_token_id = getattr(base_causallm, 'img_context_token_id', None)
        else:
            # 兼容原有的GPT2/Llama
            self.is_multimodal = False
            # if isinstance(self.base_causallm, GPT2LMHeadModel):
            #     self.embedding = self.base_causallm.transformer.get_input_embeddings()
            # else:
            #     self.embedding = self.base_causallm.get_input_embeddings()
        
        # 初始化前向传播计数（修复原代码未定义的问题）
        self.gen_forward_cnt = 0

    def custom_preprocess(self, conversations, tokenizer, num_image_token, text_only=False):
        # 补全self参数，作为类内实例方法
        if not isinstance(conversations, list):
            conversations = [conversations]
        
        # 先插入Coconut的Latent Token（根据推理步骤拆分）
        for conv in conversations:
            if conv.get('role') == 'assistant':
                # 按步骤拆分，插入<|latent|>（可自定义插入规则）
                conv['value'] = conv['value'].replace('步骤', '<|latent|>步骤')
        
        # 再调用InternVL的预处理函数，生成input_ids和labels
        from internvl.train.dataset import preprocess_internlm  # 确保该函数可导入
        return preprocess_internlm(
            template_name='internlm2-chat',
            conversations=[deepcopy(conversations)],  # 避免修改原数据
            tokenizer=tokenizer,
            num_image_tokens=[num_image_token],
            text_only=text_only  # 多模态场景设为False，纯文本设为True
        )
    
    def custom_preprocess(self, conversations, tokenizer, num_image_token, text_only=False):
        # 补全self参数，作为类内实例方法
        if not isinstance(conversations, list):
            conversations = [conversations]
        
        for conv in conversations:
            # 兼容 ShareGPT 格式 ('from') 和标准格式 ('role')
            # 你的数据示例使用的是 "from": "gpt"
            role = conv.get('from') or conv.get('role')
            
            if role in ['gpt', 'assistant']:
                content = conv.get('value', '')
                
                # 使用正则表达式匹配 <think>...内容...</think> 和 <answer>...内容...</answer>
                # re.DOTALL 模式让 '.' 也能匹配换行符，因为你的思考过程包含换行
                pattern = r"<think>(.*?)</think>\s*<answer>(.*?)</answer>"
                match = re.search(pattern, content, re.DOTALL)
                
                if match:
                    thought_content = match.group(1).strip() # 提取思考内容
                    answer_content = match.group(2).strip()  # 提取答案内容
                    
                    # 【修改核心】：插入 <|latent|> Token
                    # 方式 1: 替换 <think> 标签为 <|latent|>，并移除其他标签，构造干净的数据流
                    # 格式变为: <|latent|> 思考内容 \n 答案内容
                    # new_value = f"<|latent|>{thought_content}\n{answer_content}"
                    
                    # 保留 XML 标签结构:
                    new_value = f"<|latent|><think>{thought_content}</think>\n<answer>{answer_content}</answer>"
                    
                    conv['value'] = new_value
                
                # 兜底逻辑：如果正则没匹配到（例如格式只有 think 没有 answer），做简单替换
                elif '<think>' in content:
                    conv['value'] = content.replace('<think>', '<|latent|>')

        return conversations # 确保这一步处理完后返回，或者根据你的原有逻辑继续处理

    def forward(
        self, 
        input_ids, 
        attention_mask, 
        labels, 
        position_ids, 
        pixel_values=None,  # 多模态：图像/视频帧张量 [bs, num_patches, 3, H, W]
        image_flags=None,   # 多模态：图像有效标记 [bs, num_patches]
        **kwargs
    ):
        """
        适配多模态场景的forward方法：
        - 保留Coconut核心的Latent Token反馈逻辑
        - 兼容InternVLChatModel（多模态）和InternLM2ForCausalLM（纯文本）
        - 支持视觉输入传递、KV Cache复用、Latent Token嵌入替换
        """
        # 初始化logits列表，存储各轮pass的输出
        logits = []

        # 定位所有Latent Token的位置 (num_latent_tokens, 2) -> (batch_idx, token_idx)
        latent_indices = (input_ids == self.latent_token_id).nonzero()
        
        # 按样本分组Latent Token位置：[bs, num_latent_tokens_per_instance]
        latent_lists = [
            [idx[1].item() for idx in latent_indices if idx[0] == i]
            for i in range(input_ids.shape[0])
        ]
        
        # 计算批次中最多的Latent Token数量
        max_n_latents = max([len(l) for l in latent_lists]) if latent_indices.numel() > 0 else 0

        # 初始化首次计算范围：到最早的Latent Token位置（无Latent则到序列末尾）
        next_compute_range = (0, input_ids.shape[1])
        if max_n_latents > 0:
            next_compute_range = (0, latent_indices[:, 1].min().item())

        # 生成文本输入嵌入（多模态场景下，视觉嵌入由模型内部处理）
        inputs_embeds = self.embedding(input_ids)

        # 初始化KV Cache
        kv_cache = None

        # 逐轮处理每个Latent Token的反馈
        for pass_idx in range(max_n_latents):
            if kv_cache is None:
                # -------------------------- 首次forward（无KV Cache） --------------------------
                if self.is_multimodal and pixel_values is not None:
                    # 多模态场景：调用InternVLChatModel，传递视觉输入
                    outputs = self.base_causallm(
                        input_ids=None,  # 多模态优先用inputs_embeds，避免输入冲突
                        inputs_embeds=inputs_embeds[:, next_compute_range[0]:next_compute_range[1], :],
                        attention_mask=attention_mask[:, next_compute_range[0]:next_compute_range[1]],
                        position_ids=position_ids[:, next_compute_range[0]:next_compute_range[1]],
                        pixel_values=pixel_values,  # 传递视觉输入
                        image_flags=image_flags,    # 传递图像有效标记
                        output_hidden_states=True,  # 输出隐藏状态，用于Latent Token替换
                        use_cache=True,             # 开启KV Cache，方便后续复用
                        **kwargs                    # 传递其他模型参数（如temperature等）
                    )
                else:
                    # 纯文本场景：原有逻辑，仅传递文本输入
                    outputs = self.base_causallm(
                        inputs_embeds=inputs_embeds[:, next_compute_range[0]:next_compute_range[1], :],
                        attention_mask=attention_mask[:, next_compute_range[0]:next_compute_range[1]],
                        position_ids=position_ids[:, next_compute_range[0]:next_compute_range[1]],
                        output_hidden_states=True,
                        use_cache=True,
                        **kwargs
                    )
                # 首次pass无KV Cache，隐藏状态偏移为0
                hidden_states_offset = 0

            else:
                # -------------------------- forward（复用KV Cache） --------------------------
                # 提取并截断KV Cache到当前计算起始位置（仅复用文本部分）
                past_key_values = [
                    (k[:, :, :next_compute_range[0], :], v[:, :, :next_compute_range[0], :])
                    for k, v in kv_cache
                ]

                if self.is_multimodal and pixel_values is not None:
                    # 多模态场景：复用文本KV Cache，保持视觉输入传递
                    outputs = self.base_causallm(
                        input_ids=None,
                        inputs_embeds=inputs_embeds[:, next_compute_range[0]:next_compute_range[1], :],
                        attention_mask=attention_mask[:, :next_compute_range[1]],  # 注意mask范围扩展到当前结束位置
                        position_ids=position_ids[:, next_compute_range[0]:next_compute_range[1]],
                        past_key_values=past_key_values,  # 复用文本KV Cache
                        pixel_values=pixel_values,        # 视觉特征仅首次传递即可，此处可省略（模型内部缓存）
                        image_flags=image_flags,
                        output_hidden_states=True,
                        use_cache=True,
                        **kwargs
                    )
                else:
                    # 纯文本场景：复用KV Cache，原有逻辑
                    outputs = self.base_causallm(
                        inputs_embeds=inputs_embeds[:, next_compute_range[0]:next_compute_range[1], :],
                        attention_mask=attention_mask[:, :next_compute_range[1]],
                        position_ids=position_ids[:, next_compute_range[0]:next_compute_range[1]],
                        past_key_values=past_key_values,
                        output_hidden_states=True,
                        use_cache=True,
                        **kwargs
                    )
                # 复用KV Cache时，隐藏状态偏移为当前计算起始位置（跳过已计算部分）
                hidden_states_offset = next_compute_range[0]

            # 记录当前pass的logits
            logits.append(outputs.logits)

            # 更新下一轮计算范围：当前结束位置 -> 下一个Latent Token位置（最后一轮到序列末尾）
            next_compute_range = (
                next_compute_range[1],
                input_ids.shape[1] if (pass_idx + 1) >= max_n_latents else (next_compute_range[1] + 1)
            )

            # 提取最后一层隐藏状态（用于Latent Token嵌入替换）
            hidden_states = outputs.hidden_states[-1]
            # 更新KV Cache
            kv_cache = outputs.past_key_values

            # -------------------------- Latent Token反馈：替换输入嵌入 --------------------------
            # 确定当前pass需要替换的Latent Token位置
            filling_indices = [
                (instance_idx, mask_list[pass_idx])
                for instance_idx, mask_list in enumerate(latent_lists)
                if len(mask_list) > pass_idx
            ]

            # 拆解inputs_embeds为列表（避免原地操作）
            tensor_list = [
                [inputs_embeds[batch_idx, pos, :] for pos in range(inputs_embeds.shape[1])]
                for batch_idx in range(inputs_embeds.shape[0])
            ]

            # 替换Latent Token的嵌入为前一位置的隐藏状态
            for batch_idx, token_idx in filling_indices:
                tensor_list[batch_idx][token_idx] = hidden_states[
                    batch_idx, token_idx - 1 - hidden_states_offset, :
                ]

            # 重新组装inputs_embeds
            inputs_embeds = torch.stack([
                torch.stack(tensor_list[batch_idx]) for batch_idx in range(inputs_embeds.shape[0])
            ])

        # -------------------------- 最终pass：处理剩余序列 --------------------------
        if self.is_multimodal and pixel_values is not None:
            # 多模态场景：最终pass，传递视觉输入+复用KV Cache
            outputs = self.base_causallm(
                input_ids=None,
                inputs_embeds=inputs_embeds[:, next_compute_range[0]:next_compute_range[1], :],
                attention_mask=attention_mask[:, :next_compute_range[1]],
                position_ids=position_ids[:, next_compute_range[0]:next_compute_range[1]],
                past_key_values=(
                    [(k[:, :, :next_compute_range[0], :], v[:, :, :next_compute_range[0], :]) for k, v in kv_cache]
                    if kv_cache is not None else None
                ),
                pixel_values=pixel_values,
                image_flags=image_flags,
                output_hidden_states=True,
                use_cache=True,
                **kwargs
            )
        else:
            # 纯文本场景：最终pass，原有逻辑
            outputs = self.base_causallm(
                inputs_embeds=inputs_embeds[:, next_compute_range[0]:next_compute_range[1], :],
                attention_mask=attention_mask[:, :next_compute_range[1]],
                position_ids=position_ids[:, next_compute_range[0]:next_compute_range[1]],
                past_key_values=(
                    [(k[:, :, :next_compute_range[0], :], v[:, :, :next_compute_range[0], :]) for k, v in kv_cache]
                    if kv_cache is not None else None
                ),
                output_hidden_states=True,
                use_cache=True,
                **kwargs
            )

        # 记录最终pass的logits
        logits.append(outputs.logits)

        # 更新前向传播计数
        self.gen_forward_cnt += max_n_latents + 1

        # -------------------------- 损失计算 --------------------------
        # 拼接所有pass的logits -> [bs, seq_len, vocab_size]
        logits = torch.cat(logits, dim=-2)
        # 位移logits和labels（预测下一个token）
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # 计算交叉熵损失（忽略padding部分，若labels有-100需用ignore_index参数）
        loss_fct = CrossEntropyLoss(ignore_index=-100)  # 新增ignore_index，适配InternVL的label格式
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # 返回损失、最终输入嵌入、完整logits
        return Outputs(loss=loss, inputs_embeds=inputs_embeds, logits=logits)
        

class Coconut_internVL_Same_Word_Embedding(nn.Module):
    def __init__(
        self,
        base_causallm,          # 适配为InternVLChatModel（多模态）/InternLM2ForCausalLM（纯文本）
        expainable_llm,         # 可解释性LLM（通常为纯文本LLM，如InternLM2）
        tokenizer,
        latent_token_id,
        start_latent_id,
        end_latent_id,
        eos_token_id,
        step_start_id,
        c_thought,
        configs,
    ):
        super(Coconut_internVL_Same_Word_Embedding, self).__init__()
        
        # -------------------------- 核心计数/参数初始化（保留原逻辑） --------------------------
        self.gen_forward_cnt = 0
        self.base_causallm = base_causallm
        self.expainable_llm = expainable_llm
        self.tokenizer = tokenizer
        self.latent_token_id = latent_token_id
        self.eos_token_id = eos_token_id
        self.start_latent_id = start_latent_id
        self.end_latent_id = end_latent_id
        self.step_start_id = step_start_id
        self.c_thought = c_thought
        self.config = configs

        # -------------------------- 训练策略：参数梯度控制（保留原逻辑） --------------------------
        if hasattr(self.config, "training_method"):
            training_method = self.config.training_method
            if training_method == 'only_expainable_llm':
                # 仅训练可解释模型，冻结基础模型
                for param in self.base_causallm.parameters():
                    param.requires_grad = False
            elif training_method == 'only_base_causallm':
                # 仅训练基础模型，冻结可解释模型
                for param in self.expainable_llm.parameters():
                    param.requires_grad = False
            elif training_method == 'full':
                # 全量训练，不冻结
                pass
            elif training_method == 'freeze_backbone':
                # 冻结所有骨干参数
                for param in self.base_causallm.parameters():
                    param.requires_grad = False
                for param in self.expainable_llm.parameters():
                    param.requires_grad = False
            else:
                raise ValueError(f"不支持的训练模式: {training_method=}")

        # -------------------------- 词嵌入层提取（适配InternVL多模态） --------------------------
        # 1. 区分基础模型类型：InternVLChatModel（多模态）/InternLM2ForCausalLM（纯文本）/GPT2（兼容）
        if isinstance(self.base_causallm, InternVLChatModel):
            # InternVL多模态模型：词嵌入在language_model下
            self.embedding = self.base_causallm.language_model.get_input_embeddings()
            self.is_multimodal = True  # 标记为多模态模型
            # 记录InternVL的图像上下文token ID（若模型有则赋值）
            self.img_context_token_id = getattr(self.base_causallm, 'img_context_token_id', None)
        elif isinstance(self.base_causallm, InternLM2ForCausalLM):
            # InternLM2纯文本模型：直接提取词嵌入
            self.embedding = self.base_causallm.get_input_embeddings()
            self.is_multimodal = False
        # elif isinstance(self.base_causallm, GPT2LMHeadModel):
        #     # 兼容原GPT2模型
        #     self.embedding = self.base_causallm.transformer.get_input_embeddings()
        #     self.is_multimodal = False
        else:
            # 其他LLM默认逻辑
            self.embedding = self.base_causallm.get_input_embeddings()
            self.is_multimodal = False

    def save_jsonl_line(jsonl_path, data):
        """原代码中可视化保存用的辅助函数（示例实现）"""
        import json
        with open(jsonl_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')

    def prepare_4d_attention_mask(attention_mask, dtype):
        """原代码中packing模式下的4D attention mask准备（示例实现）"""
        if attention_mask.dim() == 2:
            batch_size, seq_len = attention_mask.shape
            attention_mask = attention_mask[:, None, None, :]
        attention_mask = attention_mask.to(dtype=dtype)
        return attention_mask
    
    def forward(
        self, 
        input_ids, 
        attention_mask, 
        labels, 
        position_ids, 
        pixel_values=None,  # 新增：多模态视觉输入 [bs, num_patches, 3, H, W]
        image_flags=None,   # 新增：多模态图像有效标记 [bs, num_patches]
        **kwargs
    ):
        """
        适配InternVL多模态的forward方法：
        - 完全保留原扩展版所有核心逻辑（双LLM、可解释损失、可视化、packing、explain_mode等）
        - 新增多模态视觉输入处理，兼容InternVLChatModel
        - 可解释模型（expainable_llm）仍为纯文本逻辑，无需视觉输入
        """
        logits = []
        loss = 0.0

        # -------------------------- 1. Latent Token定位（原逻辑完全保留） --------------------------
        latent_indices = (input_ids == self.latent_token_id).nonzero()  # (num_latent_tokens, 2)
        latent_lists = [
            [idx[1].item() for idx in latent_indices if idx[0] == i]
            for i in range(input_ids.shape[0])
        ]  # bs, num_latent_tokens_per_instance
        max_n_latents = max([len(l) for l in latent_lists]) if latent_indices.numel() > 0 else 0

        # 初始化计算范围和输入嵌入
        next_compute_range = (0, input_ids.shape[1])
        inputs_embeds = self.embedding(input_ids)
        if max_n_latents > 0:
            next_compute_range = (0, latent_indices[:, 1].min().item())

        kv_cache = None

        # -------------------------- 2. 多轮KV Cache循环（核心：多模态适配） --------------------------
        for pass_idx in range(max_n_latents):
            if kv_cache is None:
                # -------------------------- 首次forward（无KV Cache） --------------------------
                if self.is_multimodal and pixel_values is not None:
                    # 多模态场景：调用InternVLChatModel，传递视觉输入
                    outputs = self.base_causallm(
                        input_ids=None,  # 多模态优先用inputs_embeds，避免输入冲突
                        inputs_embeds=inputs_embeds[:, next_compute_range[0]:next_compute_range[1], :],
                        attention_mask=attention_mask[:, next_compute_range[0]:next_compute_range[1]],
                        position_ids=position_ids[:, next_compute_range[0]:next_compute_range[1]],
                        pixel_values=pixel_values,  # 传递视觉输入
                        image_flags=image_flags,    # 传递图像有效标记
                        output_hidden_states=True,
                        use_cache=True,  # 开启KV Cache
                        **kwargs
                    )
                else:
                    # 纯文本场景：原逻辑不变
                    outputs = self.base_causallm(
                        inputs_embeds=inputs_embeds[:, next_compute_range[0]:next_compute_range[1], :],
                        attention_mask=attention_mask[:, next_compute_range[0]:next_compute_range[1]],
                        position_ids=position_ids[:, next_compute_range[0]:next_compute_range[1]],
                        output_hidden_states=True,
                        use_cache=True,
                        **kwargs
                    )
                hidden_states_offset = 0

            else:
                # -------------------------- 后续forward（复用KV Cache） --------------------------
                # 提取文本KV Cache（视觉特征仅首次传递，模型内部缓存）
                past_key_values = [
                    (k[:, :, :next_compute_range[0], :], v[:, :, :next_compute_range[0], :])
                    for k, v in kv_cache
                ]

                if self.is_multimodal and pixel_values is not None:
                    # 多模态场景：复用文本KV Cache + 保持视觉输入传递
                    outputs = self.base_causallm(
                        input_ids=None,
                        inputs_embeds=inputs_embeds[:, next_compute_range[0]:next_compute_range[1], :],
                        attention_mask=attention_mask[:, :next_compute_range[1]],
                        position_ids=position_ids[:, next_compute_range[0]:next_compute_range[1]],
                        past_key_values=past_key_values,
                        pixel_values=pixel_values,
                        image_flags=image_flags,
                        output_hidden_states=True,
                        use_cache=True,
                        **kwargs
                    )
                else:
                    # 纯文本场景：原逻辑不变
                    outputs = self.base_causallm(
                        inputs_embeds=inputs_embeds[:, next_compute_range[0]:next_compute_range[1], :],
                        attention_mask=attention_mask[:, :next_compute_range[1]],
                        position_ids=position_ids[:, next_compute_range[0]:next_compute_range[1]],
                        past_key_values=past_key_values,
                        output_hidden_states=True,
                        use_cache=True,
                        **kwargs
                    )
                hidden_states_offset = next_compute_range[0]

            # -------------------------- Latent Token反馈（原逻辑完全保留） --------------------------
            logits.append(outputs.logits)
            # 更新计算范围
            next_compute_range = (
                next_compute_range[1],
                input_ids.shape[1] if (pass_idx + 1) >= max_n_latents else (next_compute_range[1] + 1)
            )
            # 提取隐藏状态和KV Cache
            hidden_states = outputs.hidden_states[-1]
            kv_cache = outputs.past_key_values

            # 替换Latent Token嵌入为前一位置隐藏状态
            filling_indices = [
                (instance_idx, mask_list[pass_idx])
                for instance_idx, mask_list in enumerate(latent_lists)
                if len(mask_list) > pass_idx
            ]
            # 拆解嵌入避免原地操作
            tensor_list = [
                [inputs_embeds[batch_idx, pos, :] for pos in range(inputs_embeds.shape[1])]
                for batch_idx in range(inputs_embeds.shape[0])
            ]
            # 替换嵌入
            for batch_idx, token_idx in filling_indices:
                tensor_list[batch_idx][token_idx] = hidden_states[
                    batch_idx, token_idx - 1 - hidden_states_offset, :
                ]
            # 重新组装嵌入
            inputs_embeds = torch.stack([
                torch.stack(tensor_list[batch_idx]) for batch_idx in range(inputs_embeds.shape[0])
            ])

        # -------------------------- 3. 最终pass（多模态适配） --------------------------
        if self.is_multimodal and pixel_values is not None:
            outputs = self.base_causallm(
                input_ids=None,
                inputs_embeds=inputs_embeds[:, next_compute_range[0]:next_compute_range[1], :],
                attention_mask=attention_mask[:, :next_compute_range[1]],
                position_ids=position_ids[:, next_compute_range[0]:next_compute_range[1]],
                past_key_values=(
                    [(k[:, :, :next_compute_range[0], :], v[:, :, :next_compute_range[0], :]) for k, v in kv_cache]
                    if kv_cache is not None else None
                ),
                pixel_values=pixel_values,
                image_flags=image_flags,
                output_hidden_states=True,
                use_cache=True,
                **kwargs
            )
        else:
            outputs = self.base_causallm(
                inputs_embeds=inputs_embeds[:, next_compute_range[0]:next_compute_range[1], :],
                attention_mask=attention_mask[:, :next_compute_range[1]],
                position_ids=position_ids[:, next_compute_range[0]:next_compute_range[1]],
                past_key_values=(
                    [(k[:, :, :next_compute_range[0], :], v[:, :, :next_compute_range[0], :]) for k, v in kv_cache]
                    if kv_cache is not None else None
                ),
                output_hidden_states=True,
                use_cache=True,
                **kwargs
            )
        logits.append(outputs.logits)
        self.gen_forward_cnt += max_n_latents + 1

        # -------------------------- 4. 基础模型损失计算（原逻辑保留） --------------------------
        logits = torch.cat(logits, dim=-2)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss(ignore_index=-100)  # 适配InternVL label格式
        if self.config.training_method in ['only_base_causallm', 'full']:
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

        # -------------------------- 5. 可视化逻辑（原逻辑完全保留） --------------------------
        if hasattr(self.config, 'visualize') and self.config.visualize:
            debug_predictions = []
            for debug_idx in range(0, len(latent_lists[0]), self.config.c_thought):
                continuous_embeds = inputs_embeds[:, latent_lists[0][debug_idx: debug_idx + self.config.c_thought], :].to(self.expainable_llm.device)
                # 拼接prompt（原逻辑）
                if hasattr(self.config, 'w_prompt') and self.config.w_prompt:
                    if hasattr(self.config, 'explain_mode') and self.config.explain_mode == 'v1_aug':
                        thought_idx = debug_idx // 2
                        if thought_idx != 2:
                            prompt = f'Step {thought_idx + 1} of the solution'
                        else:
                            prompt = f'Step 3 and all the remaining steps of the solution'
                        input_explain_pre_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
                        bz = continuous_embeds.shape[0]
                        input_explain_pre_embeds = self.embedding(torch.tensor(input_explain_pre_ids).to(self.expainable_llm.device))[None, ...].repeat(bz, 1, 1)
                        continuous_embeds = torch.cat([input_explain_pre_embeds, continuous_embeds], dim=1)
                # 生成可解释文本（原逻辑）
                debug_ids = torch.empty((1, 0), dtype=torch.long, device=self.expainable_llm.device)
                while True:
                    debug_embeds = torch.cat([continuous_embeds, self.embedding(debug_ids)], dim=1) if debug_ids.shape[0] != 0 else continuous_embeds
                    explainable_outputs = self.expainable_llm(
                        inputs_embeds=debug_embeds,
                        attention_mask=torch.ones(debug_embeds.shape[:2]).to(self.expainable_llm.device),
                        position_ids=torch.arange(1, debug_embeds.shape[1] + 1).unsqueeze(dim=0).to(self.expainable_llm.device),
                        output_hidden_states=True,
                    )
                    debug_logits = explainable_outputs.logits[:, -1, :] / .98
                    probs = torch.softmax(debug_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    debug_ids = torch.cat([debug_ids, next_token], dim=1)
                    print(self.tokenizer.decode(debug_ids[0]))
                    if torch.all(next_token == self.eos_token_id) or debug_ids.shape[-1] > 512:
                        break
                debug_predictions.append(self.tokenizer.decode(debug_ids[0]))
            # 保存可视化结果（原逻辑）
            if hasattr(self.config, 'visualize_jsonl') and self.config.visualize_jsonl != '':
                save_jsonl_line(self.config.visualize_jsonl, {"predictiion": debug_predictions})

        # -------------------------- 6. 可解释模型损失计算（原逻辑完全保留） --------------------------
        if hasattr(self.config, 'explain_mode') and self.config.explain_mode == 'v1_aug' and 'explainable_ids_list' in kwargs:
            c_thought_num = len(latent_lists[0]) // self.c_thought
            input_united_tokens = []
            
            # 辅助函数（原逻辑）
            def safe_token_id(x):
                return x[0] if isinstance(x, list) else x
            start_token = safe_token_id(self.tokenizer.encode('<<', add_special_tokens=False))
            end_token = safe_token_id(self.tokenizer.encode('>>', add_special_tokens=False))
            separator_token = safe_token_id(self.tokenizer.encode('\n', add_special_tokens=False))

            def trim_trailing_zeros(group):
                while group and group[-1] == 0:
                    group.pop()
                return group

            def replace_llama_special_tokens(x, merged_token, end_token, separator_token):
                out = []
                for seq in x:
                    new_seq = []
                    for t in seq:
                        if t.item() == merged_token:
                            new_seq.extend([end_token, separator_token])
                        elif t.item() != 0 or len(new_seq) > 0:
                            new_seq.append(t.item())
                    out.append(torch.tensor(new_seq, device=x.device))
                return out

            # 特殊token替换（原逻辑）
            if len(self.tokenizer.encode('>>\n', add_special_tokens=False)) == 1:
                merge_token = self.tokenizer.encode('>>\n', add_special_tokens=False)[0]
                kwargs['explainable_ids_list'] = copy.deepcopy(replace_llama_special_tokens(kwargs['explainable_ids_list'], merge_token, end_token, separator_token))
            
            # 构建输入组（原逻辑）
            for j, seq in enumerate(kwargs['explainable_ids_list']):
                i = 0
                groups = []
                while i < len(seq):
                    if seq[i] == start_token:
                        group = [start_token]
                        i += 1
                        while i < len(seq):
                            group.append(seq[i])
                            if seq[i] == end_token:
                                break
                            i += 1
                        group = trim_trailing_zeros(group)
                        groups.append(group)
                    else:
                        i += 1
                # 补充伪思考（原逻辑）
                if len(groups) < self.config.max_latent_stage:
                    input_ids_j = input_ids[j].tolist()
                    try:
                        start_idx = len(input_ids_j) - 1 - input_ids_j[::-1].index(self.end_latent_id)
                    except ValueError:
                        continue
                    try:
                        end_idx = input_ids_j.index(self.eos_token_id, start_idx + 1)
                    except ValueError:
                        end_idx = len(input_ids_j)
                    pseudo_thought = input_ids_j[start_idx + 1:end_idx]
                    if not pseudo_thought:
                        continue
                    if hasattr(self.config, 'format_pseudo_thought') and self.config.format_pseudo_thought:
                        tmp_num = self.tokenizer.decode(pseudo_thought).replace('### ', '')
                        pseudo_thought = self.tokenizer.encode(f'<<{tmp_num}={tmp_num}>>', add_special_tokens=False)
                    while len(groups) < c_thought_num:
                        groups.append(pseudo_thought)
                # 合并组（原逻辑）
                input_united_groups = []
                combined_group = []
                group_count = 0
                for group in groups:
                    group_count += 1
                    if group_count <= self.config.max_latent_stage - 1:
                        group = [-570] * self.c_thought + group + [self.eos_token_id]
                        cleaned_group = [int(x) if torch.is_tensor(x) else x for x in group]
                        input_united_groups.append(cleaned_group)
                    else:
                        if combined_group and combined_group[-1] == end_token and group[0] == start_token:
                            combined_group.append(separator_token)
                        combined_group.extend(group)
                if combined_group:
                    final_group = [-570] * self.c_thought + combined_group + [self.eos_token_id]
                    cleaned_group = [int(x) if torch.is_tensor(x) else x for x in final_group]
                    input_united_groups.append(cleaned_group)
                input_united_tokens.append(copy.deepcopy(input_united_groups))
            
            # Packing模式处理（原逻辑）
            loss_explain_all = 0.0
            bz = len(input_united_tokens)
            if hasattr(self.config, 'packing') and self.config.packing == True:
                # 计算最大长度（原逻辑）
                max_pad_len = 0
                for bz_idx in range(bz):
                    for thought_idx in range(c_thought_num):
                        continuous_embeds = inputs_embeds[bz_idx, latent_lists[bz_idx][self.c_thought * thought_idx]: latent_lists[bz_idx][self.c_thought * thought_idx + 1] + 1, :]
                        other_embeds = self.embedding(torch.tensor(input_united_tokens[bz_idx][thought_idx][self.c_thought:]).to(self.expainable_llm.device))
                        max_pad_len = max(max_pad_len, continuous_embeds.size(0) + other_embeds.size(0))
                # 初始化batch容器（原逻辑）
                input_explain_input_embeds_batch = [[] for _ in range(c_thought_num)]
                input_explain_attention_mask_batch = [[] for _ in range(c_thought_num)]
                input_explain_position_ids_batch = [[] for _ in range(c_thought_num)]
                input_explain_labels_batch = [[] for _ in range(c_thought_num)]
                # 填充batch（原逻辑）
                for thought_idx in range(c_thought_num):
                    for bz_idx in range(bz):
                        continuous_embeds = inputs_embeds[bz_idx, latent_lists[bz_idx][self.c_thought * thought_idx]: latent_lists[bz_idx][self.c_thought * thought_idx + 1] + 1, :]
                        other_embeds = self.embedding(torch.tensor(input_united_tokens[bz_idx][thought_idx][self.c_thought:]).to(self.expainable_llm.device))
                        input_explain_input_embeds_batch[thought_idx].append(torch.cat([continuous_embeds, other_embeds], dim=0))
                        # 构建attention mask（原逻辑）
                        attention_eos_index = input_united_tokens[bz_idx][thought_idx].index(self.eos_token_id)
                        attention_explain_mask = torch.zeros(len(input_united_tokens[bz_idx][thought_idx]), dtype=int)
                        attention_explain_mask[:attention_eos_index + 1] = 1
                        input_explain_attention_mask_batch[thought_idx].append(attention_explain_mask)
                        # 构建position ids（原逻辑）
                        input_explain_position_ids_batch[thought_idx].append(torch.arange(1, len(input_united_tokens[bz_idx][thought_idx]) + 1, dtype=int))
                        # 构建labels（原逻辑）
                        explain_labels = torch.tensor(input_united_tokens[bz_idx][thought_idx], dtype=int)
                        explain_labels_mask = (explain_labels != -570) & (explain_labels != self.eos_token_id)
                        explain_labels_mask[attention_eos_index] = True
                        explain_labels[~explain_labels_mask] = -100
                        input_explain_labels_batch[thought_idx].append(explain_labels)
                # 构建padded tensor（原逻辑）
                input_explain_input_embeds_batch_tensor = torch.zeros(bz, c_thought_num, max_pad_len, continuous_embeds.size(-1), device=self.expainable_llm.device)
                input_explain_attention_mask_batch_tensor = torch.zeros(bz, c_thought_num, max_pad_len, device=self.expainable_llm.device)
                input_explain_position_ids_batch_tensor = torch.zeros(bz, c_thought_num, max_pad_len, device=self.expainable_llm.device)
                input_explain_labels_batch_tensor = torch.full((bz, c_thought_num, max_pad_len), -100, device=self.expainable_llm.device)
                # 填充padded tensor（原逻辑）
                for bz_idx in range(bz):
                    for thought_idx in range(c_thought_num):
                        input_explain_input_embeds_batch_tensor[bz_idx, thought_idx, :input_explain_input_embeds_batch[thought_idx][bz_idx].size(0)] = input_explain_input_embeds_batch[thought_idx][bz_idx]
                        input_explain_attention_mask_batch_tensor[bz_idx, thought_idx, :input_explain_attention_mask_batch[thought_idx][bz_idx].size(0)] = input_explain_attention_mask_batch[thought_idx][bz_idx]
                        input_explain_position_ids_batch_tensor[bz_idx, thought_idx, :input_explain_position_ids_batch[thought_idx][bz_idx].size(0)] = input_explain_position_ids_batch[thought_idx][bz_idx]
                        input_explain_labels_batch_tensor[bz_idx, thought_idx, :input_explain_labels_batch[thought_idx][bz_idx].size(0)] = input_explain_labels_batch[thought_idx][bz_idx]
                # 重塑tensor（原逻辑）
                input_explain_input_embeds_batch_tensor = input_explain_input_embeds_batch_tensor.view(bz, -1, input_explain_input_embeds_batch_tensor.size(-1))
                input_explain_attention_mask_batch_tensor = input_explain_attention_mask_batch_tensor.view(bz, -1)
                input_explain_position_ids_batch_tensor = input_explain_position_ids_batch_tensor.view(bz, -1)
                input_explain_labels_batch_tensor = input_explain_labels_batch_tensor.view(bz, -1)
                # 构建4D attention mask（原逻辑）
                input_explain_attention_mask_batch_tensor = prepare_4d_attention_mask(input_explain_attention_mask_batch_tensor, dtype=self.expainable_llm.dtype)
                # 可解释模型前向（原逻辑）
                explainable_outputs = self.expainable_llm(
                    inputs_embeds=input_explain_input_embeds_batch_tensor,
                    attention_mask=input_explain_attention_mask_batch_tensor,
                    position_ids=input_explain_position_ids_batch_tensor.to(torch.long),
                    output_hidden_states=True,
                )
                # 损失计算（原逻辑）
                explainable_logits = explainable_outputs.logits
                effective_loss_num = float((input_explain_labels_batch_tensor != -100).sum(dim=1).bool().sum().item())
                shift_explain_logits = explainable_logits[..., :-1, :].contiguous()
                shift_explain_labels = input_explain_labels_batch_tensor[..., 1:].to(torch.long).contiguous()
                loss_explain_fct = CrossEntropyLoss(reduction='sum')
                loss_explain = loss_explain_fct(
                    shift_explain_logits.view(-1, shift_explain_logits.size(-1)),
                    shift_explain_labels.view(-1)
                )
                loss_explain /= effective_loss_num
                loss_explain_all += loss_explain
            else:
                # 非Packing模式（原逻辑）
                for thought_idx in range(c_thought_num):
                    input_explain_input_embeds = []
                    input_explain_attention_mask, input_explain_position_ids, input_explain_labels = [], [], []
                    for bz_idx in range(bz):
                        latent_len = len(latent_lists[bz_idx])
                        start_idx = thought_idx * self.c_thought
                        end_idx = min(start_idx + self.c_thought, latent_len)
                        continuous_embeds = inputs_embeds[bz_idx, latent_lists[bz_idx][start_idx:end_idx], :]
                        other_embeds = self.embedding(torch.tensor(input_united_tokens[bz_idx][thought_idx][self.c_thought:]).to(self.expainable_llm.device))
                        input_explain_input_embeds.append(torch.cat([continuous_embeds, other_embeds], dim=0))
                        # 构建attention mask（原逻辑）
                        attention_eos_index = input_united_tokens[bz_idx][thought_idx].index(self.eos_token_id)
                        attention_explain_mask = torch.zeros(len(input_united_tokens[bz_idx][thought_idx]), dtype=int)
                        attention_explain_mask[:attention_eos_index+1] = 1
                        input_explain_attention_mask.append(attention_explain_mask)
                        # 构建position ids（原逻辑）
                        input_explain_position_ids.append(torch.arange(1, len(input_united_tokens[bz_idx][thought_idx]) + 1, dtype=int))
                        # 构建labels（原逻辑）
                        explain_labels = torch.tensor(input_united_tokens[bz_idx][thought_idx], dtype=int)
                        explain_labels_mask = (explain_labels != -570) & (explain_labels != self.eos_token_id)
                        explain_labels_mask[attention_eos_index] = True
                        explain_labels[~explain_labels_mask] = -100
                        input_explain_labels.append(explain_labels)
                    # 堆叠tensor（原逻辑）
                    input_explain_input_embeds = torch.stack(input_explain_input_embeds)
                    input_explain_attention_mask = torch.stack(input_explain_attention_mask)
                    input_explain_position_ids = torch.stack(input_explain_position_ids)
                    input_explain_labels = torch.stack(input_explain_labels)
                    # 可解释模型前向（原逻辑）
                    explainable_outputs = self.expainable_llm(
                        inputs_embeds=input_explain_input_embeds.to(self.expainable_llm.device),
                        attention_mask=input_explain_attention_mask.to(self.expainable_llm.device),
                        position_ids=input_explain_position_ids.to(self.expainable_llm.device),
                        output_hidden_states=True,
                    )
                    # 投影层（原逻辑）
                    if hasattr(self.config, "use_prj") and self.config.use_prj:
                        explainable_logits = self.base_causallm.lm_head(self.projector2(explainable_outputs.hidden_states[-1]))
                    else:
                        explainable_logits = explainable_outputs.logits
                    # 有效token计数（原逻辑）
                    if hasattr(self.config, "loss_level") and self.config.loss_level == 'token_level':
                        effective_token_count = (input_explain_labels != -100).sum()
                    else:
                        effective_token_count = float((input_explain_labels != -100).sum(dim=1).bool().sum().item())
                    # 损失计算（原逻辑）
                    shift_explain_logits = explainable_logits[..., :-1, :].contiguous()
                    shift_explain_labels = input_explain_labels[..., 1:].contiguous()
                    loss_explain_fct = CrossEntropyLoss(reduction='sum')
                    loss_explain = loss_explain_fct(
                        shift_explain_logits.view(-1, shift_explain_logits.size(-1)).to(self.expainable_llm.device),
                        shift_explain_labels.view(-1).to(self.expainable_llm.device)
                    )
                    loss_explain /= effective_token_count
                    loss_explain_all += loss_explain
        
        # -------------------------- 7. 总损失合并（原逻辑保留） --------------------------
        if 'explainable_ids_list' in kwargs:
            if loss is None:
                loss = 0.0
            loss += 1.0 * loss_explain_all / c_thought_num

        # -------------------------- 8. 返回结果（原逻辑保留） --------------------------
        return Outputs(loss=loss, inputs_embeds=inputs_embeds, logits=logits)

