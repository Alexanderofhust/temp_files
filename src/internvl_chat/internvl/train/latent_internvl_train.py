import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import json
import math
from typing import List, Optional, Tuple, Union, Dict,Literal
from dataclasses import dataclass, field
from functools import partial
from copy import deepcopy
import numpy as np
import random
import traceback
import warnings
import re

# 直接导入隐式学生模型
from internvl.model.internvl_chat.latent_internvl_chat import ImplicitCoTDriverStudent

# 导入 InternVL 官方组件
from internvl.model.internvl_chat.modeling_internvl_chat import (
    InternVLChatModel,
    InternVLChatConfig
)
from torch.utils.data import ConcatDataset
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from internvl.train.dataset import  WeightedConcatDataset
from internvl.patch import (
    concat_pad_data_collator,
    replace_llama_rmsnorm_with_fused_rmsnorm,
    replace_train_sampler,
    replace_train_dataloader,
    replace_internlm2_attention_class,
    replace_qwen2_attention_class,
    replace_phi3_attention_class,
    replace_llama_attention_class
)
from internvl.train.constants import IMG_CONTEXT_TOKEN
from internvl.dist_utils import init_dist

import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed
)
from transformers.utils.logging import set_verbosity_info, enable_default_handler, enable_explicit_format

from internvl.train.constants import (BOX_END_TOKEN, BOX_START_TOKEN,
                                      IMG_CONTEXT_TOKEN, IMG_END_TOKEN,
                                      IMG_START_TOKEN, QUAD_END_TOKEN,
                                      QUAD_START_TOKEN, REF_END_TOKEN,
                                      REF_START_TOKEN, LOC_START_TOKEN, LOC_END_TOKEN,
                                      FRONT_VIEW_TOKEN, FRONT_LEFT_VIEW_TOKEN, FRONT_RIGHT_VIEW_TOKEN,
                                      BACK_LEFT_VIEW_TOKEN, BACK_RIGHT_VIEW_TOKEN, BACK_VIEW_TOKEN)
from internvl.train.dataset import (ConcatDataset, TCSLoader,
                                    WeightedConcatDataset, build_transform,
                                    check_conversations_repetition,
                                    dynamic_preprocess, preprocess,
                                    preprocess_internlm,
                                    preprocess_internvl2_5, preprocess_mpt,read_frames_decord,read_frames_gif,
                                    preprocess_phi3)
from internvl.train.dataset_packed import PackedDataset, packed_collate_fn
from PIL import Image, ImageFile, PngImagePlugin, UnidentifiedImageError
from torch.utils.data import Dataset
logger = logging.getLogger(__name__)

# ============================说明区================================
# 蒸馏学生模型需要的训练参数和数据参数在 ModelArguments 和 DataTrainingArguments 中定义。
# 输入数据集为jsonl文件，包括：scene_token,image,conversations,hidden_state_file,teacher_text_ids

@dataclass
class ModelArguments:
    """
    Arguments for specifying model, tokenizer, and configurations.
    """
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to a pretrained model (local or from huggingface.co/models).'}
    )
    vision_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to a pretrained model (local or from huggingface.co/models).'}
    )
    llm_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to a pretrained model (local or from huggingface.co/models).'}
    )
    mlp_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to a pretrained model (local or from huggingface.co/models).'}
    )
    freeze_llm: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the LLM. Default is False.'},
    )
    freeze_backbone: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the ViT. Default is False.'},
    )
    freeze_mlp: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the MLP. Default is False.'},
    )
    unfreeze_vit_layers: int = field(
        default=0,
        metadata={'help': 'Specify the number of ViT layers to unfreeze. Default is 0.'},
    )
    vision_select_layer: int = field(
        default=-1,
        metadata={'help': 'Specify the layer of ViT feature map to use. Default is -1 for the last layer.'},
    )
    use_backbone_lora: int = field(
        default=0,
        metadata={'help': 'Set the LoRA adapter rank for the ViT. Default is 0.'}
    )
    use_llm_lora: int = field(
        default=0,
        metadata={'help': 'Set the LoRA adapter rank for the LLM. Default is 0.'}
    )
    unfreeze_lm_head: bool = field(
        default=False,
        metadata={'help': 'Set to True to unfreeze the head of LLM. Default is False.'},
    )
    grad_checkpoint: bool = field(
        default=True,
        metadata={'help': 'Set to True to use gradient checkpointing. Default is True.'},
    )
    drop_path_rate: float = field(
        default=0.0,
        metadata={'help': 'Set the drop path rate for the ViT. Default is 0.'},
    )
    ps_version: Literal['v1', 'v2'] = field(
        default='v2',
        metadata={'help': 'Specify the version of pixel shuffle implementation. Default is v2.'}
    )
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={'help': 'Set to True to use the fast mode of the tokenizer.'}
    )
    use_liger: bool = field(
        default=False,
        metadata={'help': 'Set to True to use the liger kernel.'}
    )
    M: int = field(
        default=4,
        metadata={'help': 'Alias for num_thought_tokens, number of latent thought tokens. Default is 4.'}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments for specifying data input for training and evaluation.
    """
    max_seq_length: int = field(
        default=8192,
        metadata={
            'help': (
                'The maximum total input sequence length after tokenization. Sequences longer '
                'than this will be truncated, sequences shorter will be padded.'
            )
        },
    )
    force_image_size: int = field(
        default=448,
        metadata={'help': 'Set the desired size for the image. Default is 448.'},
    )
    down_sample_ratio: float = field(
        default=0.5,
        metadata={'help': 'Set the desired down-sampling ratio for the image. Default is 0.5.'},
    )
    pad2square: bool = field(
        default=False,
        metadata={'help': 'Pad the image to a square shape if set to True. Default is False.'},
    )
    conv_style: str = field(
        default='internlm2-chat', metadata={'help': 'Prompt style for a conversation.'}
    )
    meta_path: str = field(
        default=None,
        metadata={'help': 'The path of the meta file of datasets.'},
    )
    use_data_resampling: bool = field(
        default=False,
        metadata={'help': 'Set to True to use data resampling. Default is False.'},
    )
    dynamic_image_size: bool = field(
        default=False,
        metadata={'help': 'Set to True to use dynamic high resolution strategy. Default is False.'},
    )
    use_thumbnail: bool = field(
        default=False,
        metadata={'help': 'Set to True to add a thumbnail image. Default is False.'},
    )
    min_dynamic_patch: int = field(
        default=1,
        metadata={'help': 'The minimum number of dynamic patches. Default is 1.'},
    )
    max_dynamic_patch: int = field(
        default=12,
        metadata={'help': 'The maximum number of dynamic patches. Default is 12.'},
    )
    min_num_frame: int = field(
        default=8,
        metadata={'help': 'The minimum number of frames for video data. Default is 8.'},
    )
    max_num_frame: int = field(
        default=32,
        metadata={'help': 'The maximum number of frames for video data. Default is 32.'},
    )
    normalize_type: Literal['imagenet', 'clip', 'siglip'] = field(
        default='imagenet',
        metadata={'help': 'The normalization type for the image. Default is imagenet.'},
    )
    use_packed_ds: bool = field(
        default=False,
        metadata={'help': 'Whether to use packed dataset for efficient training. Default is False.'},
    )
    num_images_expected: int = field(
        default=40,
        metadata={'help': 'The maximum number of images per packed sample. Default is 40.'},
    )
    max_packed_tokens: int = field(
        default=8192,
        metadata={'help': 'The required token length of per packed sample. Default is 8192.'},
    )
    max_buffer_size: int = field(
        default=20,
        metadata={'help': 'The buffer size of the packed dataset. Default is 20.'},
    )
    log_freq: int = field(
        default=1000,
        metadata={'help': 'The log frequency of the packed dataset. Default is 1000.'},
    )
    strict_mode: bool = field(
        default=True,
        metadata={'help': 'Whether to pad the number of images to satisfy num_images_expected. Default is True.'},
    )
    replacement: bool = field(
        default=False,
        metadata={'help': 'Whether to restart the dataset after it is exhausted. Default is False.'},
    )
    allow_overflow: bool = field(
        default=False,
        metadata={'help': 'Whether to drop the sample over the specified max_packed_tokens. Default is False.'},
    )
    loss_reduction: str = field(
        default='token',
        metadata={'help': 'Loss reduction method. Default is token.'},
    )
    loss_reduction_all_gather: bool = field(
        default=False,
        metadata={'help': 'Whether to gather all during loss reduction. Default is False.'},
    )

    # ==========================================================================
    # --- New Implicit CoT & Driving Task Arguments ---
    # ==========================================================================
    num_thought_tokens: int = field(
        default=4,
        metadata={'help': 'The number of latent thought tokens (M) generated by the student. Default is 4.'},
    )
    num_trajectory_points: int = field(
        default=6,
        metadata={'help': 'The number of future trajectory points (x,y) to predict. Default is 6.'},
    )
    teacher_feature_dim: int = field(
        default=2048,
        metadata={'help': 'The hidden dimension size of the teacher model for distillation. Default is 2048.'},
    )
    distill_hs_weight: float = field(
        default=1.0,
        metadata={'help': 'Weight for the step-level hidden state alignment loss. Default is 1.0.'},
    )
    verbalizer_weight: float = field(
        default=0.5,
        metadata={'help': 'Weight for the verbalizer semantic anchoring loss. Default is 0.5.'},
    )
    trajectory_weight: float = field(
        default=2.0,
        metadata={'help': 'Weight for the trajectory prediction loss. Default is 2.0.'},
    )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        template_name,
        meta,
        tokenizer,
        tcs_loader,
        ds_name,
        num_image_token,
        image_size=448,
        is_train=True,
        pad2square=False,
        group_by_length=False,
        dynamic_image_size=False,
        use_thumbnail=False,
        min_dynamic_patch=1,
        max_dynamic_patch=6,
        min_num_frame=8,  # for video data
        max_num_frame=32,  # for video data
        sampling_method='rand',  # for video data
        repeat_time=1,
        normalize_type='imagenet',
        # hyperparameters for packed training
        use_packed_ds=False,
        data_rank=0,
        data_world_size=1,
        distributed_mode=False,
        force_shuffle=False,
        random_seed=0,
    ):
        super(LazySupervisedDataset, self).__init__()
        self.ds_name = ds_name
        self.tokenizer = tokenizer
        self.template_name = template_name
        self.num_image_token = num_image_token
        logger.info(f'[Dataset] num_image_token: {num_image_token}')
        logger.info(f'[Dataset] dynamic_image_size: {dynamic_image_size}')
        logger.info(f'[Dataset] use_thumbnail: {use_thumbnail}')
        logger.info(f'[Dataset] min_dynamic_patch: {min_dynamic_patch}, max_dynamic_patch: {max_dynamic_patch}')

        self.image_size = image_size
        self.is_train = is_train
        self.pad2square = pad2square
        self.max_num_frame = max_num_frame
        self.min_num_frame = min_num_frame
        self.sampling_method = sampling_method

        # hyperparameters for distributed training
        self.use_packed_ds = use_packed_ds
        self.data_rank = data_rank
        self.data_world_size = data_world_size
        self.worker_id = None
        self.worker_state_key = None
        self.worker_distributed = False
        self.distributed_mode = distributed_mode
        # hyperparameters for packed dataset
        self.dataset_type = 'pair'
        self.max_num_images = 1
        self.max_tokens = tokenizer.model_max_length
        self.force_shuffle = force_shuffle
        # TODO: quick resume
        self._state_dict = {}

        logger.info('Formatting inputs...Skip in lazy mode')
        assert meta['annotation'].endswith('jsonl'), f'annotation must be jsonl, but got {meta["annotation"]}'

        with open(meta['annotation'], 'r') as f:
            self.raw_data = f.readlines()
            if repeat_time < 1:
                # If repeat_time is less than 1, select a portion of the data
                self.raw_data = self.raw_data[:int(len(self.raw_data) * repeat_time)]
            if repeat_time > 1:
                assert isinstance(repeat_time, int)
                # Repeat the list if repeat_time is greater than 1
                self.raw_data = self.raw_data * repeat_time

        self.rng = np.random.default_rng(seed=random_seed)
        if self.force_shuffle:
            self.rng.shuffle(self.raw_data)

        self.root = meta['root']
        self.cached_data_dict = {}
        self.tcs_loader = tcs_loader
        self.group_by_length = group_by_length
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        self.normalize_type = normalize_type

        # If the precomputed length does not exist, roughly estimate the length of
        # each sample to improve the efficiency of group_by_length.
        if self.group_by_length:
            self.conv2length = {}  # Using a dictionary to speed up token length calculation
            self.length = []
            for data_item in self.raw_data:
                data_item = json.loads(data_item)
                if 'length' in data_item:
                    token_length = data_item['length']  # Use precomputed length if available
                else:
                    # Compute token length using the tokenizer
                    conversations = '\n'.join([temp['value'] for temp in data_item['conversations']])
                    str_length = len(conversations)
                    if str_length not in self.conv2length:
                        token_length = tokenizer(
                            conversations, return_tensors='pt', padding=False, truncation=False,
                        ).input_ids.size(1)
                        self.conv2length[str_length] = token_length + num_image_token * (
                                    max_dynamic_patch + use_thumbnail)
                    else:
                        token_length = self.conv2length[str_length]
                self.length.append(token_length)

    def __len__(self):
        return len(self.raw_data)

    def get_preprocess_function(self):
        # Select the appropriate preprocessing function based on the template name
        if self.template_name == 'Hermes-2':
            preprocess_function = preprocess_mpt
        elif self.template_name == 'internlm2-chat':
            preprocess_function = preprocess_internlm
        elif self.template_name == 'phi3-chat':
            preprocess_function = preprocess_phi3
        elif self.template_name == 'internvl2_5':
            preprocess_function = preprocess_internvl2_5
        else:
            preprocess_function = preprocess
        return preprocess_function

    def load_image(self, image_path):
        # Load the image using tcs_loader if available, otherwise use PIL
        if self.tcs_loader is not None and 's3://' in image_path:
            return self.tcs_loader(image_path)
        return Image.open(image_path).convert('RGB')

    def get_image_path(self, image_path):
        if image_path.startswith('s3://'):  # for ceph
            image_path = self.root + image_path
        else:  # for local image
            image_path = os.path.join(self.root, image_path)
        return image_path

    def get_transform(self):
        # Build transformation function
        transform = build_transform(is_train=self.is_train, input_size=self.image_size,
                                    pad2square=self.pad2square, normalize_type=self.normalize_type)
        return transform

    def multi_modal_get_item(self, data_item):
        # Build transformation function
        transform = self.get_transform()

        # Ensure the first conversation contains an image placeholder
        if '<image>' not in data_item['conversations'][0]['value']:
            data_item['conversations'][0]['value'] = '<image>\n' + data_item['conversations'][0]['value']

        # Merge the image path
        image_path = self.get_image_path(data_item['image'])

        # Load the image using tcs_loader if available, otherwise use PIL
        image = self.load_image(image_path)

        if self.dynamic_image_size:  # If dynamic image size is enabled, preprocess the image dynamically
            images = dynamic_preprocess(image, min_num=self.min_dynamic_patch, max_num=self.max_dynamic_patch,
                                        image_size=self.image_size, use_thumbnail=self.use_thumbnail)
        else:  # Otherwise, use the original image as a single patch
            images = [image]

        # Apply the transformation to each image and stack the results into a tensor
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)

        # Ensure that there is only one patch if dynamic image size is not enabled
        num_patches = pixel_values.size(0)
        if not self.dynamic_image_size:
            assert num_patches == 1, f'The number of patches should be 1, but got {num_patches}.'

        # Select the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function()

        # Preprocess the conversations and generate the return dictionary
        ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                                  self.tokenizer, [self.num_image_token * num_patches],
                                  group_by_length=self.group_by_length,
                                  use_packed_ds=self.use_packed_ds, ds_name=self.ds_name)

        # Calculate position_ids for packed dataset
        position_ids = ret['attention_mask'].long().cumsum(-1) - 1
        position_ids.masked_fill_(ret['attention_mask'] == 0, 1)
        image_end_token_id = self.tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
        assert (ret['input_ids'][0] == image_end_token_id).sum() == 1, f'image tokens are truncated, this dataset is {self.ds_name}'

        # Create the final return dictionary
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            position_ids=position_ids[0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long)
        )
        return ret

    def multi_modal_multi_image_get_item(self, data_item):
        # Build transformation function
        transform = self.get_transform()

        images, num_tiles = [], []
        num_image = len(data_item['image'])
        for image_path in data_item['image']:
            # Merge the image path
            image_path = self.get_image_path(image_path)
            # Load the image using tcs_loader if available, otherwise use PIL
            image = self.load_image(image_path)
            if self.dynamic_image_size:  # If dynamic image size is enabled, preprocess the image dynamically
                image = dynamic_preprocess(image, min_num=self.min_dynamic_patch,
                                           max_num=max(1, self.max_dynamic_patch // num_image),
                                           image_size=self.image_size, use_thumbnail=self.use_thumbnail)
                images += image
                num_tiles.append(len(image))
            else:  # Otherwise, use the original image as a single patch
                images.append(image)
                num_tiles.append(1)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)

        # Select the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function()

        # Preprocess the conversations and generate the return dictionary
        num_image_tokens = [self.num_image_token * num_tile for num_tile in num_tiles]
        ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                                  self.tokenizer, num_image_tokens, group_by_length=self.group_by_length,
                                  use_packed_ds=self.use_packed_ds, ds_name=self.ds_name, num_image=num_image)

        # Calculate position_ids for packed dataset
        position_ids = ret['attention_mask'].long().cumsum(-1) - 1
        position_ids.masked_fill_(ret['attention_mask'] == 0, 1)
        image_end_token_id = self.tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
        assert (ret['input_ids'][0] == image_end_token_id).sum() == num_image, f'image tokens are truncated, this dataset is {self.ds_name}'

        # Create the final return dictionary
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            position_ids=position_ids[0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long)
        )
        return ret

    def video_get_item(self, data_item):
        # Build transformation function
        transform = self.get_transform()

        # Ensure the first conversation contains a video placeholder
        if '<video>' not in data_item['conversations'][0]['value']:
            data_item['conversations'][0]['value'] = '<video>\n' + data_item['conversations'][0]['value']

        # Get the video file path
        video_file = data_item['video']
        video_path = os.path.join(self.root, video_file)

        # Load the video frames using tcs_loader
        # TODO: Load videos without using tcsloader.
        # image_list = self.tcs_loader(
        #     video_path,
        #     image_type='video',
        #     max_num_frames=self.max_num_frame,
        #     min_num_frames=self.min_num_frame,
        #     sample=self.sampling_method,
        #     clip=data_item.get('clip', None))
        if video_path.endswith('.gif'):
            image_list = read_frames_gif(
            video_path,
            num_frames=self.max_num_frame,
            min_num_frames=self.min_num_frame,
            sample=self.sampling_method)
        else:
            image_list = read_frames_decord(
            video_path,
            num_frames=self.max_num_frame,
            min_num_frames=self.min_num_frame,
            sample=self.sampling_method,
            clip=data_item.get('clip', None))
        # image_list = read_frames_decord(
        #     video_path,
        #     num_frames=self.max_num_frame,
        #     min_num_frames=self.min_num_frame,
        #     sample=self.sampling_method,
        #     clip=data_item.get('clip', None))
        # Generate special tokens for each video frame
        special_tokens = '\n'.join(['Frame-{}: <image>'.format(i + 1) for i in range(len(image_list))])
        data_item['conversations'][0]['value'] = data_item['conversations'][0]['value'].replace(
            '<video>\n', special_tokens + '\n')

        # Transform each frame image and stack them into a tensor
        pixel_values = [transform(image) for image in image_list]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)

        # Select the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function()

        # Preprocess the conversations and generate the return dictionary
        num_image_tokens = [self.num_image_token] * num_patches
        ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                                  self.tokenizer, num_image_tokens, group_by_length=self.group_by_length,
                                  use_packed_ds=self.use_packed_ds, ds_name=self.ds_name, num_image=num_patches)

        # Calculate position_ids for packed dataset
        position_ids = ret['attention_mask'].long().cumsum(-1) - 1
        position_ids.masked_fill_(ret['attention_mask'] == 0, 1)

        # Create the final return dictionary
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            position_ids=position_ids[0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long)
        )
        return ret

    def pure_text_get_item(self, data_item):
        # Build transformation function
        transform = self.get_transform()

        # Create a blank white image
        image = Image.new('RGB', (224, 224), (255, 255, 255))

        # Dynamically preprocess the image to generate patches
        images = dynamic_preprocess(image, min_num=self.min_dynamic_patch, max_num=1,
                                    image_size=self.image_size, use_thumbnail=self.use_thumbnail)

        # Apply the transformation to each image patch and stack them into a tensor
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)

        # Ensure there is only one patch
        assert num_patches == 1, f'The number of patches should be 1, but got {num_patches}.'

        # Select the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function()

        # Preprocess the conversations and generate the return dictionary
        ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                                  self.tokenizer, [self.num_image_token * num_patches], text_only=True,
                                  group_by_length=self.group_by_length, use_packed_ds=self.use_packed_ds,
                                  ds_name=self.ds_name)

        # Calculate position_ids for packed dataset
        position_ids = ret['attention_mask'].long().cumsum(-1) - 1
        position_ids.masked_fill_(ret['attention_mask'] == 0, 1)

        # Create the final return dictionary
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            position_ids=position_ids[0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([0] * num_patches, dtype=torch.long)
        )
        return ret

    def _enable_worker_distributed(self):
        if (
            self.distributed_mode
            and not self.worker_distributed
            and self.worker_id is not None
        ):
            self.worker_distributed = True
            self.raw_data = self.raw_data[self.worker_id::self.num_workers]
            logger.info(f'worker_distributed is enabled, {self.num_workers=}, {len(self.raw_data)=}')

    # def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # if i >= len(self.raw_data):
        #     if self.use_packed_ds:
        #         raise NotImplementedError
        #     else:
        #         i = i % len(self.raw_data)

        # try_cnt, max_try = 0, 10
        # while True:
        #     if try_cnt > max_try:
        #         raise StopIteration
        #     try:
        #         data_item = json.loads(self.raw_data[i])
        #         # conversations = data_item['conversations']
        #         # check_conversations_repetition(conversations, repeat_threshold=0.4, ngram=10)
        #         if 'image' in data_item and len(data_item['image']) != 0:
        #             if type(data_item['image']) == list:
        #                 ret = self.multi_modal_multi_image_get_item(data_item)
        #             else:
        #                 ret = self.multi_modal_get_item(data_item)
        #         elif 'video' in data_item and data_item['video'] is not None and data_item['video'] != '':
        #             ret = self.video_get_item(data_item)
        #         else:
        #             ret = self.pure_text_get_item(data_item)
        #         break
        #     except Exception as e:
        #         try_cnt += 1
        #         print(e, self.ds_name, flush=True)
        #         if not isinstance(e, (UnidentifiedImageError, FileNotFoundError)):
        #             traceback.print_exc()
        #         data_item = json.loads(self.raw_data[i])
        #         if 'image' in data_item:
        #             if type(data_item['image']) == list:
        #                 images = [self.root + item for item in data_item['image']]
        #                 print(f'Failed to load image: {images}, the dataset is: {self.ds_name}')
        #             else:
        #                 if data_item['image'].startswith('s3://'):
        #                     data_path = self.root + data_item['image']
        #                 else:
        #                     data_path = os.path.join(self.root, data_item['image'])
        #                 print(f'Failed to load image: {data_path}, the dataset is: {self.ds_name}')
        #         elif 'video' in data_item:
        #             data_path = os.path.join(self.root, data_item['video'])
        #             print(f'Failed to load video: {data_path}, the dataset is: {self.ds_name}')
        #         i = random.randint(0, len(self.raw_data) - 1)
        # return ret

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # 基础越界检查
        if i >= len(self.raw_data):
            i = i % len(self.raw_data)

        try_cnt, max_try = 0, 10
        while True:
            if try_cnt > max_try:
                raise StopIteration
            try:
                # 1. 解析原始字符串为字典
                data_item = json.loads(self.raw_data[i])
                
                # 2. 路由逻辑
                if 'image' in data_item and len(data_item['image']) != 0:
                    if isinstance(data_item['image'], list):
                        # 你的数据走这里：处理多张图
                        ret = self.multi_modal_multi_image_get_item(data_item)
                    else:
                        # 处理单张图
                        ret = self.multi_modal_get_item(data_item)
                elif 'video' in data_item and data_item['video']:
                    ret = self.video_get_item(data_item)
                else:
                    # 纯文本
                    ret = self.pure_text_get_item(data_item)
                break
            except Exception as e:
                # 错误处理逻辑（图片损坏等）
                try_cnt += 1
                print(f"Error loading index {i}: {e}")
                i = random.randint(0, len(self.raw_data) - 1)
        
        # 返回 ret 字典，包含 input_ids, labels, pixel_values 等
        return ret
    
    def __iter__(self):
        self._enable_worker_distributed()
        start_idx = 0

        assert self.worker_state_key is not None
        if self.worker_state_key in self._state_dict and len(self._state_dict[self.worker_state_key]) > 0:
            start_idx = self._state_dict[self.worker_state_key]['current_idx']

            self._state_dict.pop(self.worker_state_key)

        if self.worker_id == 0:
            logger.info(
                f'[{self.ds_name}] [Worker id {self.worker_id}] '
                f'begin to iter with {start_idx=}'
            )

        for i in range(start_idx, len(self)):
            yield self[i]


# ==============================================================================
# 2. 数据加载与 Collator
# ==============================================================================

# class ImplicitCoTDataset(LazySupervisedDataset):
#     def __init__(self, *args, hidden_state_dir=None, **kwargs):
#         # 注意：先调用 super().__init__，把通用的参数(args, kwargs)传给基类
#         # 基类 LazySupervisedDataset 的第一个参数是 template_name
#         super().__init__(*args, **kwargs)
#         # 再处理你自己特有的参数
#         self.hidden_state_dir = hidden_state_dir

#     def __getitem__(self, i):
#         # 1. 直接调用父类逻辑，获取 input_ids, labels, pixel_values 等
#         data_dict = super().__getitem__(i)
        
#         # 2. 重新解析原始 JSON 获取自定义字段
#         raw_item = json.loads(self.raw_data[i])
        
#         # 3. 解析轨迹 (从 GPT 的回复文本中提取)
#         # 示例内容: <answer> (-0.00,0.16), (-0.00,0.60) ... </answer>
#         gpt_response = raw_item['conversations'][1]['value']
#         # 使用正则提取括号内的数字对
#         traj_matches = re.findall(r'\(([-+]?\d*\.\d+|\d+),\s*([-+]?\d*\.\d+|\d+)\)', gpt_response)
#         if traj_matches:
#             # 转换为 [[x1, y1], [x2, y2], ...] 格式
#             gt_trajectory = [[float(x), float(y)] for x, y in traj_matches]
#             data_dict['gt_trajectory'] = torch.tensor(gt_trajectory, dtype=torch.float32)
        
#         # 4. 加载 Teacher Hidden States (.npy 文件)
#         if 'hidden_state_file' in raw_item:
#             hs_path = raw_item['hidden_state_file']
#             # 如果路径不是绝对路径，尝试拼接目录
#             if self.hidden_state_dir and not hs_path.startswith('/'):
#                 hs_path = os.path.join(self.hidden_state_dir, hs_path)
            
#             if os.path.exists(hs_path):
#                 # 转换为 float32 张量以便计算 MSE Loss
#                 hs_data = np.load(hs_path)
#                 data_dict['teacher_hidden_states'] = torch.from_numpy(hs_data).float()
#             else:
#                 # 建议在训练初期打印一次，确认路径无误
#                 if i == 0:
#                     print(f"Warning: Teacher hidden state not found at {hs_path}")

#         # 5. 处理 Teacher Text IDs
#         if 'teacher_text_ids' in raw_item:
#             data_dict['teacher_text_ids'] = torch.tensor(raw_item['teacher_text_ids'], dtype=torch.long)
            
#         return data_dict
class ImplicitCoTDataset(LazySupervisedDataset):
    def __init__(self, *args, hidden_state_dir=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.hidden_state_dir = hidden_state_dir

    def __getitem__(self, i):
        if i >= len(self.raw_data):
            i = i % len(self.raw_data)

        try_cnt, max_try = 0, 10
        while True:
            if try_cnt > max_try:
                raise StopIteration
            try:
                # 1. 直接调用父类逻辑，它会返回完整的 data_dict (含 input_ids, labels 等)
                # 父类内部已经处理了 data_item 的加载和路由
                ret = super().__getitem__(i)
                
                # 2. 检查 labels。如果没有 labels，说明该样本被截断得太厉害，无法训练
                if ret is None or 'labels' not in ret:
                    raise KeyError(f"Preprocess returned NO labels (likely truncation). Index: {i}")

                # 3. 重新解析这一行以提取子类特有的字段
                # 因为父类在 __getitem__ 失败时会随机换索引，我们需要拿到父类最终确定的那个 data_item
                # 但父类没返回 index，所以我们只能在子类自己逻辑里保底
                data_item = json.loads(self.raw_data[i])
                
                # --- 轨迹解析 (增加防御性代码) ---
                if len(data_item['conversations']) > 1:
                    gpt_response = data_item['conversations'][1]['value']
                    traj_matches = re.findall(r'\(([-+]?\d*\.\d+|\d+),\s*([-+]?\d*\.\d+|\d+)\)', gpt_response)
                    if traj_matches:
                        gt_trajectory = [[float(x), float(y)] for x, y in traj_matches]
                        ret['gt_trajectory'] = torch.tensor(gt_trajectory, dtype=torch.float32)

                # --- 隐状态解析 ---
                if 'hidden_state_file' in data_item:
                    hs_path = data_item['hidden_state_file']
                    if os.path.exists(hs_path):
                        ret['teacher_hidden_states'] = torch.from_numpy(np.load(hs_path)).float()
                
                # --- 教师 ID 解析 ---
                if 'teacher_text_ids' in data_item:
                    ret['teacher_text_ids'] = torch.tensor(data_item['teacher_text_ids'], dtype=torch.long)

                return ret

            except Exception as e:
                # 使用 stderr 确保信息在多进程下更有可能打印出来
                sys.stderr.write(f"Error at index {i}: {str(e)}\n")
                sys.stderr.flush()
                try_cnt += 1
                i = random.randint(0, len(self.raw_data) - 1)


def implicit_cot_collator(instances, tokenizer):

    for idx, inst in enumerate(instances):
        if 'labels' not in inst:
            # 这里一定会打印，因为是在崩溃前执行的
            print(f"[DEBUG] 样本 Index {idx} 缺失 labels! 所有的 Key 有: {inst.keys()}")
            # 打印一下这个样本的 input_ids 长度，看看是不是被截断了
            if 'input_ids' in inst:
                print(f"[DEBUG] input_ids 长度: {len(inst['input_ids'])}")
# 提取自定义字段
    gt_trajs = [inst.pop('gt_trajectory', None) for inst in instances]
    teacher_hs = [inst.pop('teacher_hidden_states', None) for inst in instances]
    teacher_ids = [inst.pop('teacher_text_ids', None) for inst in instances]

    # 调用常规拼接 (处理 input_ids, pixel_values 等)
    batch = concat_pad_data_collator(instances)

    # 拼装回 batch
    if gt_trajs[0] is not None:
        batch['gt_trajectory'] = torch.stack(gt_trajs) # 假设都是 6 个点
    if teacher_hs[0] is not None:
        batch['teacher_hidden_states'] = pad_sequence(teacher_hs, batch_first=True)
    if teacher_ids[0] is not None:
        batch['teacher_text_ids'] = pad_sequence(teacher_ids, batch_first=True, padding_value=tokenizer.pad_token_id)

    return batch

# ==============================================================================
# 3. 训练器与参数定义
# ==============================================================================

class ImplicitCoTTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss = outputs["loss"]
        
        # 只允许主进程 (local_rank <= 0) 打印/上报自定义 loss_items
        if self.args.local_rank <= 0 and self.state.global_step % self.args.logging_steps == 0:
            if "loss_items" in outputs:
                self.log({f"train/{k}": v.item() for k, v in outputs["loss_items"].items()})
                
        return (loss, outputs) if return_outputs else loss


# ==============================================================================
# 4. 主函数
# ==============================================================================

# 8. 精细化参数冻结逻辑（核心修改）
def _freeze_params(module):
    """通用冻结函数：冻结模块所有参数，禁止梯度更新"""
    for param in module.parameters():
        param.requires_grad = False
    # 可选：冻结后设为eval模式（避免BatchNorm/Dropout统计量变化）
    module.eval()
    module.requires_grad_(False)

def _unfreeze_params(module):
    """通用解冻函数：确保模块参数可训练"""
    for param in module.parameters():
        param.requires_grad = True
    module.train()
    module.requires_grad_(True)

def test_data_flow():
    print("\n" + "="*50)
    print("开始测试数据加载解析流程")
    print("="*50)

    # 1. 模拟初始化 Tokenizer
    # 换成你本地实际的路径，或者用 InternVL2-2B 占位
    model_path = "/data1/chenxiwu/ReCogDrive/model/internVL3_2B" 
    print(f"正在加载 Tokenizer: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # 2. 准备一个模拟的 meta 字典 (模拟 json 中的结构)
    # 注意：这里的 root 必须存在，或者确保 jsonl 里的路径是绝对路径
    mock_meta = {
        "annotation": "/data1/chenxiwu/ICoT/data/m_hs/test/merged_data.jsonl", # 你的真实 jsonl 路径
        "root": "/" 
    }

    # 3. 实例化子类数据集
    print("正在实例化 ImplicitCoTDataset...")
    dataset = ImplicitCoTDataset(
        template_name='internvl2_5',
        meta=mock_meta,
        tokenizer=tokenizer,
        tcs_loader=None,
        ds_name="test_ds",
        num_image_token=256,
        image_size=448,
        dynamic_image_size=True,
        use_thumbnail=True,
        hidden_state_dir=None # 因为你的 JSONL 里已经是绝对路径
    )

    # 4. 测试单条数据解析 (__getitem__)
    print(f"数据集读取成功，总条数: {len(dataset)}")
    
    # 抽取前 2 条进行深度检查
    samples = []
    for i in range(min(2, len(dataset))):
        print(f"\n--- 检查第 {i} 条样本 ---")
        item = dataset[i]
        
        # 验证基础字段
        print(f"基础字段: input_ids={item['input_ids'].shape}, pixel_values={item['pixel_values'].shape}")
        
        # 验证轨迹解析 (正则表达式)
        if 'gt_trajectory' in item:
            print(f"✅ 轨迹解析成功: 形状={item['gt_trajectory'].shape}")
            print(f"   样本值: {item['gt_trajectory'][:2]} ...") # 打印前两个点
        else:
            print("❌ 轨迹解析失败: 未找到 'gt_trajectory' 字段")

        # 验证隐状态加载 (.npy)
        if 'teacher_hidden_states' in item:
            print(f"✅ 隐状态加载成功: 形状={item['teacher_hidden_states'].shape}")
        else:
            print("❌ 隐状态加载失败: 请检查 .npy 路径是否正确")

        # 验证教师 Token IDs
        if 'teacher_text_ids' in item:
            print(f"✅ 教师 IDs 解析成功: 长度={len(item['teacher_text_ids'])}")
        
        samples.append(item)

    # 5. 测试批处理对齐 (Collator)
    print("\n" + "="*30)
    print("开始测试 Collator (Batching)")
    print("="*30)
    
    try:
        # 模拟 batch size = 2 的拼接
        batch = implicit_cot_collator(samples, tokenizer)
        
        print("Batch 拼接结果:")
        print(f"1. input_ids 形状: {batch['input_ids'].shape}")
        print(f"2. gt_trajectory 形状: {batch['gt_trajectory'].shape}")
        print(f"3. teacher_hidden_states 形状: {batch['teacher_hidden_states'].shape} (应该是补齐后的)")
        print(f"4. teacher_text_ids 形状: {batch['teacher_text_ids'].shape}")
        
        print("\n🎉 所有流程测试通过！解析和补齐逻辑正常。")
        
    except Exception as e:
        print(f"\n❌ Collator 测试失败!")
        traceback.print_exc()

def main():
    # 1. 初始化分布式环境 (必须在解析参数前)
    launcher = os.environ.get('LAUNCHER', 'pytorch')
    init_dist(launcher=launcher, backend='nccl')
    
    # 2. 解析参数
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # --- 关键修改：合并参数到 training_args 以便在 Trainer 中访问 ---
    # 这样在 ImplicitCoTTrainer 内部可以用 self.args.trajectory_weight 访问
    for key, value in vars(data_args).items():
        if not hasattr(training_args, key):
            setattr(training_args, key, value)
    for key, value in vars(model_args).items():
        if not hasattr(training_args, key):
            setattr(training_args, key, value)

    # 3. 设置日志
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', 
                        datefmt='%m/%d/%Y %H:%M:%S', handlers=[logging.StreamHandler(sys.stdout)])
    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()
    
    # 4. Tokenizer 处理 (参考官方：添加特殊 Token)
    tokenizer_path = model_args.model_name_or_path or model_args.llm_path
    logger.info(f'Loading Tokenizer: {tokenizer_path}')
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True, use_fast=False)
    tokenizer.tokenizer_path = tokenizer_path
    tokenizer.model_max_length = data_args.max_seq_length
    token_list = [IMG_START_TOKEN, IMG_END_TOKEN, IMG_CONTEXT_TOKEN,
                  QUAD_START_TOKEN, QUAD_END_TOKEN, REF_START_TOKEN,
                  REF_END_TOKEN, BOX_START_TOKEN, BOX_END_TOKEN, LOC_START_TOKEN, LOC_END_TOKEN,
                  FRONT_VIEW_TOKEN, FRONT_LEFT_VIEW_TOKEN, FRONT_RIGHT_VIEW_TOKEN,
                  BACK_LEFT_VIEW_TOKEN, BACK_RIGHT_VIEW_TOKEN, BACK_VIEW_TOKEN]
    num_new_tokens = tokenizer.add_tokens(token_list, special_tokens=True)
    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    
    # 5. Model Config 初始化
    config = InternVLChatConfig.from_pretrained(model_args.model_name_or_path)
    config.template = data_args.conv_style
    config.dynamic_image_size = data_args.dynamic_image_size
    config.use_thumbnail = data_args.use_thumbnail
    config.min_dynamic_patch = data_args.min_dynamic_patch
    config.max_dynamic_patch = data_args.max_dynamic_patch
    config.ps_version = model_args.ps_version
    config.force_image_size = data_args.force_image_size

    # 6. 加载模型
    model = ImplicitCoTDriverStudent.from_pretrained(
        model_args.model_name_or_path, 
        torch_dtype=torch.bfloat16, 
        config=config, 
        M=model_args.M
    )
    model.img_context_token_id = img_context_token_id
    
    # --- 关键修改：处理 ViT 位置编码和 Token 数量 ---
    patch_size = model.config.vision_config.patch_size
    if model.config.vision_config.image_size != data_args.force_image_size:
        model.vision_model.resize_pos_embeddings(
            old_size=model.config.vision_config.image_size,
            new_size=data_args.force_image_size,
            patch_size=patch_size
        )
        model.config.vision_config.image_size = data_args.force_image_size
    
    # 计算每个 patch 对应的 token 数量 (InternVL 逻辑)
    model.num_image_token = int((data_args.force_image_size // patch_size) ** 2 * (data_args.down_sample_ratio ** 2))

    # --- Resize Embedding 层 (因为添加了特殊 Token) ---
    if num_new_tokens > 0:
        model.language_model.resize_token_embeddings(len(tokenizer))
        # 官方逻辑：用平均值初始化新 Token 的 Embedding
        output_embeddings = model.language_model.get_output_embeddings().weight.data
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

    # 7. 梯度检查点与缓存设置
    model.language_model.config.use_cache = False
    if model_args.grad_checkpoint:
        model.vision_model.gradient_checkpointing = True
        model.vision_model.encoder.gradient_checkpointing = True
        model.language_model._set_gradient_checkpointing()
    
   

    if model_args.freeze_backbone: _freeze_params(model.vision_model)
    if model_args.freeze_llm: _freeze_params(model.language_model)
    if model_args.freeze_mlp: _freeze_params(model.mlp1)

   
    
    # 显式开启新模块的梯度
    model.latent_proj.requires_grad_(True)
    model.verbalizer.requires_grad_(True)

    # model.trajectory_head.requires_grad_(True)

    # 9. 构建数据集
    # ds_collections = json.loads(open(data_args.meta_path).read())
    # datasets = []
    # for ds_name in ds_collections.keys():
    #     datasets.append(ImplicitCoTDataset(
    #         template_name=data_args.conv_style,
    #         meta=ds_collections[ds_name],
    #         tokenizer=tokenizer,
    #         tcs_loader=None,
    #         ds_name=ds_name,
    #         num_image_token=model.num_image_token,
    #         image_size=data_args.force_image_size,
    #         dynamic_image_size=data_args.dynamic_image_size,
    #         use_thumbnail=data_args.use_thumbnail,
    #         min_dynamic_patch=data_args.min_dynamic_patch,
    #         max_dynamic_patch=data_args.max_dynamic_patch,
    #     ))
    # train_dataset = ConcatDataset(datasets)

    # # 10. 初始化 Trainer
    # # 此时 training_args 已经包含了 trajectory_weight 等参数
    # trainer = ImplicitCoTTrainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=train_dataset,
    #     tokenizer=tokenizer,
    #     data_collator=partial(implicit_cot_collator, tokenizer=tokenizer),
    # )

    # # 11. 开始训练
    # trainer.train()
    # trainer.save_model(training_args.output_dir)
    ds_collections = json.loads(open(data_args.meta_path).read())
    datasets = []
    for ds_name in ds_collections.keys():
        datasets.append(ImplicitCoTDataset(
            template_name=data_args.conv_style,
            meta=ds_collections[ds_name],
            tokenizer=tokenizer,
            tcs_loader=None,
            ds_name=ds_name,
            num_image_token=model.num_image_token,
            image_size=data_args.force_image_size,
            dynamic_image_size=data_args.dynamic_image_size,
            use_thumbnail=data_args.use_thumbnail,
            min_dynamic_patch=data_args.min_dynamic_patch,
            max_dynamic_patch=data_args.max_dynamic_patch,
            # 增加以下参数：
            pad2square=data_args.pad2square,
            group_by_length=training_args.group_by_length,
            normalize_type=data_args.normalize_type,
        ))
    train_dataset = ConcatDataset(datasets)

    # 10. 初始化 Trainer
    trainer = ImplicitCoTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=partial(implicit_cot_collator, tokenizer=tokenizer),
    )

    # 11. 开始训练
    trainer.train()
    trainer.save_model(training_args.output_dir)


if __name__ == '__main__':
    replace_llama_rmsnorm_with_fused_rmsnorm()
    replace_train_sampler()
    replace_train_dataloader()
    # 针对不同架构的优化补丁
    replace_internlm2_attention_class()
    replace_qwen2_attention_class()
    replace_phi3_attention_class()
    replace_llama_attention_class()
    # test_data_flow()
    main()