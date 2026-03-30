#!/bin/bash
set -x

# ==============================================================================
# 1. 硬件与分布式环境配置
# ==============================================================================
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=1,2,3
GPUS=${GPUS:-3}
BATCH_SIZE=${BATCH_SIZE:-6}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=34229
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch

# NCCL 优化（根据实际机器环境调整）
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0

# ==============================================================================
# 2. 路径配置
# ==============================================================================
MODEL_PATH="/data1/chenxiwu/ReCogDrive/model/internvl3_1B"
META_PATH="/data1/chenxiwu/ICoT/ICoT-Drive/src/internvl_chat/shell/data_info/student_train.json"
OUTPUT_DIR='/data1/chenxiwu/ICoT/student_train_output/test1'

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

# ==============================================================================
# 3. 运行训练脚本
# ==============================================================================
# 注意：请确保你的 python 文件名（例如 train_cot.py）与下方一致
torchrun \
  --nproc_per_node=${GPUS} \
  --master_port=${MASTER_PORT} \
  internvl/train/latent_internvl_train.py \
  --model_name_or_path ${MODEL_PATH} \
  --conv_style "internvl2_5" \
  --output_dir ${OUTPUT_DIR} \
  --meta_path ${META_PATH} \
  --overwrite_output_dir True \
  --force_image_size 448 \
  --max_dynamic_patch 6 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.1 \
  --max_seq_length 8192 \
  --do_train True \
  --dynamic_image_size True \
  --use_thumbnail True \
  --ps_version 'v2' \
  \
  --freeze_llm False \
  --freeze_mlp False \
  --freeze_backbone False \
  --vision_select_layer -1 \
  --grad_checkpoint True \
  \
  --num_train_epochs 2 \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps 1 \
  --learning_rate 4e-5 \
  --weight_decay 0.05 \
  --warmup_ratio 0.1 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --save_strategy "steps" \
  --save_steps 500 \
  --save_total_limit 1 \
  --evaluation_strategy "no" \
  --bf16 True \
  --dataloader_num_workers 1 \
  \
  --deepspeed "zero_stage1_config_student.json" \
  --report_to "tensorboard" \
  \
  --M 4 \
  --num_thought_tokens 4 \
  --num_trajectory_points 6 \
  --teacher_feature_dim 896 \
  --distill_hs_weight 1.0 \
  --verbalizer_weight 0.5 \
  --trajectory_weight 2.0 \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"