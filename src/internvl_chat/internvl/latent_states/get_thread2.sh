#!/bin/bash
set -e  # 出错立即退出

export CUDA_VISIBLE_DEVICES=2  # 指定使用的GPU设备（根据需要修改）
# ==========================
# 核心配置（按需修改）
# ==========================
MODEL_PATH="/data2/chenxiwu/Drive/recogdrive/tune_output/tune_gptdrive_traj_only_cot"
DATA_JSON="/data1/chenxiwu/ICoT/data/original_data/part_2.jsonl"
OUTPUT_DIR="/data1/chenxiwu/ICoT/data/m_hs/thread2"
IMAGE_ROOT=""
SAVE_HIDDEN_STATES="True"
IMG_SIZE=448
M_TOKENS=4
PYTHON_SCRIPT="/data1/chenxiwu/ICoT/ICoT-Drive/src/internvl_chat/internvl/latent_states/get_latent_state_single_conver.py"  # 替换为你的python脚本文件名

# ==========================
# 执行逻辑（无需修改）
# ==========================
# 创建输出目录
mkdir -p "${OUTPUT_DIR}/m_hidden_states"

# 启动Python脚本（传递命令行参数）
python "${PYTHON_SCRIPT}" \
    --model_path "${MODEL_PATH}" \
    --data_json "${DATA_JSON}" \
    --output_dir "${OUTPUT_DIR}" \
    --image_root "${IMAGE_ROOT}" \
    --save_hidden_states "${SAVE_HIDDEN_STATES}" \
    --img_size "${IMG_SIZE}" \
    --m_tokens "${M_TOKENS}"

echo "提取完成！结果路径: ${OUTPUT_DIR}"