# Copyright (c) 2024 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
#! /bin/bash

# dataset=$1
# numgpus=$2
dataset=navsim
numgpus=4
output_path=/data/shengzhenli/theia_navsim_datasets
dataset_root=/data/shengzhenli/theia_navsim_datasets
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0,1,2,3
# modify models below
# models=(facebook/dinov2-large google/vit-huge-patch14-224-in21k openai/clip-vit-large-patch14 LiheYoung/depth-anything-large-hf) # facebook/sam-vit-huge
models=(Qwen/Qwen3-VL-8B-Instruct) # facebook/sam-vit-huge facebook/dinov3-vitl16-pretrain-lvd1689m Qwen/Qwen3-VL-8B-Instruct
for model in ${models[@]}
do
    (
        python feature_extraction.py --dataset $dataset --dataset-root $output_path --output-path $output_path --model $model --split train --num-gpus $numgpus
        python feature_extraction.py --dataset $dataset --dataset-root $output_path --output-path $output_path --model $model --split val --num-gpus $numgpus
    ) &
done
wait
