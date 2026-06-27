# Copyright (c) 2024 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
#! /bin/bash

dataset=navsim_drivor_cameras
numgpus=4
batch_size=2
dataset_root=/data/shengzhenli/theia_navsim_drivor_datasets
output_path=/data/shengzhenli/theia_navsim_drivor_datasets

export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0,1,2,3
export USE_TF=0
export TRANSFORMERS_NO_TF=1

# Modify models below if needed.
models=(
    facebook/dinov3-vitl16-pretrain-lvd1689m
    google/siglip2-so400m-patch16-naflex
    depth-anything/DA3-LARGE
)

for model in ${models[@]}
do
    python feature_extraction.py --dataset $dataset --dataset-root $dataset_root --output-path $output_path --model $model --split train --num-gpus $numgpus --batch-size $batch_size
    python feature_extraction.py --dataset $dataset --dataset-root $dataset_root --output-path $output_path --model $model --split val --num-gpus $numgpus --batch-size $batch_size
done
