# Copyright (c) 2024 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
#! /bin/bash

# dataset=$1
# numgpus=$2
dataset=navsim_stitch
numgpus=4
dataset_root=/data/shengzhenli/theia_navsim_stitch_datasets
output_path=/data/shengzhenli/theia_navsim_stitch_datasets
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0,1,2,3
# modify models below
# models=(facebook/dinov2-large google/vit-huge-patch14-224-in21k openai/clip-vit-large-patch14 LiheYoung/depth-anything-large-hf) # facebook/sam-vit-huge
models=(depth-anything/DA3-LARGE ) # facebook/sam-vit-huge facebook/dinov3-vitl16-pretrain-lvd1689m google/siglip2-so400m-patch16-naflex
for model in ${models[@]}
do
    (
        python feature_extraction.py --dataset $dataset --dataset-root $dataset_root --output-path $output_path --model $model --split train --num-gpus $numgpus
        python feature_extraction.py --dataset $dataset --dataset-root $dataset_root --output-path $output_path --model $model --split val --num-gpus $numgpus
    ) &
done
wait
