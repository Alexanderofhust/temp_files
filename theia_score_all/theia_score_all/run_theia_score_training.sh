#!/usr/bin/env bash

export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/data/shengzhenli/navtrain/maps"
export NAVSIM_EXP_ROOT="/data/shengzhenli/DiffusionDrive/theia_score_exp"
export NAVSIM_DEVKIT_ROOT="/data/shengzhenli/DiffusionDrive"
export OPENSCENE_DATA_ROOT="/data/shengzhenli/navtrain"
export HYDRA_FULL_ERROR=1
export RAY_TMPDIR=/data/shengzhenli/ray_temp_storage
export NAVSIM_DISABLE_RAY_PROGRESS_BAR=1
export HF_ENDPOINT=https://hf-mirror.com
export LD_LIBRARY_PATH=""

TRAIN_TEST_SPLIT=navtrain
DATA_CACHE_PATH="/data/shengzhenli/DiffusionDrive/theia_score_dataset_cache/training_cache"
SCORE_CACHE_PATH="/data/shengzhenli/DiffusionDrive/theia_score_train_metric_cache/train_metric_cache"

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
    agent=theia_score_agent \
    experiment_name=training_theia_score_agent \
    train_test_split=$TRAIN_TEST_SPLIT \
    split=trainval \
    trainer.params.max_epochs=50 \
    use_cache_without_dataset=True \
    force_cache_computation=False \
    cache_path="$DATA_CACHE_PATH" \
    +agent.config.score_cache_dir="$SCORE_CACHE_PATH"
