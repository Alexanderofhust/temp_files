#!/usr/bin/env bash

export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/data/shengzhenli/navtrain/maps"
export NAVSIM_EXP_ROOT="/data/shengzhenli/DiffusionDrive/diffusiondrive_score_newloss_eval_exp"
export NAVSIM_DEVKIT_ROOT="/data/shengzhenli/DiffusionDrive"
export OPENSCENE_DATA_ROOT="/data/shengzhenli/navtest"
export HYDRA_FULL_ERROR=1
export HF_ENDPOINT=https://hf-mirror.com
export RAY_TMPDIR="/data/shengzhenli/ray_temp_eval"
export NAVSIM_DISABLE_RAY_PROGRESS_BAR=0

TRAIN_TEST_SPLIT=navtest
CHECKPOINT="/data/shengzhenli/DiffusionDrive/diffusiondrive_score_newloss_exp/lightning_logs/version_8/checkpoints/last.ckpt"
METRIC_CACHE_PATH="/data/shengzhenli/navsim-1.1/navtest_exp/metric_cache"

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
    train_test_split=$TRAIN_TEST_SPLIT \
    agent=diffusiondrive_score_newloss_agent \
    agent.checkpoint_path=$CHECKPOINT \
    metric_cache_path=$METRIC_CACHE_PATH \
    experiment_name=diffusiondrive_score_newloss_eval \
    worker=ray_distributed
