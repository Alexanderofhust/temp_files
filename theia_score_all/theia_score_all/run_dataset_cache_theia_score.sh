#!/usr/bin/env bash

export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/data/shengzhenli/navtrain/maps"
export NAVSIM_EXP_ROOT="/data/shengzhenli/DiffusionDrive/theia_score_dataset_cache"
export NAVSIM_DEVKIT_ROOT="/data/shengzhenli/DiffusionDrive"
export OPENSCENE_DATA_ROOT="/data/shengzhenli/navtrain"
export HYDRA_FULL_ERROR=1
export RAY_TMPDIR=/data/shengzhenli/ray_temp_storage
export HF_ENDPOINT=https://hf-mirror.com
export LD_LIBRARY_PATH=""

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_dataset_caching.py \
    agent=theia_score_agent \
    experiment_name=training_theia_score_agent \
    train_test_split=navtrain \
    split=trainval
