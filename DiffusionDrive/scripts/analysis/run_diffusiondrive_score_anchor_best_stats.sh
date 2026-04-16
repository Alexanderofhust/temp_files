#!/usr/bin/env bash

set -euo pipefail

source /home/shengzhenli/anaconda3/etc/profile.d/conda.sh
conda activate navsim

export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/data/shengzhenli/navtrain/maps"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/data/shengzhenli/DiffusionDrive/diffusiondrive_score_exp}"
export NAVSIM_DEVKIT_ROOT="/data/shengzhenli/DiffusionDrive"
export OPENSCENE_DATA_ROOT="/data/shengzhenli/navtrain"
export HYDRA_FULL_ERROR=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export NAVSIM_DISABLE_RAY_PROGRESS_BAR="${NAVSIM_DISABLE_RAY_PROGRESS_BAR:-1}"
export RAY_TMPDIR="${RAY_TMPDIR:-/data/shengzhenli/ray_temp_storage}"
export LD_LIBRARY_PATH=""

cd "$NAVSIM_DEVKIT_ROOT"
python "$NAVSIM_DEVKIT_ROOT/scripts/analysis/diffusiondrive_score_anchor_best_stats.py" "$@"
