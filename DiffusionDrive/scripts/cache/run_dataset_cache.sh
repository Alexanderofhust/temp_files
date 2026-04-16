export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/data/shengzhenli/navtrain/maps"
export NAVSIM_EXP_ROOT="/data/shengzhenli/DiffusionDrive/navsim/dinov3drivedrive_4view_dataset_cache"
export NAVSIM_DEVKIT_ROOT="/data/shengzhenli/DiffusionDrive/navsim"
export OPENSCENE_DATA_ROOT="/data/shengzhenli/navtrain"
# CACHE_PATH=$NAVSIM_EXP_ROOT/metric_cache_train
export RAY_TMPDIR=/data/shengzhenli/ray_temp_storage
export HF_ENDPOINT=https://hf-mirror.com

python /data/shengzhenli/DiffusionDrive/navsim/planning/script/run_dataset_caching.py \
    agent=dinov3drive_agent \
    experiment_name=training_dinov3drive_gent \
    train_test_split=navtrain \