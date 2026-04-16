export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/data/shengzhenli/navtrain/maps"
export NAVSIM_EXP_ROOT="/data/shengzhenli/DiffusionDrive/navsim/diffusiondrive_6view_dataset_cache"
export NAVSIM_DEVKIT_ROOT="/data/shengzhenli/DiffusionDrive/navsim"
export OPENSCENE_DATA_ROOT="/data/shengzhenli/navtest"
export RAY_TMPDIR=/data/shengzhenli/ray_temp_storage
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0


python /data/shengzhenli/DiffusionDrive/navsim/planning/script/run_metric_caching.py \
    train_test_split=navtest \
    cache.cache_path=$NAVSIM_EXP_ROOT/metric_cache \
