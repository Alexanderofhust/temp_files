export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/data/shengzhenli/navtrain/maps"
export NAVSIM_EXP_ROOT="/data/shengzhenli/DiffusionDrive/exp"
export NAVSIM_DEVKIT_ROOT="/data/shengzhenli/DiffusionDrive"
export OPENSCENE_DATA_ROOT="/data/shengzhenli/navtrain"
export HYDRA_FULL_ERROR=1
export RAY_TMPDIR=/data/shengzhenli/ray_temp_storage
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0,1,2,3
# Fix cuDNN version mismatch - remove system cuDNN from path
export LD_LIBRARY_PATH=""


TRAIN_TEST_SPLIT=navtrain

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
        agent=theia_agent \
        experiment_name=training_theia_navsim_stitch_radio_agent  \
        train_test_split=navtrain  \
        split=trainval   \
        trainer.params.max_epochs=100 \
        use_cache_without_dataset=True  \
        force_cache_computation=False \
        cache_path="/data/shengzhenli/navsim-1.1/exp/training_data_cache" 



      