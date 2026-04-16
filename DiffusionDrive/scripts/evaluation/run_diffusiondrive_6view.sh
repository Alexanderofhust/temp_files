export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/data/shengzhenli/navtrain/maps"
export NAVSIM_EXP_ROOT="/data/shengzhenli/navsim-1.1/navtest_exp"
export NAVSIM_DEVKIT_ROOT="/data/shengzhenli/DiffusionDrive"
export OPENSCENE_DATA_ROOT="/data/shengzhenli/navtest"
export HYDRA_FULL_ERROR=1
export HF_ENDPOINT=https://hf-mirror.com
export RAY_TMPDIR=/data/shengzhenli/ray_temp_eval


TRAIN_TEST_SPLIT=navtest
CHECKPOINT=/data/shengzhenli/DiffusionDrive/diffusiondrive_6view_exp/lightning_logs/version_2/checkpoints/epoch99-step16700.ckpt
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
train_test_split=$TRAIN_TEST_SPLIT \
agent=diffusiondrive_6view_agent \
worker.threads_per_node=12 \
agent.checkpoint_path=$CHECKPOINT \
experiment_name=diffuiondrive_6view_agent_eval \