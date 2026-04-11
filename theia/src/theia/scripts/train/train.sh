export HF_ENDPOINT=https://hf-mirror.com
export HYDRA_FULL_ERROR=1
torchrun --nproc_per_node=4 --nnodes=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:11111 \
  train_rvfm.py --config-name=train_rvfm_navsim \
  logging.notes=dinov3 \
  training/target_models=dinov3 \
  dataset.dataset_ratio=1.0 \
  model.backbone.backbone=facebook/deit-small-patch16-224 \
  logging.save_ckpt_interval=50000 \
  dataset.dataset_root=/data/shengzhenli/theia_navsim_datasets