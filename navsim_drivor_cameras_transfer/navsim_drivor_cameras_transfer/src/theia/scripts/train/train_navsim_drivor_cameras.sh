export HF_ENDPOINT=https://hf-mirror.com
export HYDRA_FULL_ERROR=1
export USE_TF=0
export TRANSFORMERS_NO_TF=1

torchrun --nproc_per_node=4 --nnodes=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:11113 \
  train_rvfm.py --config-name=train_rvfm_navsim_drivor_cameras \
  logging.notes=navsim_drivor_cameras_sdd \
  training/target_models=sdd \
  dataset.dataset_ratio=1.0 \
  model.backbone.backbone=facebook/deit-small-patch16-224 \
  training.main_loss=cos_l1 \
  training.batch_size=4 \
  training.num_workers=4 \
  logging.save_ckpt_interval=50000 \
  dataset.dataset_root=/data/shengzhenli/theia_navsim_drivor_datasets
