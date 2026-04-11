# AGENTS.md

This is the shared project context file for Codex and Claude. Read this file before doing repo-wide search. The goal is to let an agent answer most orientation questions and start focused work without re-reading the whole repository.

Generated from the current local codebase state on 2026-04-01.

## 1. Project In One Page

- Project name: Theia
- Paper/research framing: "Theia: Distilling Diverse Vision Foundation Models for Robot Learning"
- Code terminology: `Theia` in the paper is usually called `RVFM` or `RobotVisionFM` in the code.
- Main implementation root: `src/theia`
- Primary workflow:
  1. Organize raw image or video data into WebDataset shards.
  2. Extract teacher-model features into `.safetensors` shards.
  3. Compute per-teacher feature mean/std files.
  4. Train `RobotVisionFM` with Hydra + DDP.
  5. Optionally decode predicted teacher features into visual outputs such as DINO PCA, SAM masks, and depth.

At a high level, the model is:

- Backbone visual encoder, currently centered on DeiT variants.
- Translator heads that map backbone features into one or more teacher feature spaces.
- Distillation losses between predicted teacher features and pre-extracted teacher features.

## 2. Current Local Repo Reality

This repo is not a pristine "paper release only" snapshot. It contains local adaptations and partial migrations. These matter.

- `README.md` still reflects the original paper workflow and standard 224x224 usage.
- The codebase now also contains local NavSim and `navsim_stitch` pipelines.
- Some local preprocessing paths and model defaults target stitched 1024x256 images.
- Some train configs still target 224x224.
- Some teacher-model config files remain from older combinations even though the feature-size registry has been narrowed.

Treat the current code as "paper code plus local experimental extensions", not as a perfectly synchronized release.

## 3. Important Caveats Before Changing Anything

- Resolution is not globally consistent.
  - `src/theia/models/rvfm.py` and `src/theia/models/backbones.py` support tuple image sizes and currently default to non-square inputs.
  - `src/theia/scripts/preprocessing/image_datasets/organize_navsim_webdataset_stitch.py` explicitly creates 1024x256 stitched images.
  - `src/theia/configs/train_rvfm_navsim.yaml` overrides `model.backbone.image_size: [224, 224]`.
  - Before training or feature extraction changes, align image resolution, backbone patch geometry, and teacher feature shapes.

- Teacher feature-size support is currently incomplete.
  - `src/theia/foundation_models/common.py` defines `MODEL_FEATURE_SIZES`, but the active mapping currently contains only:
    - `facebook/dinov3-vitl16-pretrain-lvd1689m`
    - `google/siglip2-so400m-patch16-naflex`
    - `depth-anything/DA3-LARGE`
  - Older target-model configs such as `cdiv`, `ddsv`, and `cddsv` still exist under `src/theia/configs/training/target_models`.
  - If you use an older target-model config, verify or extend `MODEL_FEATURE_SIZES` first.

- Default paths are machine-specific.
  - `src/theia/configs/dataset/image_video_default.yaml` points to `/data/shengzhenli/theia_navsim_stitch_datasets`.
  - `src/theia/configs/logging/default.yaml` writes checkpoints to `/data/shengzhenli/theia/trained_models` and logs to `/data/shengzhenli/theia/logs`.
  - Override these on any other machine.

- Some helper scripts are stale.
  - `src/theia/scripts/train/sanity_check_train_rvfm.sh` uses config keys that do not match the current Hydra config layout.
  - Treat helper shell scripts as examples, not as guaranteed source of truth.

- Per-target loss weighting looks unfinished.
  - `RobotVisionFM.get_loss()` in `src/theia/models/rvfm.py` treats `self.target_loss_weights` as if it were a scalar, but the config naming suggests per-model weights.
  - If custom per-model weighting is needed, inspect this path before trusting it.

- Random target-model sampling is also partial.
  - In `src/theia/scripts/train/train_rvfm.py`, when `cfg.training.random_target_models > 0`, the code currently samples exactly 2 target models instead of using the config value.

- Decoding stats naming is less flexible than training stats naming.
  - Training dataset loading uses configurable `stats_prefix`.
  - `src/theia/decoding/decode.py` hardcodes `navsim_mean_*` and `navsim_var_*` naming in `load_feature_stats()`.

## 4. Top-Level Directory Map

Use this to avoid reading irrelevant folders.

| Path | Meaning | Usually read? |
| --- | --- | --- |
| `README.md` | Paper-level intro and basic usage | Yes, for top-level context |
| `pyproject.toml` | Package metadata and Python dependencies | Yes |
| `Dockerfile` | CUDA/Python container setup | Only if environment work is needed |
| `src/theia` | Actual Python package | Yes |
| `doc` | Images, GIFs, dataset format note | Only targeted files |
| `feature_stats` | Provided feature mean/std `.npy` files | Only if stats are relevant |
| `imagenet_feature_stats` | Stats for ImageNet workflows | Only if ImageNet is relevant |
| `trained_models` | Large checkpoint directory | Avoid unless loading or documenting checkpoints |
| `media` | Example media assets | Only for demo/decoding tasks |
| `decode_images` | Visual outputs/assets | Avoid unless decoding task |
| `decoding_results_navsim224*224_deit_small_cdiv` | Output artifacts | Avoid unless user asks |
| `test_images_for_decoding` | Sample decode inputs | Only for decode tasks |
| `logs` | Local run output | Usually no |

## 5. Fast File Routing By Task

Open only the files needed for the task.

| Task | Read these files first |
| --- | --- |
| Understand training flow | `src/theia/scripts/train/train_rvfm.py`, `src/theia/models/rvfm.py`, `src/theia/configs/train_rvfm_*.yaml` |
| Change backbone behavior | `src/theia/models/backbones.py`, `src/theia/models/utils.py` |
| Change translator behavior | `src/theia/models/feature_translators.py`, `src/theia/models/adapter_heads.py` |
| Change dataset loading | `src/theia/dataset/data_utils.py`, `doc/dataset_format.md` |
| Add or debug teacher feature extraction | `src/theia/scripts/preprocessing/feature_extraction.py`, `src/theia/preprocessing/feature_extraction_core/models.py`, `src/theia/foundation_models/*` |
| Organize raw datasets | `src/theia/scripts/preprocessing/image_datasets/*.py`, `src/theia/scripts/preprocessing/video_datasets/subsampling_videos.py` |
| Work on decoding and visualization | `src/theia/scripts/decoding/decoding_example.py`, `src/theia/decoding/*.py` |
| Use teacher-model wrappers directly | `src/theia/models/vfm.py`, `src/theia/foundation_models/__init__.py` |
| Inspect configs and overrides | `src/theia/configs/**` |
| Downstream CortexBench usage | `src/theia/utils/cortexbench/*` |

## 6. Package And Dependency Summary

Source package is installed from `src/` using setuptools.

Important dependencies from `pyproject.toml`:

- Python >= 3.10
- `torch`, `torchvision`, `torchaudio`
- `tensorflow==2.15.1`
- `hydra-core`, `omegaconf`
- `transformers`
- `webdataset` from a git URL branch: `elicassion/fix_shuffle_bug`
- `opencv-python`, `av`
- `wandb`
- `datasets`, `einops`, `tqdm`, `rich`, `matplotlib`, `seaborn`

Installation:

```bash
pip install -e .
```

## 7. Core Model Architecture

### 7.1 `RobotVisionFM`

Defined in `src/theia/models/rvfm.py`.

Key points:

- Backbone is built by `build_backbone()`.
- Translator is built by `build_feature_translator()`.
- `forward_feature()` returns backbone features before translator heads, after token handling.
- `forward()` returns a dict `{target_model_name: predicted_feature}`.
- `get_loss()` computes:
  - MSE loss
  - cosine embedding loss
  - Smooth L1 loss
- Current default main loss in config is `cos_l1`, implemented as:
  - `0.9 * cos_loss + 0.1 * l1_loss`

Input expectations:

- Images may be `[B, H, W, C]` or `[B, C, H, W]`
- Pixel dtype is typically `torch.uint8`
- Pixel range is typically `[0, 255]`

### 7.2 Backbones

Defined in `src/theia/models/backbones.py`.

Supported backbone families in this file:

- `DeiT`
- `DeiTNoCLS`
- `DeiTReg`

Important implementation details:

- Supports tuple image sizes such as `(256, 1024)`.
- Adds positional-embedding interpolation for non-square inputs.
- `DeiTNoCLS` removes the CLS token.
- `DeiTReg` adds register tokens and tracks `num_reg_tokens`.

If a task is about token counts, feature shapes, or non-square inputs, start here.

### 7.3 Translators

Defined in `src/theia/models/feature_translators.py`.

Translator types:

- `mlp`
- `conv`
- `lconv`
- `transformer` or `trans`

Important note:

- The code comment says `lconv` is the translator actively used now.
- `LightConvFeatureTranslator` is the current practical default.
- Target names are sanitized into module-safe names by replacing `.` with `_`.
- `_cls` target names use a linear head path.

### 7.4 Feature Output Handling

Defined in `src/theia/models/utils.py`.

`handle_feature_output()` supports:

- `mean_pooling`
- `max_pooling`
- `cls`
- `identity`
- `None`

For standard transformer output, `None` means "return spatial tokens and drop CLS".

## 8. Training Flow

Main entry:

- `src/theia/scripts/train/train_rvfm.py`

Hydra entrypoint:

- `@hydra.main(..., config_name="train_rvfm_imagenet")`

Actual flow:

1. Compose config from `src/theia/configs`.
2. Build target-model list from `cfg.training.target_models.target_model_names`.
3. Convert target model names into spatial feature sizes using `get_model_feature_size()` from `src/theia/foundation_models/common.py`.
4. Instantiate `RobotVisionFM`.
5. Wrap with DDP.
6. Load train and val datasets via `get_image_video_dataset()`.
7. Build per-column loaders via `get_frame_dataloader()`.
8. Merge column streams via `get_frame_iterator()`.
9. For each batch:
   - load `batch["image"]`
   - collect teacher `embedding` and optional `cls`
   - run `rvfm(images_batch)`
   - compute losses
   - backpropagate
   - log to wandb
   - save checkpoints periodically

Training characteristics:

- Distributed training uses NCCL.
- Learning rate scales with effective global batch size.
- Optimizer defaults to AdamW.
- Scheduler defaults to linear warmup plus constant LR.
- Logging is wandb-based.

Checkpoint naming pattern:

- `rvfm_dp{dataset_ratio}_{backbone}_{translator}_{pretrained?}_{notes}_step{step}.pth`

## 9. Hydra Config Layout

Config root:

- `src/theia/configs`

Top-level train configs:

- `train_rvfm_imagenet.yaml`
- `train_rvfm_navsim.yaml`
- `train_rvfm_navsim_stitch.yaml`

Config subtrees:

- `dataset/`
- `model/backbone/`
- `model/translator/`
- `training/`
- `logging/`

Important defaults:

- `train_rvfm_imagenet.yaml`
  - dataset: `imagenet`
  - backbone: `deit`
  - translator: `lconv`
  - training: `frame_level`
  - logging: `default`

- `train_rvfm_navsim.yaml`
  - dataset: `navsim`
  - overrides `model.backbone.image_size: [224, 224]`

- `train_rvfm_navsim_stitch.yaml`
  - dataset: `navsim_stitch`

Key config files to know:

- `src/theia/configs/dataset/image_video_default.yaml`
- `src/theia/configs/training/frame_level.yaml`
- `src/theia/configs/logging/default.yaml`
- `src/theia/configs/model/backbone/deit.yaml`
- `src/theia/configs/model/translator/lconv.yaml`

Current default values worth remembering:

- dataset root: `/data/shengzhenli/theia_navsim_stitch_datasets`
- batch size: `64`
- epochs: `25`
- warmup ratio: `0.1`
- base LR: `2e-3`
- default target-model config from `frame_level.yaml`: `cdiv`
- default main loss: `cos_l1`

## 10. Target-Model Configs

Target-model configs live in:

- `src/theia/configs/training/target_models`

Examples:

- `dinov3.yaml`
  - `facebook/dinov3-vitl16-pretrain-lvd1689m`

- `cdiv.yaml`
  - `google/vit-huge-patch14-224-in21k`
  - `facebook/dinov2-large`
  - `openai/clip-vit-large-patch14`

- `ddsv.yaml`
  - `google/vit-huge-patch14-224-in21k`
  - `facebook/dinov2-large`
  - `facebook/sam-vit-huge`
  - `LiheYoung/depth-anything-large-hf`

- `cddsv.yaml`
  - `google/vit-huge-patch14-224-in21k`
  - `facebook/dinov2-large`
  - `openai/clip-vit-large-patch14`
  - `facebook/sam-vit-huge`
  - `LiheYoung/depth-anything-large-hf`

- `cdi3v.yaml`
  - `google/vit-huge-patch14-224-in21k`
  - `facebook/dinov3-vitl16-pretrain-lvd1689m`
  - `openai/clip-vit-large-patch14`

Newer or single-model configs also exist for:

- `clip`
- `depth_anything`
- `depth_anything3`
- `dinov2`
- `dinov3`
- `sam`
- `sam3`
- `siglip2`
- `vit`

Before using a config, verify its feature sizes are covered in `src/theia/foundation_models/common.py`.

## 11. Dataset Format And Loading

Primary dataset-loading logic:

- `src/theia/dataset/data_utils.py`

Dataset format note:

- `doc/dataset_format.md`

Expected structure for image/video datasets:

```text
<dataset_root>/
  <dataset_name>/
    images/
      *-train.tar
      *-val.tar
    <model_name_with_slashes_replaced_by_underscores>/
      *-train.tar
      *-val.tar
    splits.json
```

Important details:

- Images are stored in tar shards as `.image` entries containing `.npy` RGB arrays.
- Teacher features are stored as `.safetensors`.
- A single feature sample usually contains:
  - `embedding`: `[C, H, W]`
  - optional `cls_token`: `[C]`

What `decode_sample()` does in `data_utils.py`:

- image bytes -> NumPy image -> optional image transform
- safetensors -> flatten `embedding` from `[C, H, W]` to `[(H*W), C]`
- `cls_token` is returned as `cls`

Supported dataset families:

- Image datasets: `imagenet`, `navsim`, `navsim_stitch`
- Video datasets: `ego4d_1in150`, `epic_kitchen_1in60`, `ssv2_1in32`
- OXE datasets: large RLDS/TFDS list under `src/theia/dataset/oxe/oxe_common.py`

Mixing logic:

- Dataset mixing is handled with `RandomMix`.
- Weights are normalized by dataset length.
- Train uses mixed sampling; val/eval forces equal weight of listed datasets.

Feature normalization:

- `get_image_video_dataset()` can normalize teacher features using precomputed mean/std.
- Stats filenames follow:
  - `<stats_prefix>_mean_<model_name>.npy`
  - `<stats_prefix>_var_<model_name>.npy`
- Model names in filenames replace `/` with `_`.

## 12. Preprocessing Pipeline

Main scripts:

- `src/theia/scripts/preprocessing/image_datasets/organize_imagenet_webdataset.py`
- `src/theia/scripts/preprocessing/image_datasets/organize_navsim_webdataset_stitch.py`
- `src/theia/scripts/preprocessing/video_datasets/subsampling_videos.py`
- `src/theia/scripts/preprocessing/feature_extraction.py`
- `src/theia/scripts/preprocessing/calc_feature_mean.py`
- `src/theia/scripts/preprocessing/calc_feature_mean_imagenet.py`
- `src/theia/scripts/preprocessing/calc_feature_mean_1024.py`

### 12.1 Organizing raw image data

ImageNet organizer:

- Converts imagefolder-style input into WebDataset shards.
- Resizes to 224x224.

NavSim stitch organizer:

- Stitches `CAM_L0 + CAM_F0 + CAM_R0`.
- Crops using DiffusionDrive-like logic.
- Resizes stitched output to 1024x256.
- Splits by log names into train/val.

### 12.2 Feature extraction

Main entry:

- `src/theia/scripts/preprocessing/feature_extraction.py`

Core model-dispatch logic:

- `src/theia/preprocessing/feature_extraction_core/models.py`

Teacher models supported there include:

- ViT
- SigLIP2
- SAM
- SAM3
- CLIP
- DINOv2
- DINOv3
- LLaVA vision tower
- Depth Anything
- Depth Anything 3 / DA3

Important storage detail:

- Feature output directories use model names with `/` replaced by `_`.
- Example:
  - `facebook/dinov3-vitl16-pretrain-lvd1689m`
  - becomes directory `facebook_dinov3-vitl16-pretrain-lvd1689m`

### 12.3 Feature statistics

Stats scripts compute per-channel mean and std-like values over flattened teacher tokens.

Used later for:

- teacher feature normalization during training
- denormalization during decoding

## 13. Decoding And Visualization

Primary files:

- `src/theia/decoding/decode.py`
- `src/theia/decoding/dinov2.py`
- `src/theia/decoding/sam.py`
- `src/theia/decoding/depth_anything.py`
- `src/theia/scripts/decoding/decoding_example.py`

What decoding does:

- Run a trained Theia model.
- Recover predicted teacher features.
- Denormalize those features.
- Visualize them through:
  - DINO PCA
  - SAM mask generation
  - Depth Anything decoder head

The example script:

- accepts an image or mp4
- loads a local checkpoint
- saves a `.png` or `.mp4` visualization

Be aware:

- decoding code expects `navsim_mean_*` / `navsim_var_*` stat filenames
- it is less configurable than the training data loader

## 14. Teacher-Model Wrapper Layer

If the task is not about Theia training itself but about teacher models or downstream comparison, read:

- `src/theia/models/vfm.py`
- `src/theia/foundation_models/__init__.py`

`VFMEncoder` wraps a single teacher model.

`ConcatVFMEncoder` concatenates multiple teacher features channel-wise.

These are useful for:

- direct teacher-feature extraction in downstream tasks
- comparison baselines
- downstream loading helpers

## 15. Logging, Optimizers, And Schedulers

Relevant files:

- `src/theia/utils/logging.py`
- `src/theia/optimizers/utils.py`
- `src/theia/lr_schedulers/lr_schedulers.py`

Current behavior:

- wandb is the main logging sink
- meter tracking is simple average accumulation
- optimizer parameter grouping is mostly weight-decay vs no-decay
- scheduler default is linear warmup + constant LR

There is no proper repository-wide automated test suite in the current tree. Validation is mainly script-level and run-level.

## 16. Commands You Are Most Likely To Need

### Install

```bash
pip install -e .
```

### Organize ImageNet into WebDataset

```bash
cd src/theia/scripts/preprocessing/image_datasets
python organize_imagenet_webdataset.py \
  --dataset imagenet \
  --imagenet-raw-path <raw_imagenet_dir> \
  --output-path <dataset_root>
```

### Organize stitched NavSim images

```bash
cd src/theia/scripts/preprocessing/image_datasets
python organize_navsim_webdataset_stitch.py \
  --dataset navsim_stitch \
  --image-dir <sensor_blobs/trainval> \
  --output-path <dataset_root>
```

### Extract teacher features

```bash
cd src/theia/scripts/preprocessing
python feature_extraction.py \
  --dataset <dataset_name> \
  --dataset-root <dataset_root> \
  --output-path <dataset_root> \
  --model <teacher_model_name> \
  --split train \
  --num-gpus <n>
```

### Compute feature mean/std files

```bash
cd src/theia/scripts/preprocessing
python calc_feature_mean.py \
  --dataset-path <dataset_root>/<dataset_name> \
  --output-path <stats_output_dir>
```

### Train with Hydra + DDP

```bash
cd src/theia/scripts/train
export HF_ENDPOINT=https://hf-mirror.com
export HYDRA_FULL_ERROR=1
torchrun --nproc_per_node=4 --nnodes=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:11111 \
  train_rvfm.py --config-name=train_rvfm_navsim_stitch \
  training/target_models=dinov3 \
  dataset.dataset_root=/data/shengzhenli/theia_navsim_stitch_datasets \
  logging.notes=dinov3
```

### Decode a checkpoint to visual outputs

```bash
cd src/theia/scripts/decoding
python decoding_example.py \
  --backbone facebook/deit-small-patch16-224 \
  --checkpoint-path <ckpt>.pth \
  --feature-stat-dir <stats_dir> \
  --media-to-vis-path <image_or_video>
```

## 17. Minimal Search Strategy For Future Agents

If you are Codex or Claude and need to work in this repo:

1. Read this file first.
2. Identify which of these lanes the task belongs to:
   - training
   - dataset loading
   - preprocessing
   - teacher-model extraction
   - decoding
   - downstream wrappers
3. Open only the 2 to 5 files listed for that lane in section 5.
4. Do not recursively read `trained_models`, `feature_stats`, `doc` images, or output artifact folders unless the task explicitly depends on them.
5. If the task mentions shape mismatch, always inspect:
   - `src/theia/foundation_models/common.py`
   - `src/theia/models/backbones.py`
   - `src/theia/models/adapter_heads.py`
   - active Hydra target-model config
6. If the task mentions data mismatch, always inspect:
   - `doc/dataset_format.md`
   - `src/theia/dataset/data_utils.py`
   - the specific organizer or feature-extraction script that created the data

## 18. Source Of Truth Ranking

When files disagree, trust them in this order:

1. Current Python implementation under `src/theia/`
2. Active Hydra config files under `src/theia/configs/`
3. Script-level examples under `src/theia/scripts/`
4. `README.md`

Reason:

- `README.md` and some helper scripts lag behind the current local experimental state.

## 19. If You Need More Context After This File

Only then branch into the right cluster:

- Training cluster:
  - `src/theia/scripts/train/train_rvfm.py`
  - `src/theia/models/rvfm.py`
  - `src/theia/models/backbones.py`
  - `src/theia/models/feature_translators.py`

- Data cluster:
  - `src/theia/dataset/data_utils.py`
  - `doc/dataset_format.md`
  - `src/theia/scripts/preprocessing/feature_extraction.py`

- Decode cluster:
  - `src/theia/decoding/decode.py`
  - `src/theia/scripts/decoding/decoding_example.py`

- Teacher model cluster:
  - `src/theia/preprocessing/feature_extraction_core/models.py`
  - `src/theia/foundation_models/*`

This file should be enough for first-pass orientation. Only go deeper if the task requires code changes or precise behavioral verification.
