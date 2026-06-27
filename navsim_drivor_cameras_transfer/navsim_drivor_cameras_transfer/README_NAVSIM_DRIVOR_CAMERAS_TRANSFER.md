# navsim_drivor_cameras 合并说明

这个包用于把 Theia 里的一个新数据集配置合入已有代码库：`navsim_drivor_cameras`。

目标是用 NavSim 原始相机图像生成和 DrivoR 输入对齐的四相机 WebDataset：

- 相机：`CAM_F0`, `CAM_B0`, `CAM_L0`, `CAM_R0`
- 每张图独立保存，不做 stitch
- 分辨率：`1148 x 672`，对应 DrivoR 的 `image_size: [1148, 672]`
- split：仍然按原来的 NavSim train / val 划分
- 不覆盖原来的 `navsim` 或 `navsim_stitch`

## 文件说明

新增文件可以直接按目录结构复制到目标 repo：

```text
src/theia/configs/dataset/navsim_drivor_cameras.yaml
src/theia/configs/train_rvfm_navsim_drivor_cameras.yaml
src/theia/scripts/preprocessing/image_datasets/organize_navsim_webdataset_drivor_cameras.py
src/theia/scripts/preprocessing/image_datasets/run_organize_navsim_drivor_cameras.sh
src/theia/scripts/preprocessing/iv_feature_extraction_drivor_cameras.sh
src/theia/scripts/preprocessing/calc_feature_mean_navsim_drivor_cameras.py
src/theia/scripts/preprocessing/calc_feature_mean_navsim_drivor_cameras.sh
src/theia/scripts/train/train_navsim_drivor_cameras.sh
```

这两个已有文件建议不要盲目覆盖，优先参考 `MODIFIED_FILES_MERGE_HINT.diff` 手动合并：

```text
src/theia/dataset/image/image_common.py
src/theia/scripts/train/train_rvfm.py
```

主要改动只有两处：

1. 在 `ALL_IMAGE_DATASETS` 里注册 `"navsim_drivor_cameras"`。
2. 在 `train_rvfm.py` 里支持从 dataset config 读取 `feature_size_overrides`，用于高分辨率相机输入对应的 teacher feature shape。

## 运行前需要改的路径

先根据目标服务器修改这些脚本里的路径：

```text
src/theia/scripts/preprocessing/image_datasets/run_organize_navsim_drivor_cameras.sh
src/theia/scripts/preprocessing/iv_feature_extraction_drivor_cameras.sh
src/theia/scripts/preprocessing/calc_feature_mean_navsim_drivor_cameras.sh
src/theia/scripts/train/train_navsim_drivor_cameras.sh
```

重点检查：

- `IMAGE_DIR`
- `NAVSIM_LOGS_DIR`
- `OUTPUT_PATH`
- `SPLIT_CONFIG`
- `dataset_root`
- `output_path`
- `CUDA_VISIBLE_DEVICES`
- `training.batch_size`
- `training.num_workers`

## 运行顺序

1. 生成四相机 WebDataset：

```bash
cd src/theia/scripts/preprocessing/image_datasets
bash run_organize_navsim_drivor_cameras.sh
```

2. 提取 teacher feature cache：

```bash
cd src/theia/scripts/preprocessing
bash iv_feature_extraction_drivor_cameras.sh
```

3. 计算 feature mean / std：

```bash
cd src/theia/scripts/preprocessing
bash calc_feature_mean_navsim_drivor_cameras.sh
```

4. 训练新的 distilled encoder：

```bash
cd src/theia/scripts/train
bash train_navsim_drivor_cameras.sh
```

## 兼容性备注

- 这个包默认对方代码库已经有 Theia 原本的 NavSim / `navsim_stitch` 数据处理、feature extraction、`train_rvfm.py` 等基础逻辑。
- 新 dataset 是单独命名的 `navsim_drivor_cameras`，尽量不影响原有实验。
- `train_navsim_drivor_cameras.sh` 里默认用 `training/target_models=sdd`，如果目标代码库里的 target model 名称不同，需要按本地已有配置调整。
- 训练得到的 encoder checkpoint 仍然是 Theia 的 DeiT-style backbone 权重；给 DrivoR-Theia 加载时只替换 backbone，不会引入 Theia 的 feature adapter。
