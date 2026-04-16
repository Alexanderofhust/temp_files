# DiffusionDrive Project Guide

## Project Overview

DiffusionDrive is a truncated diffusion model for real-time end-to-end autonomous driving, achieving record-breaking 88.1 PDMS on NAVSIM benchmark. It's 10x faster than vanilla diffusion policy while maintaining higher accuracy and diversity.

- **Paper**: Accepted to CVPR 2025 as Highlight
- **Institution**: Huazhong University of Science and Technology (HUST) & Horizon Robotics
- **Framework**: Built on NAVSIM (Data-Driven Non-Reactive Autonomous Vehicle Simulation)
- **Backbone**: ResNet-34 (60M parameters) for NAVSIM, ResNet-50 for nuScenes
- **Performance**: 88.1 PDMS on NAVSIM, 45 FPS real-time speed

## Architecture

### Core Components

1. **Truncated Diffusion Policy** (`navsim/agents/diffusiondrive/`)
   - `transfuser_model_v2.py`: Main model architecture
   - `modules/conditional_unet1d.py`: 1D U-Net for trajectory diffusion
   - `modules/scheduler.py`: Diffusion scheduler (DDPM/DDIM)
   - `modules/multimodal_loss.py`: Multi-modal training loss
   - `transfuser_agent.py`: Agent interface for NAVSIM
   - `transfuser_features.py`: Feature extraction and processing
   - `transfuser_loss.py`: Training loss computation
   - `transfuser_callback.py`: Training callbacks

2. **Planning & Simulation** (`navsim/planning/`)
   - `simulation/planner/pdm_planner/`: PDM (Predictive Decision Making) planner
   - `training/`: Training infrastructure with PyTorch Lightning
   - `scenario_builder/`: NAVSIM scenario construction
   - `metric_caching/`: Metric computation and caching

3. **Evaluation** (`navsim/evaluate/`)
   - `pdm_score.py`: PDM score computation for benchmarking

## Directory Structure

```
DiffusionDrive/
├── navsim/                          # Main package
│   ├── agents/                      # Agent implementations
│   │   ├── diffusiondrive/          # DiffusionDrive agent (main)
│   │   ├── transfuser/              # Baseline Transfuser agent
│   │   ├── constant_velocity_agent.py
│   │   ├── ego_status_mlp_agent.py
│   │   └── human_agent.py
│   ├── planning/                    # Planning and training
│   │   ├── script/                  # Entry point scripts
│   │   ├── training/                # Training infrastructure
│   │   ├── simulation/              # Simulation and planners
│   │   └── metric_caching/          # Metric caching utilities
│   ├── evaluate/                    # Evaluation metrics
│   ├── common/                      # Common utilities
│   └── visualization/               # Visualization tools
├── scripts/                         # Shell scripts
│   ├── training/                    # Training scripts
│   ├── evaluation/                  # Evaluation scripts
│   └── submission/                  # Submission scripts
├── docs/                            # Documentation
│   ├── install.md                   # Installation guide
│   └── train_eval.md                # Training/evaluation guide
├── exp/                             # Experiment outputs
├── outputs/                         # Hydra outputs
└── tutorial/                        # Tutorial notebooks
```

## Development Workflow

### 1. Environment Setup

```bash
# Activate NAVSIM environment
conda activate navsim

# Install DiffusionDrive dependencies
pip install diffusers einops

# Install package in development mode
pip install -e .
```

### 2. Data Caching (Required before training/evaluation)

```bash
# Cache training dataset
python navsim/planning/script/run_dataset_caching.py \
    agent=diffusiondrive_agent \
    experiment_name=training_diffusiondrive_agent \
    train_test_split=navtrain

# Cache evaluation metrics
python navsim/planning/script/run_metric_caching.py \
    train_test_split=navtest \
    cache.cache_path=$NAVSIM_EXP_ROOT/metric_cache
```

### 3. Training

```bash
python navsim/planning/script/run_training.py \
    agent=diffusiondrive_agent \
    experiment_name=training_diffusiondrive_agent \
    train_test_split=navtrain \
    split=trainval \
    trainer.params.max_epochs=100 \
    cache_path="${NAVSIM_EXP_ROOT}/training_cache/" \
    use_cache_without_dataset=True \
    force_cache_computation=False
```

**Important**: Before training, configure in `navsim/agents/diffusiondrive/transfuser_config.py`:
- `bkb_path`: Path to pretrained ResNet-34 from [HuggingFace](https://huggingface.co/timm/resnet34.a1_in1k)
- `plan_anchor_path`: Path to clustered anchors (`kmeans_navsim_traj_20.npy`)

### 4. Evaluation

```bash
export CKPT=/path/to/checkpoint.pth

python navsim/planning/script/run_pdm_score.py \
    train_test_split=navtest \
    agent=diffusiondrive_agent \
    worker=ray_distributed \
    agent.checkpoint_path=$CKPT \
    experiment_name=diffusiondrive_agent_eval
```

## Key Technologies

- **Python**: 3.9+
- **Deep Learning**: PyTorch 2.0.1, PyTorch Lightning 2.2.1, torchvision 0.15.2
- **Diffusion**: diffusers library
- **Autonomous Driving**: nuplan-devkit, NAVSIM framework
- **Configuration**: Hydra 1.2.0
- **Distributed**: Ray
- **Visualization**: matplotlib, opencv-python, bokeh
- **Geospatial**: geopandas, Shapely, Fiona, rasterio

## Configuration System

Uses Hydra for configuration management:
- Config files in `navsim/planning/script/config/`
- Override parameters via command line: `key=value`
- Experiment outputs saved to `outputs/` and `exp/`

## Important Files

- `setup.py`: Package installation configuration
- `requirements.txt`: Python dependencies
- `environment.yml`: Conda environment specification
- `kmeans_navsim_traj_20.npy`: Pre-computed trajectory anchors (2.6KB)
- `pytorch_model.bin`: Pre-trained model weights (87MB)

## Coding Conventions

1. **Agent Structure**: All agents inherit from `AbstractAgent` in `navsim/agents/abstract_agent.py`
2. **Feature Builders**: Implement `AbstractFeatureTargetBuilder` for custom features
3. **Lightning Modules**: Training uses `AgentLightningModule` wrapper
4. **Hydra Configs**: Use dataclasses with `@dataclass` decorator
5. **Type Hints**: Use type annotations throughout
6. **Docstrings**: Follow NumPy docstring style

## Common Tasks

### Add a New Agent
1. Create agent directory under `navsim/agents/`
2. Implement agent class inheriting from `AbstractAgent`
3. Create feature builder, model, loss, and callback modules
4. Add Hydra config in `navsim/planning/script/config/`
5. Register agent in `navsim/agents/__init__.py`

### Modify Model Architecture
- Edit `navsim/agents/diffusiondrive/transfuser_model_v2.py`
- Update feature extraction in `transfuser_features.py`
- Adjust loss computation in `transfuser_loss.py`

### Debug Training
- Check logs in `exp/` and `outputs/` directories
- Use TensorBoard: `tensorboard --logdir exp/`
- Enable PyTorch Lightning profiling in config

### Visualize Results
- Use visualization tools in `navsim/visualization/`
- Render camera/lidar data with `camera.py` and `lidar.py`

## Performance Optimization

- **Caching**: Always cache datasets before training/evaluation
- **Distributed Training**: Use Ray for multi-GPU training
- **Mixed Precision**: Enable in PyTorch Lightning config
- **Batch Size**: Adjust based on GPU memory (default: varies by config)

## Troubleshooting

1. **Import Errors**: Ensure `pip install -e .` was run
2. **CUDA OOM**: Reduce batch size or enable gradient checkpointing
3. **Slow Training**: Verify dataset caching is enabled
4. **Config Errors**: Check Hydra config syntax and paths
5. **Missing Weights**: Download from [HuggingFace](https://huggingface.co/hustvl/DiffusionDrive)

## Resources

- **Paper**: https://arxiv.org/abs/2411.15139
- **Weights**: https://huggingface.co/hustvl/DiffusionDrive
- **NAVSIM**: https://github.com/autonomousvision/navsim
- **Contact**: bcliao@hust.edu.cn

## Notes for Claude

- This is a research codebase for autonomous driving
- Focus on trajectory prediction and planning tasks
- Heavy use of PyTorch Lightning and Hydra
- Diffusion models require careful hyperparameter tuning
- Always test changes with small-scale experiments first
- Respect the existing code structure and conventions
