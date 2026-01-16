# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the **X3D subproject** within Facebook AI Research's PySlowFast video understanding framework. X3D (Progressive Network Expansion for Efficient Video Recognition) is an efficient video classification architecture. This specific project focuses on **fine-tuning X3D models for UCF-CRIME dataset** (14-class abnormal behavior detection).

## Common Commands

### Environment Setup
```bash
conda create -n py311 python=3.11 -y
conda activate py311
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129
pip install 'git+https://github.com/facebookresearch/fvcore'
conda install av -c conda-forge -y
pip install opencv-python scikit-learn pandas matplotlib seaborn tqdm
```

### Install SlowFast (from repo root)
```bash
cd /mnt/c/Users/USER/Desktop/jjs/SlowFast
python setup.py build develop
```

### Training with X3D
```bash
python tools/run_net.py \
  --cfg configs/Kinetics/X3D_S.yaml \
  DATA.PATH_TO_DATA_DIR /path/to/dataset \
  NUM_GPUS 1 \
  TRAIN.BATCH_SIZE 16
```

### Testing Only
```bash
python tools/run_net.py \
  --cfg configs/Kinetics/X3D_S.yaml \
  TEST.CHECKPOINT_FILE_PATH /path/to/checkpoint.pyth \
  TRAIN.ENABLE False
```

### UCF-CRIME Evaluation
```bash
cd projects/x3d
python evaluate_ucf_crime.py
```

### Debugging with Fewer Workers
```bash
python tools/run_net.py --cfg configs/Kinetics/X3D_S.yaml DATA_LOADER.NUM_WORKERS 0 NUM_GPUS 1
```

## Architecture Overview

```
SlowFast/
├── slowfast/                    # Core framework library
│   ├── config/                  # YACS-based configuration (defaults.py)
│   ├── datasets/                # Data loaders (kinetics.py, ava_dataset.py, etc.)
│   ├── models/                  # Model implementations
│   │   ├── build.py             # MODEL_REGISTRY factory pattern
│   │   ├── video_model_builder.py  # SlowFast/I3D/C2D/X3D native implementations
│   │   └── ptv_model_builder.py    # PyTorchVideo model wrappers
│   └── utils/                   # Checkpoint, logging, metrics utilities
├── tools/                       # Entry points
│   ├── run_net.py               # Main orchestrator (train/test/demo)
│   ├── train_net.py             # Training loop
│   └── test_net.py              # Testing loop
├── configs/                     # YAML configs organized by dataset
│   └── Kinetics/X3D_*.yaml      # X3D model variants (XS, S, M, L)
└── projects/x3d/                # This project
    ├── model.py                 # Custom X3D factory from PyTorchVideo
    ├── evaluate_ucf_crime.py    # UCF-CRIME evaluation script
    └── ptv_x3d_ft_v2.ipynb      # Main fine-tuning notebook
```

## Key Files in This Project

- **`model.py`**: Custom X3D model factory adapted from PyTorchVideo. Use `create_x3d()` to instantiate models with configurable `model_num_class`, `depth_factor`, `input_clip_length`, etc.
- **`evaluate_ucf_crime.py`**: Standalone evaluation script. Uses `UCFCrimeDataset` class that reads timestamp CSVs from annotation directory.
- **`ptv_x3d_ft_v2.ipynb`**: Primary fine-tuning notebook with training loop and checkpoint saving.

## X3D Model Variants

| Model | Frames | Crop Size | Use Case |
|-------|--------|-----------|----------|
| X3D-XS | 13 | 160 | Fast inference, mobile |
| X3D-S | 13 | 160 | Balanced efficiency |
| X3D-M | 16 | 224 | Higher accuracy |
| X3D-L | 16 | 312 | Best accuracy |

## Configuration System

Configs are YACS-based with hierarchical override:
1. `slowfast/config/defaults.py` (base defaults)
2. YAML config file (model-specific)
3. Command-line overrides (highest priority)

Override example:
```bash
python tools/run_net.py --cfg config.yaml TRAIN.BATCH_SIZE 32 SOLVER.BASE_LR 0.01
```

## UCF-CRIME Dataset Structure

```
UCF_Crimes/Videos/
├── train/
│   ├── 00_timestamp/           # CSV annotations per class
│   │   ├── Abuse_timestamps.csv
│   │   └── ...
│   ├── Abuse/                  # Video files
│   └── ...
└── test/
    └── (same structure)
```

CSV format: `file_name, start_time, end_time` (timestamps in HH:MM:SS or MM:SS format)

## Important Notes

- **Conv3D Performance**: Naïve channelwise 3D convolution in PyTorch is slow. See [PyTorch PR #40801](https://github.com/pytorch/pytorch/pull/40801) for optimization.
- **Multi-clip Testing**: SlowFast uses 3 spatial crops during evaluation by default for better accuracy.
- **Checkpoint Loading**: Models save as `.pyth` (SlowFast convention) or `.pt/.pth` (PyTorch convention). Use `strict=False` when loading if head dimensions differ.
