"""
Configuration for Anomaly Action Spotting Model using X3D-M backbone.

Key differences from original X3D project:
- Binary classification (Normal vs Abnormal)
- Sliding window on untrimmed videos (1-second units)
- Fixed 16-frame sampling from 30-frame windows
- Different strides for training (8) vs inference (16)
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class SpottingModelConfig:
    """X3D-M model configuration for spotting."""
    name: str = "x3d_m"
    num_classes: int = 2  # Binary: Normal vs Abnormal
    dropout_rate: float = 0.5
    pretrained: bool = True

    # X3D-M specific parameters
    side_size: int = 256
    crop_size: int = 224
    test_crop_size: int = 256
    num_frames: int = 16  # Fixed: 16 frames per 1-second unit

    # Head dimension (for reference)
    head_dim: int = 2048


@dataclass
class SpottingDataConfig:
    """Dataset configuration for spotting."""
    # Dataset paths
    train_root: str = "/mnt/c/JJS/UCF_Crimes/Videos/train"
    train_annotation: str = "/mnt/c/JJS/UCF_Crimes/Videos/train/00_timestamp"
    test_root: str = "/mnt/c/JJS/UCF_Crimes/Videos/test"
    test_annotation: str = "/mnt/c/JJS/UCF_Crimes/Videos/test/00_timestamp"

    # Normalization (Kinetics pretrained values)
    mean: Tuple[float, ...] = (0.45, 0.45, 0.45)
    std: Tuple[float, ...] = (0.225, 0.225, 0.225)

    # Video parameters
    fps: int = 30
    unit_duration: float = 1.0  # 1 second per unit

    # Sliding window strides (in frames)
    train_stride: int = 8   # 50% overlap (0.5s step)
    infer_stride: int = 16  # No overlap (1.0s step), same as CLIP_FRAMES

    # Validation split
    val_split: float = 0.1


@dataclass
class SpottingTrainConfig:
    """Training configuration for spotting model."""
    # DataLoader
    batch_size: int = 8
    num_workers: int = 0
    pin_memory: bool = True

    # Phase 1: Head-only training
    phase1_epochs: int = 5
    phase1_lr: float = 1e-3

    # Phase 2: Full fine-tuning
    phase2_epochs: int = 10
    phase2_backbone_lr: float = 1e-5
    phase2_head_lr: float = 1e-4

    # Optimizer
    optimizer: str = "adamw"
    momentum: float = 0.9
    weight_decay: float = 1e-4

    # Learning rate scheduler
    scheduler: str = "cosine"  # "cosine" or "step"
    warmup_epochs: int = 1

    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    save_name: str = "spotting_x3d"
    save_best_only: bool = True

    # Random seed
    seed: int = 42

    # Early stopping
    patience: int = 5


# Constants for frame sampling
CLIP_FRAMES = 16  # Total frames to extract per 1-second unit
UNIT_FRAMES = 30  # Frames in 1 second at 30 fps

# Frame indices to sample from 30-frame window
# Even indices [0,2,4,6,8,10,12,14,16,18,20,22,24,26,28] + frame 29
SAMPLE_INDICES: List[int] = list(range(0, 29, 2)) + [29]  # Length: 16

# Class names for binary classification
SPOTTING_CLASSES: List[str] = ["Normal", "Abnormal"]


def get_spotting_config(
    model_name: str = "x3d_m",
    num_classes: int = 2
) -> SpottingModelConfig:
    """
    Get spotting model configuration.

    Args:
        model_name: Model variant (currently only x3d_m supported).
        num_classes: Number of output classes (default: 2 for binary).

    Returns:
        SpottingModelConfig instance.
    """
    if model_name != "x3d_m":
        raise ValueError(f"Only x3d_m is supported for spotting, got: {model_name}")

    return SpottingModelConfig(
        name=model_name,
        num_classes=num_classes,
    )


def get_frame_indices(window_size: int = 30) -> List[int]:
    """
    Get frame indices to sample from a window.

    For a 30-frame window (1 second at 30fps):
    - Sample even indices: 0, 2, 4, ..., 28 (15 frames)
    - Plus the last frame: 29 (1 frame)
    - Total: 16 frames

    Args:
        window_size: Size of the window in frames (default: 30).

    Returns:
        List of frame indices to sample.
    """
    if window_size != 30:
        # For different window sizes, sample uniformly
        indices = [int(i * (window_size - 1) / 15) for i in range(16)]
        return indices

    return SAMPLE_INDICES.copy()
