"""
Configuration for X3D model fine-tuning on UCF-CRIME dataset.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


@dataclass
class ModelConfig:
    """X3D model configuration."""
    name: str = "x3d_m"
    num_classes: int = 14
    dropout_rate: float = 0.5
    pretrained: bool = True

    # Model-specific parameters (populated by get_model_config)
    side_size: int = 256
    crop_size: int = 224  # train crop size
    test_crop_size: int = 256
    num_frames: int = 16
    sampling_rate: int = 5


@dataclass
class DataConfig:
    """Dataset configuration."""
    train_root: str = "/mnt/c/JJS/UCF_Crimes/Videos/train"
    train_annotation: str = "/mnt/c/JJS/UCF_Crimes/Videos/train/00_timestamp"
    test_root: str = "/mnt/c/JJS/UCF_Crimes/Videos/test"
    test_annotation: str = "/mnt/c/JJS/UCF_Crimes/Videos/test/00_timestamp"

    # Normalization (Kinetics pretrained values)
    mean: Tuple[float, ...] = (0.45, 0.45, 0.45)
    std: Tuple[float, ...] = (0.225, 0.225, 0.225)
    fps: int = 30

    # Clip settings
    max_clip_duration: float = 4.0


@dataclass
class TrainConfig:
    """Training configuration."""
    # DataLoader
    batch_size: int = 8
    num_workers: int = 0
    pin_memory: bool = True

    # Phase 1: Head-only training
    phase1_epochs: int = 2
    phase1_lr: float = 1e-3

    # Phase 2: Full fine-tuning
    phase2_epochs: int = 3
    phase2_backbone_lr: float = 1e-5
    phase2_head_lr: float = 1e-3

    # Optimizer
    optimizer: str = "sgd"  # "sgd" or "adamw"
    momentum: float = 0.9
    weight_decay: float = 1e-4

    # Checkpointing
    checkpoint_dir: str = "."
    save_name: str = "x3d_ucfcrime"

    # Random seed
    seed: int = 42


# Model-specific transform parameters
MODEL_TRANSFORM_PARAMS: Dict[str, Dict] = {
    "x3d_xs": {
        "side_size": 182,
        "crop_size": 182,
        "test_crop_size": 182,
        "num_frames": 4,
        "sampling_rate": 12,
    },
    "x3d_s": {
        "side_size": 182,
        "crop_size": 182,
        "test_crop_size": 182,
        "num_frames": 13,
        "sampling_rate": 6,
    },
    "x3d_m": {
        "side_size": 256,
        "crop_size": 224,
        "test_crop_size": 256,
        "num_frames": 16,
        "sampling_rate": 5,
    },
    "x3d_l": {
        "side_size": 356,
        "crop_size": 312,
        "test_crop_size": 356,
        "num_frames": 16,
        "sampling_rate": 5,
    },
}

# UCF-CRIME class names (alphabetically sorted)
UCF_CRIME_CLASSES: List[str] = [
    "Abuse", "Arrest", "Arson", "Assault", "Burglary",
    "Explosion", "Fighting", "Normal", "RoadAccidents",
    "Robbery", "Shooting", "Shoplifting", "Stealing", "Vandalism"
]


def get_model_config(model_name: str = "x3d_m", num_classes: int = 14) -> ModelConfig:
    """Get model configuration with transform parameters for specified model."""
    if model_name not in MODEL_TRANSFORM_PARAMS:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(MODEL_TRANSFORM_PARAMS.keys())}")

    params = MODEL_TRANSFORM_PARAMS[model_name]
    return ModelConfig(
        name=model_name,
        num_classes=num_classes,
        side_size=params["side_size"],
        crop_size=params["crop_size"],
        test_crop_size=params["test_crop_size"],
        num_frames=params["num_frames"],
        sampling_rate=params["sampling_rate"],
    )


def get_clip_duration(model_cfg: ModelConfig, fps: int = 30) -> float:
    """Calculate clip duration in seconds based on model config."""
    return (model_cfg.num_frames * model_cfg.sampling_rate) / fps
