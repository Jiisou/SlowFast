"""
Utility functions for X3D training and evaluation.
"""

import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import CenterCropVideo, NormalizeVideo
from pytorchvideo.transforms import ApplyTransformToKey, ShortSideScale, UniformTemporalSubsample

from config import ModelConfig, DataConfig, UCF_CRIME_CLASSES


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Get available device (CUDA if available, else CPU)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def create_transform(model_cfg: ModelConfig, data_cfg: DataConfig, is_train: bool = True):
    """
    Create video transform pipeline.

    Args:
        model_cfg: Model configuration with crop sizes and frame counts.
        data_cfg: Data configuration with normalization parameters.
        is_train: Whether to create train transform (uses train crop size).

    Returns:
        ApplyTransformToKey transform for video data.
    """
    crop_size = model_cfg.crop_size if is_train else model_cfg.test_crop_size

    transform = ApplyTransformToKey(
        key="video",
        transform=Compose([
            UniformTemporalSubsample(model_cfg.num_frames),
            Lambda(lambda x: x / 255.0),
            NormalizeVideo(data_cfg.mean, data_cfg.std),
            ShortSideScale(size=model_cfg.side_size),
            CenterCropVideo(crop_size=(crop_size, crop_size)),
        ]),
    )
    return transform


def load_pretrained_model(model_name: str = "x3d_m", num_classes: int = 14) -> nn.Module:
    """
    Load pretrained X3D model from torch hub and replace classification head.

    Args:
        model_name: X3D variant ('x3d_xs', 'x3d_s', 'x3d_m', 'x3d_l').
        num_classes: Number of output classes.

    Returns:
        Model with replaced classification head.
    """
    print(f"Loading pretrained {model_name} from torch hub...")
    model = torch.hub.load('facebookresearch/pytorchvideo', model_name, pretrained=True)

    # Replace final classification layer
    in_features = model.blocks[5].proj.in_features
    model.blocks[5].proj = nn.Linear(in_features, num_classes)
    print(f"Replaced classification head: {in_features} -> {num_classes} classes")

    return model


def freeze_backbone(model: nn.Module):
    """Freeze backbone layers (blocks 0-4), keep head (block 5) trainable."""
    for param in model.blocks[:5].parameters():
        param.requires_grad = False
    for param in model.blocks[5].parameters():
        param.requires_grad = True


def unfreeze_all(model: nn.Module):
    """Unfreeze all model parameters."""
    for param in model.parameters():
        param.requires_grad = True


def count_parameters(model: nn.Module) -> dict:
    """Count trainable and total parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
    }


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list = None,
    save_path: str = "confusion_matrix.png",
    normalize: bool = False,
    title: str = "Confusion Matrix (X3D on UCF-Crime)"
):
    """
    Plot and save confusion matrix.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        class_names: List of class names.
        save_path: Path to save the figure.
        normalize: Whether to normalize the matrix by row (recall-based).
        title: Plot title.
    """
    if class_names is None:
        class_names = UCF_CRIME_CLASSES

    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = ".2f"
    else:
        fmt = "d"

    # Mask zero values for better visualization
    mask = (cm == 0) if not normalize else None

    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(
        cm, annot=True, fmt=fmt, cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        mask=mask
    )

    if mask is not None:
        ax.set_facecolor("silver")

    plt.title(title, fontsize=15)
    plt.ylabel("Actual Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"Confusion matrix saved to {save_path}")


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: str,
    save_full_model: bool = False
):
    """
    Save training checkpoint.

    Args:
        model: Model to save.
        optimizer: Optimizer state to save.
        epoch: Current epoch number.
        loss: Current loss value.
        path: Path to save checkpoint.
        save_full_model: If True, save entire model. If False, save state_dict only.
    """
    checkpoint = {
        'epoch': epoch,
        'loss': loss,
        'optimizer_state_dict': optimizer.state_dict(),
    }

    if save_full_model:
        checkpoint['model'] = model
    else:
        checkpoint['model_state_dict'] = model.state_dict()

    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(
    path: str,
    model: nn.Module = None,
    optimizer: torch.optim.Optimizer = None,
    device: torch.device = None
) -> dict:
    """
    Load training checkpoint.

    Args:
        path: Path to checkpoint file.
        model: Model to load weights into.
        optimizer: Optimizer to load state into.
        device: Device to map checkpoint to.

    Returns:
        Checkpoint dictionary with epoch, loss, etc.
    """
    if device is None:
        device = get_device()

    checkpoint = torch.load(path, map_location=device)

    if model is not None and 'model_state_dict' in checkpoint:
        result = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if result.missing_keys:
            print(f"Warning: Missing keys: {result.missing_keys}")
        if result.unexpected_keys:
            print(f"Warning: Unexpected keys: {result.unexpected_keys}")

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print(f"Loaded checkpoint from {path} (epoch {checkpoint.get('epoch', 'unknown')})")
    return checkpoint
