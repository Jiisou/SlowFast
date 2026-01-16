#!/usr/bin/env python3
"""
Training script for X3D fine-tuning on UCF-CRIME dataset.

Two-phase training strategy:
- Phase 1: Freeze backbone, train only the classification head
- Phase 2: Unfreeze all layers, fine-tune entire network with differential learning rates
"""

import argparse
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    ModelConfig, DataConfig, TrainConfig,
    get_model_config, get_clip_duration
)
from dataset import UCFCrimeDataset
from utils import (
    set_seed, get_device, create_transform,
    load_pretrained_model, freeze_backbone, unfreeze_all,
    count_parameters, save_checkpoint
)


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    desc: str = "Training"
) -> float:
    """
    Train model for one epoch.

    Args:
        model: Model to train.
        data_loader: Training data loader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to train on.
        desc: Description for progress bar.

    Returns:
        Average loss for the epoch.
    """
    model.train()
    total_loss = 0.0

    pbar = tqdm(data_loader, desc=desc, leave=True)
    for i, (inputs, labels) in enumerate(pbar):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "avg_loss": f"{total_loss / (i + 1):.4f}"
        })

    return total_loss / len(data_loader)


def create_optimizer(
    model: nn.Module,
    train_cfg: TrainConfig,
    phase: int = 1
) -> optim.Optimizer:
    """
    Create optimizer for the specified training phase.

    Args:
        model: Model to optimize.
        train_cfg: Training configuration.
        phase: Training phase (1 or 2).

    Returns:
        Configured optimizer.
    """
    if phase == 1:
        # Phase 1: Only head parameters
        params = model.blocks[5].parameters()
        lr = train_cfg.phase1_lr
    else:
        # Phase 2: Differential learning rates
        params = [
            {'params': model.blocks[:5].parameters(), 'lr': train_cfg.phase2_backbone_lr},
            {'params': model.blocks[5].proj.parameters(), 'lr': train_cfg.phase2_head_lr}
        ]
        lr = train_cfg.phase2_head_lr  # Not used for param groups

    if train_cfg.optimizer.lower() == "sgd":
        return optim.SGD(
            params,
            lr=lr if phase == 1 else train_cfg.phase2_head_lr,
            momentum=train_cfg.momentum,
            weight_decay=train_cfg.weight_decay if phase == 2 else 0
        )
    elif train_cfg.optimizer.lower() == "adamw":
        return optim.AdamW(
            params,
            lr=lr if phase == 1 else train_cfg.phase2_head_lr,
            weight_decay=train_cfg.weight_decay if phase == 2 else 0
        )
    else:
        raise ValueError(f"Unknown optimizer: {train_cfg.optimizer}")


def train(
    model_cfg: ModelConfig,
    data_cfg: DataConfig,
    train_cfg: TrainConfig,
):
    """
    Main training function with two-phase strategy.

    Args:
        model_cfg: Model configuration.
        data_cfg: Data configuration.
        train_cfg: Training configuration.
    """
    # Setup
    set_seed(train_cfg.seed)
    device = get_device()
    torch.cuda.empty_cache()

    # Create transforms
    train_transform = create_transform(model_cfg, data_cfg, is_train=True)

    # Create dataset and dataloader
    print("\nLoading training dataset...")
    train_dataset = UCFCrimeDataset(
        root_dir=data_cfg.train_root,
        annotation_dir=data_cfg.train_annotation,
        transform=train_transform,
        max_clip_duration=data_cfg.max_clip_duration,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
        pin_memory=train_cfg.pin_memory,
    )

    # Load pretrained model
    model = load_pretrained_model(model_cfg.name, model_cfg.num_classes)
    model = model.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Phase 1: Train head only
    print("\n" + "=" * 60)
    print("[Phase 1] Training Head Only")
    print("=" * 60)

    freeze_backbone(model)
    params = count_parameters(model)
    print(f"Trainable parameters: {params['trainable']:,} / {params['total']:,}")

    optimizer_p1 = create_optimizer(model, train_cfg, phase=1)

    for epoch in range(train_cfg.phase1_epochs):
        loss = train_one_epoch(
            model, train_loader, criterion, optimizer_p1, device,
            desc=f"Phase 1 Epoch {epoch + 1}/{train_cfg.phase1_epochs}"
        )
        print(f"Phase 1 - Epoch {epoch + 1}/{train_cfg.phase1_epochs}: Loss = {loss:.4f}")

    # Phase 2: Full fine-tuning
    print("\n" + "=" * 60)
    print("[Phase 2] Full Fine-tuning")
    print("=" * 60)

    unfreeze_all(model)
    params = count_parameters(model)
    print(f"Trainable parameters: {params['trainable']:,} / {params['total']:,}")

    optimizer_p2 = create_optimizer(model, train_cfg, phase=2)

    for epoch in range(train_cfg.phase2_epochs):
        loss = train_one_epoch(
            model, train_loader, criterion, optimizer_p2, device,
            desc=f"Phase 2 Epoch {epoch + 1}/{train_cfg.phase2_epochs}"
        )
        print(f"Phase 2 - Epoch {epoch + 1}/{train_cfg.phase2_epochs}: Loss = {loss:.4f}")

    # Save final model
    timestamp = datetime.now().strftime("%y%m%d%H%M")
    total_epochs = train_cfg.phase1_epochs + train_cfg.phase2_epochs

    # Save state dict
    state_dict_path = os.path.join(
        train_cfg.checkpoint_dir,
        f"{train_cfg.save_name}_{total_epochs}ep_{train_cfg.optimizer}_{timestamp}.pth"
    )
    torch.save(model.state_dict(), state_dict_path)
    print(f"\nModel state dict saved to: {state_dict_path}")

    # Optionally save full model
    full_model_path = os.path.join(
        train_cfg.checkpoint_dir,
        f"{train_cfg.save_name}_{total_epochs}ep_{train_cfg.optimizer}_{timestamp}_full.pt"
    )
    torch.save(model, full_model_path)
    print(f"Full model saved to: {full_model_path}")

    print("\nTraining complete!")
    return model


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train X3D on UCF-CRIME dataset")

    # Model
    parser.add_argument("--model", type=str, default="x3d_m",
                        choices=["x3d_xs", "x3d_s", "x3d_m", "x3d_l"],
                        help="X3D model variant")
    parser.add_argument("--num-classes", type=int, default=14,
                        help="Number of output classes")

    # Data
    parser.add_argument("--train-root", type=str,
                        default="/mnt/c/JJS/UCF_Crimes/Videos/train",
                        help="Path to training videos")
    parser.add_argument("--train-annotation", type=str,
                        default="/mnt/c/JJS/UCF_Crimes/Videos/train/00_timestamp",
                        help="Path to training annotations")

    # Training
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for training")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="Number of data loading workers")
    parser.add_argument("--phase1-epochs", type=int, default=2,
                        help="Number of epochs for phase 1 (head only)")
    parser.add_argument("--phase2-epochs", type=int, default=3,
                        help="Number of epochs for phase 2 (full fine-tuning)")
    parser.add_argument("--optimizer", type=str, default="sgd",
                        choices=["sgd", "adamw"],
                        help="Optimizer type")
    parser.add_argument("--phase1-lr", type=float, default=1e-3,
                        help="Learning rate for phase 1")
    parser.add_argument("--phase2-backbone-lr", type=float, default=1e-5,
                        help="Backbone learning rate for phase 2")
    parser.add_argument("--phase2-head-lr", type=float, default=1e-3,
                        help="Head learning rate for phase 2")

    # Output
    parser.add_argument("--checkpoint-dir", type=str, default=".",
                        help="Directory to save checkpoints")
    parser.add_argument("--save-name", type=str, default="x3d_ucfcrime",
                        help="Base name for saved models")

    # Misc
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Create configurations from args
    model_cfg = get_model_config(args.model, args.num_classes)

    data_cfg = DataConfig(
        train_root=args.train_root,
        train_annotation=args.train_annotation,
    )

    train_cfg = TrainConfig(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        phase1_epochs=args.phase1_epochs,
        phase2_epochs=args.phase2_epochs,
        phase1_lr=args.phase1_lr,
        phase2_backbone_lr=args.phase2_backbone_lr,
        phase2_head_lr=args.phase2_head_lr,
        optimizer=args.optimizer,
        checkpoint_dir=args.checkpoint_dir,
        save_name=args.save_name,
        seed=args.seed,
    )

    # Print configuration
    print("=" * 60)
    print("X3D Training Configuration")
    print("=" * 60)
    print(f"Model: {model_cfg.name}")
    print(f"Num classes: {model_cfg.num_classes}")
    print(f"Input: {model_cfg.num_frames} frames @ {model_cfg.crop_size}x{model_cfg.crop_size}")
    print(f"Batch size: {train_cfg.batch_size}")
    print(f"Optimizer: {train_cfg.optimizer}")
    print(f"Phase 1: {train_cfg.phase1_epochs} epochs @ lr={train_cfg.phase1_lr}")
    print(f"Phase 2: {train_cfg.phase2_epochs} epochs @ backbone_lr={train_cfg.phase2_backbone_lr}, head_lr={train_cfg.phase2_head_lr}")
    print("=" * 60)

    # Train
    train(model_cfg, data_cfg, train_cfg)


if __name__ == "__main__":
    main()
