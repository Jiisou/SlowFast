#!/usr/bin/env python3
"""
Evaluation script for X3D model on UCF-CRIME dataset.
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm

from config import DataConfig, get_model_config, UCF_CRIME_CLASSES, UCF_CRIME_BINARY_CLASSES
from dataset import UCFCrimeDataset
from model import create_x3d
from utils import get_device, create_transform, plot_confusion_matrix, load_pretrained_model


def evaluate_model(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device
) -> dict:
    """
    Evaluate model on the given data loader.

    Args:
        model: Model to evaluate.
        data_loader: Data loader for evaluation.
        device: Device to run evaluation on.

    Returns:
        Dictionary with evaluation results (accuracy, auc, labels, predictions, probabilities).
    """
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    print("Starting evaluation...")
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Evaluating"):
            inputs = inputs.to(device)

            # Forward pass
            outputs = model(inputs)

            # Compute probabilities for AUC
            probs = F.softmax(outputs, dim=1)

            # Get predictions
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)

    try:
        num_classes = all_probs.shape[1]
        if num_classes == 2:
            # Binary classification: use probability of positive class (index 1)
            auc = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            # Multi-class: use one-vs-rest
            auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
    except ValueError as e:
        print(f"AUC calculation error (possibly missing classes in test set): {e}")
        auc = 0.0

    return {
        'accuracy': accuracy,
        'auc': auc,
        'labels': all_labels,
        'predictions': all_preds,
        'probabilities': all_probs,
    }


def load_model(
    checkpoint_path: str,
    model_name: str = "x3d_m",
    num_classes: int = 14,
    device: torch.device = None,
    use_custom_model: bool = True
) -> torch.nn.Module:
    """
    Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file (.pt or .pth).
        model_name: X3D model variant.
        num_classes: Number of output classes.
        device: Device to load model on.
        use_custom_model: If True, use create_x3d from model.py. If False, use torch hub.

    Returns:
        Loaded model.
    """
    if device is None:
        device = get_device()

    model_cfg = get_model_config(model_name, num_classes)

    if use_custom_model:
        # Use custom model definition from model.py
        model = create_x3d(
            input_clip_length=model_cfg.num_frames,
            input_crop_size=model_cfg.crop_size,
            model_num_class=num_classes,
            depth_factor=2.2,
        )
    else:
        # Use torch hub model with replaced head
        model = load_pretrained_model(model_name, num_classes)

    # Load checkpoint
    if checkpoint_path.endswith('.pt'):
        # Full model saved
        loaded = torch.load(checkpoint_path, map_location=device)
        if isinstance(loaded, dict) and 'model_state_dict' in loaded:
            model.load_state_dict(loaded['model_state_dict'], strict=False)
        elif isinstance(loaded, torch.nn.Module):
            model = loaded
        else:
            # Assume it's a state dict
            model.load_state_dict(loaded, strict=False)
    else:
        # State dict only
        state_dict = torch.load(checkpoint_path, map_location=device)
        result = model.load_state_dict(state_dict, strict=False)
        if result.missing_keys:
            print(f"Warning: Missing keys: {result.missing_keys[:5]}...")
        if result.unexpected_keys:
            print(f"Warning: Unexpected keys: {result.unexpected_keys[:5]}...")

    print(f"Loaded checkpoint from {checkpoint_path}")
    return model.to(device)


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate X3D on UCF-CRIME dataset")

    # Model
    parser.add_argument("--checkpoint", type=str,
                        default="./x3d_ucfcrime_mulclass_2+5ep.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--model", type=str, default="x3d_m",
                        choices=["x3d_xs", "x3d_s", "x3d_m", "x3d_l"],
                        help="X3D model variant")
    parser.add_argument("--num-classes", type=int, default=14,
                        help="Number of output classes (ignored if --binary is set)")
    parser.add_argument("--binary", action="store_true",
                        help="Use binary classification (Normal vs Abnormal)")
    parser.add_argument("--use-hub-model", action="store_true",
                        help="Use torch hub model instead of custom model.py")

    # Data
    parser.add_argument("--test-root", type=str,
                        default="/mnt/c/JJS/UCF_Crimes/Videos/test",
                        help="Path to test videos")
    parser.add_argument("--test-annotation", type=str,
                        default="/mnt/c/JJS/UCF_Crimes/Videos/test/00_timestamp",
                        help="Path to test annotations")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for evaluation")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="Number of data loading workers")

    # Output
    parser.add_argument("--output-dir", type=str, default="./results",
                        help="Directory to save results")
    parser.add_argument("--save-confusion-matrix", action="store_true", default=True,
                        help="Save confusion matrix plot")
    parser.add_argument("--normalize-cm", action="store_true",
                        help="Normalize confusion matrix by row")

    args = parser.parse_args()

    # Setup
    device = get_device()

    # Determine number of classes and class names
    num_classes = 2 if args.binary else args.num_classes
    class_names = UCF_CRIME_BINARY_CLASSES if args.binary else UCF_CRIME_CLASSES

    # Get model config
    model_cfg = get_model_config(args.model, num_classes)
    data_cfg = DataConfig(
        test_root=args.test_root,
        test_annotation=args.test_annotation,
    )

    # Create test transform
    test_transform = create_transform(model_cfg, data_cfg, is_train=False)

    # Create test dataset
    print("\nLoading test dataset...")
    print(f"Classification mode: {'Binary (Normal vs Abnormal)' if args.binary else 'Multi-class'}")
    test_dataset = UCFCrimeDataset(
        root_dir=args.test_root,
        annotation_dir=args.test_annotation,
        transform=test_transform,
        binary=args.binary,
        verbose=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        return

    model = load_model(
        args.checkpoint,
        model_name=args.model,
        num_classes=num_classes,
        device=device,
        use_custom_model=not args.use_hub_model,
    )

    # Evaluate
    results = evaluate_model(model, test_loader, device)

    # Print results
    print("\n" + "=" * 40)
    print("Evaluation Results")
    print("=" * 40)
    print(f"Classification: {'Binary' if args.binary else 'Multi-class'}")
    print(f"Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    if args.binary:
        # For binary classification, use probability of positive class (Abnormal)
        try:
            binary_auc = roc_auc_score(results['labels'], results['probabilities'][:, 1])
            print(f"AUC (Binary): {binary_auc:.4f}")
        except ValueError as e:
            print(f"AUC calculation error: {e}")
    else:
        print(f"AUC (One-vs-Rest): {results['auc']:.4f}")
    print("=" * 40)

    # Save confusion matrix
    if args.save_confusion_matrix:
        cm_suffix = "_binary" if args.binary else ""
        cm_path = os.path.join(args.output_dir, f"confusion_matrix{cm_suffix}.png")
        title = "Confusion Matrix (X3D Binary Classification)" if args.binary else "Confusion Matrix (X3D on UCF-Crime)"
        plot_confusion_matrix(
            results['labels'],
            results['predictions'],
            class_names=class_names,
            save_path=cm_path,
            normalize=args.normalize_cm,
            title=title,
        )

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
