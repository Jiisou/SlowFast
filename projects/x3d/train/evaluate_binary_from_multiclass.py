#!/usr/bin/env python3
"""
Binary classification inference using a multiclass checkpoint.
Aggregates 14-class predictions into Normal (index 7) vs Abnormal (all others).
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
from utils import get_device, create_transform, plot_confusion_matrix

# Normal class index in multiclass (alphabetically sorted)
NORMAL_CLASS_INDEX = UCF_CRIME_CLASSES.index("Normal")  # = 7


def load_multiclass_model(
    checkpoint_path: str,
    model_name: str = "x3d_m",
    device: torch.device = None,
) -> torch.nn.Module:
    """Load multiclass model (14 classes) from checkpoint."""
    if device is None:
        device = get_device()

    model_cfg = get_model_config(model_name, num_classes=14)

    model = create_x3d(
        input_clip_length=model_cfg.num_frames,
        input_crop_size=model_cfg.crop_size,
        model_num_class=14,
        depth_factor=2.2,
    )

    # Load checkpoint
    state_dict = torch.load(checkpoint_path, map_location=device)
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']

    model.load_state_dict(state_dict, strict=False)
    print(f"Loaded multiclass model from {checkpoint_path}")

    return model.to(device)


def multiclass_to_binary_probs(probs: torch.Tensor) -> tuple:
    """
    Convert multiclass probabilities to binary probabilities.

    Args:
        probs: Multiclass probabilities, shape (batch, 14)

    Returns:
        (prob_normal, prob_abnormal) each of shape (batch,)
    """
    prob_normal = probs[:, NORMAL_CLASS_INDEX]
    prob_abnormal = probs[:, :NORMAL_CLASS_INDEX].sum(dim=1) + probs[:, NORMAL_CLASS_INDEX+1:].sum(dim=1)
    return prob_normal, prob_abnormal


def evaluate(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device
) -> dict:
    """Evaluate multiclass model with binary aggregation."""
    model.eval()

    all_labels = []
    all_preds = []
    all_prob_abnormal = []
    all_top_abnormal_class = []

    print("Starting evaluation...")
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Evaluating"):
            inputs = inputs.to(device)

            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)

            # Convert to binary
            prob_normal, prob_abnormal = multiclass_to_binary_probs(probs)
            binary_preds = (prob_abnormal > prob_normal).long()

            # Find top abnormal class (excluding Normal at index 7)
            abnormal_probs = probs.clone()
            abnormal_probs[:, NORMAL_CLASS_INDEX] = -1  # Mask out Normal
            top_abnormal_idx = abnormal_probs.argmax(dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(binary_preds.cpu().numpy())
            all_prob_abnormal.extend(prob_abnormal.cpu().numpy())
            all_top_abnormal_class.extend(top_abnormal_idx.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_prob_abnormal = np.array(all_prob_abnormal)
    all_top_abnormal_class = np.array(all_top_abnormal_class)

    # Metrics
    accuracy = accuracy_score(all_labels, all_preds)

    try:
        auc = roc_auc_score(all_labels, all_prob_abnormal)
    except ValueError as e:
        print(f"AUC calculation error: {e}")
        auc = 0.0

    return {
        'accuracy': accuracy,
        'auc': auc,
        'labels': all_labels,
        'predictions': all_preds,
        'prob_abnormal': all_prob_abnormal,
        'top_abnormal_class': all_top_abnormal_class,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Binary inference using multiclass checkpoint"
    )

    parser.add_argument("--checkpoint", type=str,
                        default="../ckpts/x3d_ucfcrime_mulclass_2+3ep_SGD_26011512.pth",
                        help="Path to multiclass checkpoint")
    parser.add_argument("--model", type=str, default="x3d_m",
                        choices=["x3d_xs", "x3d_s", "x3d_m", "x3d_l"])
    parser.add_argument("--test-root", type=str,
                        default="/mnt/c/JJS/UCF_Crimes/Videos/test")
    parser.add_argument("--test-annotation", type=str,
                        default="/mnt/c/JJS/UCF_Crimes/Videos/test/00_timestamp")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="./results")
    parser.add_argument("--normalize-cm", action="store_true",
                        help="Normalize confusion matrix")

    args = parser.parse_args()

    device = get_device()

    # Load multiclass model
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        return

    model = load_multiclass_model(args.checkpoint, args.model, device)

    # Create test dataset (binary mode for labels)
    model_cfg = get_model_config(args.model, num_classes=14)
    data_cfg = DataConfig(
        test_root=args.test_root,
        test_annotation=args.test_annotation,
    )

    test_transform = create_transform(model_cfg, data_cfg, is_train=False)

    print("\nLoading test dataset (binary labels)...")
    test_dataset = UCFCrimeDataset(
        root_dir=args.test_root,
        annotation_dir=args.test_annotation,
        transform=test_transform,
        binary=True,  # Labels as binary (Normal=0, Abnormal=1)
        verbose=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # Evaluate
    results = evaluate(model, test_loader, device)

    # Print results
    print("\n" + "=" * 50)
    print("Binary Inference from Multiclass Model")
    print("=" * 50)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Normal class index: {NORMAL_CLASS_INDEX} ({UCF_CRIME_CLASSES[NORMAL_CLASS_INDEX]})")
    print("-" * 50)
    print(f"Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"AUC: {results['auc']:.4f}")
    print("=" * 50)

    # Confusion matrix
    os.makedirs(args.output_dir, exist_ok=True)
    cm_path = os.path.join(args.output_dir, "confusion_matrix_binary_from_multiclass.png")
    plot_confusion_matrix(
        results['labels'],
        results['predictions'],
        class_names=UCF_CRIME_BINARY_CLASSES,
        save_path=cm_path,
        normalize=args.normalize_cm,
        title="Binary Classification (from Multiclass Model)",
    )

    # Show top predicted abnormal classes distribution
    print("\nTop predicted abnormal class distribution:")
    from collections import Counter
    top_class_counts = Counter(results['top_abnormal_class'])
    for idx, count in sorted(top_class_counts.items(), key=lambda x: -x[1]):
        print(f"  {UCF_CRIME_CLASSES[idx]:<15}: {count}")

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
