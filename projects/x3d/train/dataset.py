"""
UCF-CRIME dataset loader for X3D fine-tuning.
"""

import os
import re
import collections
from typing import Optional, Dict, List, Callable

import pandas as pd
import torch
from torch.utils.data import Dataset
from pytorchvideo.data.encoded_video import EncodedVideo


def time_to_seconds(time_str) -> float:
    """
    Convert time string to seconds.

    Supports formats:
    - 'HH:MM:SS' (e.g., '0:00:06')
    - 'MM:SS' (e.g., '00:06')
    - Numeric string (e.g., '6.5')

    Args:
        time_str: Time string or numeric value.

    Returns:
        Time in seconds as float.
    """
    if pd.isna(time_str) or time_str == "":
        return 0.0

    # Extract digits and colons only (ignore trailing text)
    time_match = re.search(r"(\d+:?\d*:?\d*)", str(time_str))
    if not time_match:
        return 0.0

    clean_time_str = time_match.group(1)
    parts = str(clean_time_str).split(':')

    if len(parts) == 3:  # HH:MM:SS
        return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    elif len(parts) == 2:  # MM:SS
        return float(parts[0]) * 60 + float(parts[1])

    # Already numeric
    try:
        return float(time_str)
    except ValueError:
        return 0.0


class UCFCrimeDataset(Dataset):
    """
    UCF-CRIME dataset for video classification.

    Loads video clips based on timestamp annotations from CSV files.
    Each class has a corresponding CSV file with columns: file_name, start_time, end_time.

    Args:
        root_dir: Path to dataset root (e.g., 'train/' or 'test/').
        annotation_dir: Path to timestamp CSV files (e.g., 'train/00_timestamp/').
        transform: Optional transform to apply to video data.
        max_clip_duration: Maximum clip duration in seconds (clips longer than this are truncated).
        binary: If True, use binary classification (Normal=0 vs Abnormal=1).
        normal_class_name: Name of the normal class for binary classification.
        verbose: Whether to print loading progress and statistics.
    """

    def __init__(
        self,
        root_dir: str,
        annotation_dir: str,
        transform: Optional[Callable] = None,
        max_clip_duration: float = 4.0,
        binary: bool = False,
        normal_class_name: str = "Normal",
        verbose: bool = True
    ):
        self.root_dir = root_dir
        self.annotation_dir = annotation_dir
        self.transform = transform
        self.max_clip_duration = max_clip_duration
        self.binary = binary
        self.normal_class_name = normal_class_name
        self.verbose = verbose

        self.clip_metadata: List[Dict] = []
        self.class_to_idx: Dict[str, int] = {}
        self.idx_to_class: Dict[int, str] = {}

        # For binary classification, store original class info
        self.original_class_to_idx: Dict[str, int] = {}
        self.original_idx_to_class: Dict[int, str] = {}

        self._load_annotations()

    def _load_annotations(self):
        """Load annotations from CSV files."""
        # Get class directories (exclude timestamp folder)
        classes = sorted([
            d for d in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, d)) and d != '00_timestamp'
        ])

        # Store original class mapping
        self.original_class_to_idx = {cls: i for i, cls in enumerate(classes)}
        self.original_idx_to_class = {i: cls for cls, i in self.original_class_to_idx.items()}

        # Set up class mapping based on classification mode
        if self.binary:
            # Binary: Normal=0, Abnormal=1
            self.class_to_idx = {"Normal": 0, "Abnormal": 1}
            self.idx_to_class = {0: "Normal", 1: "Abnormal"}
            if self.verbose:
                print(f"Binary classification mode: Normal vs Abnormal")
        else:
            # Multi-class: use original mapping
            self.class_to_idx = self.original_class_to_idx.copy()
            self.idx_to_class = self.original_idx_to_class.copy()

        missing_videos = []

        for cls in classes:
            csv_path = os.path.join(self.annotation_dir, f"{cls}_timestamps.csv")

            if not os.path.exists(csv_path):
                if self.verbose:
                    print(f"CSV file not found: {csv_path}")
                continue

            try:
                df = pd.read_csv(csv_path, on_bad_lines='skip', engine='c')
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Could not read {csv_path}. Error: {e}")
                continue

            for _, row in df.iterrows():
                raw_file_name = str(row['file_name']).strip().split()[0]
                video_path = os.path.join(self.root_dir, cls, raw_file_name)

                if os.path.exists(video_path):
                    # Determine label based on classification mode
                    if self.binary:
                        # Normal=0, Abnormal=1
                        label = 0 if cls == self.normal_class_name else 1
                    else:
                        label = self.original_class_to_idx[cls]

                    self.clip_metadata.append({
                        'video_path': video_path,
                        'start_time': time_to_seconds(row['start_time']),
                        'end_time': time_to_seconds(row['end_time']),
                        'label': label,
                        'original_class': cls  # Keep original class for reference
                    })
                else:
                    missing_videos.append(video_path)

        if self.verbose:
            if missing_videos:
                print(f"\nWarning: {len(missing_videos)} videos not found")
            print(f"\nDataset initialized with {len(self.clip_metadata)} valid clips.")
            self.print_statistics()

    def __len__(self) -> int:
        return len(self.clip_metadata)

    def __getitem__(self, idx: int):
        item = self.clip_metadata[idx]

        try:
            video = EncodedVideo.from_path(item['video_path'])

            start = item['start_time']
            end = item['end_time']

            # Truncate long clips to prevent memory issues
            if end - start > self.max_clip_duration:
                # Extract from around the event midpoint
                start = max(0, start - 2)
                end = min(video.duration, end + 2)

            # Load video clip
            video_data = video.get_clip(start_sec=start, end_sec=end)

            if self.transform:
                video_data = self.transform(video_data)

            return video_data['video'], item['label']

        except Exception as e:
            print(f"Error loading video {item['video_path']}: {e}")
            raise e

    def print_statistics(self):
        """Print dataset class distribution."""
        counts = collections.Counter([item['label'] for item in self.clip_metadata])
        total = 0

        for idx in range(len(self.idx_to_class)):
            cls_name = self.idx_to_class[idx]
            count = counts.get(idx, 0)
            total += count
            print(f"[Class] {cls_name:<20}: {count:4} clips")

        print(f"Total clips: {total}")

    def get_class_weights(self) -> torch.Tensor:
        """
        Compute class weights inversely proportional to class frequency.
        Useful for handling class imbalance in loss functions.

        Returns:
            Tensor of class weights.
        """
        counts = collections.Counter([item['label'] for item in self.clip_metadata])
        total = len(self.clip_metadata)
        num_classes = len(self.class_to_idx)

        weights = []
        for idx in range(num_classes):
            count = counts.get(idx, 1)  # Avoid division by zero
            weights.append(total / (num_classes * count))

        return torch.tensor(weights, dtype=torch.float32)
