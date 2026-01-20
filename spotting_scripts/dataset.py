"""
UCF-CRIME Spotting Dataset for sliding window anomaly detection.

Key Implementation Details:
- Inference Unit: 1 second of video (30 frames at 30fps)
- Frame Sampling: Extract 16 frames from 30-frame window
  - Even indices: [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28]
  - Plus final frame: [29]
  - Total: 16 frames
- Labeling: If 1-second window overlaps with any annotated event → label=1
- Training Mode: stride=8 frames (0.5s, 50% overlap)
- Inference Mode: stride=16 frames (1.0s, non-overlapping)
"""

import os
import re
from typing import Optional, Callable, List, Dict, Tuple

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import decord
    from decord import VideoReader, cpu
    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False

try:
    import av
    AV_AVAILABLE = True
except ImportError:
    AV_AVAILABLE = False

from config import (
    SpottingDataConfig,
    CLIP_FRAMES,
    UNIT_FRAMES,
    SAMPLE_INDICES,
    get_frame_indices,
)


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
    time_match = re.search(r"(\d+:?\d*:?\d*\.?\d*)", str(time_str))
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


def load_video_decord(video_path: str) -> Tuple[np.ndarray, float, int]:
    """
    Load video using decord library.

    Args:
        video_path: Path to video file.

    Returns:
        Tuple of (frames as numpy array [N, H, W, C], fps, total_frames).
    """
    vr = VideoReader(video_path, ctx=cpu(0))
    fps = vr.get_avg_fps()
    total_frames = len(vr)
    return vr, fps, total_frames


def load_video_pyav(video_path: str) -> Tuple[List[np.ndarray], float, int]:
    """
    Load video using PyAV library.

    Args:
        video_path: Path to video file.

    Returns:
        Tuple of (list of frames, fps, total_frames).
    """
    container = av.open(video_path)
    stream = container.streams.video[0]
    fps = float(stream.average_rate)
    total_frames = stream.frames

    frames = []
    for frame in container.decode(video=0):
        img = frame.to_ndarray(format='rgb24')
        frames.append(img)

    container.close()
    return frames, fps, total_frames


class UCFCrimeSpottingDataset(Dataset):
    """
    UCF-CRIME Spotting Dataset for sliding window anomaly detection.

    This dataset extracts 1-second video units from untrimmed clips using a
    sliding window approach. Each unit is labeled as Normal (0) or Abnormal (1)
    based on overlap with annotated anomaly events.

    Args:
        video_dir: Path to video directory containing class subdirectories.
        annotation_csv: Path to CSV file or directory with annotations.
            CSV format: file_name, start_time, end_time (for each anomaly event).
        transform: Optional transform to apply to video tensors.
        mode: 'train' (stride=8) or 'inference' (stride=16).
        fps: Video frame rate (default: 30).
        unit_duration: Duration of each unit in seconds (default: 1.0).
        verbose: Whether to print loading progress.
    """

    def __init__(
        self,
        video_dir: str,
        annotation_csv: str,
        transform: Optional[Callable] = None,
        mode: str = 'train',
        fps: int = 30,
        unit_duration: float = 1.0,
        verbose: bool = True,
    ):
        self.video_dir = video_dir
        self.annotation_csv = annotation_csv
        self.transform = transform
        self.mode = mode
        self.fps = fps
        self.unit_duration = unit_duration
        self.verbose = verbose

        # Stride in frames based on mode
        self.stride = 8 if mode == 'train' else 16
        self.window_size = int(fps * unit_duration)  # 30 frames for 1 second

        # Validate video loading library
        if not DECORD_AVAILABLE and not AV_AVAILABLE:
            raise ImportError("Please install decord or av: pip install decord or pip install av")

        # Load annotations and build sample index
        self.annotations: Dict[str, List[Tuple[float, float]]] = {}
        self.samples: List[Dict] = []

        self._load_annotations()
        self._build_samples()

        if self.verbose:
            self._print_statistics()

    def _load_annotations(self):
        """Load annotations from CSV files."""
        if os.path.isdir(self.annotation_csv):
            # Directory of CSV files (one per class)
            self._load_annotations_from_dir(self.annotation_csv)
        else:
            # Single CSV file
            self._load_annotations_from_file(self.annotation_csv)

    def _load_annotations_from_dir(self, annotation_dir: str):
        """Load annotations from directory of CSV files."""
        for csv_file in os.listdir(annotation_dir):
            if not csv_file.endswith('.csv'):
                continue

            csv_path = os.path.join(annotation_dir, csv_file)
            try:
                df = pd.read_csv(csv_path, on_bad_lines='skip')
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Could not read {csv_path}: {e}")
                continue

            self._parse_annotation_df(df)

    def _load_annotations_from_file(self, csv_path: str):
        """Load annotations from single CSV file."""
        try:
            df = pd.read_csv(csv_path, on_bad_lines='skip')
            self._parse_annotation_df(df)
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not read {csv_path}: {e}")

    def _parse_annotation_df(self, df: pd.DataFrame):
        """Parse annotation DataFrame and populate self.annotations."""
        for _, row in df.iterrows():
            file_name = str(row['file_name']).strip().split()[0]
            start_time = time_to_seconds(row.get('start_time', 0))
            end_time = time_to_seconds(row.get('end_time', 0))

            if file_name not in self.annotations:
                self.annotations[file_name] = []

            # Only add valid time ranges
            if end_time > start_time:
                self.annotations[file_name].append((start_time, end_time))

    def _build_samples(self):
        """Build sample index by scanning videos and creating sliding windows."""
        # Find all video files
        video_files = []
        for root, dirs, files in os.walk(self.video_dir):
            # Skip annotation directories
            if '00_timestamp' in root:
                continue
            for file in files:
                if file.endswith(('.mp4', '.avi', '.mkv', '.mov')):
                    video_files.append(os.path.join(root, file))

        if self.verbose:
            print(f"Found {len(video_files)} video files")

        # Process each video
        for video_path in video_files:
            self._process_video(video_path)

    def _process_video(self, video_path: str):
        """Process a single video and add samples."""
        video_name = os.path.basename(video_path)

        # Determine if this video contains anomalies
        # Check both full path components and just filename
        events = []
        for key in self.annotations:
            if key in video_path or key == video_name:
                events.extend(self.annotations[key])

        # Get video info
        try:
            if DECORD_AVAILABLE:
                vr = VideoReader(video_path, ctx=cpu(0))
                total_frames = len(vr)
                video_fps = vr.get_avg_fps()
            else:
                container = av.open(video_path)
                stream = container.streams.video[0]
                total_frames = stream.frames
                video_fps = float(stream.average_rate)
                container.close()

            # Use detected fps or fall back to config
            if video_fps <= 0:
                video_fps = self.fps

        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not read video {video_path}: {e}")
            return

        # Determine parent directory (class name)
        parent_dir = os.path.basename(os.path.dirname(video_path))
        is_normal_class = parent_dir.lower() == 'normal'

        # Create sliding window samples
        window_size = self.window_size
        stride = self.stride

        # Number of complete windows
        num_windows = max(0, (total_frames - window_size) // stride + 1)

        for i in range(num_windows):
            start_frame = i * stride
            end_frame = start_frame + window_size

            # Convert to time
            start_time = start_frame / video_fps
            end_time = end_frame / video_fps

            # Determine label based on overlap with events
            if is_normal_class:
                label = 0
            elif not events:
                # No annotations for this abnormal video - skip or use default
                # For safety, we'll label as abnormal if in abnormal class
                label = 1
            else:
                # Check overlap with any event
                label = 0
                for event_start, event_end in events:
                    # Window overlaps with event if they intersect
                    if start_time < event_end and end_time > event_start:
                        label = 1
                        break

            self.samples.append({
                'video_path': video_path,
                'start_frame': start_frame,
                'end_frame': end_frame,
                'start_time': start_time,
                'end_time': end_time,
                'label': label,
                'video_fps': video_fps,
            })

    def _print_statistics(self):
        """Print dataset statistics."""
        total = len(self.samples)
        normal = sum(1 for s in self.samples if s['label'] == 0)
        abnormal = total - normal

        print(f"\n{'='*50}")
        print(f"UCF-CRIME Spotting Dataset ({self.mode} mode)")
        print(f"{'='*50}")
        print(f"Total samples: {total}")
        print(f"  Normal:   {normal} ({100*normal/total:.1f}%)")
        print(f"  Abnormal: {abnormal} ({100*abnormal/total:.1f}%)")
        print(f"Window size: {self.window_size} frames ({self.unit_duration}s)")
        print(f"Stride: {self.stride} frames")
        print(f"{'='*50}\n")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (video_tensor [C, T, H, W], label).
        """
        # Try to load the sample, with fallback to other samples on error
        max_retries = 10
        original_idx = idx

        for retry in range(max_retries):
            try:
                sample = self.samples[idx]

                # Extract frames
                frames = self._extract_frames(
                    sample['video_path'],
                    sample['start_frame'],
                    sample['end_frame'],
                )

                # Convert to tensor (T, H, W, C)
                video_tensor = torch.from_numpy(np.stack(frames)).float()

                # Apply transform
                if self.transform is not None:
                    video_tensor = self.transform(video_tensor)

                return video_tensor, sample['label']

            except Exception as e:
                if retry == 0:
                    print(f"Warning: Error loading sample {idx} "
                          f"({sample['video_path']}): {type(e).__name__}")
                # Try a different random sample
                idx = np.random.randint(0, len(self.samples))

        # If all retries fail, raise the error
        raise RuntimeError(f"Failed to load sample after {max_retries} retries. "
                          f"Original idx: {original_idx}")

    def _extract_frames(
        self,
        video_path: str,
        start_frame: int,
        end_frame: int
    ) -> List[np.ndarray]:
        """
        Extract and sample frames from video.

        Args:
            video_path: Path to video file.
            start_frame: Start frame index.
            end_frame: End frame index.

        Returns:
            List of 16 sampled frames as numpy arrays.
        """
        window_size = end_frame - start_frame
        frame_indices = get_frame_indices(window_size)

        # Adjust indices to absolute positions
        absolute_indices = [start_frame + i for i in frame_indices]

        if DECORD_AVAILABLE:
            # Decord is more robust for corrupted videos
            vr = VideoReader(video_path, ctx=cpu(0))
            # Clamp indices to valid range
            max_idx = len(vr) - 1
            absolute_indices = [min(max(0, i), max_idx) for i in absolute_indices]
            frames = vr.get_batch(absolute_indices).asnumpy()
            frames = list(frames)
        else:
            # PyAV fallback - decode only needed frames for efficiency
            container = av.open(video_path)
            stream = container.streams.video[0]

            # Get total frames
            total_frames = stream.frames
            if total_frames <= 0:
                # Estimate from duration
                total_frames = int(stream.duration * stream.average_rate / stream.time_base)

            max_idx = max(0, total_frames - 1)
            absolute_indices = [min(max(0, i), max_idx) for i in absolute_indices]

            # Decode frames - collect all and select
            all_frames = []
            max_needed = max(absolute_indices) + 1

            for i, frame in enumerate(container.decode(video=0)):
                if i >= max_needed:
                    break
                all_frames.append(frame.to_ndarray(format='rgb24'))

            container.close()

            # Handle case where video has fewer frames than expected
            if len(all_frames) == 0:
                raise ValueError(f"No frames decoded from {video_path}")

            max_idx = len(all_frames) - 1
            absolute_indices = [min(i, max_idx) for i in absolute_indices]
            frames = [all_frames[i] for i in absolute_indices]

        return frames

    def get_sample_info(self, idx: int) -> Dict:
        """Get metadata for a sample without loading video."""
        return self.samples[idx].copy()

    def get_video_samples(self, video_path: str) -> List[int]:
        """Get all sample indices for a specific video."""
        return [i for i, s in enumerate(self.samples) if s['video_path'] == video_path]

    def get_class_weights(self) -> torch.Tensor:
        """Compute class weights for handling imbalance."""
        labels = [s['label'] for s in self.samples]
        from utils import compute_class_weights
        return compute_class_weights(labels, num_classes=2)


class UCFCrimeSpottingInferenceDataset(Dataset):
    """
    Simplified dataset for inference on a single video.

    Extracts all 1-second windows from a video with non-overlapping stride.
    """

    def __init__(
        self,
        video_path: str,
        transform: Optional[Callable] = None,
        fps: int = 30,
        unit_duration: float = 1.0,
    ):
        self.video_path = video_path
        self.transform = transform
        self.fps = fps
        self.unit_duration = unit_duration

        self.window_size = int(fps * unit_duration)
        self.stride = CLIP_FRAMES  # Non-overlapping

        # Get video info
        if DECORD_AVAILABLE:
            vr = VideoReader(video_path, ctx=cpu(0))
            self.total_frames = len(vr)
            self.video_fps = vr.get_avg_fps()
        else:
            container = av.open(video_path)
            stream = container.streams.video[0]
            self.total_frames = stream.frames
            self.video_fps = float(stream.average_rate)
            container.close()

        if self.video_fps <= 0:
            self.video_fps = fps

        # Calculate number of windows
        self.num_windows = max(0, (self.total_frames - self.window_size) // self.stride + 1)

    def __len__(self) -> int:
        return self.num_windows

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        """
        Get a window.

        Args:
            idx: Window index.

        Returns:
            Tuple of (video_tensor [C, T, H, W], metadata dict).
        """
        start_frame = idx * self.stride
        end_frame = start_frame + self.window_size

        # Extract frames
        window_size = end_frame - start_frame
        frame_indices = get_frame_indices(window_size)
        absolute_indices = [start_frame + i for i in frame_indices]

        if DECORD_AVAILABLE:
            vr = VideoReader(self.video_path, ctx=cpu(0))
            max_idx = len(vr) - 1
            absolute_indices = [min(i, max_idx) for i in absolute_indices]
            frames = vr.get_batch(absolute_indices).asnumpy()
        else:
            container = av.open(self.video_path)
            all_frames = []
            for frame in container.decode(video=0):
                all_frames.append(frame.to_ndarray(format='rgb24'))
            container.close()
            max_idx = len(all_frames) - 1
            absolute_indices = [min(i, max_idx) for i in absolute_indices]
            frames = np.stack([all_frames[i] for i in absolute_indices])

        # Convert to tensor
        video_tensor = torch.from_numpy(frames).float()

        if self.transform is not None:
            video_tensor = self.transform(video_tensor)

        metadata = {
            'start_frame': start_frame,
            'end_frame': end_frame,
            'start_time': start_frame / self.video_fps,
            'end_time': end_frame / self.video_fps,
        }

        return video_tensor, metadata
