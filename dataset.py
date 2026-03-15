# -*- coding: utf-8 -*-
"""
LRW Dataset Loader and Augmentation Module.

Description:
    Implements the PyTorch Dataset for LRW (Lip Reading in the Wild).
    Includes specific preprocessing steps mentioned in the 3DCvT paper:
    - Random/Center Crop to 88x88.
    - Horizontal Flip (p=0.5).
    - Mixup algorithm for training batches.

Paper Reference:
    "A Lip Reading Method Based on 3D Convolutional Vision Transformer"
    - Section B.1: Data Preprocessing (Crop 88x88, Flip 0.5, Normalize [0,1])
    - Section IV.A.1: LRW Dataset details.

Author: Jiafeng Wu (Reproducing 3DCvT)
Environment: Python 3.10, PyTorch 2.x
"""

import os
import pickle
import torch
import random
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from torch.utils.data import Dataset

# Setup Logging
logger = logging.getLogger(__name__)

class VideoAugmentor:
    """
    Handles video-level transformations (spatio-temporal).
    Operates on Numpy arrays (T, H, W).
    """
    
    def __init__(self, mode: str = 'train', crop_size: int = 88):
        """
        Args:
            mode (str): 'train', 'val', or 'test'.
            crop_size (int): Target spatial dimension (Paper uses 88).
        """
        self.mode = mode
        self.crop_size = crop_size

    def __call__(self, frames: np.ndarray) -> np.ndarray:
        """
        Apply transformations.
        
        Args:
            frames (np.ndarray): Shape (T, H, W), uint8 or float.
        
        Returns:
            np.ndarray: Transformed frames after mean/std normalization.
        """
        # 1. Spatial Cropping
        frames = self._crop(frames)
        
        # 2. Horizontal Flip (Train only)
        # Paper: "each video frame is flipped at a probability level of 0.5"
        if self.mode == 'train' and random.random() < 0.5:
            frames = np.flip(frames, axis=2).copy()  # .copy() ensures contiguous memory layout for torch.from_numpy

        # 3. Mean/Std Normalization
        frames = frames.astype(np.float32) / 255.0
        frames = (frames - 0.421) / 0.165
        
        return frames
 
    def _crop(self, frames: np.ndarray) -> np.ndarray:
        """Performs Random Crop (Train) or Center Crop (Val/Test)."""
        t, h, w = frames.shape
        th, tw = self.crop_size, self.crop_size
        
        # Safety check
        if w < tw or h < th:
            raise ValueError(f"Frame size ({h}x{w}) smaller than crop size ({th}x{tw})")

        if self.mode == 'train':
            # Random Crop
            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)
        else:
            # Center Crop
            x1 = int(round((w - tw) / 2.))
            y1 = int(round((h - th) / 2.))

        return frames[:, y1:y1+th, x1:x1+tw]


class LRWDataset(Dataset):
    """
    PyTorch Dataset for LRW.
    Assumes data is preprocessed into .pkl files with structure:
    {'video': (T, 96, 96), 'label': str, ...}
    """

    def __init__(self, 
                 root_dir: str, 
                 mode: str = 'train', 
                 max_len: int = 29):
        """
        Args:
            root_dir (str): Path to preprocessed data (e.g., /ssd2/3DcVT_data/data_LRW).
            mode (str): 'train', 'val', or 'test'.
            max_len (int): Maximum frame length (LRW usually has 29 frames).
        """
        self.root_dir = Path(root_dir)
        self.mode = mode
        self.max_len = max_len
        self.augmentor = VideoAugmentor(mode=mode, crop_size=88)

        # 1. Build Vocabulary (Class Label Map)
        # We scan the root dir for folder names to ensure consistent ID mapping.
        # Sorting is CRITICAL to guarantee index 0 is always the same word.
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        if len(self.classes) == 0:
            raise RuntimeError(f"No class directories found in {root_dir}")
        
        logger.info(f"Initialized LRWDataset [{mode}]. Found {len(self.classes)} classes.")

        # 2. Collect Samples
        self.samples = self._load_sample_list()
        logger.info(f"Loaded {len(self.samples)} samples for split '{mode}'.")

    def _load_sample_list(self) -> List[Path]:
        """Scans directory for .pkl files belonging to the current mode."""
        # Pattern: root/WORD/mode/*.pkl
        # This might take a few seconds on first load
        samples = []
        for cls_name in self.classes:
            cls_dir = self.root_dir / cls_name / self.mode
            if cls_dir.exists():
                # Use glob to find all pkl files
                files = list(cls_dir.glob("*.pkl"))
                samples.extend(files)
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            dict: {
                'video': (C, T, H, W) Tensor, float32,
                'label': int
            }
        """
        pkl_path = self.samples[idx]
        
        # 1. Load Data
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        frames = data['video']  # Shape (T, 96, 96)
        word_label = data['label']

        # 2. Data Augmentation (Crop, Flip, Normalize)
        frames = self.augmentor(frames) # Returns (T, 88, 88) float32

        # 3. Convert to Tensor and Adjust Dimensions
        # PyTorch 3D Conv expects (C, T, H, W)
        # Input frames are (T, H, W), so we add Channel dim at 0.
        video_tensor = torch.from_numpy(frames).unsqueeze(0) # (1, T, 88, 88)
        
        # 4. Handle Temporal Length (Padding/Truncating if necessary)
        # LRW is mostly fixed at 29, but good to be safe.
        C, T, H, W = video_tensor.shape
        if T > self.max_len:
            video_tensor = video_tensor[:, :self.max_len, :, :]
        elif T < self.max_len:
            # Pad with zeros in time dimension
            pad_len = self.max_len - T
            padding = torch.zeros((C, pad_len, H, W))
            video_tensor = torch.cat((video_tensor, padding), dim=1)

        label_idx = self.class_to_idx[word_label]

        # LRW: all 29 frames are the target word, so boundary mask is all-ones
        boundary_mask = torch.ones(self.max_len, 1)

        return {
            'video': video_tensor, # (1, 29, 88, 88)
            'label': torch.tensor(label_idx, dtype=torch.long),
            'boundary': boundary_mask,  # (29, 1)
        }

class Mixup:
    """
    Implements Mixup augmentation logic.
    Paper Ref [41]: "Mixup: Beyond empirical risk minimization".
    
    Usage:
        Should be called inside the Training Loop or Collate function.
        mixup = Mixup(alpha=0.4)
        inputs, targets_a, targets_b, lam = mixup(inputs, targets)
        loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
    """
    def __init__(self, alpha: float = 0.4):
        """
        Args:
            alpha (float): Mixup alpha parameter. 
                           Paper doesn't specify exact alpha, 0.2 or 0.4 are standard.
                           A beta distribution B(alpha, alpha) is used.
        """
        self.alpha = alpha

    def __call__(self, batch_x: torch.Tensor, batch_y: torch.Tensor, batch_boundary: torch.Tensor = None):
        """
        Args:
            batch_x: Input images/videos (B, C, T, H, W)
            batch_y: Input labels (B)
            batch_boundary: Boundary masks (B, T, 1), optional
            
        Returns:
            mixed_x: Mixed inputs
            target_a: Original labels
            target_b: Shuffled labels
            lam: Lambda mixing coefficient
            mixed_boundary: Mixed boundary masks (if provided)
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1

        batch_size = batch_x.size(0)
        index = torch.randperm(batch_size).to(batch_x.device)

        mixed_x = lam * batch_x + (1 - lam) * batch_x[index, :]
        target_a, target_b = batch_y, batch_y[index]
        
        # Mix boundary masks the same way as inputs
        if batch_boundary is not None:
            mixed_boundary = lam * batch_boundary + (1 - lam) * batch_boundary[index]
            return mixed_x, target_a, target_b, lam, mixed_boundary
        
        return mixed_x, target_a, target_b, lam, None

class LRW1000Dataset(Dataset):
    """
    PyTorch Dataset for LRW-1000 (1000-word Chinese lip reading).
    
    Assumes data is preprocessed by preprocess_lrw1000.py into .pkl files
    with structure: {'video': (T, 96, 96), 'label': str, ...}
    
    Directory layout:
        root_dir/
        ├── vocab.json          # Sorted vocabulary (1000 words)
        └── <chinese_word>/
            └── <split>/
                └── *.pkl
    """

    def __init__(self, 
                 root_dir: str, 
                 mode: str = 'train', 
                 max_len: int = 29):
        """
        Args:
            root_dir (str): Path to preprocessed data (e.g., /ssd2/3DCvT_data/data_LRW1000).
            mode (str): 'train', 'val', or 'test'.
            max_len (int): Maximum frame length (default 29, same as LRW).
        """
        self.root_dir = Path(root_dir)
        self.mode = mode
        self.max_len = max_len
        self.augmentor = VideoAugmentor(mode=mode, crop_size=88)

        # 1. Load Vocabulary
        vocab_path = self.root_dir / 'vocab.json'
        if vocab_path.exists():
            import json
            with open(vocab_path, 'r', encoding='utf-8') as f:
                self.classes = json.load(f)
        else:
            # Fallback: scan directory names (same logic as LRW)
            self.classes = sorted([d.name for d in self.root_dir.iterdir() 
                                   if d.is_dir()])
        
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        if len(self.classes) == 0:
            raise RuntimeError(f"No classes found in {root_dir}")
        
        logger.info(f"Initialized LRW1000Dataset [{mode}]. Found {len(self.classes)} classes.")

        # 2. Collect Samples
        self.samples = self._load_sample_list()
        logger.info(f"Loaded {len(self.samples)} samples for split '{mode}'.")

    def _load_sample_list(self) -> List[Path]:
        """Scans directory for .pkl files belonging to the current mode."""
        samples = []
        for cls_name in self.classes:
            cls_dir = self.root_dir / cls_name / self.mode
            if cls_dir.exists():
                files = list(cls_dir.glob("*.pkl"))
                samples.extend(files)
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            dict: {
                'video': (C, T, H, W) Tensor, float32,
                'label': int
            }
        """
        pkl_path = self.samples[idx]
        
        # 1. Load Data
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        frames = data['video']  # Shape (T, 96, 96)
        word_label = data['label']

        # 2. Data Augmentation (Crop, Flip, Normalize)
        frames = self.augmentor(frames)  # Returns (T, 88, 88) float32

        # 3. Convert to Tensor and Adjust Dimensions
        # PyTorch 3D Conv expects (C, T, H, W)
        video_tensor = torch.from_numpy(frames).unsqueeze(0)  # (1, T, 88, 88)
        
        # 4. Handle Temporal Length (Padding/Truncating)
        C, T, H, W = video_tensor.shape
        if T > self.max_len:
            video_tensor = video_tensor[:, :self.max_len, :, :]
        elif T < self.max_len:
            pad_len = self.max_len - T
            padding = torch.zeros((C, pad_len, H, W))
            video_tensor = torch.cat((video_tensor, padding), dim=1)

        label_idx = self.class_to_idx[word_label]

        # Word Boundary Mask: identify which frames contain the target word
        # Parse start/end times from filename to compute boundary position
        boundary_mask = self._build_boundary_mask(pkl_path, data)

        return {
            'video': video_tensor,  # (1, max_len, 88, 88)
            'label': torch.tensor(label_idx, dtype=torch.long),
            'boundary': boundary_mask,  # (max_len, 1)
        }

    def _build_boundary_mask(self, pkl_path: Path, data: dict) -> torch.Tensor:
        """
        Build a binary boundary mask indicating which frames contain the target word.
        
        Uses boundary info from pkl if available (from updated preprocessing),
        otherwise approximates from the filename timestamps.
        """
        mask = torch.zeros(self.max_len, 1)
        
        # Method 1: Use pre-computed boundary from pkl (preferred)
        if 'boundary' in data:
            s, e = data['boundary']
            s = max(0, min(int(s), self.max_len - 1))
            e = max(s, min(int(e), self.max_len - 1))
            mask[s:e+1] = 1.0
            return mask
        
        # Method 2: Approximate from filename timestamps
        # Filename format: {hash_id}_{start}_{end}.pkl where start/end use 'p' for '.'
        try:
            stem = pkl_path.stem  # e.g., "abc123_15p78_16p16"
            parts = stem.rsplit('_', 2)
            if len(parts) >= 3:
                start_time = float(parts[-2].replace('p', '.'))
                end_time = float(parts[-1].replace('p', '.'))
                word_dur_frames = max(1, round((end_time - start_time) * 25))
                # Word was centered in the window during preprocessing
                center = self.max_len / 2.0
                half = word_dur_frames / 2.0
                s = max(0, int(round(center - half)))
                e = min(self.max_len, int(round(center + half)))
                mask[s:e] = 1.0
                return mask
        except (ValueError, IndexError):
            pass
        
        # Fallback: assume all frames are valid
        mask[:] = 1.0
        return mask

