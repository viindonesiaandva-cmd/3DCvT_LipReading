# -*- coding: utf-8 -*-
"""
LRW Dataset Preprocessing Script for 3DCvT Lip Reading Model.

Description:
    This script processes the raw LRW dataset (mp4 files) into a format suitable for high-speed training.
    It performs resizing, grayscale conversion, and serialization.

Pipeline:
    1. Scan directory for MP4 files.
    2. Read video frames (Opencv).
    3. Transform: Resize (256x256 -> 96x96), Grayscale.
    4. Save as Pickle/Numpy files to SSD.

Author: Jiafeng Wu (Reproducing 3DCvT)
Environment: Python 3.10, OpenCV, TQDM, Numpy
"""

import os
import cv2
import glob
import pickle
import argparse
import logging
import multiprocessing
import numpy as np
from pathlib import Path
from tqdm import tqdm
from functools import partial
from typing import List, Optional, Tuple

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("preprocess.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataConfig:
    """Configuration holder for data processing parameters."""
    TARGET_SIZE: Tuple[int, int] = (96, 96)  # Paper: adjust size to 96x96
    TO_GRAYSCALE: bool = True               # Paper: transformed into a grayscale image
    EXT: str = ".mp4"                       # Source extension


def load_video(path: str) -> Optional[np.ndarray]:
    """
    Loads a video file and converts it to a numpy array of frames.

    Args:
        path (str): Absolute path to the video file.

    Returns:
        np.ndarray: Video tensor of shape (T, H, W) if grayscale, else (T, H, W, C).
                    Returns None if reading fails.
    """
    cap = cv2.VideoCapture(path)
    frames = []
    
    if not cap.isOpened():
        logger.error(f"Failed to open video: {path}")
        return None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Preprocessing Step 1: Resize
        # LRW is 256x256, Paper reduces to 96x96
        frame = cv2.resize(frame, DataConfig.TARGET_SIZE, interpolation=cv2.INTER_AREA)

        # Preprocessing Step 2: Grayscale
        if DataConfig.TO_GRAYSCALE:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        logger.warning(f"Video contains no frames: {path}")
        return None

    # Stack to numpy array
    return np.array(frames) # Shape: (T, 96, 96)


def process_item(
    video_path: str, 
    source_root: str, 
    target_root: str
) -> bool:
    """
    Worker function to process a single video file.
    
    Args:
        video_path (str): Path to source mp4.
        source_root (str): Root dir of source dataset (to calculate relative path).
        target_root (str): Root dir of target storage.

    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        # 1. Calculate output path structure
        # source: /public/dataset/LRW/lipread_mp4/ABOUT/train/ABOUT_00001.mp4
        # target: /ssd2/3DcVT_data/data_LRW/ABOUT/train/ABOUT_00001.pkl
        path_obj = Path(video_path)
        
        # Get relative path parts: e.g., ('ABOUT', 'train', 'ABOUT_00001.mp4')
        # We assume source_root ends before the word category
        # Robust relative path finding:
        try:
            rel_path = path_obj.relative_to(source_root)
        except ValueError:
            # Fallback if path manipulation fails
            logger.error(f"Path nesting error: {video_path} not in {source_root}")
            return False

        save_dir = Path(target_root) / rel_path.parent
        save_filename = rel_path.stem + ".pkl"
        save_path = save_dir / save_filename

        # Skip if already exists
        if save_path.exists():
            return True

        # 2. Create directory if not exists
        save_dir.mkdir(parents=True, exist_ok=True)

        # 3. Load and Process Video
        video_tensor = load_video(str(path_obj))
        
        if video_tensor is None:
            return False

        # 4. Save Data (Pickle is fast for python objects)
        # We create a dictionary to store metadata if needed later (like label)
        data_packet = {
            'video': video_tensor,          # (T, 96, 96) uint8
            'label': rel_path.parts[0],     # The word (e.g., "ABOUT")
            'split': rel_path.parts[1],     # train/val/test
            'duration': video_tensor.shape[0]
        }

        with open(save_path, 'wb') as f:
            pickle.dump(data_packet, f)

        return True

    except Exception as e:
        logger.error(f"Exception processing {video_path}: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(description="LRW Preprocessing Pipeline")
    parser.add_argument('--source_dir', type=str, default='/public/dataset/LRW/lipread_mp4', 
                        help='Path to raw LRW dataset')
    parser.add_argument('--target_dir', type=str, default='/ssd2/3DcVT_data/data_LRW', 
                        help='Path to save processed data')
    parser.add_argument('--workers', type=int, default=16, 
                        help='Number of multiprocessing workers')
    
    args = parser.parse_args()

    # Verify Paths
    source_path = Path(args.source_dir)
    target_path = Path(args.target_dir)

    if not source_path.exists():
        raise FileNotFoundError(f"Source directory not found: {source_path}")
    
    logger.info(f"Scanning files in {source_path}...")
    
    # Fast glob scanning
    # Pattern: WORD/SPLIT/VIDEO.mp4
    video_files = list(source_path.glob(f"**/*{DataConfig.EXT}"))
    
    logger.info(f"Found {len(video_files)} video files.")
    logger.info(f"Target Output: {target_path}")
    logger.info(f"Preprocessing Settings: Resize={DataConfig.TARGET_SIZE}, Gray={DataConfig.TO_GRAYSCALE}")

    # Create partial function for multiprocessing
    worker_func = partial(
        process_item, 
        source_root=source_path, 
        target_root=target_path
    )

    # Multiprocessing Pool
    # Using 'imap_unordered' is usually faster and allows TQDM to update smoothly
    with multiprocessing.Pool(processes=args.workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(worker_func, [str(f) for f in video_files]), 
            total=len(video_files),
            desc="Preprocessing Videos",
            unit="vid"
        ))

    success_count = sum(results)
    logger.info(f"Processing Complete. Success: {success_count}/{len(video_files)}")

if __name__ == "__main__":
    main()