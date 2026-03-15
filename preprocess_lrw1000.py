# -*- coding: utf-8 -*-
"""
LRW-1000 Dataset Preprocessing Script for 3DCvT Lip Reading Model.

Description:
    Processes the raw LRW-1000 dataset (lip images in JPEG format) into
    a format suitable for training, consistent with the LRW preprocessing pipeline.

    LRW-1000 Dataset Structure:
        /public/dataset/LRW_1000_Full_Version/
        ├── info/
        │   ├── trn_1000.txt    (603194 samples)
        │   ├── val_1000.txt    (63238 samples)
        │   └── tst_1000.txt    (51589 samples)
        └── lip_images/lip_images/
            └── <md5_hash>/
                ├── 1.jpg
                ├── 2.jpg
                └── ...N.jpg

    Annotation Format (CSV per line):
        <hash_id>,<chinese_word>,<pinyin>,<start_time>,<end_time>
        e.g.: 57f7d1d11ca308f3f24728cb5c930574,扫描,sao miao,15.78,16.16

Pipeline:
    1. Parse annotation files (trn/val/tst_1000.txt).
    2. Build vocabulary (sorted list of 1000 Chinese words).
    3. For each sample:
       a. Compute frame range from start/end times (at 25 FPS).
       b. Extract a centered window of `target_frames` around the word.
       c. Load lip images, resize to 96x96, convert to grayscale.
       d. Save as .pkl (same format as LRW pipeline).
    4. Save vocabulary mapping (vocab.json).

Output Structure:
    /ssd2/3DCvT_data/data_LRW1000/
    ├── vocab.json
    └── <chinese_word>/
        └── <split>/
            └── <hash_id>_<start>_<end>.pkl

Usage:
    python preprocess_lrw1000.py [--source_dir ...] [--target_dir ...] [--workers 16]

Author: Jiafeng Wu (Reproducing 3DCvT)
"""

import os
import cv2
import json
import pickle
import argparse
import logging
import multiprocessing
import numpy as np
from pathlib import Path
from tqdm import tqdm
from functools import partial
from typing import List, Tuple, Optional, Dict

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("preprocess_lrw1000.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
FPS = 25
TARGET_SIZE = (96, 96)
TARGET_FRAMES = 29  # Fixed temporal length (same as LRW)


def parse_annotation_file(filepath: str) -> List[dict]:
    """
    Parse an LRW-1000 annotation file.

    Each line: hash_id,chinese_word,pinyin,start_time,end_time
    
    Returns:
        List of dicts with keys: hash_id, word, pinyin, start, end
    """
    samples = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) < 5:
                logger.warning(f"Skipping malformed line {line_num}: {line}")
                continue
            
            hash_id = parts[0].strip()
            word = parts[1].strip()
            pinyin = parts[2].strip()
            try:
                start = float(parts[3].strip())
                end = float(parts[4].strip())
            except ValueError:
                logger.warning(f"Invalid time at line {line_num}: {line}")
                continue
            
            samples.append({
                'hash_id': hash_id,
                'word': word,
                'pinyin': pinyin,
                'start': start,
                'end': end,
            })
    return samples


def build_vocabulary(all_samples: List[dict]) -> List[str]:
    """
    Builds a sorted vocabulary from all annotation samples.
    
    Returns:
        Sorted list of unique Chinese words.
    """
    words = set()
    for s in all_samples:
        words.add(s['word'])
    vocab = sorted(list(words))
    return vocab


def get_total_frames(clip_dir: Path) -> int:
    """Count the number of JPEG frames in a clip directory."""
    count = 0
    for f in clip_dir.iterdir():
        if f.suffix.lower() in ('.jpg', '.jpeg', '.png'):
            count += 1
    return count


def load_frames_from_range(clip_dir: Path, start_frame: int, end_frame: int) -> Optional[np.ndarray]:
    """
    Load a range of frames from a clip directory.
    Frame files are named 1.jpg, 2.jpg, ... (1-indexed).
    
    Args:
        clip_dir: Path to the hash directory containing JPGs.
        start_frame: Start frame number (1-indexed, inclusive).
        end_frame: End frame number (1-indexed, inclusive).
    
    Returns:
        np.ndarray of shape (T, 96, 96), uint8, grayscale. None if failed.
    """
    frames = []
    for i in range(start_frame, end_frame + 1):
        img_path = clip_dir / f"{i}.jpg"
        if not img_path.exists():
            # If frame doesn't exist, pad with black
            frames.append(np.zeros(TARGET_SIZE[::-1], dtype=np.uint8))
            continue
        
        img = cv2.imread(str(img_path))
        if img is None:
            frames.append(np.zeros(TARGET_SIZE[::-1], dtype=np.uint8))
            continue
        
        # Resize to 96x96
        img = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)
        
        # Convert to grayscale
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        frames.append(img)
    
    if len(frames) == 0:
        return None
    
    return np.array(frames, dtype=np.uint8)  # (T, 96, 96)


def process_sample(
    sample: dict,
    lip_images_root: str,
    target_root: str,
    split: str,
    target_frames: int = TARGET_FRAMES
) -> bool:
    """
    Process a single LRW-1000 sample.
    
    Strategy:
        1. Compute frame range from start/end times.
        2. Center a window of `target_frames` around the word midpoint.
        3. If window exceeds clip boundaries, shift/clamp.
        4. If clip has fewer frames than needed, pad with zeros.
    
    Args:
        sample: Dict with hash_id, word, pinyin, start, end.
        lip_images_root: Root path to lip_images/lip_images/.
        target_root: Root path for output .pkl files.
        split: 'train', 'val', or 'test'.
        target_frames: Number of output frames.
    
    Returns:
        True if successful.
    """
    try:
        hash_id = sample['hash_id']
        word = sample['word']
        start_time = sample['start']
        end_time = sample['end']
        
        # Build paths
        clip_dir = Path(lip_images_root) / hash_id
        if not clip_dir.exists():
            return False
        
        # Build output path
        # Sanitize filename: use start_end with 'p' for decimal point
        start_str = f"{start_time:.2f}".replace('.', 'p')
        end_str = f"{end_time:.2f}".replace('.', 'p')
        save_dir = Path(target_root) / word / split
        save_filename = f"{hash_id}_{start_str}_{end_str}.pkl"
        save_path = save_dir / save_filename
        
        # Skip if already exists
        if save_path.exists():
            return True
        
        # Count total frames in clip
        total_frames = get_total_frames(clip_dir)
        if total_frames == 0:
            return False
        
        # Compute word frame range (1-indexed)
        word_start = max(1, round(start_time * FPS))
        word_end = max(word_start, round(end_time * FPS))
        
        # Clamp to actual clip range
        word_start = min(word_start, total_frames)
        word_end = min(word_end, total_frames)
        
        # Center window of `target_frames` around word midpoint
        word_mid = (word_start + word_end) / 2.0
        window_start = int(round(word_mid - target_frames / 2.0))
        window_end = window_start + target_frames - 1
        
        # Clamp window to clip boundaries
        if window_start < 1:
            window_start = 1
            window_end = window_start + target_frames - 1
        if window_end > total_frames:
            window_end = total_frames
            window_start = max(1, window_end - target_frames + 1)
        
        # Load frames
        video = load_frames_from_range(clip_dir, window_start, window_end)
        if video is None:
            return False
        
        # Pad if we still don't have enough frames (very short clips)
        actual_len = video.shape[0]
        if actual_len < target_frames:
            pad = np.zeros((target_frames - actual_len, TARGET_SIZE[1], TARGET_SIZE[0]), dtype=np.uint8)
            video = np.concatenate([video, pad], axis=0)
        
        # Truncate if too many (shouldn't happen but safety)
        video = video[:target_frames]
        
        # Compute word boundary position relative to the output window (0-indexed)
        # word_start, word_end are 1-indexed frame numbers in the original clip
        # window_start is also 1-indexed
        boundary_start = max(0, word_start - window_start)  # 0-indexed in output
        boundary_end = min(target_frames - 1, word_end - window_start)  # 0-indexed in output
        
        # Save
        save_dir.mkdir(parents=True, exist_ok=True)
        data_packet = {
            'video': video,             # (T, 96, 96) uint8
            'label': word,              # Chinese word
            'split': split,
            'duration': actual_len,     # Actual word duration before padding
            'boundary': (boundary_start, boundary_end),  # Word position in window (0-indexed)
            'pinyin': sample['pinyin'],
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(data_packet, f)
        
        return True
    
    except Exception as e:
        logger.error(f"Error processing {sample.get('hash_id', 'unknown')}: {e}")
        return False


def process_sample_wrapper(args_tuple):
    """Wrapper for multiprocessing (unpacks tuple args)."""
    sample, lip_images_root, target_root, split, target_frames = args_tuple
    return process_sample(sample, lip_images_root, target_root, split, target_frames)


def main():
    parser = argparse.ArgumentParser(description="LRW-1000 Preprocessing Pipeline")
    parser.add_argument('--source_dir', type=str, 
                        default='/public/dataset/LRW_1000_Full_Version',
                        help='Path to raw LRW-1000 dataset root')
    parser.add_argument('--target_dir', type=str,
                        default='/ssd2/3DCvT_data/data_LRW1000',
                        help='Path to save processed data')
    parser.add_argument('--workers', type=int, default=16,
                        help='Number of multiprocessing workers')
    parser.add_argument('--target_frames', type=int, default=TARGET_FRAMES,
                        help=f'Fixed temporal length per sample (default: {TARGET_FRAMES})')
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    target_dir = Path(args.target_dir)
    lip_images_root = source_dir / 'lip_images' / 'lip_images'
    info_dir = source_dir / 'info'

    # Verify source paths
    if not source_dir.exists():
        raise FileNotFoundError(f"Source dir not found: {source_dir}")
    if not lip_images_root.exists():
        raise FileNotFoundError(f"Lip images dir not found: {lip_images_root}")

    # Define split mapping
    split_files = {
        'train': info_dir / 'trn_1000.txt',
        'val':   info_dir / 'val_1000.txt',
        'test':  info_dir / 'tst_1000.txt',
    }

    # 1. Parse all annotation files
    logger.info("Parsing annotation files...")
    all_samples = []
    split_samples = {}
    for split_name, filepath in split_files.items():
        if not filepath.exists():
            logger.warning(f"Annotation file not found: {filepath}")
            continue
        samples = parse_annotation_file(str(filepath))
        split_samples[split_name] = samples
        all_samples.extend(samples)
        logger.info(f"  {split_name}: {len(samples)} samples from {filepath.name}")

    logger.info(f"Total samples across all splits: {len(all_samples)}")

    # 2. Build and save vocabulary
    vocab = build_vocabulary(all_samples)
    logger.info(f"Vocabulary size: {len(vocab)} classes")

    target_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = target_dir / 'vocab.json'
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    logger.info(f"Vocabulary saved to: {vocab_path}")

    # 3. Process each split
    logger.info(f"Target directory: {target_dir}")
    logger.info(f"Settings: FPS={FPS}, Target size={TARGET_SIZE}, "
                f"Target frames={args.target_frames}")

    for split_name, samples in split_samples.items():
        logger.info(f"\n--- Processing split: {split_name} ({len(samples)} samples) ---")
        
        # Prepare args for multiprocessing
        task_args = [
            (s, str(lip_images_root), str(target_dir), split_name, args.target_frames)
            for s in samples
        ]
        
        # Multiprocess
        with multiprocessing.Pool(processes=args.workers) as pool:
            results = list(tqdm(
                pool.imap_unordered(process_sample_wrapper, task_args),
                total=len(task_args),
                desc=f"Preprocessing [{split_name}]",
                unit="sample"
            ))
        
        success_count = sum(results)
        fail_count = len(results) - success_count
        logger.info(f"[{split_name}] Done. Success: {success_count}, Failed: {fail_count}")

    logger.info("\n=== Preprocessing Complete ===")


if __name__ == "__main__":
    main()
