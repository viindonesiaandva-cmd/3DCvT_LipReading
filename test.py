# -*- coding: utf-8 -*-
"""
Evaluation Script for 3DCvT Lip Reading Model.

Description:
    Evaluates the trained model on the validation or test set.
    Outputs:
      1. Overall Top-1 ~ Top-5 accuracy (printed to console).
      2. Per-class accuracy sorted from high to low (saved to --output file).
    Compatible with both LRW (500 classes) and LRW-1000 (1184 classes).

Usage:
    python test.py --checkpoint experiments/exp/ckpts/best_model.pth
    python test.py --checkpoint best.pth --output results/per_class_acc.csv

Author: Jiafeng Wu (Reproducing 3DCvT)
"""

import argparse
import csv
import logging
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import DATASET_REGISTRY, LipReading3DCvT, load_weights

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def evaluate(model, loader, device, num_classes):
    """
    Evaluate model and return overall Top-k accuracy and per-class stats.

    Returns:
        topk_acc: dict  e.g. {'Top-1': 83.5, 'Top-2': 90.1, ...}
        per_class_correct: np.ndarray of shape (num_classes,)
        per_class_total:   np.ndarray of shape (num_classes,)
    """
    model.eval()
    topk = (1, 2, 3, 4, 5)
    top_correct = np.zeros(len(topk))
    per_class_correct = np.zeros(num_classes)
    per_class_total = np.zeros(num_classes)
    total_samples = 0

    with torch.no_grad():
        for batch_data in tqdm(loader, desc="Evaluating"):
            inputs = batch_data['video'].to(device)
            targets = batch_data['label'].to(device)
            boundary = batch_data['boundary'].to(device) if 'boundary' in batch_data else None

            outputs = model(inputs, boundary)

            # --- Top-k accuracy ---
            maxk = max(topk)
            _, pred = outputs.topk(maxk, 1, True, True)
            pred_t = pred.t()
            correct_mat = pred_t.eq(targets.view(1, -1).expand_as(pred_t))

            bs = targets.size(0)
            for i, k in enumerate(topk):
                top_correct[i] += correct_mat[:k].reshape(-1).float().sum().item()
            total_samples += bs

            # --- Per-class accuracy ---
            top1_pred = pred[:, 0]
            for t, p in zip(targets, top1_pred):
                cls = t.item()
                per_class_total[cls] += 1
                if p.item() == cls:
                    per_class_correct[cls] += 1

    topk_acc = {f'Top-{k}': top_correct[i] / total_samples * 100
                for i, k in enumerate(topk)}
    return topk_acc, per_class_correct, per_class_total


def main():
    parser = argparse.ArgumentParser(description="3DCvT Evaluation")
    parser.add_argument('--dataset', type=str, default='lrw',
                        choices=['lrw', 'lrw1000'])
    parser.add_argument('--data_root', type=str, default=None)
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to .pth checkpoint')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--num_classes', type=int, default=None)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--mode', type=str, default='val',
                        choices=['val', 'test'])
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for per-class accuracy (csv/txt). '
                             'Default: per_class_acc_<dataset>_<mode>.csv')
    args = parser.parse_args()

    # Auto-configure
    DatasetClass, default_root, default_classes = DATASET_REGISTRY[args.dataset]
    if args.data_root is None:
        args.data_root = default_root
    if args.num_classes is None:
        args.num_classes = default_classes
    if args.output is None:
        args.output = f'per_class_acc_{args.dataset}_{args.mode}.csv'

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Data
    test_set = DatasetClass(args.data_root, mode=args.mode)
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    logger.info(f"Dataset: {args.dataset} [{args.mode}], {len(test_set)} samples, "
                f"{args.num_classes} classes")

    # Get class names from dataset
    class_names = test_set.classes  # Both LRWDataset and LRW1000Dataset have this

    # Model
    model = LipReading3DCvT(num_classes=args.num_classes).to(device)
    load_weights(args.checkpoint, model, device)

    # Evaluate
    topk_acc, per_class_correct, per_class_total = evaluate(
        model, test_loader, device, args.num_classes
    )

    # --- Print overall Top-k ---
    logger.info("=" * 50)
    logger.info(f"Overall Results — {args.dataset.upper()} [{args.mode}]")
    logger.info("-" * 50)
    for name, acc in topk_acc.items():
        logger.info(f"  {name} Accuracy: {acc:.2f}%")
    logger.info("=" * 50)

    # --- Per-class accuracy, sorted high → low ---
    per_class_acc = []
    for i, name in enumerate(class_names):
        total = int(per_class_total[i])
        correct = int(per_class_correct[i])
        acc = (correct / total * 100) if total > 0 else 0.0
        per_class_acc.append((name, acc, correct, total))

    per_class_acc.sort(key=lambda x: x[1], reverse=True)

    # Write to file
    output_path = args.output
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Rank', 'Class', 'Accuracy(%)', 'Correct', 'Total'])
        for rank, (name, acc, correct, total) in enumerate(per_class_acc, 1):
            writer.writerow([rank, name, f'{acc:.2f}', correct, total])

    logger.info(f"Per-class accuracy saved to: {output_path}")

    # Print top-10 and bottom-10 for quick overview
    logger.info(f"\nTop-10 classes:")
    for rank, (name, acc, correct, total) in enumerate(per_class_acc[:10], 1):
        logger.info(f"  {rank:>3}. {name:<20s} {acc:6.2f}%  ({correct}/{total})")

    logger.info(f"\nBottom-10 classes:")
    for rank_offset, (name, acc, correct, total) in enumerate(per_class_acc[-10:]):
        rank = len(per_class_acc) - 9 + rank_offset
        logger.info(f"  {rank:>3}. {name:<20s} {acc:6.2f}%  ({correct}/{total})")


if __name__ == "__main__":
    main()