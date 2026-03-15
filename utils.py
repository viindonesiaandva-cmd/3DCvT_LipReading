# -*- coding: utf-8 -*-
"""
Shared utilities for 3DCvT training scripts.

Contains:
    - DATASET_REGISTRY: Dataset name -> (class, default_root, num_classes) mapping.
    - AverageMeter: Running average tracker for loss/accuracy.
    - CurvePlotter: Training curve visualization and saving.
    - _save_checkpoint / _load_checkpoint: Unified checkpoint format with auto-detection.

Used by both train.py (single-GPU) and train_ddp.py (DDP multi-GPU).
"""

import os
import logging
import torch
from pathlib import Path

from dataset import LRWDataset, LRW1000Dataset
from model import LipReading3DCvT  # noqa: F401 — re-exported for convenience


# ============================================================================
# Dataset Registry
# ============================================================================

DATASET_REGISTRY = {
    'lrw':     (LRWDataset,     '/ssd2/3DCvT_data/data_LRW',    500),
    'lrw1000': (LRW1000Dataset, '/ssd2/3DCvT_data/data_LRW1000', 1184),
}


# ============================================================================
# Utility Classes
# ============================================================================

class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CurvePlotter:
    """Handles plotting and saving of training curves."""

    def __init__(self, save_dir):
        self.save_dir = Path(save_dir)
        self.train_loss = []
        self.val_loss = []
        self.val_acc = []
        self.epochs = []

    def update(self, epoch, train_loss, val_loss, val_acc):
        self.epochs.append(epoch)
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)
        self.val_acc.append(val_acc)
        self._plot()

    def _plot(self):
        import matplotlib.pyplot as plt

        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        l1, = ax1.plot(self.epochs, self.train_loss, label='Train Loss', color='tab:blue')
        l2, = ax1.plot(self.epochs, self.val_loss, label='Val Loss', color='tab:orange')

        ax2 = ax1.twinx()
        ax2.set_ylabel('Accuracy (%)')
        l3, = ax2.plot(self.epochs, self.val_acc, label='Val Acc', color='tab:green', linestyle='--')

        lines = [l1, l2, l3]
        ax1.legend(lines, [l.get_label() for l in lines], loc='upper center')
        plt.title('Training Metrics')
        plt.grid(True, alpha=0.3)
        plt.savefig(self.save_dir / "learning_curve.png")
        plt.close()


# ============================================================================
# Checkpoint Save / Load  (Unified format, interchangeable between scripts)
# ============================================================================

def save_checkpoint(epoch, model, optimizer, scheduler, scaler, val_acc, plotter=None):
    """
    Build a unified checkpoint dict.

    Standardized keys so that checkpoints saved by train.py and train_ddp.py
    are fully interchangeable for resuming training.

    Keys:
        epoch, model_state_dict, optimizer_state_dict, scheduler_state_dict,
        scaler_state_dict, acc, curve_history (optional).

    DDP ``module.`` prefix is stripped automatically.
    """
    # Extract raw model weights, stripping DDP 'module.' wrapper if present
    if hasattr(model, 'module'):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()

    state = {
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'acc': val_acc,
    }

    if plotter is not None:
        state['curve_history'] = {
            'epochs': plotter.epochs,
            'train_loss': plotter.train_loss,
            'val_loss': plotter.val_loss,
            'val_acc': plotter.val_acc,
        }

    return state


def load_checkpoint(checkpoint_path, model, optimizer, scheduler, scaler,
                    device, plotter=None):
    """
    Load a checkpoint with automatic format detection.

    Supports three formats:
        1. **Unified** (``model_state_dict``, ``optimizer_state_dict``, …)
        2. **Legacy DDP** (``model``, ``optimizer``, …)
        3. **Raw state_dict** (weights only, no optimizer/scheduler)

    Handles DDP ``module.`` prefix stripping automatically.

    Args:
        checkpoint_path: Path to ``.pth`` file.
        model: Model (may be DDP-wrapped or ``torch.compile``-wrapped).
        optimizer, scheduler, scaler: Training state objects.
        device: Target device for tensor mapping.
        plotter: Optional ``CurvePlotter`` to restore history into.

    Returns:
        The loaded checkpoint dict (for extracting ``epoch``, ``acc``, etc.).
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logging.info(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # --- Detect format and extract model weights ---
    if not isinstance(checkpoint, dict):
        logging.warning("Checkpoint contains only raw model weights. "
                        "Optimizer/scheduler states not restored.")
        checkpoint = {'model_state_dict': checkpoint}

    # Resolve model weights key (unified vs legacy)
    if 'model_state_dict' in checkpoint:
        raw_state = checkpoint['model_state_dict']
    elif 'model' in checkpoint:
        raw_state = checkpoint['model']
        logging.info("Detected legacy DDP checkpoint format (key='model').")
    else:
        raw_state = checkpoint
        logging.warning("Checkpoint has no standard model key. "
                        "Treating entire dict as state_dict.")

    # Strip 'module.' prefix if present (from DDP-saved weights)
    cleaned_state = {}
    for k, v in raw_state.items():
        cleaned_state[k[7:] if k.startswith('module.') else k] = v

    # Load into model (handle DDP / compile wrappers)
    target_model = model.module if hasattr(model, 'module') else model
    # torch.compile wraps in _orig_mod
    if hasattr(target_model, '_orig_mod'):
        target_model = target_model._orig_mod
    # strict=False: tolerates 'boundary_token' in old checkpoints (now removed from model)
    # and any new parameters that don't exist in the checkpoint yet.
    missing, unexpected = target_model.load_state_dict(cleaned_state, strict=False)
    if missing:
        logging.warning(f"Missing keys in checkpoint (will use random init): {missing}")
    if unexpected:
        logging.warning(f"Unexpected keys in checkpoint (ignored): {unexpected}")
        if 'boundary_token' in unexpected:
            logging.warning(
                "Checkpoint contains legacy key 'boundary_token'. "
                "This usually means the weights were saved from an older model variant. "
                "Loading will continue for compatibility, but predictions may be unreliable "
                "unless you use the matching code/checkpoint pair."
            )
    logging.info("Model weights loaded successfully.")
    opt_state = checkpoint.get('optimizer_state_dict') or checkpoint.get('optimizer')
    if opt_state is not None:
        optimizer.load_state_dict(opt_state)
        logging.info("Optimizer state restored.")
    else:
        logging.warning("No optimizer state in checkpoint. Optimizer starts fresh.")

    # --- Restore scheduler ---
    sched_state = (checkpoint.get('scheduler_state_dict')
                   or checkpoint.get('scheduler'))
    if sched_state is not None:
        scheduler.load_state_dict(sched_state)
        logging.info("Scheduler state restored.")
    else:
        resume_epoch = checkpoint.get('epoch', 0)
        logging.warning(f"No scheduler state in checkpoint. "
                        f"Stepping {resume_epoch} times to catch up...")
        for _ in range(resume_epoch):
            scheduler.step()

    # --- Restore AMP scaler ---
    scaler_state = (checkpoint.get('scaler_state_dict')
                    or checkpoint.get('scaler'))
    if scaler_state is not None:
        scaler.load_state_dict(scaler_state)
        logging.info("AMP scaler state restored.")

    # --- Restore curve history ---
    if plotter is not None and 'curve_history' in checkpoint:
        h = checkpoint['curve_history']
        plotter.epochs = h['epochs']
        plotter.train_loss = h['train_loss']
        plotter.val_loss = h['val_loss']
        plotter.val_acc = h['val_acc']
        logging.info(f"Curve history restored ({len(h['epochs'])} epochs).")

    return checkpoint


def load_weights(checkpoint_path, model, device):
    """
    Load model weights only (no optimizer/scheduler/scaler).

    Convenience wrapper for evaluation and inference scripts.
    Supports the same 3 checkpoint formats and DDP prefix stripping
    as ``load_checkpoint``.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logging.info(f"Loading weights from: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    if not isinstance(checkpoint, dict):
        checkpoint = {'model_state_dict': checkpoint}

    if 'model_state_dict' in checkpoint:
        raw_state = checkpoint['model_state_dict']
    elif 'model' in checkpoint:
        raw_state = checkpoint['model']
    else:
        raw_state = checkpoint

    cleaned_state = {}
    for k, v in raw_state.items():
        cleaned_state[k[7:] if k.startswith('module.') else k] = v

    target_model = model.module if hasattr(model, 'module') else model
    if hasattr(target_model, '_orig_mod'):
        target_model = target_model._orig_mod
    missing, unexpected = target_model.load_state_dict(cleaned_state, strict=False)
    if missing:
        logging.warning(f"Missing keys in checkpoint (will use random init): {missing}")
    if unexpected:
        logging.warning(f"Unexpected keys in checkpoint (ignored): {unexpected}")
        if 'boundary_token' in unexpected:
            logging.warning(
                "Checkpoint contains legacy key 'boundary_token'. "
                "This usually means the weights were saved from an older model variant. "
                "Loading will continue for compatibility, but predictions may be unreliable "
                "unless you use the matching code/checkpoint pair."
            )
    logging.info("Model weights loaded successfully.")

    return checkpoint
