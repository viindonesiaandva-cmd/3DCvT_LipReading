# -*- coding: utf-8 -*-
"""
Training Script for 3DCvT Lip Reading Model.

Description:
    Main entry point for training.
    Features:
    - Automatic experiment folder creation (ckpts, curves, logs).
    - Mixed Precision Training (AMP).
    - Mixup Data Augmentation integration.
    - Label Smoothing Cross Entropy.
    - Cosine Learning Rate Scheduler.
    - Periodical Loss/Accuracy Curve plotting.

Usage:
    python train.py --exp_name baseline_run --batch_size 32 --lr 6e-4

Author: Jiafeng Wu (Reproducing 3DCvT)
Environment: Python 3.10, PyTorch 2.x
"""

import os

import argparse
import logging
import json
import signal
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import time
import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.amp import GradScaler
from pathlib import Path

from dataset import Mixup
from utils import (
    DATASET_REGISTRY, AverageMeter, CurvePlotter, LipReading3DCvT,
    save_checkpoint, load_checkpoint,
)

# --------------------------------------------------------------------------------
# Main Training Functions
# --------------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, scheduler, criterion, scaler, mixup_fn, device, epoch, accum_steps=1):
    model.train()
    losses = AverageMeter()
    num_batches = len(loader)
    
    # Wrap loader with tqdm, ncols=0 auto-adjusts to terminal width
    pbar = tqdm(loader, desc=f"Epoch {epoch}", ncols=0)

    optimizer.zero_grad(set_to_none=True)
    
    for batch_idx, batch_data in enumerate(pbar):
        inputs = batch_data['video'].to(device) # (B, 1, T, H, W)
        targets = batch_data['label'].to(device)
        boundary = batch_data['boundary'].to(device) # (B, T, 1)

        # Apply Mixup
        if mixup_fn is not None:
            inputs, targets_a, targets_b, lam, boundary = mixup_fn(inputs, targets, boundary)
        
        # Determine if this is the last micro-batch in the accumulation window
        is_accumulating = ((batch_idx + 1) % accum_steps != 0) and (batch_idx + 1 != num_batches)

        # Mixed Precision Forward
        with torch.amp.autocast('cuda'):
            outputs = model(inputs, boundary) # (B, Num_Classes)
            
            # Mixup Loss Calculation
            if mixup_fn is not None:
                loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
            else:
                loss = criterion(outputs, targets)
            # Scale loss by accumulation steps to get correct mean
            loss = loss / accum_steps

        # Backward (accumulate gradients)
        scaler.scale(loss).backward()
        
        # Record unscaled loss for display
        losses.update((loss * accum_steps).item(), inputs.size(0))

        # Step optimizer only after accumulating enough micro-batches
        if not is_accumulating:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # Update progress bar postfix with current Loss and LR
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix(loss=f"{losses.avg:.4f}", lr=f"{current_lr:.6f}")
    
    # Step scheduler after each epoch
    scheduler.step()
    
    return losses.avg


def validate(model, loader, criterion, device):
    model.eval()
    losses = AverageMeter()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_data in loader:
            inputs = batch_data['video'].to(device)
            targets = batch_data['label'].to(device)
            boundary = batch_data['boundary'].to(device)

            # Standard Forward (No Mixup in Val)
            outputs = model(inputs, boundary)
            loss = criterion(outputs, targets)

            # Accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            losses.update(loss.item(), inputs.size(0))

    acc = 100 * correct / total
    return losses.avg, acc

# --------------------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="3DCvT Training")
    parser.add_argument('--dataset', type=str, default='lrw', choices=['lrw', 'lrw1000'],
                        help='Dataset to use: lrw (500 classes) or lrw1000 (1000 classes)')
    parser.add_argument('--data_root', type=str, default=None,
                        help='Path to processed .pkl data (auto-set based on --dataset if not provided)')
    parser.add_argument('--exp_name', type=str, default='default_exp', help='Name of the experiment')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size (Paper uses 256 for 4 GPUs)')
    parser.add_argument('--epochs', type=int, default=120, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=6e-4, help='Initial learning rate')
    parser.add_argument('--num_workers', type=int, default=8, help='DataLoader workers')
    parser.add_argument('--num_classes', type=int, default=None,
                        help='Number of classes (auto-set based on --dataset if not provided)')
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID to use')
    parser.add_argument('--resume', type=str, default='', help='Path to checkpoint for resuming training (e.g., experiments/exp1/ckpts/latest.pth)')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Number of warmup epochs')
    parser.add_argument('--accum_steps', type=int, default=1,
                        help='Gradient accumulation steps. Effective batch = batch_size * accum_steps')
    parser.add_argument('--use_compile', action='store_true',
                        help='Enable torch.compile optimization (disabled by default for stability on some GPUs)')
    args = parser.parse_args()

    # Auto-configure based on dataset choice
    DatasetClass, default_root, default_classes = DATASET_REGISTRY[args.dataset]
    if args.data_root is None:
        args.data_root = default_root
    if args.num_classes is None:
        args.num_classes = default_classes

    # 1. Directories Setup
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    # Enable cuDNN auto-tuner: finds the fastest convolution algorithms for fixed input sizes.
    # Since lip reading frames are always (1, 29, 88, 88), this gives a significant speedup.
    torch.backends.cudnn.benchmark = True
    exp_dir = Path(f"./experiments/{args.exp_name}")
    ckpt_dir = exp_dir / "ckpts"
    curve_dir = exp_dir / "curves"
    log_dir = exp_dir / "logs"
    
    for d in [ckpt_dir, curve_dir, log_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 2. Logging Setup
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "train.log"),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Experiment Started: {args.exp_name}")
    logging.info(f"Config: {json.dumps(vars(args), indent=2)}")
    effective_bs = args.batch_size * args.accum_steps
    logging.info(f"Effective batch size: {args.batch_size} x {args.accum_steps} accum = {effective_bs}")
    if not args.use_compile:
        logging.info("torch.compile: disabled (recommended for stability on RTX 20xx / checkpointing).")

    # 3. Data Setup
    logging.info(f"Initializing Datasets ({args.dataset})...")
    train_set = DatasetClass(args.data_root, mode='train')
    val_set = DatasetClass(args.data_root, mode='val')
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              drop_last=True,  # Ensures every batch is full-sized; avoids incorrect loss scaling with accum_steps
                              persistent_workers=True, prefetch_factor=3)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, 
                            num_workers=args.num_workers, pin_memory=True,
                            persistent_workers=True, prefetch_factor=3)

    # Mixup Augmentation (alpha=0.2 for small batch size, 0.4 for large batch)
    mixup_fn = Mixup(alpha=0.4)
    # mixup_fn = None  # Uncomment to disable Mixup
    
    # 4. Model Setup
    logging.info("Building Model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LipReading3DCvT(num_classes=args.num_classes).to(device)
    
    # torch.compile is opt-in: on some setups (especially RTX 20xx + checkpointing),
    # eager mode is more stable and avoids graph/autocast edge cases.
    if args.use_compile:
        model = torch.compile(model, mode='default')
        logging.info("torch.compile: enabled (mode=default)")
    
    # 5. Optimizer & Loss & Scheduler
    # Paper: Adam optimizer, Label Smoothing
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Paper: Cosine learning rate scheduling with Warmup
    # Warmup: LR linearly increases from 1e-5 to args.lr over warmup_epochs
    # Then: Cosine annealing from args.lr to 1e-6
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-5/args.lr, end_factor=1.0, total_iters=args.warmup_epochs
    )
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=1e-6
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[args.warmup_epochs]
    )
    
    # Paper: Label Smoothing (Formula 6) -> PyTorch built-in
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)
    
    scaler = GradScaler('cuda') # For Mixed Precision
    plotter = CurvePlotter(curve_dir)
    
    best_acc = 0.0
    start_epoch = 1
    
    # Resume from checkpoint
    if args.resume:
        checkpoint = load_checkpoint(
            checkpoint_path=args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            plotter=plotter
        )
        
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_acc = checkpoint.get('acc', 0.0)
        logging.info(f"Resume successful. Starting from Epoch {start_epoch}, Best Acc: {best_acc:.2f}%")


    # 6. Training Loop
    logging.info("Start Training...")
    start_time_total = time.time()  # Record total start time

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start_time = time.time()  # Record epoch start time
        
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, scaler, mixup_fn, device, epoch,
            accum_steps=args.accum_steps
        )
        
        # Validation
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Calculate timing
        epoch_duration = time.time() - epoch_start_time
        remaining_epochs = args.epochs - epoch
        estimated_remaining_time = remaining_epochs * epoch_duration
        eta_str = str(datetime.timedelta(seconds=int(estimated_remaining_time)))
        
        # Log with timing and ETA
        logging.info(f"Epoch [{epoch}/{args.epochs}] Completed in {int(epoch_duration)}s | ETA: {eta_str}")
        logging.info(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Update Plots
        plotter.update(epoch, train_loss, val_loss, val_acc)
        
        # Save Checkpoints
        # 1. Save Latest (full state for resume)
        state = save_checkpoint(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            val_acc=val_acc,
            plotter=plotter
        )
        torch.save(state, ckpt_dir / "latest.pth")
        
        # 2. Save Best (raw weights only for smaller file size)
        if val_acc > best_acc:
            best_acc = val_acc
            logging.info(f"New Best Accuracy: {best_acc:.2f}% - Saving Model...")
            torch.save(state['model_state_dict'], ckpt_dir / "best_model.pth")
            
        # 3. Save Periodic (e.g., every 10 epochs)
        if epoch % 10 == 0:
            torch.save(state['model_state_dict'], ckpt_dir / f"epoch_{epoch}.pth")

    total_time = str(datetime.timedelta(seconds=int(time.time() - start_time_total)))
    logging.info(f"Training Complete. Total Time: {total_time}. Best Accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    signal.signal(signal.SIGINT, lambda s, f: sys.exit(1))
    signal.signal(signal.SIGTERM, lambda s, f: sys.exit(1))
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.", flush=True)