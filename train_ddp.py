# -*- coding: utf-8 -*-
"""
DDP Training Script for 3DCvT (Multi-GPU).

Description:
    Distributed Data Parallel (DDP) training script for 2x RTX 4090.
    
    Key Features:
    - SyncBatchNorm: Synchronize BN stats across GPUs.
    - DistributedSampler: Shard data across GPUs.
    - Reduce Tensor: Aggregate loss/acc for correct logging.
    - Rank 0 Logging: Only the main process writes to disk.

Usage:
    torchrun --nproc_per_node=2 train_ddp.py --exp_name ddp_run_01 --batch_size 16

Author: Jiafeng Wu (Reproducing 3DCvT)
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
import torch.distributed as dist
import time
import datetime
from contextlib import nullcontext
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from pathlib import Path

from dataset import Mixup
from utils import (
    DATASET_REGISTRY, CurvePlotter, LipReading3DCvT,
    save_checkpoint, load_checkpoint,
)

# --------------------------------------------------------------------------------
# DDP Utilities
# --------------------------------------------------------------------------------

def init_distributed_mode():
    """Initializes the process group for DDP."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        gpu = int(os.environ["LOCAL_RANK"])
    else:
        print('Not using Distributed Mode')
        return False, 0, 0, 1

    torch.cuda.set_device(gpu)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    dist.barrier(device_ids=[gpu])
    
    # Enable cuDNN auto-tuner: finds the fastest convolution algorithms for fixed input sizes.
    # Since lip reading frames are always (1, 29, 88, 88), this gives a significant speedup.
    torch.backends.cudnn.benchmark = True
    
    # Setup print/logging only for master process
    setup_for_distributed(rank == 0)
    return True, rank, gpu, world_size

def setup_for_distributed(is_master):
    """Disable printing when not in master process"""
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def reduce_mean(tensor, nprocs):
    """Average a tensor across all GPUs"""
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt

# --------------------------------------------------------------------------------
# Training Loop
# --------------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, scheduler, criterion, scaler, mixup_fn, device, epoch, nprocs, rank, accum_steps=1):
    model.train()
    total_loss = 0.0
    num_batches = len(loader)
    
    # DDP requires setting sampler epoch to ensure proper shuffle randomness
    loader.sampler.set_epoch(epoch)

    # --- Only enable TQDM on Rank 0 ---
    if rank == 0:
        # ncols=0 auto-adjusts width, mininterval=0.5 prevents excessive refresh
        pbar = tqdm(loader, desc=f"Epoch {epoch}", ncols=0, mininterval=0.5)
    else:
        pbar = loader
    # --- End ---

    optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch_data in enumerate(pbar):
        inputs = batch_data['video'].to(device, non_blocking=True)
        targets = batch_data['label'].to(device, non_blocking=True)
        boundary = batch_data['boundary'].to(device, non_blocking=True)

        if mixup_fn is not None:
            inputs, targets_a, targets_b, lam, boundary = mixup_fn(inputs, targets, boundary)
        
        # Determine if this is the last micro-batch in the accumulation window
        is_accumulating = ((batch_idx + 1) % accum_steps != 0) and (batch_idx + 1 != num_batches)

        # Skip gradient sync during accumulation (DDP optimization)
        # Only sync gradients on the final micro-batch before optimizer.step()
        context = model.no_sync if is_accumulating else nullcontext

        with context():
            with torch.amp.autocast('cuda'):
                outputs = model(inputs, boundary)
                if mixup_fn is not None:
                    loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
                else:
                    loss = criterion(outputs, targets)
                # Scale loss by accumulation steps to get correct mean
                loss = loss / accum_steps

            scaler.scale(loss).backward()

        # Aggregate unscaled loss across GPUs (for accurate display)
        reduced_loss = reduce_mean(loss * accum_steps, nprocs)  # Undo /accum_steps for display
        total_loss += reduced_loss.item()

        # Step optimizer only after accumulating enough micro-batches
        if not is_accumulating:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # --- Rank 0: update progress bar postfix ---
        if rank == 0:
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix(loss=f"{reduced_loss.item():.4f}", lr=f"{current_lr:.6f}")
        # --- End ---
    
    scheduler.step()
    
    # Close pbar to prevent output misalignment
    if rank == 0:
        pbar.close()

    return total_loss / num_batches


def validate(model, loader, criterion, device, nprocs):
    model.eval()
    total_loss = 0.0
    total_correct = 0.0
    total_samples = 0.0

    with torch.no_grad():
        for batch_data in loader:
            inputs = batch_data['video'].to(device, non_blocking=True)
            targets = batch_data['label'].to(device, non_blocking=True)
            boundary = batch_data['boundary'].to(device, non_blocking=True)

            outputs = model(inputs, boundary)
            loss = criterion(outputs, targets)

            # Metrics
            _, predicted = torch.max(outputs.data, 1)
            batch_correct = (predicted == targets).sum()
            
            # Aggregate metrics across GPUs
            reduced_loss = reduce_mean(loss, nprocs)
            
            # Gather correct predictions and sample counts
            # Note: Simple sum is enough here if batches are even, 
            # but reduce_mean on loss is safer for uneven last batches
            total_loss += reduced_loss.item()
            
            # We need to sum up correct predictions from all GPUs to calculate global Acc
            dist.all_reduce(batch_correct, op=dist.ReduceOp.SUM)
            total_correct += batch_correct.item()
            
            # We also need total samples seen by all GPUs
            batch_size_tensor = torch.tensor(targets.size(0), device=device)
            dist.all_reduce(batch_size_tensor, op=dist.ReduceOp.SUM)
            total_samples += batch_size_tensor.item()

    avg_loss = total_loss / len(loader)
    acc = 100 * total_correct / total_samples
    return avg_loss, acc

# --------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="3DCvT DDP Training")
    parser.add_argument('--dataset', type=str, default='lrw', choices=['lrw', 'lrw1000'],
                        help='Dataset to use: lrw (500 classes) or lrw1000 (1000 classes)')
    parser.add_argument('--data_root', type=str, default=None,
                        help='Path to processed .pkl data (auto-set based on --dataset if not provided)')
    parser.add_argument('--exp_name', type=str, default='ddp_exp')
    # Batch size per GPU. If set to 16, total batch size = 16 * 2 = 32
    parser.add_argument('--batch_size', type=int, default=16) 
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--lr', type=float, default=6e-4) # Base LR, maybe scale by world_size?
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--num_classes', type=int, default=None,
                        help='Number of classes (auto-set based on --dataset if not provided)')
    parser.add_argument('--resume', type=str, default=None, help='Path to latest checkpoint (default: None)')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Number of warmup epochs')
    parser.add_argument('--accum_steps', type=int, default=1,
                        help='Gradient accumulation steps. Effective batch = batch_size * world_size * accum_steps')
    args = parser.parse_args()

    # Auto-configure based on dataset choice
    DatasetClass, default_root, default_classes = DATASET_REGISTRY[args.dataset]
    if args.data_root is None:
        args.data_root = default_root
    if args.num_classes is None:
        args.num_classes = default_classes

    # 1. Initialize DDP
    is_ddp, rank, local_rank, world_size = init_distributed_mode()
    
    # 2. Setup Directories (Only on Rank 0)
    if rank == 0:
        exp_dir = Path(f"./experiments/{args.exp_name}")
        ckpt_dir = exp_dir / "ckpts"
        curve_dir = exp_dir / "curves"
        log_dir = exp_dir / "logs"
        for d in [ckpt_dir, curve_dir, log_dir]:
            d.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "train.log"),
                logging.StreamHandler()
            ]
        )
        logging.info(f"DDP Initialized. World Size: {world_size}")
        logging.info(f"Config: {json.dumps(vars(args), indent=2)}")
        effective_bs = args.batch_size * world_size * args.accum_steps
        logging.info(f"Effective batch size: {args.batch_size} x {world_size} GPUs x {args.accum_steps} accum = {effective_bs}")
    
    device = torch.device(f"cuda:{local_rank}")

    # 3. Datasets & DistributedSampler
    if rank == 0: logging.info(f"Initializing Datasets ({args.dataset})...")
    
    train_set = DatasetClass(args.data_root, mode='train')
    val_set = DatasetClass(args.data_root, mode='val')

    # Sampler is crucial for DDP
    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_set, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=True,   # Keep worker processes alive between epochs (avoids re-fork overhead)
        prefetch_factor=3,         # Pre-load 3 batches per worker for smoother GPU feeding
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, sampler=val_sampler,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
        persistent_workers=True,
        prefetch_factor=3,
    )

    # Mixup Augmentation (alpha=0.2 for small batch size, 0.4 for large batch)
    mixup_fn = Mixup(alpha=0.4)
    # mixup_fn = None  # Uncomment to disable Mixup

    # 4. Model & SyncBN
    model = LipReading3DCvT(num_classes=args.num_classes).to(device)
    
    # Convert BatchNorm to SyncBatchNorm for better Multi-GPU convergence
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    # IMPORTANT: Gradient checkpointing re-runs the forward pass during backward.
    # SyncBatchNorm triggers all_reduce during forward, and the re-run causes
    # mismatched collectives across processes → NaN or deadlock.
    # Fix: revert SyncBN back to regular BatchNorm inside checkpointed Stage 3 blocks.
    def _revert_sync_bn(module):
        """Recursively convert SyncBatchNorm back to BatchNorm2d."""
        for name, child in module.named_children():
            if isinstance(child, nn.SyncBatchNorm):
                bn = nn.BatchNorm2d(
                    child.num_features, eps=child.eps, momentum=child.momentum,
                    affine=child.affine, track_running_stats=child.track_running_stats
                )
                if child.affine:
                    bn.weight = child.weight
                    bn.bias = child.bias
                bn.running_mean = child.running_mean
                bn.running_var = child.running_var
                bn.num_batches_tracked = child.num_batches_tracked
                setattr(module, name, bn)
            else:
                _revert_sync_bn(child)
    
    if model.use_checkpoint:
        for blk in model.stage3_blocks:
            _revert_sync_bn(blk)
        if rank == 0:
            logging.info("Reverted SyncBN → BatchNorm in Stage 3 blocks (checkpoint compatibility).")
    
    # NOTE: torch.compile is disabled for DDP — it conflicts with both gradient
    # checkpointing (autocast context loss) and NCCL collectives (CUDA Graph warnings).
    # SDPA (FlashAttention) already provides the main kernel-level speedup.
    
    # Wrap with DDP
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # 5. Optimizer
    # Scale LR based on World Size? Some papers do, some don't.
    # Paper uses 6e-4 for Total Batch Size 256.
    # With 2x4090, batch ~32-64. 6e-4 is likely safe, or slightly lower.
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
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)
    scaler = torch.amp.GradScaler('cuda')
    
    plotter = CurvePlotter(curve_dir) if rank == 0 else None
    best_acc = 0.0
    start_epoch = 1  # Default: start from epoch 1

    # --- Resume from checkpoint ---
    if args.resume:
        if rank == 0:
            logging.info(f"Resuming training from checkpoint: {args.resume}")
        
        checkpoint = load_checkpoint(
            checkpoint_path=args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device
        )
        
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_acc = checkpoint.get('acc', 0.0)

        if rank == 0:
            logging.info(f"Resume successful. Starting from Epoch {start_epoch}, Best Acc: {best_acc:.2f}%")


    # 6. Loop
    if rank == 0: 
        logging.info("Start DDP Training...")
        start_time_total = time.time() # Record total start time

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start_time = time.time() # Record epoch start time
        
        # Note: train_one_epoch takes an additional rank argument for DDP
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, scaler, 
            mixup_fn, device, epoch, world_size, rank, accum_steps=args.accum_steps
        )
        
        val_loss, val_acc = validate(model, val_loader, criterion, device, world_size)
        
        epoch_duration = time.time() - epoch_start_time
        
        if rank == 0:
            # Calculate estimated remaining time
            remaining_epochs = args.epochs - epoch
            estimated_remaining_time = remaining_epochs * epoch_duration
            eta_str = str(datetime.timedelta(seconds=int(estimated_remaining_time)))
            
            # Log detailed timing stats
            logging.info(f"Epoch [{epoch}/{args.epochs}] Completed in {int(epoch_duration)}s | ETA: {eta_str}")
            logging.info(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
            plotter.update(epoch, train_loss, val_loss, val_acc)
            
            # Save Checkpoints
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
            
            if val_acc > best_acc:
                best_acc = val_acc
                logging.info(f"New Best Accuracy: {best_acc:.2f}%")
                # Save best model (raw state_dict only for smaller file size)
                torch.save(state['model_state_dict'], ckpt_dir / "best_model.pth")

            if epoch % 10 == 0:
                torch.save(state['model_state_dict'], ckpt_dir / f"epoch_{epoch}.pth")

    if rank == 0:
        total_time = str(datetime.timedelta(seconds=int(time.time() - start_time_total)))
        logging.info(f"Training Complete. Total Time: {total_time}. Best Accuracy: {best_acc:.2f}%")

def _cleanup():
    """Clean up DDP process group for graceful shutdown."""
    if dist.is_initialized():
        dist.destroy_process_group()


def _signal_handler(signum, frame):
    """Handle Ctrl+C (SIGINT) and SIGTERM for clean exit."""
    _cleanup()
    sys.exit(1)


if __name__ == "__main__":
    # Register signal handlers BEFORE main() — ensures Ctrl+C works even
    # when the process is blocked in NCCL collectives or DataLoader workers.
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.", flush=True)
    finally:
        _cleanup()