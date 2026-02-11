"""
train.py
========
Main training script for the RCAN super-resolution model.

Usage:
    python train.py                          # Default config (BSD500, all defaults)
    python train.py --config custom_conf.py  # Custom config file

The training loop:
  1. Load dataset (BSD500 by default; swap to DIV2K in config)
  2. Build RCAN model, optimizer, LR scheduler, and loss function
  3. For each epoch:
       a. Train: forward → compute loss → backward → optimizer step
       b. Validate every 5 epochs: compute PSNR/SSIM on val set
       c. Save checkpoint every SAVE_EVERY epochs
       d. Save best model (by val PSNR) whenever a new best is found
  4. Save training summary (CSV + JSON logs)

Resuming:
    Set RESUME_CHECKPOINT in config to a .pth path, and training will
    pick up from where it left off (epoch, optimizer state, etc.).

Hardware:
    Automatically uses CUDA if available, else CPU.
    To force CPU: set DEVICE = "cpu" in config.
"""

import os
import sys
import time
import random
import argparse
import importlib.util
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to path so relative imports work
sys.path.insert(0, str(Path(__file__).parent))

import configs.rcan_config as config

from models.rcan import build_rcan
from models.losses import build_loss
from data.dataset import build_dataset
from utils.metrics import compute_metrics, AverageMeter
from utils.checkpoint import save_checkpoint, load_checkpoint, BestCheckpointTracker
from utils.logger import TrainingLogger


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int):
    """Fix all random seeds for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Deterministic ops (may slow training slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Training step (one epoch)
# ---------------------------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    device: torch.device,
    epoch: int,
    logger: TrainingLogger,
) -> dict:
    """
    Run one full training epoch.

    For each batch:
      1. Move LR and HR tensors to device
      2. Forward pass: model(lr) → sr
      3. Compute loss(sr, hr)
      4. Backward pass: loss.backward()
      5. Gradient clipping (prevents exploding gradients in deep networks)
      6. Optimizer step

    Args
    ----
    model      : RCAN model
    dataloader : training DataLoader
    optimizer  : Adam optimizer
    loss_fn    : CombinedLoss instance
    device     : torch.device
    epoch      : current epoch number (for logging)
    logger     : TrainingLogger

    Returns
    -------
    avg_losses : dict with average 'total' and 'l1' losses for this epoch
    """
    model.train()

    loss_meter = AverageMeter("Loss")
    n_batches = len(dataloader)

    for batch_idx, (lr, hr) in enumerate(dataloader):
        lr = lr.to(device, non_blocking=True)
        hr = hr.to(device, non_blocking=True)

        # Forward pass
        sr = model(lr)

        # Compute loss
        total_loss, loss_dict = loss_fn(sr, hr)

        # Backward + gradient clip + step
        optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping: prevents very large gradients in 400+ layer network
        # Max norm of 1.0 is a safe default for SR models.
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        loss_meter.update(loss_dict["total"], n=lr.size(0))

        # Print progress every 10% of batches
        if (batch_idx + 1) % max(1, n_batches // 10) == 0:
            print(
                f"    Batch [{batch_idx+1}/{n_batches}] "
                f"Loss: {loss_dict['total']:.4f} "
                f"(avg: {loss_meter.avg:.4f})"
            )

    return {"total": loss_meter.avg, "l1": loss_meter.avg}


# ---------------------------------------------------------------------------
# Validation step
# ---------------------------------------------------------------------------
@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn,
    device: torch.device,
    scale: int,
) -> tuple:
    """
    Run validation: compute average PSNR, SSIM, and loss over the val set.

    The @torch.no_grad() decorator disables gradient computation, saving
    memory and speeding up inference.

    Args
    ----
    model      : RCAN model
    dataloader : validation DataLoader
    loss_fn    : loss function
    device     : torch.device
    scale      : SR scale factor (for border crop in metrics)

    Returns
    -------
    (avg_psnr, avg_ssim, avg_loss_dict) : tuple
    """
    model.eval()

    psnr_meter = AverageMeter("PSNR")
    ssim_meter = AverageMeter("SSIM")
    loss_meter = AverageMeter("Loss")

    for lr, hr in dataloader:
        lr = lr.to(device)
        hr = hr.to(device)

        sr = model(lr)

        # Clamp to valid range before computing metrics
        sr_clamped = torch.clamp(sr, 0, 1)

        # Compute metrics
        psnr, ssim = compute_metrics(sr_clamped, hr, scale=scale)
        psnr_meter.update(psnr, n=lr.size(0))
        ssim_meter.update(ssim, n=lr.size(0))

        # Compute loss
        _, loss_dict = loss_fn(sr, hr)
        loss_meter.update(loss_dict["total"], n=lr.size(0))

    return psnr_meter.avg, ssim_meter.avg, {"total": loss_meter.avg}


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------
def train():
    """Main training entry point."""

    # -------------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------------
    set_seed(config.SEED)

    # Device selection
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"\n{'='*60}")
    print(f"RCAN Training — Phase 1 (Microscope)")
    print(f"{'='*60}")
    print(f"Device      : {device}")
    print(f"Dataset     : {config.DATASET_NAME}")
    print(f"Scale       : ×{config.SCALE_FACTOR}")
    print(f"Epochs      : {config.NUM_EPOCHS}")
    print(f"Batch size  : {config.BATCH_SIZE}")
    print(f"LR          : {config.LEARNING_RATE}")

    # -------------------------------------------------------------------------
    # Create output directories
    # -------------------------------------------------------------------------
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)
    os.makedirs(config.SAMPLE_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # Datasets and DataLoaders
    # -------------------------------------------------------------------------
    print("\nLoading datasets...")
    train_dataset = build_dataset(config, split="train")
    val_dataset = build_dataset(config, split="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY and device.type == "cuda",
        drop_last=True,  # Drop last batch if smaller than batch_size
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # Val one image at a time for precise metrics
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=False,
    )

    print(
        f"  Train: {len(train_dataset)} samples, " f"{len(train_loader)} batches/epoch"
    )
    print(f"  Val  : {len(val_dataset)} samples")

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------
    print("\nBuilding RCAN model...")
    model = build_rcan(config).to(device)
    n_params = model.count_parameters()
    print(f"  Parameters: {n_params:,}")
    print(
        f"  Architecture: {config.N_RESGROUPS} groups × "
        f"{config.N_RESBLOCKS} RCAB blocks × {config.N_FEATS} channels"
    )

    # -------------------------------------------------------------------------
    # Loss, Optimizer, Scheduler
    # -------------------------------------------------------------------------
    loss_fn = build_loss(config)

    # Adam is the standard optimizer for SR. Betas and LR from the paper.
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        betas=(config.ADAM_BETA1, config.ADAM_BETA2),
    )

    # StepLR: halve the learning rate every LR_DECAY_STEP epochs.
    # This is exactly the paper's schedule (LR halved every 200 epochs).
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.LR_DECAY_STEP,
        gamma=config.LR_DECAY_GAMMA,
    )

    # -------------------------------------------------------------------------
    # Resume from checkpoint (optional)
    # -------------------------------------------------------------------------
    start_epoch = 1
    resume_path = getattr(config, "RESUME_CHECKPOINT", None)
    if resume_path and os.path.exists(resume_path):
        print(f"\nResuming from: {resume_path}")
        ckpt = load_checkpoint(resume_path, model, optimizer, str(device))
        start_epoch = ckpt.get("epoch", 0) + 1
        # Advance scheduler to match resumed epoch
        for _ in range(start_epoch - 1):
            scheduler.step()

    # -------------------------------------------------------------------------
    # Logging and checkpoint tracking
    # -------------------------------------------------------------------------
    logger = TrainingLogger(config.LOG_DIR)
    tracker = BestCheckpointTracker(config.CHECKPOINT_DIR, config.MAX_KEEP_CHECKPOINTS)

    # -------------------------------------------------------------------------
    # Training Loop
    # -------------------------------------------------------------------------
    print(f"\n{'─'*60}")
    print(f"Starting training from epoch {start_epoch}...")
    print(f"{'─'*60}\n")

    for epoch in range(start_epoch, config.NUM_EPOCHS + 1):
        epoch_start = time.time()

        # --- Train ---
        print(f"Epoch {epoch}/{config.NUM_EPOCHS}")
        train_losses = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, epoch, logger
        )

        # Advance LR scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        epoch_time = time.time() - epoch_start

        # --- Log training metrics ---
        logger.log_train(epoch, train_losses, current_lr, epoch_time)

        # --- Validate every 5 epochs (or on last epoch) ---
        if epoch % 5 == 0 or epoch == config.NUM_EPOCHS:
            val_psnr, val_ssim, val_losses = validate(
                model, val_loader, loss_fn, device, config.SCALE_FACTOR
            )
            logger.log_val(epoch, val_psnr, val_ssim, val_losses)

            # --- Save checkpoint ---
            if epoch % config.SAVE_EVERY == 0 or epoch == config.NUM_EPOCHS:
                ckpt_filename = f"checkpoint_epoch{epoch:04d}_psnr{val_psnr:.2f}.pth"
                ckpt_path = save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    metrics={"psnr": val_psnr, "ssim": val_ssim},
                    checkpoint_dir=config.CHECKPOINT_DIR,
                    filename=ckpt_filename,
                    is_best=tracker.update(val_psnr, ckpt_filename),
                )

        print()  # blank line between epochs

    # -------------------------------------------------------------------------
    # Final summary
    # -------------------------------------------------------------------------
    logger.save_summary(
        {
            "dataset": config.DATASET_NAME,
            "scale": config.SCALE_FACTOR,
            "n_resgroups": config.N_RESGROUPS,
            "n_resblocks": config.N_RESBLOCKS,
            "n_feats": config.N_FEATS,
            "epochs": config.NUM_EPOCHS,
            "batch_size": config.BATCH_SIZE,
        }
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RCAN for image SR")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to a custom config .py file. "
        "Defaults to configs/rcan_config.py",
    )
    args = parser.parse_args()

    if args.config:
        # Load a custom config module if provided
        spec = importlib.util.spec_from_file_location("custom_config", args.config)
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)
        print(f"Using custom config: {args.config}")

    train()
