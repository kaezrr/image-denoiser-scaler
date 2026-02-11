"""
utils/checkpoint.py
===================
Checkpoint management utilities.

Handles:
  - Saving model state, optimizer state, and training metadata
  - Loading checkpoints for resuming training or inference
  - Tracking the best N checkpoints by PSNR and removing old ones
"""

import os
import json
import shutil
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn


def save_checkpoint(
    model:          nn.Module,
    optimizer:      torch.optim.Optimizer,
    epoch:          int,
    metrics:        Dict[str, float],
    checkpoint_dir: str,
    filename:       str = "checkpoint.pth",
    is_best:        bool = False,
) -> str:
    """
    Save a training checkpoint to disk.

    Checkpoint contains:
      - model state_dict          (all weights)
      - optimizer state_dict      (momentum, adaptive LR state)
      - epoch                     (for resuming)
      - metrics                   (PSNR, SSIM, loss at this epoch)
      - metadata                  (timestamp, architecture info)

    Args
    ----
    model          : the RCAN model
    optimizer      : optimizer instance
    epoch          : current epoch number
    metrics        : dict with at minimum {'psnr': float, 'ssim': float}
    checkpoint_dir : directory to save checkpoints
    filename       : filename for the checkpoint
    is_best        : if True, also save a copy as 'best_model.pth'

    Returns
    -------
    path : full path to the saved checkpoint
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    state = {
        "epoch":          epoch,
        "model_state":    model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "metrics":        metrics,
    }

    path = os.path.join(checkpoint_dir, filename)
    torch.save(state, path)

    if is_best:
        best_path = os.path.join(checkpoint_dir, "best_model.pth")
        shutil.copyfile(path, best_path)
        print(f"  ✓ Best model saved → {best_path}  (PSNR: {metrics.get('psnr', 0):.2f} dB)")

    return path


def load_checkpoint(
    checkpoint_path: str,
    model:           nn.Module,
    optimizer:       Optional[torch.optim.Optimizer] = None,
    device:          str = "cpu",
) -> Dict[str, Any]:
    """
    Load a checkpoint from disk.

    Args
    ----
    checkpoint_path : path to the .pth file
    model           : model instance to load weights into
    optimizer       : (optional) optimizer to restore state into
    device          : device to map tensors to

    Returns
    -------
    state dict containing 'epoch', 'metrics', and other metadata
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state"])

    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])

    epoch   = checkpoint.get("epoch", 0)
    metrics = checkpoint.get("metrics", {})

    print(f"  ✓ Checkpoint loaded from: {checkpoint_path}")
    print(f"    Epoch: {epoch} | "
          f"PSNR: {metrics.get('psnr', 'N/A'):.2f} dB | "
          f"SSIM: {metrics.get('ssim', 'N/A'):.4f}")

    return checkpoint


class BestCheckpointTracker:
    """
    Keeps track of the best N checkpoints by PSNR.
    Automatically deletes older checkpoints when the limit is exceeded.

    Args
    ----
    checkpoint_dir  : directory where checkpoints are stored
    max_keep        : maximum number of checkpoints to retain
    metric          : which metric to compare ('psnr' or 'ssim')
    """

    def __init__(
        self,
        checkpoint_dir: str,
        max_keep:       int = 3,
        metric:         str = "psnr",
    ):
        self.checkpoint_dir = checkpoint_dir
        self.max_keep       = max_keep
        self.metric         = metric
        self.history: list  = []  # [(metric_value, filepath), ...]
        self.best_value     = -float("inf")

        os.makedirs(checkpoint_dir, exist_ok=True)

    def update(self, value: float, filepath: str) -> bool:
        """
        Register a new checkpoint and prune old ones if necessary.

        Args
        ----
        value    : metric value for this checkpoint (e.g., PSNR)
        filepath : path to the checkpoint file

        Returns
        -------
        is_best : True if this is the best checkpoint seen so far
        """
        is_best = value > self.best_value
        if is_best:
            self.best_value = value

        self.history.append((value, filepath))
        # Sort descending: best checkpoints first
        self.history.sort(key=lambda x: x[0], reverse=True)

        # Remove checkpoints beyond max_keep
        if len(self.history) > self.max_keep:
            _, old_path = self.history.pop()   # Remove worst
            if os.path.exists(old_path) and "best_model" not in old_path:
                os.remove(old_path)

        return is_best
