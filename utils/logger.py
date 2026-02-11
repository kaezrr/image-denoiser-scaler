"""
utils/logger.py
===============
Training logger that writes metrics to:
  - Console (human-readable)
  - CSV file  (for plotting in Excel / matplotlib later)
  - JSON summary file

Designed to be lightweight and dependency-free (no TensorBoard required).
"""

import os
import csv
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional


class TrainingLogger:
    """
    Logs training and validation metrics to console and CSV.

    Creates two files:
      - {log_dir}/train_log.csv   : per-epoch train metrics
      - {log_dir}/val_log.csv     : per-epoch validation metrics
      - {log_dir}/summary.json    : final training summary

    Args
    ----
    log_dir    : directory to write log files to
    experiment : name for this training run (used in summary)
    """

    # Column names for CSV files
    TRAIN_COLUMNS = ["epoch", "loss_total", "loss_l1", "lr", "epoch_time_s"]
    VAL_COLUMNS   = ["epoch", "psnr", "ssim", "loss_total"]

    def __init__(self, log_dir: str, experiment: str = "rcan_training"):
        self.log_dir    = log_dir
        self.experiment = experiment
        self.start_time = time.time()

        os.makedirs(log_dir, exist_ok=True)

        self.train_csv_path = os.path.join(log_dir, "train_log.csv")
        self.val_csv_path   = os.path.join(log_dir, "val_log.csv")
        self.summary_path   = os.path.join(log_dir, "summary.json")

        self._init_csv(self.train_csv_path, self.TRAIN_COLUMNS)
        self._init_csv(self.val_csv_path,   self.VAL_COLUMNS)

        self.best_psnr   = 0.0
        self.best_epoch  = 0
        self.epoch_count = 0

    def _init_csv(self, path: str, columns: list):
        """Create CSV file with header row (only if it doesn't exist)."""
        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=columns)
                writer.writeheader()

    def log_train(
        self,
        epoch:       int,
        loss_dict:   Dict[str, float],
        lr:          float,
        epoch_time:  float,
    ):
        """
        Log training metrics for one epoch.

        Args
        ----
        epoch      : current epoch number
        loss_dict  : dict with 'total', 'l1' (and optionally 'perceptual')
        lr         : current learning rate
        epoch_time : time taken for this epoch in seconds
        """
        row = {
            "epoch":        epoch,
            "loss_total":   f"{loss_dict.get('total', 0):.6f}",
            "loss_l1":      f"{loss_dict.get('l1', 0):.6f}",
            "lr":           f"{lr:.2e}",
            "epoch_time_s": f"{epoch_time:.1f}",
        }
        with open(self.train_csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.TRAIN_COLUMNS)
            writer.writerow(row)

        # Console output
        print(
            f"  [Epoch {epoch:4d}] "
            f"Loss: {loss_dict.get('total', 0):.4f} | "
            f"LR: {lr:.2e} | "
            f"Time: {epoch_time:.1f}s"
        )

    def log_val(
        self,
        epoch:     int,
        psnr:      float,
        ssim:      float,
        loss_dict: Dict[str, float],
    ):
        """
        Log validation metrics for one epoch.

        Args
        ----
        epoch     : current epoch number
        psnr      : validation PSNR (dB)
        ssim      : validation SSIM [0, 1]
        loss_dict : dict with 'total' key
        """
        is_best = psnr > self.best_psnr
        if is_best:
            self.best_psnr  = psnr
            self.best_epoch = epoch

        row = {
            "epoch":      epoch,
            "psnr":       f"{psnr:.4f}",
            "ssim":       f"{ssim:.6f}",
            "loss_total": f"{loss_dict.get('total', 0):.6f}",
        }
        with open(self.val_csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.VAL_COLUMNS)
            writer.writerow(row)

        best_marker = "  ← BEST" if is_best else ""
        print(
            f"  [Val   {epoch:4d}] "
            f"PSNR: {psnr:.2f} dB | "
            f"SSIM: {ssim:.4f}{best_marker}"
        )

    def save_summary(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        Save a JSON summary of the training run.

        Args
        ----
        config_dict : optional dict of config values to include in the summary
        """
        total_time = time.time() - self.start_time
        summary = {
            "experiment":     self.experiment,
            "completed_at":   datetime.now().isoformat(),
            "total_time_min": round(total_time / 60, 2),
            "best_psnr_db":   round(self.best_psnr, 4),
            "best_epoch":     self.best_epoch,
            "log_dir":        self.log_dir,
        }
        if config_dict:
            summary["config"] = config_dict

        with open(self.summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'='*60}")
        print(f"Training complete.")
        print(f"  Best PSNR  : {self.best_psnr:.2f} dB (epoch {self.best_epoch})")
        print(f"  Total time : {total_time/60:.1f} min")
        print(f"  Summary    : {self.summary_path}")
        print(f"{'='*60}")
