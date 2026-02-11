"""
evaluate.py
===========
Evaluation script for a trained RCAN checkpoint.

Computes:
  - Average PSNR (dB) over the validation set
  - Average SSIM over the validation set
  - Bicubic baseline PSNR/SSIM for comparison

Saves visual comparison grids:
  [LR input | Bicubic upscale | RCAN output | HR ground truth]

Usage:
    python evaluate.py --checkpoint outputs/checkpoints/best_model.pth
    python evaluate.py --checkpoint best_model.pth --num_samples 16
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

import configs.rcan_config as config
from models.rcan      import build_rcan
from data.dataset     import build_dataset
from utils.metrics    import compute_metrics, AverageMeter
from utils.checkpoint import load_checkpoint


# ---------------------------------------------------------------------------
# Bicubic upscaling baseline
# ---------------------------------------------------------------------------
def bicubic_upsample(lr: torch.Tensor, scale: int) -> torch.Tensor:
    """
    Upsample LR tensor using bicubic interpolation (non-learned baseline).

    This represents the "dumb" upscaling that any image viewer can do.
    Comparing RCAN vs. bicubic quantifies how much value the model adds.

    Args
    ----
    lr    : (B, 3, H, W) LR tensor
    scale : upscale factor

    Returns
    -------
    upsampled : (B, 3, H*scale, W*scale) tensor
    """
    return torch.nn.functional.interpolate(
        lr,
        scale_factor  = scale,
        mode          = "bicubic",
        align_corners = False,
    )


# ---------------------------------------------------------------------------
# Save visual comparison grid
# ---------------------------------------------------------------------------
def save_comparison(
    lr:       torch.Tensor,
    bicubic:  torch.Tensor,
    sr:       torch.Tensor,
    hr:       torch.Tensor,
    save_path: str,
    psnr:     float,
    ssim:     float,
):
    """
    Save a side-by-side comparison image: LR | Bicubic | RCAN | HR.

    Note: LR is shown upscaled to HR size for visual alignment.

    Args
    ----
    lr, bicubic, sr, hr : single-image tensors (3, H, W)
    save_path : where to save the PNG
    psnr, ssim : metrics for this image (shown in filename)
    """
    def to_pil(t: torch.Tensor) -> Image.Image:
        """Convert (3, H, W) tensor [0,1] to PIL Image."""
        arr = (t.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
        return Image.fromarray(arr)

    h, w = hr.shape[1], hr.shape[2]

    # Upscale LR to HR size for visual comparison
    lr_up = torch.nn.functional.interpolate(
        lr.unsqueeze(0), size=(h, w), mode="nearest"
    ).squeeze(0)

    pil_lr      = to_pil(lr_up)
    pil_bicubic = to_pil(bicubic)
    pil_sr      = to_pil(sr)
    pil_hr      = to_pil(hr)

    # Create horizontal strip: [LR | Bicubic | RCAN | HR]
    strip = Image.new("RGB", (w * 4, h))
    strip.paste(pil_lr,      (0,       0))
    strip.paste(pil_bicubic, (w,       0))
    strip.paste(pil_sr,      (w * 2,   0))
    strip.paste(pil_hr,      (w * 3,   0))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    strip.save(save_path)


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(checkpoint_path: str, num_samples: int = 8):
    """
    Evaluate a trained RCAN checkpoint on the validation set.

    Args
    ----
    checkpoint_path : path to the .pth checkpoint file
    num_samples     : number of visual comparison images to save
    """
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"\n{'='*60}")
    print(f"RCAN Evaluation")
    print(f"{'='*60}")
    print(f"Checkpoint : {checkpoint_path}")
    print(f"Device     : {device}")

    # -------------------------------------------------------------------------
    # Build and load model
    # -------------------------------------------------------------------------
    model = build_rcan(config).to(device)
    load_checkpoint(checkpoint_path, model, device=str(device))
    model.eval()

    # -------------------------------------------------------------------------
    # Dataset
    # -------------------------------------------------------------------------
    val_dataset = build_dataset(config, split="val")
    val_loader  = DataLoader(val_dataset, batch_size=1, shuffle=False)

    print(f"Val images : {len(val_dataset)}")

    # -------------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------------
    rcan_psnr_meter    = AverageMeter("RCAN PSNR")
    rcan_ssim_meter    = AverageMeter("RCAN SSIM")
    bicubic_psnr_meter = AverageMeter("Bicubic PSNR")
    bicubic_ssim_meter = AverageMeter("Bicubic SSIM")

    sample_dir   = os.path.join(config.SAMPLE_DIR, "eval_comparison")
    samples_saved = 0

    for idx, (lr, hr) in enumerate(val_loader):
        lr = lr.to(device)
        hr = hr.to(device)

        # RCAN prediction
        sr          = model(lr)
        sr_clamped  = torch.clamp(sr, 0, 1)

        # Bicubic baseline
        bicubic     = bicubic_upsample(lr, config.SCALE_FACTOR)
        bicubic_clamped = torch.clamp(bicubic, 0, 1)

        # Compute metrics for this image
        psnr_rcan,    ssim_rcan    = compute_metrics(sr_clamped,      hr, config.SCALE_FACTOR)
        psnr_bicubic, ssim_bicubic = compute_metrics(bicubic_clamped, hr, config.SCALE_FACTOR)

        rcan_psnr_meter.update(psnr_rcan)
        rcan_ssim_meter.update(ssim_rcan)
        bicubic_psnr_meter.update(psnr_bicubic)
        bicubic_ssim_meter.update(ssim_bicubic)

        # Save visual comparison for first `num_samples` images
        if samples_saved < num_samples:
            save_path = os.path.join(
                sample_dir,
                f"sample_{idx:04d}_psnr{psnr_rcan:.2f}_ssim{ssim_rcan:.4f}.png"
            )
            save_comparison(
                lr=lr[0], bicubic=bicubic_clamped[0],
                sr=sr_clamped[0], hr=hr[0],
                save_path=save_path,
                psnr=psnr_rcan, ssim=ssim_rcan,
            )
            samples_saved += 1

    # -------------------------------------------------------------------------
    # Print summary
    # -------------------------------------------------------------------------
    print(f"\n{'─'*60}")
    print(f"{'Method':<20} {'PSNR (dB)':>12} {'SSIM':>12}")
    print(f"{'─'*60}")
    print(f"{'Bicubic':<20} {bicubic_psnr_meter.avg:>12.2f} {bicubic_ssim_meter.avg:>12.4f}")
    print(f"{'RCAN':<20} {rcan_psnr_meter.avg:>12.2f} {rcan_ssim_meter.avg:>12.4f}")
    print(f"{'─'*60}")

    psnr_gain = rcan_psnr_meter.avg - bicubic_psnr_meter.avg
    ssim_gain = rcan_ssim_meter.avg - bicubic_ssim_meter.avg
    print(f"{'RCAN gain':<20} {psnr_gain:>+12.2f} {ssim_gain:>+12.4f}")
    print(f"{'─'*60}")

    print(f"\nVisual comparisons saved to: {sample_dir}")
    print("(Each image shows: LR → Bicubic → RCAN → HR ground truth)")

    return {
        "rcan_psnr":    rcan_psnr_meter.avg,
        "rcan_ssim":    rcan_ssim_meter.avg,
        "bicubic_psnr": bicubic_psnr_meter.avg,
        "bicubic_ssim": bicubic_ssim_meter.avg,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RCAN checkpoint")
    parser.add_argument(
        "--checkpoint",
        type     = str,
        required = True,
        help     = "Path to the trained model checkpoint (.pth)"
    )
    parser.add_argument(
        "--num_samples",
        type    = int,
        default = 8,
        help    = "Number of visual comparison images to save (default: 8)"
    )
    args = parser.parse_args()

    results = evaluate(args.checkpoint, args.num_samples)
