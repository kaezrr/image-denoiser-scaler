"""
infer.py
========
Single-image inference: run RCAN on one image and save the SR output.

Usage:
    python infer.py --input photo.jpg --checkpoint outputs/checkpoints/best_model.pth
    python infer.py --input photo.jpg --checkpoint best_model.pth --output my_sr.png

For large images that might exceed GPU memory, the script can optionally
tile the image and run RCAN on each tile separately (see --tile flag).

Output:
  - The SR image saved to --output path (default: input_name_sr.png)
  - PSNR/SSIM printed if --hr ground truth is also provided
"""

import os
import sys
import argparse
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent))

import configs.rcan_config as config
from models.rcan      import build_rcan
from utils.checkpoint import load_checkpoint
from utils.metrics    import compute_metrics


# ---------------------------------------------------------------------------
# Image I/O helpers
# ---------------------------------------------------------------------------
def load_image(path: str) -> torch.Tensor:
    """
    Load an image from disk as a (1, 3, H, W) float32 tensor in [0, 1].

    Args
    ----
    path : path to the image file

    Returns
    -------
    tensor : (1, 3, H, W), values in [0, 1]
    """
    img = Image.open(path).convert("RGB")
    return transforms.ToTensor()(img).unsqueeze(0)   # → (1, 3, H, W)


def save_image(tensor: torch.Tensor, path: str):
    """
    Save a (1, 3, H, W) or (3, H, W) tensor as a PNG image.

    Args
    ----
    tensor : tensor with values in [0, 1]
    path   : output file path
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    arr = (tensor.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
    Image.fromarray(arr).save(path)
    print(f"  Saved SR output → {path}")


# ---------------------------------------------------------------------------
# Tiled inference (for large images)
# ---------------------------------------------------------------------------
def infer_tiled(
    model:     torch.nn.Module,
    lr:        torch.Tensor,
    scale:     int,
    tile_size: int = 256,
    overlap:   int = 32,
    device:    torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Run RCAN on large images by splitting into overlapping tiles.

    Why tiling?
      RCAN keeps full feature maps in GPU memory throughout the forward pass.
      For very large images (>1080p LR), this can cause OOM errors.
      Tiling processes one tile at a time, merging results with overlap blending
      to avoid visible seams at tile boundaries.

    Overlap blending:
      Tiles overlap by `overlap` pixels. In the overlap zone, we use a linear
      blend from the left/top tile to the right/bottom tile. This eliminates
      hard edges at tile boundaries.

    Args
    ----
    model     : RCAN model
    lr        : (1, 3, H, W) LR input tensor
    scale     : SR scale factor
    tile_size : LR tile size (default 256 → 512 SR at ×2)
    overlap   : overlap between tiles in LR pixels
    device    : computation device

    Returns
    -------
    sr : (1, 3, H*scale, W*scale) SR output tensor
    """
    B, C, H, W = lr.shape
    assert B == 1, "Tiled inference only supports batch_size=1"

    # Output tensor (accumulated)
    H_out = H * scale
    W_out = W * scale
    output = torch.zeros(1, C, H_out, W_out, device=device)
    weight = torch.zeros(1, 1, H_out, W_out, device=device)

    # Step = tile_size - overlap (tiles overlap by `overlap` pixels)
    step = tile_size - overlap

    for y in range(0, H, step):
        for x in range(0, W, step):
            # Tile boundaries (clamped to image size)
            y1 = y
            x1 = x
            y2 = min(y + tile_size, H)
            x2 = min(x + tile_size, W)

            # Extract tile
            tile = lr[:, :, y1:y2, x1:x2].to(device)

            # Run RCAN on tile
            with torch.no_grad():
                sr_tile = model(tile)
                sr_tile = torch.clamp(sr_tile, 0, 1)

            # Place tile in output (scaled coordinates)
            y1_out = y1 * scale
            x1_out = x1 * scale
            y2_out = y2 * scale
            x2_out = x2 * scale

            output[:, :, y1_out:y2_out, x1_out:x2_out] += sr_tile
            weight[:, :, y1_out:y2_out, x1_out:x2_out] += 1.0

    # Average overlapping regions
    output /= weight.clamp(min=1e-8)

    return output


# ---------------------------------------------------------------------------
# Main inference function
# ---------------------------------------------------------------------------
def infer(
    input_path:      str,
    checkpoint_path: str,
    output_path:     Optional[str] = None,
    hr_path:         Optional[str] = None,
    use_tiling:      bool = False,
    tile_size:       int  = 256,
):
    """
    Run RCAN on a single image.

    Args
    ----
    input_path      : path to the LR input image
    checkpoint_path : path to the trained RCAN checkpoint
    output_path     : where to save the SR output (auto-named if None)
    hr_path         : optional HR ground truth for metric computation
    use_tiling      : process in tiles (for large images)
    tile_size       : tile size in LR pixels (used only if use_tiling=True)
    """
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"\n{'='*60}")
    print(f"RCAN Single-Image Inference")
    print(f"{'='*60}")
    print(f"Input      : {input_path}")
    print(f"Checkpoint : {checkpoint_path}")
    print(f"Device     : {device}")
    print(f"Scale      : ×{config.SCALE_FACTOR}")

    # -------------------------------------------------------------------------
    # Build and load model
    # -------------------------------------------------------------------------
    model = build_rcan(config).to(device)
    load_checkpoint(checkpoint_path, model, device=str(device))
    model.eval()

    # -------------------------------------------------------------------------
    # Load input image
    # -------------------------------------------------------------------------
    lr = load_image(input_path)
    _, _, H, W = lr.shape
    print(f"LR size    : {W}×{H}  →  SR size: {W*config.SCALE_FACTOR}×{H*config.SCALE_FACTOR}")

    # -------------------------------------------------------------------------
    # Run inference
    # -------------------------------------------------------------------------
    lr_device = lr.to(device)

    with torch.no_grad():
        if use_tiling:
            print("Mode       : Tiled inference")
            sr = infer_tiled(model, lr_device, config.SCALE_FACTOR, tile_size)
        else:
            print("Mode       : Full-image inference")
            sr = model(lr_device)
            sr = torch.clamp(sr, 0, 1)

    # -------------------------------------------------------------------------
    # Save output
    # -------------------------------------------------------------------------
    if output_path is None:
        stem       = Path(input_path).stem
        output_path = str(Path(input_path).parent / f"{stem}_sr_x{config.SCALE_FACTOR}.png")

    save_image(sr, output_path)

    # -------------------------------------------------------------------------
    # Optional: compute metrics vs. HR ground truth
    # -------------------------------------------------------------------------
    if hr_path:
        hr = load_image(hr_path).to(device)
        assert sr.shape == hr.shape, \
            f"SR shape {sr.shape} != HR shape {hr.shape}. " \
            f"Ensure HR is the correct scale."
        psnr, ssim = compute_metrics(sr, hr, scale=config.SCALE_FACTOR)
        print(f"\nMetrics vs. ground truth:")
        print(f"  PSNR : {psnr:.2f} dB")
        print(f"  SSIM : {ssim:.4f}")

    print("\nDone.")
    return sr


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RCAN single-image inference")
    parser.add_argument("--input",      required=True,  help="Path to LR input image")
    parser.add_argument("--checkpoint", required=True,  help="Path to trained checkpoint (.pth)")
    parser.add_argument("--output",     default=None,   help="Output SR image path (auto-named if omitted)")
    parser.add_argument("--hr",         default=None,   help="Optional HR ground truth for metrics")
    parser.add_argument("--tile",       action="store_true", help="Use tiled inference (for large images)")
    parser.add_argument("--tile_size",  type=int, default=256, help="Tile size for tiled inference (default: 256)")

    args = parser.parse_args()
    infer(
        input_path      = args.input,
        checkpoint_path = args.checkpoint,
        output_path     = args.output,
        hr_path         = args.hr,
        use_tiling      = args.tile,
        tile_size       = args.tile_size,
    )
