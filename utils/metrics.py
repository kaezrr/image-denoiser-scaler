"""
utils/metrics.py
================
Image quality metrics for evaluating super-resolution models.

Implements:
  - PSNR  (Peak Signal-to-Noise Ratio)
  - SSIM  (Structural Similarity Index Measure)
  - AverageMeter  (convenience class for tracking running averages)

Both metrics follow the standard SR evaluation protocol:
  - Compute on the Y channel (luminance) of YCbCr colour space, not RGB.
  - Crop the border by `scale` pixels to avoid boundary artefacts.
  - Input tensors in [0, 1] range; internally scaled to [0, 255] for PSNR.

Why Y channel only?
  Human perception is much more sensitive to luminance (Y) changes than
  colour (CbCr) changes. All SR papers report metrics on Y to be fair and
  directly comparable to each other.

Why crop the border?
  Upsampling methods introduce artefacts at image boundaries. Cropping
  `scale` pixels from each edge removes these from the evaluation.
"""

import math
import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Union


# ---------------------------------------------------------------------------
# RGB → Y channel conversion
# ---------------------------------------------------------------------------
def rgb_to_y(img: torch.Tensor) -> torch.Tensor:
    """
    Convert an RGB tensor to the Y (luminance) channel of YCbCr.

    Follows the standard BT.601 conversion:
        Y = 16 + 65.481*R + 128.553*G + 24.966*B  (in [0,255] scale)

    Args
    ----
    img : (B, 3, H, W) or (3, H, W), values in [0, 1]

    Returns
    -------
    y   : (B, 1, H, W) or (1, H, W), values in [16/255, 235/255] range
    """
    if img.dim() == 3:
        img = img.unsqueeze(0)  # add batch dim temporarily

    r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
    # Scale to [0, 255] before applying the YCbCr formula
    y = 16.0 + (65.481 * r + 128.553 * g + 24.966 * b) * 255.0 / 255.0
    # Normalise back to [0, 1] range (Y in YCbCr spans [16, 235])
    y = y / 255.0

    return y


# ---------------------------------------------------------------------------
# Border crop
# ---------------------------------------------------------------------------
def crop_border(img: torch.Tensor, border: int) -> torch.Tensor:
    """
    Crop `border` pixels from all four sides of an image tensor.

    Args
    ----
    img    : (B, C, H, W) or (C, H, W)
    border : number of pixels to crop from each edge

    Returns
    -------
    cropped tensor with spatial dimensions reduced by 2*border each side
    """
    if border == 0:
        return img
    if img.dim() == 3:
        return img[:, border:-border, border:-border]
    return img[:, :, border:-border, border:-border]


# ---------------------------------------------------------------------------
# PSNR
# ---------------------------------------------------------------------------
def compute_psnr(
    sr:        torch.Tensor,
    hr:        torch.Tensor,
    max_val:   float = 1.0,
    crop_border: int = 0,
    y_channel: bool = True,
) -> float:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR) in dB.

    PSNR = 10 * log10(MAX^2 / MSE)

    Where:
      MAX = maximum possible pixel value (max_val, typically 1.0)
      MSE = mean squared error between SR and HR

    Higher PSNR = lower distortion. Typical values:
      - Bicubic ×2:  ~33-34 dB
      - RCAN ×2:     ~38-39 dB (paper results)

    Args
    ----
    sr          : (B, 3, H, W) predicted super-resolved image, [0, 1]
    hr          : (B, 3, H, W) ground truth high-res image, [0, 1]
    max_val     : maximum pixel value (1.0 for [0,1] tensors)
    crop_border : pixels to crop from border before computing (default: scale)
    y_channel   : compute on Y channel only (standard SR protocol)

    Returns
    -------
    psnr : float (in dB)
    """
    assert sr.shape == hr.shape, \
        f"SR and HR shapes must match: {sr.shape} vs {hr.shape}"

    if y_channel:
        sr = rgb_to_y(sr)
        hr = rgb_to_y(hr)

    if crop_border > 0:
        sr = crop_border_fn(sr, crop_border)
        hr = crop_border_fn(hr, crop_border)

    mse = torch.mean((sr - hr) ** 2)

    if mse == 0:
        return float("inf")

    psnr = 10.0 * math.log10(max_val ** 2 / mse.item())
    return psnr


# Alias with cleaner name (avoid shadowing built-in)
def crop_border_fn(img: torch.Tensor, border: int) -> torch.Tensor:
    return crop_border(img, border)


# ---------------------------------------------------------------------------
# SSIM
# ---------------------------------------------------------------------------
def _gaussian_kernel(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """
    Create a 2D Gaussian kernel for SSIM computation.

    Uses a separable 1D Gaussian outer-producted to 2D.
    Kernel is normalised to sum to 1.
    """
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel = g.unsqueeze(1) @ g.unsqueeze(0)  # outer product → 2D
    return kernel


def compute_ssim(
    sr:          torch.Tensor,
    hr:          torch.Tensor,
    window_size: int   = 11,
    sigma:       float = 1.5,
    c1:          float = (0.01 ** 2),
    c2:          float = (0.03 ** 2),
    crop_border: int   = 0,
    y_channel:   bool  = True,
) -> float:
    """
    Compute Structural Similarity Index Measure (SSIM).

    SSIM measures perceptual similarity across three dimensions:
      - Luminance:  how similar are the mean brightness values?
      - Contrast:   how similar are the standard deviations?
      - Structure:  how similar is the local correlation pattern?

    SSIM(x, y) = [l(x,y)]^α · [c(x,y)]^β · [s(x,y)]^γ
    With α=β=γ=1, simplifies to:
    SSIM = (2μ_xμ_y + C1)(2σ_xy + C2) / (μ_x² + μ_y² + C1)(σ_x² + σ_y² + C2)

    SSIM ∈ [-1, 1], with 1 = identical images.
    Typical values: bicubic ~0.93, RCAN ~0.96+ (on ×2 SR).

    Args
    ----
    sr          : (B, 3, H, W) SR image, [0, 1]
    hr          : (B, 3, H, W) HR image, [0, 1]
    window_size : Gaussian window size (standard: 11)
    sigma       : Gaussian sigma (standard: 1.5)
    c1, c2      : stability constants (standard: 0.01², 0.03²)
    crop_border : pixels to crop from border
    y_channel   : compute on Y channel only (standard SR protocol)

    Returns
    -------
    ssim : float in [-1, 1]
    """
    assert sr.shape == hr.shape, \
        f"SR and HR shapes must match: {sr.shape} vs {hr.shape}"

    if y_channel:
        sr = rgb_to_y(sr)
        hr = rgb_to_y(hr)

    if crop_border > 0:
        sr = crop_border_fn(sr, crop_border)
        hr = crop_border_fn(hr, crop_border)

    # Build Gaussian window: (1, 1, window_size, window_size)
    kernel_2d = _gaussian_kernel(window_size, sigma).to(sr.device)
    kernel    = kernel_2d.unsqueeze(0).unsqueeze(0)  # (1, 1, W, W)

    # Ensure 4D: (B, C, H, W)
    if sr.dim() == 3:
        sr = sr.unsqueeze(0)
    if hr.dim() == 3:
        hr = hr.unsqueeze(0)

    channels = sr.shape[1]

    # Expand kernel for each channel
    kernel_expanded = kernel.expand(channels, 1, window_size, window_size)

    # Padding to maintain spatial dimensions
    pad = window_size // 2

    # Compute local statistics using Gaussian-weighted convolution
    mu_x  = F.conv2d(sr, kernel_expanded, padding=pad, groups=channels)
    mu_y  = F.conv2d(hr, kernel_expanded, padding=pad, groups=channels)

    mu_x2 = mu_x ** 2
    mu_y2 = mu_y ** 2
    mu_xy = mu_x * mu_y

    # Variance and covariance
    sigma_x2  = F.conv2d(sr * sr, kernel_expanded, padding=pad, groups=channels) - mu_x2
    sigma_y2  = F.conv2d(hr * hr, kernel_expanded, padding=pad, groups=channels) - mu_y2
    sigma_xy  = F.conv2d(sr * hr, kernel_expanded, padding=pad, groups=channels) - mu_xy

    # SSIM formula
    numerator   = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)

    ssim_map = numerator / denominator
    return ssim_map.mean().item()


# ---------------------------------------------------------------------------
# Convenience: compute both metrics at once
# ---------------------------------------------------------------------------
def compute_metrics(
    sr:          torch.Tensor,
    hr:          torch.Tensor,
    scale:       int = 2,
    y_channel:   bool = True,
) -> Tuple[float, float]:
    """
    Compute PSNR and SSIM with the standard SR evaluation protocol:
      - Y channel only
      - Border cropped by `scale` pixels

    Args
    ----
    sr     : (B, 3, H*scale, W*scale) predicted SR image
    hr     : (B, 3, H*scale, W*scale) ground truth HR image
    scale  : SR scale factor (used as crop border size)
    y_channel: use Y channel (default True, follows SR papers)

    Returns
    -------
    (psnr, ssim) : float tuple
    """
    with torch.no_grad():
        psnr = compute_psnr(sr, hr, crop_border=scale, y_channel=y_channel)
        ssim = compute_ssim(sr, hr, crop_border=scale, y_channel=y_channel)
    return psnr, ssim


# ---------------------------------------------------------------------------
# AverageMeter
# ---------------------------------------------------------------------------
class AverageMeter:
    """
    Tracks running average, sum, count, and recent value of a scalar metric.

    Usage:
        meter = AverageMeter("PSNR")
        for batch in dataloader:
            psnr = compute_psnr(...)
            meter.update(psnr)
        print(meter.avg)

    Args
    ----
    name : display name (for logging/printing)
    fmt  : format string for __str__ (default: ":f" for float)
    """

    def __init__(self, name: str = "", fmt: str = ":f"):
        self.name = name
        self.fmt  = fmt
        self.reset()

    def reset(self):
        """Reset all tracked values to zero."""
        self.val   = 0.0
        self.avg   = 0.0
        self.sum   = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        """
        Update with a new value.

        Args
        ----
        val : new value (e.g., PSNR for this batch)
        n   : number of samples this value is averaged over (usually batch_size)
        """
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count

    def __str__(self) -> str:
        fmtstr = f"{{name}} {{val{self.fmt}}} ({{avg{self.fmt}}})"
        return fmtstr.format(**self.__dict__)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Metrics self-test")

    # Identical images → PSNR should be inf, SSIM should be 1.0
    x = torch.rand(2, 3, 96, 96)
    psnr_identical = compute_psnr(x, x, crop_border=2)
    ssim_identical = compute_ssim(x, x, crop_border=2)
    print(f"  Identical: PSNR={psnr_identical}, SSIM={ssim_identical:.4f}")
    assert ssim_identical > 0.999, "SSIM of identical images should be ~1.0"
    print("  ✓ Identical images test passed.")

    # Different images → finite PSNR, SSIM < 1
    y = torch.rand(2, 3, 96, 96)
    psnr_diff, ssim_diff = compute_metrics(x, y, scale=2)
    print(f"  Different: PSNR={psnr_diff:.2f} dB, SSIM={ssim_diff:.4f}")
    assert psnr_diff < 50 and psnr_diff > 0, "PSNR should be finite positive"
    assert 0 < ssim_diff < 1, "SSIM of different images should be in (0, 1)"
    print("  ✓ Different images test passed.")

    # AverageMeter test
    meter = AverageMeter("PSNR")
    for v in [30.0, 32.0, 34.0]:
        meter.update(v)
    print(f"  AverageMeter avg: {meter.avg:.2f}  (expected: 32.00)")
    assert abs(meter.avg - 32.0) < 1e-6
    print("  ✓ AverageMeter test passed.")
    print("\nAll metrics tests passed.")
