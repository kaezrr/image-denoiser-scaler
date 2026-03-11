"""
models/metrics.py
PSNR and SSIM. Both take (B, C, H, W) tensors in [0, 1].
"""
import torch
import torch.nn.functional as F


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = F.mse_loss(pred.clamp(0, 1), target.clamp(0, 1))
    return float("inf") if mse == 0 else (10 * torch.log10(1.0 / mse)).item()


def ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> float:
    C1, C2 = 0.01**2, 0.03**2
    B, C, H, W = pred.shape
    sigma  = 1.5
    coords = torch.arange(window_size, dtype=torch.float32, device=pred.device) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2)); g /= g.sum()
    win = (g.unsqueeze(1) * g.unsqueeze(0)).unsqueeze(0).unsqueeze(0).expand(C, 1, -1, -1)
    pad = window_size // 2
    mu_x  = F.conv2d(pred,          win, padding=pad, groups=C)
    mu_y  = F.conv2d(target,        win, padding=pad, groups=C)
    sx    = F.conv2d(pred * pred,   win, padding=pad, groups=C) - mu_x**2
    sy    = F.conv2d(target*target, win, padding=pad, groups=C) - mu_y**2
    sxy   = F.conv2d(pred * target, win, padding=pad, groups=C) - mu_x * mu_y
    num   = (2 * mu_x * mu_y + C1) * (2 * sxy + C2)
    den   = (mu_x**2 + mu_y**2 + C1) * (sx + sy + C2)
    return (num / den).mean().item()
