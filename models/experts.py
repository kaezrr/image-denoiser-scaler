"""
models/experts.py
Loads pretrained frozen SR experts and wraps them in a unified interface.

Expert 1: Real-ESRGAN (x4plus) — strong perceptual quality, handles real degradations
Expert 2: EDSR-base x4 (from super-image HuggingFace) — faithful reconstruction, high PSNR
Expert 3: Bicubic — fast baseline, included so fusion can learn to ignore it in hard cases

All experts:
  - Input:  (B, 3, H, W) float32 tensor in [0, 1]
  - Output: (B, 3, H*4, W*4) float32 tensor in [0, 1]
  - Parameters: ALL FROZEN (requires_grad=False)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image


# ── Expert 1: Real-ESRGAN ────────────────────────────────────────────────────

class RealESRGANExpert(nn.Module):
    """
    Wraps Real-ESRGAN x4plus pretrained model.
    Downloads weights automatically on first use (~67MB).
    Input/output: (B, 3, H, W) float tensor in [0, 1].
    """
    def __init__(self, device):
        super().__init__()
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                        num_block=23, num_grow_ch=32, scale=4)
        self.upsampler = RealESRGANer(
            scale=4, model_path="weights/RealESRGAN_x4plus.pth",
            model=model, tile=0, tile_pad=10, pre_pad=0,
            half=(device.type == "cuda"), device=device,
        )
        # Freeze
        for p in self.upsampler.model.parameters():
            p.requires_grad_(False)
        self._device = device

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process batch by iterating (RealESRGAN doesn't support batching natively)."""
        results = []
        for img_t in x:
            # tensor (3,H,W) [0,1] -> numpy uint8 BGR
            np_img = (img_t.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            np_bgr = np_img[:, :, ::-1].copy()
            sr_bgr, _ = self.upsampler.enhance(np_bgr, outscale=4)
            sr_rgb = sr_bgr[:, :, ::-1].astype(np.float32) / 255.0
            results.append(torch.from_numpy(sr_rgb).permute(2, 0, 1))
        return torch.stack(results).to(self._device)


# ── Expert 2: EDSR (via super-image / HuggingFace) ───────────────────────────

class EDSRExpert(nn.Module):
    """
    Wraps EDSR-base x4 from HuggingFace super-image.
    Downloads weights automatically (~5MB).
    Input/output: (B, 3, H, W) float tensor in [0, 1].
    """
    def __init__(self, device):
        super().__init__()
        from super_image import EdsrModel
        self.model = EdsrModel.from_pretrained("eugenesiow/edsr-base", scale=4).to(device)
        self._device = device
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ── Expert 3: Bicubic (no weights needed) ────────────────────────────────────

class BicubicExpert(nn.Module):
    """Bicubic upsampling baseline. No parameters."""
    def __init__(self, scale: int = 4):
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False)


# ── Expert Pool ───────────────────────────────────────────────────────────────

class ExpertPool(nn.Module):
    """
    Holds all frozen experts. Runs inference and returns stacked outputs.
    Output: (B, 3*num_experts, H*scale, W*scale)
    """
    def __init__(self, cfg: dict, device):
        super().__init__()
        self._device = device
        scale = cfg["data"]["scale"]

        print("Loading Expert 1: Real-ESRGAN …")
        self.expert1 = RealESRGANExpert(device)

        print("Loading Expert 2: EDSR-base x4 …")
        self.expert2 = EDSRExpert(device)

        print("Loading Expert 3: Bicubic baseline …")
        self.expert3 = BicubicExpert(scale)

        # Freeze everything
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, lr: torch.Tensor):
        """
        Args:
            lr: (B, 3, H, W) LR input in [0, 1]
        Returns:
            stacked: (B, 9, H*4, W*4) — concatenation of all expert outputs
            outputs: list of (B, 3, H*4, W*4) individual expert outputs
        """
        o1 = self.expert1(lr).clamp(0, 1)
        o2 = self.expert2(lr).clamp(0, 1)
        o3 = self.expert3(lr).clamp(0, 1)
        outputs = [o1, o2, o3]
        return torch.cat(outputs, dim=1), outputs


def download_realesrgan_weights():
    """Download Real-ESRGAN x4plus weights if not present."""
    import os, requests
    from pathlib import Path
    url  = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
    dest = Path("weights/RealESRGAN_x4plus.pth")
    if dest.exists():
        print(f"✓ Real-ESRGAN weights already present: {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading Real-ESRGAN weights (~67MB)…")
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in r.iter_content(1 << 20):
            f.write(chunk)
    print(f"✓ Saved: {dest}")
