"""
data/dataset.py
Loads HR images from DIV2K valid_HR, degrades them on-the-fly to produce
(lr_tensor, hr_tensor) pairs. lr is passed through frozen experts at training time.
"""
import io, random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageFilter


# ── Inline degradation ───────────────────────────────────────────────────────

def _noise(img, sigma_range):
    s = random.uniform(*sigma_range)
    a = np.array(img, np.float32)
    return Image.fromarray(np.clip(a + np.random.randn(*a.shape) * s, 0, 255).astype(np.uint8))

def _jpeg(img, q_range):
    q = random.randint(*q_range)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=q); buf.seek(0)
    return Image.open(buf).copy()

def _blur(img, k_range):
    k = random.choice([x for x in range(k_range[0], k_range[1]+1, 2)])
    return img.filter(ImageFilter.GaussianBlur(k / 6.0))

def _downscale(img, scale):
    w, h = img.size
    return img.resize((w // scale, h // scale), Image.BICUBIC)


class FusionDataset(Dataset):
    """
    Returns (lr_tensor, hr_tensor).
    lr_tensor: degraded LR image, shape (3, H/scale, W/scale)
    hr_tensor: clean HR image,    shape (3, H, W)
    """

    EXT = {".png", ".jpg", ".jpeg", ".bmp"}

    def __init__(self, root_dir: str, cfg: dict, indices: list):
        self.deg_cfg    = cfg["degradation"]
        self.scale      = cfg["data"]["scale"]
        self.patch_size = cfg["data"]["patch_size"]
        self.to_tensor  = transforms.ToTensor()

        root = Path(root_dir)
        all_paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in self.EXT)
        assert len(all_paths) > 0, f"No images found in {root_dir}"
        self.paths = [all_paths[i] for i in indices]

    def __len__(self):
        return len(self.paths)

    def _crop(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        s = self.patch_size
        if w < s or h < s:
            img = img.resize((max(w, s), max(h, s)), Image.BICUBIC)
            w, h = img.size
        x, y = random.randint(0, w - s), random.randint(0, h - s)
        return img.crop((x, y, x + s, y + s))

    def _degrade(self, hr: Image.Image) -> Image.Image:
        img = hr.copy()
        fns = [
            lambda x: _noise(x, self.deg_cfg["noise_sigma_range"]),
            lambda x: _jpeg(x,  self.deg_cfg["jpeg_quality_range"]),
            lambda x: _blur(x,  self.deg_cfg["blur_kernel_range"]),
        ]
        n = min(random.randint(1, self.deg_cfg["max_degradations"]), len(fns))
        for fn in random.sample(fns, n):
            img = fn(img)
        return _downscale(img, self.scale)

    def __getitem__(self, idx):
        hr = Image.open(self.paths[idx]).convert("RGB")
        hr = self._crop(hr)
        lr = self._degrade(hr)
        return self.to_tensor(lr), self.to_tensor(hr)


def make_splits(cfg: dict):
    """Return (train_dataset, val_dataset) using the split ratio in config."""
    root   = cfg["data"]["val_dir"]
    split  = cfg["data"]["train_split"]
    paths  = sorted(p for p in Path(root).rglob("*")
                    if p.suffix.lower() in FusionDataset.EXT)
    n      = len(paths)
    k      = int(n * split)
    indices = list(range(n))
    random.seed(42)
    random.shuffle(indices)
    return (FusionDataset(root, cfg, indices[:k]),
            FusionDataset(root, cfg, indices[k:]))
