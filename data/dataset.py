"""
data/dataset.py
===============
Dataset classes for the RCAN training pipeline.

Two datasets are implemented:
  - BSD500SRDataset  : Uses BSD500 (auto-downloadable via torchvision).
                       Small and fast — perfect for Phase 1 development.
                       ~200 images when using train+val splits.

  - DIV2KDataset     : Uses the DIV2K dataset (must be downloaded manually).
                       800 HR training images — the production benchmark.

Both datasets:
  1. Load a high-resolution (HR) image.
  2. Synthesise a degraded low-resolution (LR) version on the fly:
       a. Add Gaussian noise  (simulates sensor noise → RCAN's "Microscope" role)
       b. Bicubic downsample  (creates the LR input)
  3. Return (lr_patch, hr_patch) tensor pairs for training.

SWAPPING TO DIV2K:
  In configs/rcan_config.py, change:
      DATASET_NAME = "div2k"
      DATA_ROOT    = "/path/to/your/DIV2K"

  DIV2K folder structure expected:
      DIV2K/
        DIV2K_train_HR/   ← 0001.png ... 0800.png
        DIV2K_valid_HR/   ← 0801.png ... 0900.png
"""

import os
import random
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


# ---------------------------------------------------------------------------
# BSD500 Super-Resolution Dataset  (development / small scale)
# ---------------------------------------------------------------------------
class BSD500SRDataset(Dataset):
    """
    BSD500-based super-resolution dataset.

    BSD500 (Berkeley Segmentation Dataset 500) is a classic image quality
    benchmark. It's small (~200 images for training), easy to download via
    torchvision, and widely used for SR testing. It is NOT as comprehensive
    as DIV2K but is ideal for verifying the pipeline quickly.

    Each item returns:
      lr  : torch.Tensor (3, LR_H, LR_W) — noisy, downsampled patch, [0,1]
      hr  : torch.Tensor (3, HR_H, HR_W) — clean, full-resolution patch, [0,1]

    The HR patch size is LR_PATCH_SIZE * SCALE_FACTOR.
    For default settings: LR=48×48, HR=96×96.

    Args
    ----
    root        : path where BSD500 data is/will be stored
    split       : 'train' | 'val' | 'test'
    scale       : SR scale factor
    lr_patch_size : size of the LR crop to extract during training
    noise_sigma_range : (min, max) range for Gaussian noise sigma
    augment     : whether to apply random flip/rotation augmentation
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        scale: int = 2,
        lr_patch_size: int = 48,
        noise_sigma_range: Tuple[float, float] = (0, 50),
        augment: bool = True,
    ):
        super().__init__()
        self.scale = scale
        self.lr_patch_size = lr_patch_size
        self.hr_patch_size = lr_patch_size * scale
        self.noise_sigma_range = noise_sigma_range
        self.augment = augment
        self.split = split

        # Load BSD500 via torchvision — downloads automatically if needed
        self.image_paths = self._load_bsd500_paths(root, split)

        if len(self.image_paths) == 0:
            raise RuntimeError(
                f"No images found for BSD500 split='{split}'. "
                f"Check that torchvision downloaded correctly to: {root}"
            )

        print(
            f"[BSD500SRDataset] split={split} | "
            f"{len(self.image_paths)} images | "
            f"LR patch: {lr_patch_size}×{lr_patch_size} → "
            f"HR patch: {self.hr_patch_size}×{self.hr_patch_size} | "
            f"scale: ×{scale}"
        )

    def _load_bsd500_paths(self, root: str, split: str) -> List[str]:
        """
        Download BSD500 via torchvision and return list of image paths.

        torchvision.datasets.SBDataset wraps the BSD500 boundary dataset,
        but we only need the raw images. We use a simpler approach:
        download via datasets.VOCDetection which includes the BSD images,
        OR fall back to downloading directly from the official URL.

        For simplicity and reliability, we use torchvision's BSDS500 via
        the Cityscapes/SBDataset API, or generate synthetic data if unavailable.
        """
        # Try to use torchvision's built-in dataset loaders
        try:
            from torchvision.datasets import SBDataset

            # SBDataset is the standard torchvision wrapper for BSD500
            # image_set: 'train' (300 images) | 'val' (200 images) | 'test' (200 images)
            ts = split if split in ("train", "val") else "val"
            dataset = SBDataset(root=root, image_set=ts, download=True)
            paths = [dataset.images[i] for i in range(len(dataset))]
            return paths
        except Exception:
            pass

        # Fallback: search for any PNG/JPG files in the root directory
        # This handles the case where the user has manually placed images
        extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
        root_path = Path(root)
        if root_path.exists():
            all_paths = [
                str(p) for p in root_path.rglob("*") if p.suffix.lower() in extensions
            ]
            if all_paths:
                # Simple train/val split by index
                n_total = len(all_paths)
                if split == "train":
                    return all_paths[: int(n_total * 0.8)]
                elif split == "val":
                    return all_paths[int(n_total * 0.8) :]
                else:
                    return all_paths

        # Ultimate fallback: generate a small synthetic dataset so the code
        # runs and can be verified even without network access.
        print(
            f"[BSD500SRDataset] WARNING: Could not download BSD500. "
            f"Using {50 if split=='train' else 10} synthetic images for testing."
        )
        return self._generate_synthetic_paths(root, n=50 if split == "train" else 10)

    def _generate_synthetic_paths(self, root: str, n: int = 50) -> List[str]:
        """
        Generate synthetic test images when the real dataset is unavailable.

        Creates simple gradient/pattern images with enough structure that
        the SR model can actually learn something meaningful from them.
        These are saved to disk so they persist across dataset instantiations.
        """
        synth_dir = Path(root) / "synthetic"
        synth_dir.mkdir(parents=True, exist_ok=True)

        paths = []
        rng = np.random.RandomState(42)

        for i in range(n):
            path = synth_dir / f"synth_{i:04d}.png"
            if not path.exists():
                # Create a 256×256 image with structured noise + gradients
                img = np.zeros((256, 256, 3), dtype=np.uint8)
                # Gradient background
                for c in range(3):
                    base = rng.randint(50, 200)
                    img[:, :, c] = (
                        np.linspace(base, 255 - base, 256).reshape(1, -1)
                        * np.ones((256, 1))
                    ).astype(np.uint8)
                # Add some edges (rectangles) for the model to learn from
                for _ in range(rng.randint(3, 8)):
                    x1, y1 = rng.randint(0, 200, 2)
                    x2, y2 = x1 + rng.randint(20, 60), y1 + rng.randint(20, 60)
                    color = rng.randint(0, 255, 3)
                    img[y1:y2, x1:x2] = color
                Image.fromarray(img).save(path)
            paths.append(str(path))

        return paths

    def _load_image(self, path: str) -> Image.Image:
        """Load an image as RGB PIL image."""
        try:
            return Image.open(path).convert("RGB")
        except Exception as e:
            raise IOError(f"Failed to load image: {path}\nError: {e}")

    def _random_crop(self, img: Image.Image) -> Image.Image:
        """
        Randomly crop a HR_PATCH_SIZE × HR_PATCH_SIZE patch from the image.
        If the image is smaller than needed, it is first resized.
        """
        w, h = img.size
        min_size = self.hr_patch_size

        # Ensure image is large enough to crop from
        if w < min_size or h < min_size:
            scale_up = max(min_size / w, min_size / h) + 0.1
            new_w = int(w * scale_up)
            new_h = int(h * scale_up)
            img = img.resize((new_w, new_h), Image.BICUBIC)
            w, h = img.size

        # Random top-left corner for the crop
        x = random.randint(0, w - min_size)
        y = random.randint(0, h - min_size)

        return img.crop((x, y, x + min_size, y + min_size))

    def _augment(self, hr_patch: Image.Image) -> Image.Image:
        """
        Apply random geometric augmentations to the HR patch.
        Augmentations are applied before downsampling, so they affect both
        HR and the derived LR consistently.

        Augmentations:
          - Random horizontal flip (50% probability)
          - Random vertical flip   (50% probability)
          - Random 90° rotation    (applied 0, 1, 2, or 3 times randomly)
        """
        if random.random() < 0.5:
            hr_patch = hr_patch.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            hr_patch = hr_patch.transpose(Image.FLIP_TOP_BOTTOM)
        # Random 90° rotation
        k = random.randint(0, 3)
        for _ in range(k):
            hr_patch = hr_patch.transpose(Image.ROTATE_90)
        return hr_patch

    def _degrade(self, hr_patch: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Synthesise a degraded LR version of the HR patch.

        Pipeline:
          1. HR PIL image  →  float32 numpy [0, 1]
          2. Add Gaussian noise  (sigma randomly drawn from noise_sigma_range)
          3. Bicubic downsample to LR size
          4. Convert both HR and LR to tensors

        The noise is added to the HR image BEFORE downsampling because:
          - RCAN's "Microscope" role: it must remove fine-grained noise from textures.
          - Noise added before downsampling is partially averaged out during
            downsampling (realistic sensor noise behaviour).

        Returns
        -------
        lr_tensor : (3, LR_H, LR_W) float32 tensor, values in [0,1]
        hr_tensor : (3, HR_H, HR_W) float32 tensor, values in [0,1]
        """
        # HR → numpy float [0, 1]
        hr_np = np.array(hr_patch, dtype=np.float32) / 255.0  # (H, W, 3)

        # 1. Add Gaussian noise to HR (sigma from range, applied in [0,255] scale)
        sigma = random.uniform(*self.noise_sigma_range) / 255.0
        if sigma > 0:
            noise = np.random.randn(*hr_np.shape).astype(np.float32) * sigma
            hr_noisy_np = np.clip(hr_np + noise, 0, 1)
        else:
            hr_noisy_np = hr_np

        # 2. Convert noisy HR to PIL for bicubic downsampling
        hr_noisy_pil = Image.fromarray((hr_noisy_np * 255).astype(np.uint8))
        lr_size = (self.lr_patch_size, self.lr_patch_size)
        lr_pil = hr_noisy_pil.resize(lr_size, Image.BICUBIC)

        # 3. Convert to tensors (C, H, W), values in [0, 1]
        to_tensor = transforms.ToTensor()
        lr_tensor = to_tensor(lr_pil)  # (3, LR_H, LR_W)
        hr_tensor = to_tensor(hr_patch)  # (3, HR_H, HR_W) — clean original

        return lr_tensor, hr_tensor

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        lr : (3, LR_H, LR_W) tensor — degraded low-resolution input
        hr : (3, HR_H, HR_W) tensor — clean high-resolution target
        """
        img = self._load_image(self.image_paths[idx])
        hr_patch = self._random_crop(img)
        if self.augment and self.split == "train":
            hr_patch = self._augment(hr_patch)
        lr, hr = self._degrade(hr_patch)
        return lr, hr


# ---------------------------------------------------------------------------
# DIV2K Dataset (production)
# ---------------------------------------------------------------------------
class DIV2KDataset(Dataset):
    """
    DIV2K super-resolution dataset.

    DIV2K is the standard benchmark for SR models. 800 high-quality 2K images
    for training, 100 for validation. The LR images can either be:
      a) Pre-generated and stored alongside HR (if available), or
      b) Synthesised on-the-fly (this implementation uses option b).

    Download instructions:
      https://data.vision.ee.ethz.ch/cvl/DIV2K/

    Expected folder structure:
        DATA_ROOT/
          DIV2K_train_HR/   ← 0001.png to 0800.png
          DIV2K_valid_HR/   ← 0801.png to 0900.png

    Args
    ----
    root       : path to the DIV2K root folder (containing DIV2K_train_HR etc.)
    split      : 'train' (0001-0800) | 'val' (0801-0900)
    scale      : SR scale factor
    lr_patch_size     : LR patch size
    noise_sigma_range : (min, max) noise sigma range
    augment    : data augmentation flag
    """

    # Image index ranges per split
    SPLIT_RANGES = {
        "train": (1, 800),
        "val": (801, 900),
    }

    def __init__(
        self,
        root: str,
        split: str = "train",
        scale: int = 2,
        lr_patch_size: int = 48,
        noise_sigma_range: Tuple[float, float] = (0, 50),
        augment: bool = True,
    ):
        super().__init__()
        self.scale = scale
        self.lr_patch_size = lr_patch_size
        self.hr_patch_size = lr_patch_size * scale
        self.noise_sigma_range = noise_sigma_range
        self.augment = augment
        self.split = split

        self.hr_dir = Path(root) / f"DIV2K_{split}_HR"
        if not self.hr_dir.exists():
            raise FileNotFoundError(
                f"DIV2K HR directory not found: {self.hr_dir}\n"
                f"Please download DIV2K from: "
                f"https://data.vision.ee.ethz.ch/cvl/DIV2K/ "
                f"and place it at: {root}"
            )

        start, end = self.SPLIT_RANGES[split]
        self.image_paths = [
            str(self.hr_dir / f"{i:04d}.png")
            for i in range(start, end + 1)
            if (self.hr_dir / f"{i:04d}.png").exists()
        ]

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No DIV2K images found in: {self.hr_dir}")

        print(
            f"[DIV2KDataset] split={split} | "
            f"{len(self.image_paths)} images | "
            f"scale: ×{scale}"
        )

    def _random_crop(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        size = self.hr_patch_size
        x = random.randint(0, w - size)
        y = random.randint(0, h - size)
        return img.crop((x, y, x + size, y + size))

    def _augment(self, patch: Image.Image) -> Image.Image:
        if random.random() < 0.5:
            patch = patch.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            patch = patch.transpose(Image.FLIP_TOP_BOTTOM)
        k = random.randint(0, 3)
        for _ in range(k):
            patch = patch.transpose(Image.ROTATE_90)
        return patch

    def _degrade(self, hr_patch: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        hr_np = np.array(hr_patch, dtype=np.float32) / 255.0
        sigma = random.uniform(*self.noise_sigma_range) / 255.0
        if sigma > 0:
            noisy = np.clip(
                hr_np + np.random.randn(*hr_np.shape).astype(np.float32) * sigma, 0, 1
            )
        else:
            noisy = hr_np
        lr_pil = Image.fromarray((noisy * 255).astype(np.uint8)).resize(
            (self.lr_patch_size, self.lr_patch_size), Image.BICUBIC
        )
        to_tensor = transforms.ToTensor()
        return to_tensor(lr_pil), to_tensor(hr_patch)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = Image.open(self.image_paths[idx]).convert("RGB")
        hr_patch = self._random_crop(img)
        if self.augment and self.split == "train":
            hr_patch = self._augment(hr_patch)
        return self._degrade(hr_patch)


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------
def build_dataset(config, split: str = "train"):
    """
    Build the appropriate dataset from config.

    Args
    ----
    config : config module (configs/rcan_config.py)
    split  : 'train' | 'val'

    Returns
    -------
    Dataset instance (BSD500SRDataset or DIV2KDataset)
    """
    common_kwargs = dict(
        scale=config.SCALE_FACTOR,
        lr_patch_size=config.LR_PATCH_SIZE,
        noise_sigma_range=(config.NOISE_SIGMA_MIN, config.NOISE_SIGMA_MAX),
        augment=(split == "train"),
    )

    if config.DATASET_NAME == "bsd500":
        return BSD500SRDataset(
            root=config.DATA_ROOT,
            split=split,
            **common_kwargs,
        )
    elif config.DATASET_NAME == "div2k":
        return DIV2KDataset(
            root=config.DATA_ROOT,
            split=split,
            **common_kwargs,
        )
    else:
        raise ValueError(
            f"Unknown dataset: {config.DATASET_NAME}. " f"Supported: 'bsd500', 'div2k'"
        )
