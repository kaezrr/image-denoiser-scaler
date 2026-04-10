#!/usr/bin/env python3
"""
Inference pipeline: Noisy image → Denoise → Super-Resolution upscale.

Flow
----
    Clean HR (300×300)
        │
        ▼  add_gaussian_noise
    Noisy HR (300×300)
        │
        ▼  denoiser model
    Denoised HR (300×300)
        │
        ▼  bicubic downsample ÷2
    Denoised LR (150×150)
        │
        ▼  SR model
    SR Output (300×300)

Why this order?
    The denoiser was trained at 300×300. The SR model expects 150×150 LR input.
    So we denoise first at full patch resolution, downsample to feed SR,
    then SR reconstructs back to 300×300.
    This mimics a real scenario: noisy low-res sensor image → clean HR output.

Usage
-----
    python pipeline.py                                     # default model names
    python pipeline.py --denoiser my_denoiser --sr my_sr  # custom model names
    python pipeline.py --n 8                               # show 8 examples
"""

import argparse
import os
import sys

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless — safe for servers without a display
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensorflow.keras.models import load_model  # type: ignore

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, _THIS_DIR)

from archive.dataset_de import prepare_data
from archive.noise import add_gaussian_to_dataset
from archive.model_de import CropToMatch, ResizeTo
from archive.model_up import SubPixelConv2D
from archive.utils import plot_rgb_img

MODELS_DIR  = os.path.join(_PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(_PROJECT_ROOT, "Results")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def downsample_batch(images: np.ndarray, scale: int = 2) -> np.ndarray:
    """Bicubic downsample a batch of float32 [0,1] images by `scale`.

    Parameters
    ----------
    images : np.ndarray
        Shape (N, H, W, 3), float32 in [0, 1].
    scale : int
        Downscale factor — 2 turns 300×300 into 150×150.

    Returns
    -------
    np.ndarray
        Shape (N, H//scale, W//scale, 3), float32 in [0, 1].
    """
    h, w = images.shape[1], images.shape[2]
    target_h, target_w = h // scale, w // scale
    # Cast to float32 — cv2.resize does not support float16 (mixed precision output)
    images = images.astype(np.float32)
    out = []
    for img in tqdm(images, desc="Downsampling to LR"):
        lr = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        out.append(lr)
    return np.array(out, dtype=np.float32)


def save_pipeline_results(
    noisy_hr: np.ndarray,
    denoised_hr: np.ndarray,
    sr_output: np.ndarray,
    original_hr: np.ndarray,
    n: int = 5,
    save_path: str = "Results/pipeline_results.png",
) -> None:
    """Save a 4-column grid: Noisy HR | Denoised HR | SR Output | Original HR."""
    n = min(n, len(noisy_hr))
    fig, axes = plt.subplots(n, 4, figsize=(20, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    titles = ["Noisy HR", "Denoised HR", "SR Output", "Original HR"]
    for i in range(n):
        for j, img in enumerate([noisy_hr[i], denoised_hr[i], sr_output[i], original_hr[i]]):
            axes[i, j].imshow(plot_rgb_img(img))
            axes[i, j].set_title(titles[j], fontsize=12)
            axes[i, j].axis("off")

    plt.suptitle("Pipeline: Noisy → Denoise → SR Upscale → Compare", fontsize=14, y=1.01)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[pipeline] Results saved to {save_path}")


def compute_psnr(original: np.ndarray, output: np.ndarray) -> float:
    """PSNR in dB between two float32 [0,1] image batches."""
    mse = float(np.mean((original - output) ** 2))
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(1.0 / mse)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Noisy → Denoise → SR pipeline")
    parser.add_argument("--denoiser", default="inception_resnet",
                        help="Denoiser model name under models/ (default: inception_resnet)")
    parser.add_argument("--sr", default="sr_edsr_model",
                        help="SR model name under models/ (default: sr_edsr_model)")
    parser.add_argument("--n", type=int, default=5,
                        help="Number of examples to visualize (default: 5)")
    args = parser.parse_args()

    denoiser_path = os.path.join(MODELS_DIR, f"{args.denoiser}.keras")
    sr_path       = os.path.join(MODELS_DIR, f"{args.sr}.keras")
    results_path  = os.path.join(RESULTS_DIR, "pipeline_results.png")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load both models
    # ------------------------------------------------------------------
    print("\n===== Step 1: Loading models =====")

    if not os.path.exists(denoiser_path):
        raise FileNotFoundError(
            f"Denoiser not found: {denoiser_path}\n"
            f"  Train it first:  python train.py --name {args.denoiser}"
        )
    if not os.path.exists(sr_path):
        raise FileNotFoundError(
            f"SR model not found: {sr_path}\n"
            f"  Train it first:  python train_sr.py --name {args.sr}"
        )

    print(f"  Loading denoiser : {denoiser_path}")
    denoiser = load_model(denoiser_path, custom_objects={"CropToMatch": CropToMatch, "ResizeTo": ResizeTo})

    print(f"  Loading SR model : {sr_path}")
    sr_model = load_model(sr_path, custom_objects={"SubPixelConv2D": SubPixelConv2D})

    # ------------------------------------------------------------------
    # 2. Prepare data — we only need test_data (clean HR 300×300)
    # ------------------------------------------------------------------
    print("\n===== Step 2: Preparing dataset =====")
    _, test_data = prepare_data()           # (N, 300, 300, 3) float32 [0,1]
    clean_hr = test_data[:args.n]           # (n, 300, 300, 3)

    # ------------------------------------------------------------------
    # 3. Add Gaussian noise to the clean HR images
    # ------------------------------------------------------------------
    print("\n===== Step 3: Adding Gaussian noise =====")
    noisy_hr = add_gaussian_to_dataset(clean_hr)    # (n, 300, 300, 3)

    # ------------------------------------------------------------------
    # 4. Denoise at 300×300
    # ------------------------------------------------------------------
    print("\n===== Step 4: Denoising (300×300) =====")
    denoised_hr = denoiser.predict(noisy_hr, verbose=1)   # (n, 300, 300, 3)

    # ------------------------------------------------------------------
    # 5. Downsample denoised output to LR (150×150) for the SR model
    # ------------------------------------------------------------------
    print("\n===== Step 5: Downsampling denoised HR → LR (150×150) =====")
    denoised_lr = downsample_batch(denoised_hr, scale=2)  # (n, 150, 150, 3)

    # ------------------------------------------------------------------
    # 6. Super-resolve back to 300×300
    # ------------------------------------------------------------------
    print("\n===== Step 6: Super-Resolution (150×150 → 300×300) =====")
    sr_output = sr_model.predict(denoised_lr, verbose=1)  # (n, 300, 300, 3)

    # ------------------------------------------------------------------
    # 7. Metrics — compare each stage against clean ground truth
    # ------------------------------------------------------------------
    print("\n===== Step 7: Metrics =====")
    psnr_noisy    = compute_psnr(clean_hr, noisy_hr)
    psnr_denoised = compute_psnr(clean_hr, denoised_hr)
    psnr_sr       = compute_psnr(clean_hr, sr_output)

    print(f"  PSNR  noisy HR  vs original : {psnr_noisy:.2f} dB  ← baseline (noise damage)")
    print(f"  PSNR  denoised  vs original : {psnr_denoised:.2f} dB  ← after denoiser")
    print(f"  PSNR  SR output vs original : {psnr_sr:.2f} dB  ← after full pipeline")

    # ------------------------------------------------------------------
    # 8. Save visualisation grid
    # ------------------------------------------------------------------
    print("\n===== Step 8: Saving results =====")
    save_pipeline_results(
        noisy_hr=noisy_hr,
        denoised_hr=denoised_hr,
        sr_output=sr_output,
        original_hr=clean_hr,
        n=args.n,
        save_path=results_path,
    )

    print("\n[pipeline] Done.")
    print(f"  Grid saved to : {results_path}")


if __name__ == "__main__":
    main()