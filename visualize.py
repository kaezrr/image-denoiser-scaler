"""Visualization helpers for comparing noisy, denoised and original images."""

import os

import matplotlib.pyplot as plt
import numpy as np

from utils import plot_rgb_img


def show_denoising_results(
    noisy_images: np.ndarray,
    denoised_images: np.ndarray,
    original_images: np.ndarray,
    n: int = 5,
    save_path: str | None = "Results/denoising_results.png",
) -> None:
    """Display a grid of noisy → denoised → original image triplets.

    Parameters
    ----------
    noisy_images : np.ndarray
        Noisy input images (N, H, W, 3).
    denoised_images : np.ndarray
        Model predictions (N, H, W, 3).
    original_images : np.ndarray
        Ground-truth clean images (N, H, W, 3).
    n : int
        Number of examples to display.
    save_path : str or None
        If given, save the figure to this path. Parent directory is created
        automatically if it does not exist.
    """
    n = min(n, len(noisy_images))
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    titles = ["Noisy", "Denoised", "Original"]
    for i in range(n):
        for j, img in enumerate([noisy_images[i], denoised_images[i], original_images[i]]):
            axes[i, j].imshow(plot_rgb_img(img))
            axes[i, j].set_title(titles[j])
            axes[i, j].axis("off")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=120)
        print(f"[visualize] Results saved to {save_path}")
    plt.show()
    