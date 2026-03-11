"""Utility helpers for the image denoiser project."""

import cv2
import numpy as np


def plot_rgb_img(image: np.ndarray):
    """Convert a BGR image to RGB for correct matplotlib display and return it.

    Parameters
    ----------
    image : np.ndarray
        Image in BGR format (OpenCV default) with values in [0, 1] or [0, 255].

    Returns
    -------
    np.ndarray
        Image converted to RGB colour space.
    """
    img = image.copy()
    # If the image is float and in [0,1], scale to [0,255] uint8 for cvtColor
    if img.dtype in (np.float32, np.float64):
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return rgb
