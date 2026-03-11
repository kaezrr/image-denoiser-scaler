"""Noise generation functions for image denoising training."""

import cv2
import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Individual image noise functions
# ---------------------------------------------------------------------------

def gaussian_blur(image: np.ndarray) -> np.ndarray:
    """Apply Gaussian blur to a single image.

    Parameters
    ----------
    image : np.ndarray
        Image array (H, W, 3).  Can be uint8 or float.

    Returns
    -------
    np.ndarray
        Blurred image with same dtype as input.
    """
    return cv2.GaussianBlur(image, (35, 35), cv2.BORDER_DEFAULT)


def gaussian_noise(image: np.ndarray) -> np.ndarray:
    """Add Gaussian noise to a single image.

    The image is expected to be **uint8** (or will be converted internally).

    Parameters
    ----------
    image : np.ndarray
        Image array (H, W, 3), dtype uint8 or float32 in [0,1].

    Returns
    -------
    np.ndarray
        Noisy image with the same dtype as the input.
    """
    was_float = image.dtype in (np.float32, np.float64)
    if was_float:
        img = np.clip(image * 255, 0, 255).astype(np.uint8)
    else:
        img = image.copy()

    mean = (10, 10, 10)
    std = (50, 50, 50)
    noise = np.random.normal(mean, std, img.shape).astype(np.uint8)
    noisy = img + noise

    if was_float:
        return noisy.astype(np.float32) / 255.0
    return noisy


def salt_and_pepper_noise(image: np.ndarray, p: float = 0.05) -> np.ndarray:
    """Apply salt-and-pepper noise to a single image.

    Parameters
    ----------
    image : np.ndarray
        Image array (H, W, 3).
    p : float
        Total probability of noise (half salt, half pepper).

    Returns
    -------
    np.ndarray
        Noisy image with same dtype as input.
    """
    out = image.copy()
    h, w, c = out.shape
    rand = np.random.random((h, w))

    # Determine white/black value based on dtype
    if out.dtype in (np.float32, np.float64):
        black, white = 0.0, 1.0
    else:
        black, white = 0, 255

    out[rand < p / 2] = black        # pepper
    out[(rand >= p / 2) & (rand < p)] = white  # salt
    return out


# ---------------------------------------------------------------------------
# Dataset-level helpers
# ---------------------------------------------------------------------------

def add_gaussian_to_dataset(data: np.ndarray) -> np.ndarray:
    """Apply Gaussian noise to every image in a dataset.

    Parameters
    ----------
    data : np.ndarray
        Array of shape (N, H, W, 3).

    Returns
    -------
    np.ndarray
        Noisy dataset with same shape and dtype.
    """
    noisy = []
    for img in tqdm(data, desc="Adding Gaussian noise"):
        noisy.append(gaussian_noise(img))
    return np.array(noisy)


def add_gaussian_blur_to_dataset(data: np.ndarray) -> np.ndarray:
    """Apply Gaussian blur to every image in a dataset.

    Parameters
    ----------
    data : np.ndarray
        Array of shape (N, H, W, 3).

    Returns
    -------
    np.ndarray
        Blurred dataset with same shape and dtype.
    """
    blurred = []
    for img in tqdm(data, desc="Adding Gaussian blur"):
        blurred.append(gaussian_blur(img))
    return np.array(blurred)
