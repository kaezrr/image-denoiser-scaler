"""Dataset download, extraction and loading for the DIV2K dataset via TensorFlow Datasets."""

import os

import cv2
import numpy as np
from tqdm import tqdm
import tensorflow_datasets as tfds

_PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATASET_DIR = os.path.join(_PROJECT_ROOT, "data")

IMG_SIZE = 300
PATCH_SIZE = IMG_SIZE  # alias for clarity


def _random_crop(img: np.ndarray, crop_size: int, rng: np.random.RandomState) -> np.ndarray:
    """Extract a random square crop from an image.

    DIV2K images are high-resolution (2K+), so we crop rather than
    resize to preserve detail and generate multiple patches per image.
    """
    h, w = img.shape[:2]
    if h < crop_size or w < crop_size:
        # Fallback: just resize if image is somehow too small
        return cv2.resize(img, (crop_size, crop_size))
    y = rng.randint(0, h - crop_size)
    x = rng.randint(0, w - crop_size)
    return img[y:y + crop_size, x:x + crop_size]


def _load_images(max_images: int = 5000, patches_per_image: int = 6) -> np.ndarray:
    """Load DIV2K images via TFDS and extract random crops.

    DIV2K has only 800 training images, so we extract multiple random
    patches per image to reach the desired dataset size.

    Parameters
    ----------
    max_images : int
        Maximum number of patches to produce in total.
    patches_per_image : int
        How many random crops to extract from each source image.

    Returns
    -------
    np.ndarray
        Array of shape (N, PATCH_SIZE, PATCH_SIZE, 3), dtype uint8.
    """
    print("[dataset] Loading DIV2K via TensorFlow Datasets (auto download if needed)...")
    ds = tfds.load(
        "div2k/bicubic_x2",
        split="train",
        data_dir=DATASET_DIR,
        as_supervised=False,
        shuffle_files=False,
    )

    rng = np.random.RandomState(42)
    patches: list[np.ndarray] = []

    for sample in tqdm(ds, desc="Loading images"):
        if len(patches) >= max_images:
            break
        # TFDS returns tf.Tensor — convert to numpy, RGB→BGR for opencv consistency
        img = sample["hr"].numpy()                    # shape (H, W, 3), RGB uint8
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        for _ in range(patches_per_image):
            if len(patches) >= max_images:
                break
            patches.append(_random_crop(img, PATCH_SIZE, rng))

    data = np.array(patches, dtype=np.uint8)
    print(f"[dataset] Loaded {len(data)} patches with shape {data.shape}")
    return data


def prepare_data():
    """Full pipeline: download → load → split → normalise.

    TFDS handles download, extraction and caching automatically.

    Returns
    -------
    train_data : np.ndarray   (2500, 300, 300, 3) float32 in [0, 1]
    test_data  : np.ndarray   (500, 300, 300, 3) float32 in [0, 1]
    """
    data = _load_images(max_images=5000, patches_per_image=6)

    train_data = data[:2500].astype(np.float32) / 255.0
    test_data = data[3000:3500].astype(np.float32) / 255.0

    del data

    print(f"[dataset] train_data shape: {train_data.shape}, test_data shape: {test_data.shape}")
    return train_data, test_data