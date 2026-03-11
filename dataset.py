"""Dataset download, extraction and loading for the DIV2K dataset."""

import os
import zipfile
import glob

import cv2
import numpy as np
import requests
from tqdm import tqdm

# DIV2K high-resolution training images (800 images, ~4GB)
DATASET_URL = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
DATASET_DIR = os.path.join(os.path.dirname(__file__), "data")
ZIP_PATH = os.path.join(DATASET_DIR, "DIV2K_train_HR.zip")
EXTRACT_DIR = os.path.join(DATASET_DIR, "DIV2K_train_HR")

IMG_SIZE = 300
PATCH_SIZE = IMG_SIZE  # alias for clarity


def _download_dataset() -> None:
    """Download the DIV2K zip archive if it is not already present."""
    if os.path.exists(ZIP_PATH):
        print("[dataset] Zip already downloaded, skipping.")
        return

    os.makedirs(DATASET_DIR, exist_ok=True)
    print(f"[dataset] Downloading DIV2K dataset from {DATASET_URL} ...")
    resp = requests.get(DATASET_URL, stream=True)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(ZIP_PATH, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc="Download") as bar:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            f.write(chunk)
            bar.update(len(chunk))
    print("[dataset] Download complete.")


def _extract_dataset() -> None:
    """Extract the zip archive if the extraction directory does not exist."""
    if os.path.isdir(EXTRACT_DIR):
        print("[dataset] Already extracted, skipping.")
        return

    print("[dataset] Extracting archive ...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(DATASET_DIR)
    print("[dataset] Extraction complete.")


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
    """Load DIV2K images and extract random crops, returning as an array.

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
    image_paths = sorted(glob.glob(os.path.join(EXTRACT_DIR, "*.png")))
    if not image_paths:
        raise FileNotFoundError(f"No .png images found in {EXTRACT_DIR}")

    rng = np.random.RandomState(42)
    rng.shuffle(image_paths)

    patches: list[np.ndarray] = []
    for path in tqdm(image_paths, desc="Loading images"):
        if len(patches) >= max_images:
            break
        img = cv2.imread(path)
        if img is None:
            continue  # skip corrupt files
        for _ in range(patches_per_image):
            if len(patches) >= max_images:
                break
            patch = _random_crop(img, PATCH_SIZE, rng)
            patches.append(patch)

    data = np.array(patches, dtype=np.uint8)
    print(f"[dataset] Loaded {len(data)} patches with shape {data.shape}")
    return data


def prepare_data():
    """Full pipeline: download → extract → load → split → normalise.

    Returns
    -------
    train_data : np.ndarray   (2500, 300, 300, 3) float32 in [0, 1]
    test_data  : np.ndarray   (500, 300, 300, 3) float32 in [0, 1]
    """
    _download_dataset()
    _extract_dataset()
    data = _load_images(max_images=5000, patches_per_image=6)

    train_data = data[:2500].astype(np.float32) / 255.0
    test_data = data[3000:3500].astype(np.float32) / 255.0

    del data

    print(f"[dataset] train_data shape: {train_data.shape}, test_data shape: {test_data.shape}")
    return train_data, test_data