"""Dataset download, extraction and loading for the Cats vs Dogs dataset."""

import os
import zipfile
import glob

import cv2
import numpy as np
import requests
from tqdm import tqdm

# Microsoft Cats vs Dogs direct download URL
DATASET_URL = "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip"
DATASET_DIR = os.path.join(os.path.dirname(__file__), "data")
ZIP_PATH = os.path.join(DATASET_DIR, "kagglecatsanddogs_5340.zip")
EXTRACT_DIR = os.path.join(DATASET_DIR, "PetImages")

IMG_SIZE = 300


def _download_dataset() -> None:
    """Download the Cats vs Dogs zip archive if it is not already present."""
    if os.path.exists(ZIP_PATH):
        print("[dataset] Zip already downloaded, skipping.")
        return

    os.makedirs(DATASET_DIR, exist_ok=True)
    print(f"[dataset] Downloading dataset from {DATASET_URL} ...")
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


def _load_images(max_images: int = 5000) -> np.ndarray:
    """Load cat and dog images, resize to IMG_SIZE×IMG_SIZE, return as array.

    Parameters
    ----------
    max_images : int
        Maximum number of images to load (from both categories combined).

    Returns
    -------
    np.ndarray
        Array of shape (N, IMG_SIZE, IMG_SIZE, 3), dtype uint8.
    """
    cat_dir = os.path.join(EXTRACT_DIR, "Cat")
    dog_dir = os.path.join(EXTRACT_DIR, "Dog")

    image_paths: list[str] = []
    for d in (cat_dir, dog_dir):
        image_paths.extend(sorted(glob.glob(os.path.join(d, "*.jpg"))))

    # Shuffle deterministically so we get a mix of cats and dogs
    rng = np.random.RandomState(42)
    rng.shuffle(image_paths)

    images: list[np.ndarray] = []
    for path in tqdm(image_paths, desc="Loading images", total=min(len(image_paths), max_images)):
        if len(images) >= max_images:
            break
        img = cv2.imread(path)
        if img is None:
            continue  # skip corrupt files
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        images.append(img)

    data = np.array(images, dtype=np.uint8)
    print(f"[dataset] Loaded {len(data)} images with shape {data.shape}")
    return data


def prepare_data():
    """Full pipeline: download → extract → load → split → normalise.

    Returns
    -------
    train_data : np.ndarray   (2500, 300, 300, 3) float32 in [0,1]
    test_data  : np.ndarray   (500, 300, 300, 3) float32 in [0,1]
    """
    _download_dataset()
    _extract_dataset()
    data = _load_images(max_images=5000)

    train_data = data[:2500].astype(np.float32) / 255.0
    test_data = data[3000:3500].astype(np.float32) / 255.0

    # Free the large uint8 array as soon as possible
    del data

    print(f"[dataset] train_data shape: {train_data.shape}, test_data shape: {test_data.shape}")
    return train_data, test_data
