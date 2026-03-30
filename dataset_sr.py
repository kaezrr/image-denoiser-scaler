"""
Dataset download, extraction, and synchronized cropping for Super-Resolution.
This module uses TensorFlow Datasets to fetch DIV2K and extracts perfect 
pixel-aligned pairs for training.
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
import tensorflow_datasets as tfds

# Directory to cache the heavy DIV2K dataset so it doesn't redownload
DATASET_DIR = os.path.join(os.path.dirname(__file__), "data")

# Upscaling factor. If scaling by 2x, HR is 2x larger than LR.
SCALE = 2
HR_PATCH_SIZE = 300
LR_PATCH_SIZE = HR_PATCH_SIZE // SCALE  # 150


def _random_sync_crop(hr_img: np.ndarray, lr_img: np.ndarray, rng: np.random.RandomState):
    """
    Extract perfectly aligned crops from both HR and LR images.
    
    Why this is critical:
    If the LR image is 1000x1000 and the HR image is 2000x2000, coordinate (50, 50) 
    in the LR space corresponds exactly to coordinate (100, 100) in the HR space.
    """
    hr_h, hr_w = hr_img.shape[:2]
    lr_h, lr_w = lr_img.shape[:2]

    # Edge-case safety: If an image is unexpectedly smaller than our target patch size
    if lr_h < LR_PATCH_SIZE or lr_w < LR_PATCH_SIZE:
        lr_crop = cv2.resize(lr_img, (LR_PATCH_SIZE, LR_PATCH_SIZE))
        hr_crop = cv2.resize(hr_img, (HR_PATCH_SIZE, HR_PATCH_SIZE))
        return lr_crop, hr_crop

    # Step 1: Pick a random top-left anchor coordinate in the Low-Res image
    # We subtract LR_PATCH_SIZE to ensure the crop doesn't go out of bounds
    lr_y = rng.randint(0, lr_h - LR_PATCH_SIZE)
    lr_x = rng.randint(0, lr_w - LR_PATCH_SIZE)

    # Step 2: Calculate the exact matching coordinate in High-Res space
    # We simply multiply by the scale factor to maintain mathematical alignment
    hr_y = lr_y * SCALE
    hr_x = lr_x * SCALE

    # Step 3: Slice the numpy arrays to extract the patches
    lr_crop = lr_img[lr_y : lr_y + LR_PATCH_SIZE, lr_x : lr_x + LR_PATCH_SIZE]
    hr_crop = hr_img[hr_y : hr_y + HR_PATCH_SIZE, hr_x : hr_x + HR_PATCH_SIZE]

    return lr_crop, hr_crop


def _load_sr_images(max_images: int = 5000, patches_per_image: int = 6):
    """
    Load DIV2K LR/HR pairs via TFDS and extract synchronized crops.
    We extract multiple patches per image because DIV2K only has 800 training images,
    which isn't enough for deep learning without aggressive cropping.
    """
    print("[dataset_sr] Loading DIV2K (bicubic_x2) via TensorFlow Datasets...")
    
    # Load the specific "bicubic_x2" split, which provides pre-downscaled LR images
    ds = tfds.load(
        "div2k/bicubic_x2",
        split="train",
        data_dir=DATASET_DIR,
        as_supervised=False,
        shuffle_files=False,
    )

    rng = np.random.RandomState(42) # Seed for reproducible random crops
    lr_patches = []
    hr_patches = []

    for sample in tqdm(ds, desc="Extracting LR/HR pairs"):
        if len(lr_patches) >= max_images:
            break
        
        # TFDS loads images in RGB. OpenCV uses BGR natively. 
        # We convert to BGR here so it matches cv2.imwrite/imshow later.
        hr_img = cv2.cvtColor(sample["hr"].numpy(), cv2.COLOR_RGB2BGR)
        lr_img = cv2.cvtColor(sample["lr"].numpy(), cv2.COLOR_RGB2BGR)

        # Extract multiple pairs from this single image
        for _ in range(patches_per_image):
            if len(lr_patches) >= max_images:
                break
            lr_p, hr_p = _random_sync_crop(hr_img, lr_img, rng)
            lr_patches.append(lr_p)
            hr_patches.append(hr_p)

    # Convert lists to dense numpy arrays for Keras
    lr_data = np.array(lr_patches, dtype=np.uint8)
    hr_data = np.array(hr_patches, dtype=np.uint8)
    print(f"[dataset_sr] Loaded {len(lr_data)} patch pairs.")
    return lr_data, hr_data


def prepare_sr_data():
    """
    Full pipeline: load, split into train/test, and normalize.
    Normalizing to [0, 1] is crucial for neural network stability.
    """
    lr_data, hr_data = _load_sr_images(max_images=5000, patches_per_image=6)

    # First 2500 patches for training
    train_lr = lr_data[:2500].astype(np.float32) / 255.0
    train_hr = hr_data[:2500].astype(np.float32) / 255.0

    # Patches 3000-3500 for testing (leaving a gap prevents overlapping crops)
    test_lr = lr_data[3000:3500].astype(np.float32) / 255.0
    test_hr = hr_data[3000:3500].astype(np.float32) / 255.0

    # Free up RAM
    del lr_data, hr_data

    return train_lr, train_hr, test_lr, test_hr