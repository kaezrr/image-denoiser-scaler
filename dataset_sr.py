"""Dataset loading for Super-Resolution — reads DIV2K directly from disk.

Bypasses TensorFlow Datasets entirely to avoid the protobuf version
conflict. Reads PNG files straight from the extracted zip folders.

Expected folder layout (auto-detected from EXTRACTED_DIR)
----------------------------------------------------------
data/downloads/extracted/
    ZIP.<hash_train_HR>.zip/
        DIV2K_train_HR/
            0001.png ... 0800.png
    ZIP.<hash_train_LR>.zip/
        DIV2K_train_LR_bicubic/
            X2/
                0001x2.png ... 0800x2.png
    ZIP.<hash_valid_HR>.zip/
        DIV2K_valid_HR/
            0801.png ... 0900.png
    ZIP.<hash_valid_LR>.zip/
        DIV2K_valid_LR_bicubic/
            X2/
                0801x2.png ... 0900x2.png

If auto-detection fails, set environment variables:
    export DIV2K_TRAIN_HR=/path/to/DIV2K_train_HR
    export DIV2K_TRAIN_LR=/path/to/DIV2K_train_LR_bicubic/X2
    export DIV2K_VALID_HR=/path/to/DIV2K_valid_HR
    export DIV2K_VALID_LR=/path/to/DIV2K_valid_LR_bicubic/X2
"""

import os
import glob
import cv2
import numpy as np
from tqdm import tqdm

_BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
_EXTRACTED_DIR = os.path.join(_BASE_DIR, "data", "downloads", "extracted")

SCALE         = 2
HR_PATCH_SIZE = 300
LR_PATCH_SIZE = HR_PATCH_SIZE // SCALE   # 150


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def _find_folder(keyword: str, subpath: str) -> str | None:
    """Search extracted zip folders for a subfolder matching keyword+subpath."""
    if not os.path.isdir(_EXTRACTED_DIR):
        return None
    for entry in sorted(os.listdir(_EXTRACTED_DIR)):
        if keyword.lower() in entry.lower():
            # Direct match
            candidate = os.path.join(_EXTRACTED_DIR, entry, subpath)
            if os.path.isdir(candidate):
                return candidate
            # One extra nesting level (some extractors add a wrapper folder)
            top = os.path.join(_EXTRACTED_DIR, entry)
            for sub in os.listdir(top):
                candidate2 = os.path.join(top, sub, subpath)
                if os.path.isdir(candidate2):
                    return candidate2
    return None


def _resolve_dirs():
    """Return (train_hr, train_lr, valid_hr, valid_lr) directory paths."""
    train_hr = os.environ.get("DIV2K_TRAIN_HR") or _find_folder(
        "trai_HR", "DIV2K_train_HR"
    )
    train_lr = os.environ.get("DIV2K_TRAIN_LR") or _find_folder(
        "trai_LR", os.path.join("DIV2K_train_LR_bicubic", "X2")
    )
    valid_hr = os.environ.get("DIV2K_VALID_HR") or _find_folder(
        "vali_HR", "DIV2K_valid_HR"
    )
    valid_lr = os.environ.get("DIV2K_VALID_LR") or _find_folder(
        "vali_LR", os.path.join("DIV2K_valid_LR_bicubic", "X2")
    )

    missing = [
        name for name, val in [
            ("train HR", train_hr), ("train LR", train_lr),
            ("valid HR", valid_hr), ("valid LR", valid_lr),
        ] if not val
    ]
    if missing:
        raise FileNotFoundError(
            f"\n[dataset_sr] Could not locate DIV2K folders: {missing}\n"
            f"  Searched under: {_EXTRACTED_DIR}\n\n"
            f"  Run this to see what's there:\n"
            f"    find data/downloads/extracted -type d | head -40\n\n"
            f"  Then point to the folders manually:\n"
            f"    export DIV2K_TRAIN_HR=/path/to/DIV2K_train_HR\n"
            f"    export DIV2K_TRAIN_LR=/path/to/DIV2K_train_LR_bicubic/X2\n"
            f"    export DIV2K_VALID_HR=/path/to/DIV2K_valid_HR\n"
            f"    export DIV2K_VALID_LR=/path/to/DIV2K_valid_LR_bicubic/X2\n"
        )

    return train_hr, train_lr, valid_hr, valid_lr


# ---------------------------------------------------------------------------
# Synchronized crop
# ---------------------------------------------------------------------------

def _random_sync_crop(
    hr_img: np.ndarray,
    lr_img: np.ndarray,
    rng: np.random.RandomState,
):
    """Pixel-aligned random crop: LR (x,y) ↔ HR (x×scale, y×scale)."""
    lr_h, lr_w = lr_img.shape[:2]

    if lr_h < LR_PATCH_SIZE or lr_w < LR_PATCH_SIZE:
        return (
            cv2.resize(lr_img, (LR_PATCH_SIZE, LR_PATCH_SIZE)),
            cv2.resize(hr_img, (HR_PATCH_SIZE, HR_PATCH_SIZE)),
        )

    lr_y = rng.randint(0, lr_h - LR_PATCH_SIZE)
    lr_x = rng.randint(0, lr_w - LR_PATCH_SIZE)
    hr_y, hr_x = lr_y * SCALE, lr_x * SCALE

    return (
        lr_img[lr_y : lr_y + LR_PATCH_SIZE, lr_x : lr_x + LR_PATCH_SIZE],
        hr_img[hr_y : hr_y + HR_PATCH_SIZE, hr_x : hr_x + HR_PATCH_SIZE],
    )


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def _sorted_pngs(folder: str) -> list[str]:
    paths = glob.glob(os.path.join(folder, "*.png"))
    if not paths:
        raise FileNotFoundError(f"No PNG files found in: {folder}")
    return sorted(paths)


def _load_pairs(
    hr_dir: str,
    lr_dir: str,
    max_patches: int,
    patches_per_image: int,
    rng: np.random.RandomState,
    label: str = "Loading",
) -> tuple[list, list]:
    """Load HR/LR pairs and extract synchronized random crops."""
    hr_paths = _sorted_pngs(hr_dir)
    lr_paths = _sorted_pngs(lr_dir)

    # DIV2K filenames sort into matching order (0001.png ↔ 0001x2.png)
    pairs = list(zip(hr_paths, lr_paths))

    lr_patches, hr_patches = [], []

    for hr_path, lr_path in tqdm(pairs, desc=label):
        if len(lr_patches) >= max_patches:
            break

        hr_img = cv2.imread(hr_path, cv2.IMREAD_COLOR)
        lr_img = cv2.imread(lr_path, cv2.IMREAD_COLOR)

        if hr_img is None or lr_img is None:
            print(f"  [warn] Could not read: {hr_path} or {lr_path}")
            continue

        for _ in range(patches_per_image):
            if len(lr_patches) >= max_patches:
                break
            lr_p, hr_p = _random_sync_crop(hr_img, lr_img, rng)
            lr_patches.append(lr_p)
            hr_patches.append(hr_p)

    return lr_patches, hr_patches


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def prepare_sr_data():
    """Load DIV2K from disk, split into train/test, normalise to [0,1].

    No TFDS or protobuf required. Reads PNGs directly via OpenCV.

    Returns
    -------
    train_lr : (N, 150, 150, 3) float32
    train_hr : (N, 300, 300, 3) float32
    test_lr  : (M, 150, 150, 3) float32
    test_hr  : (M, 300, 300, 3) float32
    """
    train_hr_dir, train_lr_dir, valid_hr_dir, valid_lr_dir = _resolve_dirs()

    print(f"[dataset_sr] Train HR : {train_hr_dir}")
    print(f"[dataset_sr] Train LR : {train_lr_dir}")
    print(f"[dataset_sr] Valid HR : {valid_hr_dir}")
    print(f"[dataset_sr] Valid LR : {valid_lr_dir}")

    rng = np.random.RandomState(42)

    lr_train, hr_train = _load_pairs(
        train_hr_dir, train_lr_dir,
        max_patches=2500, patches_per_image=4,
        rng=rng, label="Train patches",
    )
    lr_test, hr_test = _load_pairs(
        valid_hr_dir, valid_lr_dir,
        max_patches=500, patches_per_image=5,
        rng=rng, label="Test patches",
    )

    train_lr = np.array(lr_train, dtype=np.float32) / 255.0
    train_hr = np.array(hr_train, dtype=np.float32) / 255.0
    test_lr  = np.array(lr_test,  dtype=np.float32) / 255.0
    test_hr  = np.array(hr_test,  dtype=np.float32) / 255.0

    print(f"[dataset_sr] train: {train_lr.shape},  test: {test_lr.shape}")
    return train_lr, train_hr, test_lr, test_hr