"""Noise and degradation functions for robust SR training.

Single-image functions
----------------------
gaussian_noise          — additive Gaussian noise
gaussian_blur           — Gaussian blur
salt_and_pepper_noise   — random pixel corruption
jpeg_compression        — JPEG encode/decode artefacts (blocking, ringing)
random_degrade          — randomly chains 1–3 of the above per image

Dataset helpers
---------------
add_gaussian_to_dataset       — apply gaussian_noise to every image
add_gaussian_blur_to_dataset  — apply gaussian_blur to every image

Keras Sequence generators
--------------------------
NoisyImageSequence  — original denoising generator (noisy → clean)
RobustSRSequence    — SR generator (degraded LR, augmented) → clean HR
"""

import cv2
import numpy as np
from tqdm import tqdm
import tensorflow as tf


# ---------------------------------------------------------------------------
# Individual image degradation functions
# All accept float32 [0,1] or uint8 and return the same dtype.
# ---------------------------------------------------------------------------

def gaussian_blur(image: np.ndarray) -> np.ndarray:
    """Apply Gaussian blur to a single image."""
    return cv2.GaussianBlur(image, (35, 35), cv2.BORDER_DEFAULT)


def gaussian_noise(image: np.ndarray) -> np.ndarray:
    """Add Gaussian noise to a single image.

    Parameters
    ----------
    image : np.ndarray
        (H, W, 3) float32 in [0,1] or uint8.

    Returns
    -------
    np.ndarray
        Noisy image, same dtype as input.
    """
    was_float = image.dtype in (np.float32, np.float64)
    img = np.clip(image * 255, 0, 255).astype(np.uint8) if was_float else image.copy()

    mean = (10, 10, 10)
    std  = (50, 50, 50)
    noise = np.random.normal(mean, std, img.shape).astype(np.int16)
    noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return noisy.astype(np.float32) / 255.0 if was_float else noisy


def salt_and_pepper_noise(image: np.ndarray, p: float = 0.05) -> np.ndarray:
    """Apply salt-and-pepper noise to a single image.

    Parameters
    ----------
    image : np.ndarray
        (H, W, 3), float32 or uint8.
    p : float
        Total probability of noise (half salt, half pepper).
    """
    out = image.copy()
    h, w, _ = out.shape
    rand = np.random.random((h, w))

    black, white = (0.0, 1.0) if out.dtype in (np.float32, np.float64) else (0, 255)
    out[rand < p / 2]                    = black   # pepper
    out[(rand >= p / 2) & (rand < p)]    = white   # salt
    return out


def jpeg_compression(image: np.ndarray, quality: int = None) -> np.ndarray:
    """Simulate JPEG compression artefacts (blocking, ringing, colour shifts).

    This is a very common real-world degradation — most images on the web
    have been JPEG-compressed at least once. Training on this makes the
    model robust to blocking artefacts without needing a separate deblocking
    stage.

    Parameters
    ----------
    image : np.ndarray
        (H, W, 3) float32 in [0,1] or uint8.
    quality : int or None
        JPEG quality level 1–100. Lower = more artefacts.
        If None, a random quality between 10 and 60 is chosen each call,
        ensuring the model sees the full spectrum of JPEG damage.

    Returns
    -------
    np.ndarray
        JPEG-compressed image, same dtype as input.
    """
    was_float = image.dtype in (np.float32, np.float64)
    img = np.clip(image * 255, 0, 255).astype(np.uint8) if was_float else image.copy()

    if quality is None:
        quality = int(np.random.randint(10, 61))  # random quality per call

    # cv2.imencode compresses to a JPEG byte buffer in memory (no disk I/O)
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    _, buf = cv2.imencode(".jpg", img, encode_params)
    degraded = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    return degraded.astype(np.float32) / 255.0 if was_float else degraded


def random_degrade(image: np.ndarray, rng: np.random.RandomState = None) -> np.ndarray:
    """Apply a random chain of 1–3 degradations to a single image.

    This is the core of the "brutal training pipeline" idea. Each call
    picks a different combination and severity, so the model is forced to
    generalise across many degradation types rather than overfitting to one.

    Degradation pool
    ----------------
    - gaussian_noise      (always available)
    - gaussian_blur       (always available)
    - salt_and_pepper     (p sampled from [0.01, 0.08])
    - jpeg_compression    (quality sampled from [10, 60])

    Parameters
    ----------
    image : np.ndarray
        (H, W, 3) float32 in [0,1].
    rng : np.random.RandomState or None
        Optional RNG for reproducibility. Uses global numpy RNG if None.

    Returns
    -------
    np.ndarray
        Degraded image, float32 in [0,1].
    """
    if rng is None:
        rng = np.random.RandomState()

    # All degradation functions, with lambda wrappers for parameterisation
    pool = [
        lambda img: gaussian_noise(img),
        lambda img: gaussian_blur(img),
        lambda img: salt_and_pepper_noise(img, p=float(rng.uniform(0.01, 0.08))),
        lambda img: jpeg_compression(img, quality=int(rng.randint(10, 61))),
    ]

    # Pick 1–3 unique degradations and chain them
    n_degrade = rng.randint(1, 4)
    chosen = rng.choice(len(pool), size=n_degrade, replace=False)

    out = image.copy()
    for idx in chosen:
        out = pool[idx](out)
        out = np.clip(out, 0.0, 1.0)   # keep values in range after each step

    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset-level helpers (kept for backward compatibility)
# ---------------------------------------------------------------------------

def add_gaussian_to_dataset(data: np.ndarray) -> np.ndarray:
    """Apply gaussian_noise to every image in a dataset."""
    noisy = [gaussian_noise(img) for img in tqdm(data, desc="Adding Gaussian noise")]
    return np.array(noisy)


def add_gaussian_blur_to_dataset(data: np.ndarray) -> np.ndarray:
    """Apply gaussian_blur to every image in a dataset."""
    blurred = [gaussian_blur(img) for img in tqdm(data, desc="Adding Gaussian blur")]
    return np.array(blurred)


# ---------------------------------------------------------------------------
# Keras Sequence generators
# ---------------------------------------------------------------------------

class NoisyImageSequence(tf.keras.utils.Sequence):
    """Original denoising generator — yields (noisy, clean) batches.

    Parameters
    ----------
    clean_data : np.ndarray
        Clean images (N, H, W, 3), float32 in [0,1].
    batch_size : int
    noise_fn : callable, optional
        Single-image noise function. Defaults to gaussian_noise.
    """

    def __init__(self, clean_data: np.ndarray, batch_size: int = 32, noise_fn=None):
        self.clean_data = clean_data
        self.batch_size = batch_size
        self.noise_fn   = noise_fn or gaussian_noise

    def __len__(self) -> int:
        return int(np.ceil(len(self.clean_data) / self.batch_size))

    def __getitem__(self, idx: int):
        start = idx * self.batch_size
        end   = min(start + self.batch_size, len(self.clean_data))
        clean_batch = self.clean_data[start:end]
        noisy_batch = np.array([self.noise_fn(img) for img in clean_batch])
        return noisy_batch, clean_batch


class RobustSRSequence(tf.keras.utils.Sequence):
    """Robust SR generator — yields (degraded+augmented LR, clean HR) batches.

    This is the main workhorse for the "heavy data, light model" strategy.
    Every batch goes through two stages:

    1. Augmentation — the LR/HR pair is jointly flipped/rotated so the
       model learns rotation/reflection invariance without extra data.

    2. Degradation — the LR patch is passed through random_degrade, so the
       model receives a different combination of noise, blur, and JPEG
       artefacts every single batch. The HR target is always clean.

    The model is therefore trained to do three things at once:
        - Remove Gaussian noise
        - Remove JPEG blocking artefacts
        - Upscale 2×

    At inference time, a single forward pass handles all of this.

    Parameters
    ----------
    lr_data : np.ndarray
        Low-res patches (N, 150, 150, 3), float32 in [0,1].
    hr_data : np.ndarray
        High-res patches (N, 300, 300, 3), float32 in [0,1].
    batch_size : int
    augment : bool
        Whether to apply random flips/rotations. Default True.
    degrade : bool
        Whether to apply random_degrade to each LR patch. Default True.
        Set to False for a clean validation split.
    seed : int
        Base seed — each epoch the sequence shuffles with a new seed so
        augmentation combinations vary across epochs.
    """

    def __init__(
        self,
        lr_data: np.ndarray,
        hr_data: np.ndarray,
        batch_size: int = 16,
        augment: bool = True,
        degrade: bool = True,
        seed: int = 42,
    ):
        self.lr_data    = lr_data
        self.hr_data    = hr_data
        self.batch_size = batch_size
        self.augment    = augment
        self.degrade    = degrade
        self.seed       = seed
        self.epoch      = 0
        # Shuffle indices once at init; on_epoch_end reshuffles
        self.indices    = np.arange(len(lr_data))
        self._shuffle()

    def _shuffle(self):
        rng = np.random.RandomState(self.seed + self.epoch)
        rng.shuffle(self.indices)

    def on_epoch_end(self):
        """Called by Keras after each epoch — reshuffles for the next pass."""
        self.epoch += 1
        self._shuffle()

    def __len__(self) -> int:
        return int(np.ceil(len(self.lr_data) / self.batch_size))

    def __getitem__(self, idx: int):
        start   = idx * self.batch_size
        end     = min(start + self.batch_size, len(self.lr_data))
        batch_i = self.indices[start:end]

        lr_batch = self.lr_data[batch_i].copy()
        hr_batch = self.hr_data[batch_i].copy()

        rng = np.random.RandomState(self.seed + self.epoch * 10000 + idx)

        out_lr, out_hr = [], []
        for lr, hr in zip(lr_batch, hr_batch):

            # ----------------------------------------------------------
            # Stage 1 — Joint augmentation (same transform on LR and HR)
            # ----------------------------------------------------------
            if self.augment:
                # Horizontal flip
                if rng.random() > 0.5:
                    lr = lr[:, ::-1, :]
                    hr = hr[:, ::-1, :]
                # Vertical flip
                if rng.random() > 0.5:
                    lr = lr[::-1, :, :]
                    hr = hr[::-1, :, :]
                # 90° rotation (k = 0,1,2,3 quarter turns)
                k = rng.randint(0, 4)
                if k > 0:
                    lr = np.rot90(lr, k)
                    hr = np.rot90(hr, k)

            # ----------------------------------------------------------
            # Stage 2 — Random degradation on LR only (HR stays clean)
            # ----------------------------------------------------------
            if self.degrade:
                lr = random_degrade(lr, rng=rng)

            out_lr.append(lr.astype(np.float32))
            out_hr.append(hr.astype(np.float32))

        return np.array(out_lr), np.array(out_hr)