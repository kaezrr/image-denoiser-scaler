# Image Denoising & Upscaling — Phase 1: RCAN Implementation

## Project Overview

This is the **first 20% of a two-architecture deep learning image restoration system**. It implements the **Residual Channel Attention Network (RCAN)**, which will serve as the "Microscope" component — responsible for fine-grained texture denoising before the SwinIR "Architect" handles global structural consistency.

**Full pipeline (future phases):**
```
Noisy LR Image → [RCAN: Denoising + Upscaling] → [SwinIR: Global Structure] → Clean HR Image
```

---

## Repository Structure

```
rcan_project/
├── README.md                  ← You are here
├── configs/
│   └── rcan_config.py         ← All hyperparameters in one place
├── data/
│   ├── dataset.py             ← Dataset class (BSR500 subset by default, swap for DIV2K)
│   └── transforms.py          ← Image augmentation & degradation pipeline
├── models/
│   ├── rcan.py                ← Full RCAN architecture (ECCV 2018)
│   └── losses.py              ← L1, Perceptual, and combined losses
├── utils/
│   ├── metrics.py             ← PSNR and SSIM implementations
│   ├── logger.py              ← Training logger with CSV export
│   └── checkpoint.py         ← Save/load checkpoint utilities
├── train.py                   ← Main training loop
├── evaluate.py                ← Evaluation script (PSNR/SSIM on test set)
├── infer.py                   ← Single-image inference script
├── requirements.txt           ← All Python dependencies
└── outputs/                   ← Generated images, logs, checkpoints go here
    └── checkpoints/
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train on the small dataset (BSR500 subset — auto-downloads)
```bash
python train.py
```

### 3. Swap to DIV2K for full training
Edit `configs/rcan_config.py`:
```python
DATASET_NAME = "div2k"          # change from "bsr500"
DATA_ROOT    = "/path/to/DIV2K" # point to downloaded DIV2K folder
```

### 4. Evaluate
```bash
python evaluate.py --checkpoint outputs/checkpoints/best_model.pth
```

### 5. Run inference on a single image
```bash
python infer.py --input my_image.jpg --checkpoint outputs/checkpoints/best_model.pth
```

---

## Architecture: RCAN (Residual Channel Attention Network)

**Paper:** [Image Super-Resolution Using Very Deep Residual Channel Attention Networks](https://arxiv.org/abs/1807.02758) — ECCV 2018

### Core Idea

Standard CNNs treat all feature channels equally. RCAN introduces **Channel Attention (CA)** which learns *which channels carry the most useful information* and adaptively re-weights them. This is especially useful for SR because:
- Low-frequency channels (background, smooth areas) are abundant but carry little new info
- High-frequency channels (edges, textures) are rare but critical for sharp outputs

### Architecture Layers (top-down)

```
Input LR Image (3 × H × W)
        │
[Shallow Feature Extraction]     ← 1 Conv layer, maps RGB → n_feats channels
        │
[Residual in Residual (RIR)]     ← n_resgroups × [Residual Group + Long Skip Connection]
        │
  ┌─────────────────────────────────────────────┐
  │  Residual Group (×n_resgroups)               │
  │  ┌───────────────────────────────────────┐   │
  │  │  RCAB (Residual Channel Attention     │   │
  │  │         Block) ×n_resblocks           │   │
  │  │  ┌─────────────────────────────────┐  │   │
  │  │  │  Conv → ReLU → Conv → CA → +   │  │   │
  │  │  └─────────────────────────────────┘  │   │
  │  │  + Short Skip (within group)          │   │
  │  └───────────────────────────────────────┘   │
  │  + Long Skip (input → output of RG)          │
  └─────────────────────────────────────────────┘
        │
[Upsampling Module]              ← PixelShuffle sub-pixel convolution (×scale)
        │
[Final Reconstruction]           ← 1 Conv layer, maps n_feats → 3 channels
        │
Output SR Image (3 × sH × sW)
```

### Channel Attention Mechanism

```
Feature Map (C × H × W)
        │
  Global Average Pooling         ← Squeezes spatial dims → (C × 1 × 1)
        │
  FC → ReLU → FC → Sigmoid       ← Learns channel importance weights [0,1]
        │
  Element-wise multiply           ← Re-weights each channel
        │
Scaled Feature Map (C × H × W)
```

The two FC layers follow a bottleneck: `C → C//reduction → C`, with `reduction=16` by default.

---

## Dataset Strategy

### Development Phase (this implementation): BSD500 subset
- ~200 images, auto-downloaded via `torchvision.datasets`
- Small and fast — good for verifying the training loop works
- LR images are synthesized on-the-fly with bicubic downsampling + Gaussian noise

### Production Phase (swap in later): DIV2K
- 800 HR training images, 100 validation images
- Industry standard benchmark for SR
- Just change `DATASET_NAME = "div2k"` in the config

### Data Degradation Pipeline (for "Microscope" role)
```
HR Patch (48×48 at ×2 scale → corresponds to 96×96 HR)
    │
Gaussian Noise (σ drawn from [0, 50])   ← RCAN's denoising role
    │
Bicubic Downscale (÷ scale_factor)      ← Creates the LR input
    │
LR Patch (48×48)                        ← Fed to RCAN
```

---

## Metrics

| Metric | What it measures | Higher is better? |
|--------|-----------------|-------------------|
| **PSNR** | Pixel-level fidelity (dB) | ✅ Yes |
| **SSIM** | Perceptual structure similarity [0,1] | ✅ Yes |

**Typical values on Set5 @ ×2:**
- Bicubic baseline: ~33.7 dB PSNR
- RCAN (paper): ~38.3 dB PSNR

---

## Connecting to SwinIR (Phase 2)

When you implement SwinIR, the daisy-chain will look like:

```python
# Phase 2 pipeline integration point
rcan_output = rcan_model(noisy_lr_image)   # RCAN: fine detail denoising
final_output = swinir_model(rcan_output)   # SwinIR: global structural consistency
```

RCAN's output is already a standard `(B, 3, H*scale, W*scale)` tensor — SwinIR can consume it directly without any adapter layer.
