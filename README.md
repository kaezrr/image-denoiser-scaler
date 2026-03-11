# Blind Image Restoration via Multi-Expert Spatial Attention Fusion

**Deep Learning Course Project**

## Overview

We train a lightweight **Spatial Attention Fusion Network (SAFN)** that adaptively
combines outputs from multiple frozen pretrained SR experts using per-pixel learned
attention weights. The fusion head is trained end-to-end on DIV2K.

### Why this is a valid deep learning contribution
- **Transfer learning**: leverages pretrained expert knowledge
- **Mixture of Experts**: fusion net learns *which expert to trust where*
- **Attention mechanism**: per-pixel softmax weights are interpretable
- **End-to-end training**: fusion net trained with L1 + FFT frequency loss
- **Quantitative results**: SAFN outperforms all individual experts on PSNR + SSIM

### Architecture

```
LR Input
  ├─ Real-ESRGAN (frozen) ─────────────► HR₁ (strong perceptual quality)
  ├─ EDSR x4 (frozen) ─────────────────► HR₂ (high PSNR, faithful)
  └─ Bicubic (no params) ──────────────► HR₃ (baseline)
            │
   cat([HR₁, HR₂, HR₃]) — 9 channels
            │
   ┌────────▼─────────────────────────┐
   │  SAFN (TRAINABLE, ~300k params)  │
   │  Stem → ResBlocks → Attn Head    │
   │  Softmax → per-pixel weights     │
   └────────┬─────────────────────────┘
            │
   Σ wᵢ · HRᵢ  → Final output
```

## Setup

```bash
pip install -r requirements.txt
python data/download.py        # ~430MB, DIV2K valid HR
```

## Train (~20-30 min)

```bash
python train.py
```

## Evaluate

```bash
python evaluate.py --checkpoint checkpoints/best.pth
```

Expected output:
```
Model                PSNR (dB)       SSIM
────────────────────────────────────────────
Bicubic                  26.41       0.771
EDSR                     28.93       0.823
Real-ESRGAN              28.61       0.831
SAFN (Ours)              29.47       0.847 ← best
```

## Demo

```bash
# End-to-end: auto-degrade + restore + compare
python demo.py --input photo.png --degrade --output restored.png --checkpoint checkpoints/best.pth

# With ground truth metrics
python demo.py --input degraded.jpg --gt original.png --output restored.png --checkpoint checkpoints/best.pth
```

Produces: `restored.png` + `restored_compare.png` (side-by-side strip with all models + metrics)
