# image-denoiser-scaler

A learning-focused deep learning project built on DIV2K for:
- image denoising (archived pipeline), and
- image super-resolution (current SR pipeline).

The repo contains both legacy experiments in `archive/` and newer SR-focused scripts in the root.

## Why this project exists

This is a hands-on learning project to compare model design tradeoffs:
- simple vs deeper denoising architectures,
- clean-trained vs robust-trained SR models,
- quality vs speed vs power.

## Setup

```bash
pip install -r requirements.txt
```

## Project layout (important)

- `archive/`:
    - legacy denoising + baseline SR training scripts,
    - includes the denoiser models and baseline EDSR-lite training flow.
- root (`train_sr.py`, `benchmark_sr.py`, `plots.py`, `model_sr.py`, ...):
    - newer SR workflow,
    - robust EDSR-lite training + SR benchmark + plotting suite.

## Models covered (4 total)

| # | Model | Task | Location | Train script | Benchmark |
|---|---|---|---|---|---|
| 1 | Simple Denoiser | Denoising | archived/registered (`models/simple_denoiser.keras`) | (pretrained baseline in registry) | `python archive/benchmark.py` |
| 2 | Inception-ResNet | Denoising | `archive/model_de.py` | `python archive/train_de.py --name inception_resnet --train` | `python archive/benchmark.py` |
| 3 | EDSR-lite (clean trained) | Super-resolution | `archive/model_up.py` | `python archive/train_up.py --name sr_edsr_model --train` | `python benchmark_sr.py` |
| 4 | EDSR-lite (robust trained) | Super-resolution | `model_sr.py` | `python train_sr.py --name sr_robust_model --train` | `python benchmark_sr.py` |

## Training commands

### Denoising (archive)

```bash
# Inception-ResNet denoiser
python archive/train_de.py --name inception_resnet --train

# Demo from saved model
python archive/train_de.py --name inception_resnet --demo
```

### Super-resolution

```bash
# Baseline EDSR-lite (clean-trained, archive pipeline)
python archive/train_up.py --name sr_edsr_model --train

# Robust EDSR-lite (new SR pipeline)
python train_sr.py --name sr_robust_model --train

# Demo robust model
python train_sr.py --name sr_robust_model --demo
```

## Benchmark commands

```bash
# Denoiser benchmark report (Simple Denoiser + Inception-ResNet)
python archive/benchmark.py

# SR benchmark report (EDSR-lite clean vs robust)
python benchmark_sr.py
```

Generated reports:
- `benchmark_results/report.md`
- `sr_benchmark_results/report.md`

## Plotting and analysis

Use `plots.py` for SR diagnostics (training curves, robustness charts, qualitative visuals):

```bash
# Generate full SR plot suite
python plots.py

# Single model focus
python plots.py --model sr_robust_model

# Skip inference-heavy plots
python plots.py --no-model
```

Outputs are saved in `plots/`.

## Current results snapshot

### 1) Simple Denoiser (denoising)
- PSNR: **21.34 dB**
- SSIM: **0.6337**
- Inference: **26.0 ms/img**

### 2) Inception-ResNet (denoising)
- PSNR: **22.96 dB**
- SSIM: **0.6382**
- Inference: **221.1 ms/img**

### 3) EDSR-lite (clean trained, SR)
- PSNR clean: **31.84 dB**
- PSNR degraded: **17.24 dB**
- Robustness drop: **14.60 dB**

### 4) EDSR-lite (robust trained, SR)
- PSNR clean: **26.08 dB**
- PSNR degraded: **22.79 dB**
- Robustness drop: **3.29 dB**

Interpretation:
- clean-trained EDSR-lite reaches higher clean PSNR,
- robust-trained EDSR-lite is much more stable under noisy/JPEG-degraded LR inputs.

## Visual results

- Denoiser examples: `benchmark_results/images/`
- SR examples: `sr_benchmark_results/images/`
- Training/analysis plots: `plots/`

### Denoiser comparison

#### Simple Denoiser

![Simple Denoiser comparison](<benchmark_results/images/Simple_Denoiser/comparison.png>)

#### Inception-ResNet

![Inception-ResNet comparison](<benchmark_results/images/Inception-ResNet/comparison.png>)

### Super-resolution comparison

#### EDSR-lite (clean trained)

Clean LR input (LR → SR → HR)

![EDSR-lite clean-trained clean input](<sr_benchmark_results/images/EDSR-lite_(clean_trained)/clean_comparison.png>)

Degraded LR input (noisy/JPEG LR → SR → HR)

![EDSR-lite clean-trained degraded input](<sr_benchmark_results/images/EDSR-lite_(clean_trained)/degraded_comparison.png>)

#### EDSR-lite (robust trained)

Clean LR input (LR → SR → HR)

![EDSR-lite robust-trained clean input](<sr_benchmark_results/images/EDSR-lite_(robust_trained)/robust_clean_comparison.png>)

Degraded LR input (noisy/JPEG LR → SR → HR)

![EDSR-lite robust-trained degraded input](<sr_benchmark_results/images/EDSR-lite_(robust_trained)/robust_degraded_comparison.png>)

## Notes

- Dataset is cached under `data/` after first run.
- Model metadata is tracked in:
    - `models/model_registry.json` (denoising)
    - `models/sr_model_registry.json` (super-resolution)