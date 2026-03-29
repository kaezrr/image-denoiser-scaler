# image-denoiser-upscaler

Convolutional autoencoder that removes Gaussian noise from images (for now). Trained on DIV2K.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python train.py                          # train with default name "denoiser_model"
python train.py --name my_model          # train and save as models/my_model.keras
python train.py --name my_model --train  # force re-train even if model exists
python train.py --name my_model --demo   # load models/my_model.keras and visualize

python benchmark.py                      # benchmark all models saved in models/
```

Saves `models/<my_model>.keras` and `Results/<my_model>_results.png` after training.


## Results

![Results](Results/inception_resnet_results.png)

## How it works

DIV2K's 800 training images are cropped into 300×300 patches (6 per image) to get ~5000 samples. Gaussian noise is added on-the-fly during training so we don't need to store a second copy of the dataset in memory.

The model is a Hybrid Inception-ResNet autoencoder for image denoising / super-resolution.

`benchmark.py` evaluates all registered models and saves results to `benchmark_results/report.md`.

### Architecture

**Encoder**

    Stage 1 : InceptionBlock(64)   → MaxPool  [300→150]  — skip_1
    Stage 2 : ResidualBlock(128)×2 → MaxPool  [150→75]   — skip_2
    Stage 3 : InceptionBlock(256)  → MaxPool  [75→38]    — skip_3

**Bottleneck**

    Conv(512) → BN → ReLU                    [38×38]

**Decoder** (U-Net style — concat skip at each level)

    Stage 1 : UpSample → concat(skip_3) → Conv(256)  [38→75]
    Stage 2 : UpSample → concat(skip_2) → Conv(128)  [75→150]
    Stage 3 : UpSample → concat(skip_1) → Conv(64)   [150→300]

**Output** : Conv(3, sigmoid)                [300×300×3]

Adam + MSE loss. Early stopping with patience=3.

## Files

```
image-denoiser-upscaler/
├── train.py               main script
├── benchmark.py           multi-model benchmark + Markdown report
├── model.py               autoencoder architecture
├── dataset.py             DIV2K download + patch extraction via TFDS
├── noise.py               noise functions + Keras Sequence generator
├── visualize.py           results grid (noisy / denoised / original)
├── utils.py               BGR→RGB helper for matplotlib
├── requirements.txt
├── models/
│   └── model_registry.json    model details and training times
├── benchmark_results/
│   └── report.md
├── Results/
│   ├── simple_denoiser.png
│   └── inception_resnet_results.png
└── README.md
```

Dataset is cached to `./data/` on first run.


---

## Roadmap

- [ ] Super-resolution upscaling (DIV2K already has bicubic-downscaled pairs so the data side is sorted)
- [ ] Web GUI for live super-resolution
- [ ] Model comparison — training time, denoising speed, and upscaling speed across all implemented models