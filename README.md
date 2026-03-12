# image-denoiser-upscaler

Convolutional autoencoder that removes Gaussian noise from images (for now). Trained on DIV2K.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python train.py          # train from scratch (downloads dataset automatically)
python train.py --train  # force retrain even if saved model exists
python train.py --demo   # skip training, load saved model and visualize
```

Saves `denoiser_model.keras` and `Results/denoising_results.png` after training.


## Results

![Results](Results/denoising_results.png)

## How it works

DIV2K's 800 training images are cropped into 300×300 patches (6 per image) to get ~5000 samples. Gaussian noise is added on-the-fly during training so we don't need to store a second copy of the dataset in memory.

The model is an encoder-decoder:
- **Encoder** — Conv2D(64) → MaxPool → BN → Conv2D(32) → MaxPool → BN
- **Decoder** — Conv2D(32) → Upsample → Conv2D(64) → Upsample → Conv2D(3, sigmoid)

Adam + MSE loss. Early stopping with patience=3.

## Files

```
image-denoiser-upscaler/
├── train.py               main script
├── model.py               autoencoder architecture
├── dataset.py             DIV2K download + patch extraction via TFDS
├── noise.py               noise functions + Keras Sequence generator
├── visualize.py           results grid (noisy / denoised / original)
├── utils.py               BGR→RGB helper for matplotlib
├── requirements.txt
├── denoising_results.png
└── README.md
```

Dataset is cached to `./data/` on first run.


---

## Roadmap

- [ ] Super-resolution upscaling (DIV2K already has bicubic-downscaled pairs so the data side is sorted)
- [ ] Web GUI for live super-resolution
- [ ] Model comparison — training time, denoising speed, and upscaling speed across all implemented models