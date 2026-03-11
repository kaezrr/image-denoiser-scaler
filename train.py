#!/usr/bin/env python3
"""Main training script for the image denoising autoencoder.

Usage
-----
    python train.py            # train (or skip if saved model exists)
    python train.py --train    # force re-training even if model exists
    python train.py --demo     # demo only – load saved model & visualize

The script will:
1. Download the DIV2K (if not already present)
2. Preprocess and split images
3. Generate noisy training / test images
4. Build and train the convolutional autoencoder (or load a saved one)
5. Evaluate the model and display denoising results
"""

import argparse
import os
import sys

# Ensure this module's directory is on the path so sibling imports work
# regardless of the working directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
from tensorflow.keras.models import load_model  # type: ignore

from dataset import prepare_data
from noise import add_gaussian_to_dataset, NoisyImageSequence, gaussian_noise
from model import build_autoencoder
from visualize import show_denoising_results

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "denoiser_model.keras")


def main() -> None:
    parser = argparse.ArgumentParser(description="Image denoising autoencoder")
    parser.add_argument("--train", action="store_true",
                        help="Force re-training even if a saved model exists")
    parser.add_argument("--demo", action="store_true",
                        help="Demo only – load saved model and visualize results")
    args = parser.parse_args()

    need_training = args.train or (not args.demo and not os.path.exists(MODEL_PATH))

    # ------------------------------------------------------------------
    # 1. Download & prepare dataset
    # ------------------------------------------------------------------
    print("\n===== Step 1: Preparing dataset =====")
    train_data, test_data = prepare_data()

    if need_training:
        # --------------------------------------------------------------
        # 2. Prepare noisy data generator (on-the-fly to save RAM)
        # --------------------------------------------------------------
        print("\n===== Step 2: Preparing noise generator =====")
        batch_size = 32
        train_gen = NoisyImageSequence(train_data, batch_size=batch_size)
        print(f"  Training batches : {len(train_gen)} (generated on-the-fly)")

        # --------------------------------------------------------------
        # 3. Build model
        # --------------------------------------------------------------
        print("\n===== Step 3: Building autoencoder =====")
        model = build_autoencoder(input_shape=(300, 300, 3))

        # --------------------------------------------------------------
        # 4. Train (using generator to avoid OOM)
        # --------------------------------------------------------------
        print("\n===== Step 4: Training =====")
        model.fit(
            train_gen,
            epochs=50,
            callbacks=[EarlyStopping(monitor="loss", patience=3)],
        )

        # --------------------------------------------------------------
        # 4b. Save trained model
        # --------------------------------------------------------------
        model.save(MODEL_PATH)
        print(f"\n[train] Model saved to {MODEL_PATH}")
    else:
        # --------------------------------------------------------------
        # Load previously saved model
        # --------------------------------------------------------------
        print(f"\n===== Loading saved model from {MODEL_PATH} =====")
        model = load_model(MODEL_PATH)

    # ------------------------------------------------------------------
    # 5. Evaluate
    # ------------------------------------------------------------------
    print("\n===== Step 5: Evaluating =====")
    gaussian_test_data = add_gaussian_to_dataset(test_data)
    loss = model.evaluate(gaussian_test_data, test_data)
    print(f"  Test MSE loss: {loss:.6f}")

    # ------------------------------------------------------------------
    # 6. Visualize results
    # ------------------------------------------------------------------
    print("\n===== Step 6: Visualizing denoising results =====")
    denoised = model.predict(gaussian_test_data[:5])
    show_denoising_results(
        noisy_images=gaussian_test_data[:5],
        denoised_images=denoised,
        original_images=test_data[:5],
        n=5,
    )


if __name__ == "__main__":
    main()
