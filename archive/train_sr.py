#!/usr/bin/env python3
"""Main training script for the Super-Resolution model."""

import argparse
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)

# Ensure imports work regardless of current working directory.
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, _THIS_DIR)

from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
from tensorflow.keras.models import load_model  # type: ignore

from archive.dataset_up import prepare_sr_data
from archive.model_up import build_sr_model, SubPixelConv2D
from visualize import show_sr_results

# Define where models and images will be saved (root-level)
MODELS_DIR = os.path.join(_PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(_PROJECT_ROOT, "Results")

def main() -> None:
    # Command line argument parsing
    parser = argparse.ArgumentParser(description="Super-Resolution EDSR-lite")
    parser.add_argument("--name", default="sr_edsr_model",
                        help="Model name (default: sr_edsr_model)")
    parser.add_argument("--train", action="store_true", help="Force re-training")
    parser.add_argument("--demo", action="store_true", help="Demo only")
    args = parser.parse_args()

    MODEL_NAME   = args.name
    MODEL_PATH   = os.path.join(MODELS_DIR, f"{MODEL_NAME}.keras")
    RESULTS_PATH = os.path.join(RESULTS_DIR, f"{MODEL_NAME}_results.png")

    # Ensure directories exist so Keras doesn't crash when saving
    os.makedirs(MODELS_DIR,  exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Determine if we need to train or just load an existing model
    need_training = args.train or (not args.demo and not os.path.exists(MODEL_PATH))

    print("\n===== Step 1: Preparing SR dataset =====")
    train_lr, train_hr, test_lr, test_hr = prepare_sr_data()

    if need_training:
        print("\n===== Step 2: Building SR model =====")
        # We specify scale=2 to match our dataset crop math
        model = build_sr_model(scale=2, num_res_blocks=8, input_shape=(150, 150, 3))

        print("\n===== Step 3: Training =====")
        model.fit(
            x=train_lr,   # Input: Low-Res images
            y=train_hr,   # Target Ground Truth: High-Res images
            batch_size=16,
            epochs=50,
            validation_split=0.1, # Use 10% of data to check for overfitting
            # Stop training early if val_loss doesn't improve for 5 epochs
            callbacks=[EarlyStopping(monitor="val_loss", patience=5)],
        )

        # Save the full model graph, weights, and custom layer references
        model.save(MODEL_PATH)
        print(f"\n[train] Model saved to {MODEL_PATH}")
    else:
        print(f"\n===== Loading saved model from {MODEL_PATH} =====")
        # We MUST pass the custom layer in custom_objects so Keras knows how to build it
        model = load_model(MODEL_PATH, custom_objects={"SubPixelConv2D": SubPixelConv2D})

    print("\n===== Step 4: Evaluating =====")
    # Evaluate against the untouched test set
    loss = model.evaluate(test_lr, test_hr)
    print(f"  Test MAE loss: {loss:.6f}")

    print("\n===== Step 5: Visualizing SR results =====")
    # Generate 5 predictions and pass them to our visualization function
    sr_images = model.predict(test_lr[:5])
    show_sr_results(test_lr[:5], sr_images, test_hr[:5], n=5, save_path=RESULTS_PATH)

if __name__ == "__main__":
    main()