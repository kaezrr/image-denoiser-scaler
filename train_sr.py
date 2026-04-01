#!/usr/bin/env python3
"""Training script for the robust EDSR-lite Super-Resolution model.

What changed from the baseline
--------------------------------
1. Uses RobustSRSequence instead of passing raw arrays to model.fit().
   Every batch receives a different random combination of:
       - Gaussian noise
       - Gaussian blur
       - Salt-and-pepper noise
       - JPEG compression artefacts
   plus random flip/rotation augmentation.

2. Validation uses a clean (non-degraded) split so the val_loss curve
   reflects true SR quality, not how well the model fights noise.

3. Evaluates on BOTH clean and degraded test inputs so you can see
   exactly how much robustness the training pipeline added.

4. PSNR and SSIM are logged every epoch (defined in model_sr.py).

Usage
-----
    python train_sr.py                          # default name
    python train_sr.py --name robust_sr         # custom name
    python train_sr.py --name robust_sr --train # force retrain
    python train_sr.py --name robust_sr --demo  # load and visualize
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # type: ignore
from tensorflow.keras.models import load_model  # type: ignore

from dataset_sr import prepare_sr_data
from model_sr import build_sr_model, SubPixelConv2D, combined_loss, PSNRMetric, SSIMMetric
from noise_sr import RobustSRSequence, random_degrade
from visualize import show_sr_results

_BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR  = os.path.join(_BASE_DIR, "models")
RESULTS_DIR = os.path.join(_BASE_DIR, "Results")


def _degrade_batch(data: np.ndarray) -> np.ndarray:
    """Apply random_degrade to every image in a batch for test evaluation."""
    rng = np.random.RandomState(0)   # fixed seed so test results are reproducible
    return np.array([random_degrade(img, rng=rng) for img in data], dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Robust EDSR-lite SR training")
    parser.add_argument("--name",  default="sr_robust_model",
                        help="Model name — saved as models/<name>.keras "
                             "(default: sr_robust_model)")
    parser.add_argument("--train", action="store_true",
                        help="Force re-training even if a saved model exists")
    parser.add_argument("--demo",  action="store_true",
                        help="Demo only — load saved model and visualize")
    parser.add_argument("--blocks", type=int, default=8,
                        help="Number of residual blocks (default: 8)")
    args = parser.parse_args()

    MODEL_PATH   = os.path.join(MODELS_DIR,  f"{args.name}.keras")
    RESULTS_PATH = os.path.join(RESULTS_DIR, f"{args.name}_results.png")

    os.makedirs(MODELS_DIR,  exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    need_training = args.train or (not args.demo and not os.path.exists(MODEL_PATH))

    # ------------------------------------------------------------------
    # 1. Dataset
    # ------------------------------------------------------------------
    print("\n===== Step 1: Preparing SR dataset =====")
    train_lr, train_hr, test_lr, test_hr = prepare_sr_data()

    if need_training:
        # --------------------------------------------------------------
        # 2. Build model
        # --------------------------------------------------------------
        print("\n===== Step 2: Building robust SR model =====")
        model = build_sr_model(
            scale=2,
            num_res_blocks=args.blocks,
            input_shape=(150, 150, 3),
        )

        # --------------------------------------------------------------
        # 3. Generators
        #
        # train_gen  — degradation ON, augmentation ON
        # val_gen    — degradation OFF, augmentation OFF
        #
        # Keeping validation clean lets val_loss reflect true SR quality
        # so EarlyStopping fires on meaningful signal, not noise variance.
        # --------------------------------------------------------------
        print("\n===== Step 3: Setting up data generators =====")

        # Reserve last 10% of training data for validation
        n_val   = int(len(train_lr) * 0.1)
        val_lr  = train_lr[-n_val:]
        val_hr  = train_hr[-n_val:]
        trn_lr  = train_lr[:-n_val]
        trn_hr  = train_hr[:-n_val]

        train_gen = RobustSRSequence(
            trn_lr, trn_hr,
            batch_size=16,
            augment=True,
            degrade=True,
        )
        val_gen = RobustSRSequence(
            val_lr, val_hr,
            batch_size=16,
            augment=False,
            degrade=False,   # clean validation
        )

        print(f"  Train batches : {len(train_gen)}  (degraded + augmented)")
        print(f"  Val batches   : {len(val_gen)}   (clean)")

        # --------------------------------------------------------------
        # 4. Train
        # --------------------------------------------------------------
        print("\n===== Step 4: Training =====")
        callbacks = [
            # Stop when val_loss plateaus for 5 epochs
            EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True,
                verbose=1,
            ),
            # Halve LR when val_loss stalls for 3 epochs
            # Helps the model squeeze out the last few dB of PSNR
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=1,
            ),
        ]

        model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=50,
            callbacks=callbacks,
        )

        model.save(MODEL_PATH)
        print(f"\n[train] Model saved → {MODEL_PATH}")

    else:
        # --------------------------------------------------------------
        # Load existing model
        # --------------------------------------------------------------
        print(f"\n===== Loading saved model from {MODEL_PATH} =====")
        model = load_model(
            MODEL_PATH,
            custom_objects={
                "SubPixelConv2D": SubPixelConv2D,
                "combined_loss":  combined_loss,
                "PSNRMetric":     PSNRMetric,
                "SSIMMetric":     SSIMMetric,
            },
        )

    # ------------------------------------------------------------------
    # 5. Evaluate
    #    Run on BOTH clean and degraded test inputs to measure robustness.
    # ------------------------------------------------------------------
    print("\n===== Step 5: Evaluating =====")

    print("  [5a] Clean LR input (best-case):")
    results_clean = model.evaluate(test_lr, test_hr, verbose=1)

    print("  [5b] Degraded LR input (real-world robustness check):")
    test_lr_degraded = _degrade_batch(test_lr)
    results_degraded = model.evaluate(test_lr_degraded, test_hr, verbose=1)

    metric_names = ["loss", "psnr", "ssim"]
    print("\n  ┌─────────────────────────┬────────────────┬─────────────────┐")
    print(  "  │ Metric                  │ Clean input    │ Degraded input  │")
    print(  "  ├─────────────────────────┼────────────────┼─────────────────┤")
    for name, c, d in zip(metric_names, results_clean, results_degraded):
        print(f"  │ {name:<23}  │ {c:>12.4f}   │ {d:>13.4f}   │")
    print(  "  └─────────────────────────┴────────────────┴─────────────────┘")

    # ------------------------------------------------------------------
    # 6. Visualize — show clean and degraded side by side
    # ------------------------------------------------------------------
    print("\n===== Step 6: Visualizing results =====")

    n_show = 5

    print("  Generating SR predictions on clean LR...")
    sr_clean = model.predict(test_lr[:n_show], verbose=0)
    show_sr_results(
        test_lr[:n_show], sr_clean, test_hr[:n_show],
        n=n_show, save_path=RESULTS_PATH,
    )

    degraded_path = RESULTS_PATH.replace("_results.png", "_degraded_results.png")
    print("  Generating SR predictions on degraded LR...")
    sr_degraded = model.predict(test_lr_degraded[:n_show], verbose=0)
    show_sr_results(
        test_lr_degraded[:n_show], sr_degraded, test_hr[:n_show],
        n=n_show, save_path=degraded_path,
    )

    print(f"\n[train_sr] Done.")
    print(f"  Clean results    → {RESULTS_PATH}")
    print(f"  Degraded results → {degraded_path}")


if __name__ == "__main__":
    main()