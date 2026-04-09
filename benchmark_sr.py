#!/usr/bin/env python3
"""Benchmark multiple Super-Resolution models and produce a Markdown report.

Mirrors the structure of benchmark.py but is SR-specific:
  - Loads LR/HR pairs from dataset_sr (no noise dataset needed)
  - Tests each model on BOTH clean and degraded LR inputs
  - Reports PSNR and SSIM (not MSE) as the primary quality metrics
  - Same power sampler as benchmark.py (nvidia / RAPL / AMD hwmon)

Usage
-----
    python sr_benchmark.py                          # uses models/sr_model_registry.json
    python sr_benchmark.py --registry path/to/sr_registry.json
    python sr_benchmark.py --no-power               # skip power sampling

Registry format  (models/sr_model_registry.json)
-------------------------------------------------
{
  "models": [
    {
      "name": "EDSR-lite (clean trained)",
      "model_path": "models/sr_edsr_model.keras",
      "train_time": "18 min",
      "description": "Baseline — trained on clean bicubic LR only."
    },
    {
      "name": "EDSR-lite (robust trained)",
      "model_path": "models/sr_robust_model.keras",
      "train_time": "22 min",
      "description": "Trained with random noise + JPEG + augmentation."
    }
  ]
}

Output
------
    sr_benchmark_results/
        report.md                        Markdown report with embedded images
        images/<model_name>/
            clean_comparison.png         LR → SR → HR  (clean input)
            degraded_comparison.png      degraded LR → SR → HR  (stress test)
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from skimage.metrics import structural_similarity as ssim_skimage
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("[warn] scikit-image not found — SSIM will use tf.image.ssim instead.")

import tensorflow as tf
from tensorflow.keras.models import load_model  # type: ignore

from dataset_sr import prepare_sr_data
from noise import random_degrade
from model_sr import SubPixelConv2D, combined_loss, PSNRMetric, SSIMMetric
from utils import plot_rgb_img

# ---------------------------------------------------------------------------
# Output dirs
# ---------------------------------------------------------------------------
_BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(_BASE_DIR, "sr_benchmark_results")
IMAGES_DIR  = os.path.join(RESULTS_DIR, "images")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR,  exist_ok=True)

N_SAMPLES       = 5      # image triplets saved per model
POWER_SAMPLE_HZ = 0.1   # seconds between power readings


# ===========================================================================
# Power sampling  (identical to benchmark.py)
# ===========================================================================

_RAPL_PATH = "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj"


def _find_amd_hwmon_path() -> str | None:
    hwmon_root = "/sys/class/hwmon"
    if not os.path.isdir(hwmon_root):
        return None
    for hw in os.listdir(hwmon_root):
        name_file = os.path.join(hwmon_root, hw, "name")
        try:
            with open(name_file) as f:
                if "k10temp" in f.read():
                    p = os.path.join(hwmon_root, hw, "power1_input")
                    if os.path.exists(p):
                        return p
        except OSError:
            continue
    return None


_AMD_HWMON_PATH = _find_amd_hwmon_path()


def _detect_power_source() -> str:
    try:
        subprocess.check_output(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, timeout=2,
        )
        return "nvidia"
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        pass
    if os.path.exists(_RAPL_PATH):
        try:
            with open(_RAPL_PATH):
                pass
            return "rapl"
        except PermissionError:
            print(
                "[power] Intel RAPL found but not readable.\n"
                "  Fix:  sudo chmod a+r /sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj"
            )
    if _AMD_HWMON_PATH:
        return "amd_hwmon"
    return "none"


POWER_SOURCE = _detect_power_source()
print(f"[power] Source detected: {POWER_SOURCE}")


def _read_watts_nvidia() -> float | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
        )
        return float(out.strip())
    except Exception:
        return None


def _read_watts_rapl() -> float | None:
    try:
        def _read():
            with open(_RAPL_PATH) as f:
                return int(f.read().strip())
        e0 = _read()
        time.sleep(POWER_SAMPLE_HZ)
        e1 = _read()
        delta_uj = e1 - e0
        if delta_uj < 0:
            delta_uj += 2 ** 32
        return delta_uj / 1e6 / POWER_SAMPLE_HZ
    except Exception:
        return None


def _read_watts_amd() -> float | None:
    try:
        with open(_AMD_HWMON_PATH) as f:
            uw = int(f.read().strip())
        return uw / 1e6
    except Exception:
        return None


_POWER_READERS = {
    "nvidia"    : _read_watts_nvidia,
    "rapl"      : _read_watts_rapl,
    "amd_hwmon" : _read_watts_amd,
    "none"      : lambda: None,
}


class PowerSampler:
    """Background-thread power sampler — identical to benchmark.py."""

    def __init__(self):
        self.readings: list[float] = []
        self._stop_event = threading.Event()
        self._reader = _POWER_READERS[POWER_SOURCE]

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join()

    def _loop(self):
        while not self._stop_event.is_set():
            val = self._reader()
            if val is not None:
                self.readings.append(val)
            if POWER_SOURCE != "rapl":
                time.sleep(POWER_SAMPLE_HZ)

    @property
    def mean_watts(self) -> float | None:
        return float(np.mean(self.readings)) if self.readings else None

    @property
    def available(self) -> bool:
        return POWER_SOURCE != "none"


# ===========================================================================
# Metrics
# ===========================================================================

def compute_psnr(original: np.ndarray, output: np.ndarray) -> float:
    """PSNR in dB. Both float32 in [0,1]."""
    mse = float(np.mean((original.astype(np.float32) - output.astype(np.float32)) ** 2))
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(1.0 / mse)


def compute_ssim(original: np.ndarray, output: np.ndarray) -> float:
    """Mean SSIM over a batch. Falls back to tf.image.ssim if skimage missing."""
    original = original.astype(np.float32)
    output   = output.astype(np.float32)

    if HAS_SKIMAGE:
        scores = [
            ssim_skimage(o, d, data_range=1.0, channel_axis=-1)
            for o, d in zip(original, output)
        ]
        return float(np.mean(scores))

    # TF fallback
    scores = tf.image.ssim(
        tf.constant(original), tf.constant(output), max_val=1.0
    )
    return float(tf.reduce_mean(scores).numpy())


def compute_mse(original: np.ndarray, output: np.ndarray) -> float:
    return float(np.mean((original.astype(np.float32) - output.astype(np.float32)) ** 2))


def compute_inference_and_power(
    model,
    sample_batch: np.ndarray,
    measure_power: bool = True,
) -> tuple[float, float | None]:
    """Timed inference pass with concurrent power sampling.

    Returns (ms_per_image, mean_watts).
    A warm-up call is made first to trigger XLA compilation so timing
    reflects steady-state inference, not graph compilation.
    """
    model.predict(sample_batch[:1], verbose=0)   # warm-up

    sampler = PowerSampler()
    if measure_power and sampler.available:
        sampler.start()

    t0 = time.perf_counter()
    model.predict(sample_batch, verbose=0)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    if measure_power and sampler.available:
        sampler.stop()

    return elapsed_ms / len(sample_batch), sampler.mean_watts


# ===========================================================================
# Degraded test input
# ===========================================================================

def _make_degraded(lr_data: np.ndarray) -> np.ndarray:
    """Apply random_degrade to every LR image with a fixed seed.

    Fixed seed ensures the degraded test set is identical across all
    model benchmarks so comparisons are fair.
    """
    rng = np.random.RandomState(0)
    return np.array(
        [random_degrade(img, rng=rng) for img in lr_data],
        dtype=np.float32,
    )


# ===========================================================================
# Comparison grid saving
# ===========================================================================

def _to_uint8(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    return np.clip(img * 255, 0, 255).astype(np.uint8)


def save_sr_grid(
    model_name: str,
    lr_input: np.ndarray,
    sr_output: np.ndarray,
    hr_target: np.ndarray,
    suffix: str,
    n: int = N_SAMPLES,
) -> str:
    """Save a 3-column grid: LR input | SR output | HR ground truth.

    Parameters
    ----------
    suffix : str
        "clean" or "degraded" — appended to the filename.
    """
    n = min(n, len(lr_input))
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    hr_h, hr_w = hr_target.shape[1], hr_target.shape[2]
    titles = [
        f"LR input ({suffix})",
        "SR output",
        "HR ground truth",
    ]

    for i in range(n):
        # Upscale LR visually to HR size for side-by-side comparison
        lr_vis = cv2.resize(
            _to_uint8(lr_input[i]), (hr_w, hr_h),
            interpolation=cv2.INTER_NEAREST,
        )
        imgs = [lr_vis / 255.0, sr_output[i], hr_target[i]]
        for j, img in enumerate(imgs):
            axes[i, j].imshow(plot_rgb_img(img.astype(np.float32)))
            axes[i, j].set_title(titles[j], fontsize=12)
            axes[i, j].axis("off")

    plt.suptitle(f"{model_name}  —  {suffix} input", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    model_img_dir = os.path.join(IMAGES_DIR, model_name.replace(" ", "_"))
    os.makedirs(model_img_dir, exist_ok=True)
    save_path = os.path.join(model_img_dir, f"{suffix}_comparison.png")
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return save_path


# ===========================================================================
# Markdown report
# ===========================================================================

def _rel(path: str) -> str:
    return os.path.relpath(path, RESULTS_DIR)


def generate_report(results: list[dict], output_path: str) -> None:
    has_power = any(r.get("mean_watts") is not None for r in results)

    lines = [
        "# Super-Resolution Model Benchmark Report",
        "",
        "> Auto-generated by `sr_benchmark.py`",
        "",
        "---",
        "",
        "## Summary Table",
        "",
        "> PSNR and SSIM are reported for **clean LR input** and **degraded LR input** separately.",
        "> The gap between them shows how robust each model is to real-world degradation.",
        "",
    ]

    # --- Summary table ---
    cols = [
        "Model", "Train Time",
        "PSNR clean", "SSIM clean",
        "PSNR degraded", "SSIM degraded",
        "PSNR drop", "Inference (ms/img)",
    ]
    if has_power:
        cols += ["Avg Power (W)", "PSNR/W (clean)", "FPS/W"]

    lines += ["| " + " | ".join(cols) + " |"]
    lines += ["|" + "|".join(["---"] * len(cols)) + "|"]

    for r in results:
        fps        = 1000.0 / r["inference_ms_per_img"]
        psnr_drop  = r["psnr_clean"] - r["psnr_degraded"]

        row = [
            r["name"],
            r.get("train_time_human", "—"),
            f"{r['psnr_clean']:.2f}",
            f"{r['ssim_clean']:.4f}",
            f"{r['psnr_degraded']:.2f}",
            f"{r['ssim_degraded']:.4f}",
            f"{psnr_drop:.2f}",
            f"{r['inference_ms_per_img']:.1f}",
        ]
        if has_power:
            if r.get("mean_watts") is not None:
                row += [
                    f"{r['mean_watts']:.1f}",
                    f"{r['psnr_per_watt']:.3f}",
                    f"{fps / r['mean_watts']:.3f}",
                ]
            else:
                row += ["—", "—", "—"]

        lines.append("| " + " | ".join(row) + " |")

    lines += ["", "---", "", "## Per-Model Results", ""]

    for r in results:
        fps       = 1000.0 / r["inference_ms_per_img"]
        psnr_drop = r["psnr_clean"] - r["psnr_degraded"]

        lines += [
            f"### {r['name']}",
            "",
            f"- **Train time**         : {r.get('train_time_human', '—')}",
            f"- **MSE (clean)**        : {r['mse_clean']:.6f}",
            f"- **PSNR (clean)**       : {r['psnr_clean']:.2f} dB",
            f"- **SSIM (clean)**       : {r['ssim_clean']:.4f}",
            f"- **MSE (degraded)**     : {r['mse_degraded']:.6f}",
            f"- **PSNR (degraded)**    : {r['psnr_degraded']:.2f} dB",
            f"- **SSIM (degraded)**    : {r['ssim_degraded']:.4f}",
            f"- **PSNR drop**          : {psnr_drop:.2f} dB  ← robustness gap",
            f"- **Inference speed**    : {r['inference_ms_per_img']:.1f} ms / image  "
            f"({fps:.1f} FPS)",
        ]

        if r.get("mean_watts") is not None:
            lines += [
                f"- **Avg power draw**     : {r['mean_watts']:.1f} W  "
                f"(source: `{POWER_SOURCE}`)",
                f"- **PSNR / W (clean)**   : {r['psnr_per_watt']:.3f} dB/W",
                f"- **FPS / W**            : {fps / r['mean_watts']:.3f}",
            ]
        else:
            lines.append(f"- **Power draw**         : — (source `{POWER_SOURCE}` unavailable)")

        lines.append("")

        if r.get("description"):
            lines += [f"> {r['description']}", ""]

        clean_rel    = _rel(r["clean_grid"])
        degraded_rel = _rel(r["degraded_grid"])

        lines += [
            "#### Clean LR input  (LR → SR → HR)",
            "",
            f"![{r['name']} clean]({clean_rel})",
            "",
            "#### Degraded LR input  (noisy/JPEG LR → SR → HR)",
            "",
            f"![{r['name']} degraded]({degraded_rel})",
            "",
            "---",
            "",
        ]

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"[sr_benchmark] Report written → {output_path}")


# ===========================================================================
# Per-model benchmark runner
# ===========================================================================

def benchmark_sr_model(
    entry: dict,
    test_lr_clean: np.ndarray,
    test_lr_degraded: np.ndarray,
    test_hr: np.ndarray,
    measure_power: bool = True,
) -> dict | None:

    name       = entry["name"]
    model_path = entry["model_path"]

    print(f"\n{'='*60}")
    print(f"  Benchmarking : {name}")
    print(f"  Path         : {model_path}")
    print(f"{'='*60}")

    if not os.path.exists(model_path):
        print(f"  [skip] File not found: {model_path}")
        return None

    # Load with all custom objects so both old and new model variants work
    model = load_model(
        model_path,
        custom_objects={
            "SubPixelConv2D": SubPixelConv2D,
            "combined_loss" : combined_loss,
            "PSNRMetric"    : PSNRMetric,
            "SSIMMetric"    : SSIMMetric,
        },
    )

    # ------------------------------------------------------------------
    # Inference on CLEAN LR
    # ------------------------------------------------------------------
    sr_clean = model.predict(test_lr_clean[:N_SAMPLES], verbose=0)

    mse_clean  = compute_mse(test_hr[:N_SAMPLES],  sr_clean)
    psnr_clean = compute_psnr(test_hr[:N_SAMPLES], sr_clean)
    ssim_clean = compute_ssim(test_hr[:N_SAMPLES], sr_clean)

    # Inference time + power — measured on clean input
    inf_ms, mean_watts = compute_inference_and_power(
        model, test_lr_clean[:N_SAMPLES], measure_power=measure_power
    )

    # ------------------------------------------------------------------
    # Inference on DEGRADED LR  (robustness stress test)
    # ------------------------------------------------------------------
    sr_degraded = model.predict(test_lr_degraded[:N_SAMPLES], verbose=0)

    mse_degraded  = compute_mse(test_hr[:N_SAMPLES],  sr_degraded)
    psnr_degraded = compute_psnr(test_hr[:N_SAMPLES], sr_degraded)
    ssim_degraded = compute_ssim(test_hr[:N_SAMPLES], sr_degraded)

    psnr_per_watt = (psnr_clean / mean_watts) if mean_watts else None
    fps           = 1000.0 / inf_ms
    psnr_drop     = psnr_clean - psnr_degraded

    # ------------------------------------------------------------------
    # Console output
    # ------------------------------------------------------------------
    print(f"\n  {'Metric':<22}  {'Clean LR':>12}  {'Degraded LR':>13}")
    print(f"  {'-'*50}")
    print(f"  {'MSE':<22}  {mse_clean:>12.6f}  {mse_degraded:>13.6f}")
    print(f"  {'PSNR (dB)':<22}  {psnr_clean:>12.2f}  {psnr_degraded:>13.2f}")
    print(f"  {'SSIM':<22}  {ssim_clean:>12.4f}  {ssim_degraded:>13.4f}")
    print(f"  {'PSNR drop':<22}  {psnr_drop:>12.2f} dB  ← robustness gap")
    print(f"  {'Inference':<22}  {inf_ms:>10.1f} ms/img  ({fps:.1f} FPS)")
    if mean_watts is not None:
        print(f"  {'Power':<22}  {mean_watts:>10.1f} W  (source: {POWER_SOURCE})")
        print(f"  {'PSNR/W (clean)':<22}  {psnr_per_watt:>10.3f} dB/W")
        print(f"  {'FPS/W':<22}  {fps / mean_watts:>10.3f}")
    else:
        print(f"  Power: not available (source: {POWER_SOURCE})")

    # ------------------------------------------------------------------
    # Save comparison grids
    # ------------------------------------------------------------------
    clean_grid = save_sr_grid(
        name, test_lr_clean[:N_SAMPLES], sr_clean, test_hr[:N_SAMPLES],
        suffix="clean",
    )
    degraded_grid = save_sr_grid(
        name, test_lr_degraded[:N_SAMPLES], sr_degraded, test_hr[:N_SAMPLES],
        suffix="degraded",
    )

    # Write inference + power back to history file so plots.py can read them
    model_stem = os.path.basename(model_path).replace(".keras", "")
    hpath = os.path.join(_BASE_DIR, "history", f"{model_stem}_history.json")
    if os.path.exists(hpath):
        with open(hpath) as _f:
            _payload = json.load(_f)
        if "eval" in _payload:
            _payload["eval"]["inference_ms"] = inf_ms
            _payload["eval"]["mean_watts"]   = mean_watts
            with open(hpath, "w") as _f:
                json.dump(_payload, _f, indent=2)

    return {
        "name"                : name,
        "model_path"          : model_path,
        "train_time_human"    : entry.get("train_time", "—"),
        "description"         : entry.get("description", ""),
        "mse_clean"           : mse_clean,
        "psnr_clean"          : psnr_clean,
        "ssim_clean"          : ssim_clean,
        "mse_degraded"        : mse_degraded,
        "psnr_degraded"       : psnr_degraded,
        "ssim_degraded"       : ssim_degraded,
        "inference_ms_per_img": inf_ms,
        "mean_watts"          : mean_watts,
        "psnr_per_watt"       : psnr_per_watt,
        "clean_grid"          : clean_grid,
        "degraded_grid"       : degraded_grid,
    }


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark SR models")
    parser.add_argument(
        "--registry",
        default=os.path.join(_BASE_DIR, "models", "sr_model_registry.json"),
        help="Path to sr_model_registry.json",
    )
    parser.add_argument(
        "--no-power",
        action="store_true",
        help="Skip power sampling — useful for a quick quality-only run",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Registry
    # ------------------------------------------------------------------
    if not os.path.exists(args.registry):
        # Helpful bootstrap: create a template registry if none exists
        os.makedirs(os.path.dirname(args.registry), exist_ok=True)
        template = {
            "models": [
                {
                    "name"       : "EDSR-lite (clean trained)",
                    "model_path" : "models/sr_edsr_model.keras",
                    "train_time" : "—",
                    "description": "Baseline — trained on clean bicubic LR only.",
                },
                {
                    "name"       : "EDSR-lite (robust trained)",
                    "model_path" : "models/sr_robust_model.keras",
                    "train_time" : "—",
                    "description": "Trained with random noise + JPEG + augmentation.",
                },
            ]
        }
        with open(args.registry, "w") as f:
            json.dump(template, f, indent=2)
        print(f"[sr_benchmark] Created template registry → {args.registry}")
        print("  Edit it to match your model paths, then re-run.")
        sys.exit(0)

    with open(args.registry) as f:
        registry = json.load(f)

    print(f"[sr_benchmark] {len(registry['models'])} model(s) in registry")

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    print("\n[sr_benchmark] Preparing SR test dataset...")
    _, _, test_lr_clean, test_hr = prepare_sr_data()

    print("[sr_benchmark] Generating degraded test inputs (fixed seed)...")
    test_lr_degraded = _make_degraded(test_lr_clean)

    measure_power = not args.no_power

    # ------------------------------------------------------------------
    # Benchmark each model
    # ------------------------------------------------------------------
    results = []
    for entry in registry["models"]:
        result = benchmark_sr_model(
            entry,
            test_lr_clean,
            test_lr_degraded,
            test_hr,
            measure_power=measure_power,
        )
        if result:
            results.append(result)

    if not results:
        print("[sr_benchmark] No models were successfully benchmarked.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    report_path = os.path.join(RESULTS_DIR, "report.md")
    generate_report(results, report_path)
    print(f"\n[sr_benchmark] Done. Open sr_benchmark_results/report.md to view results.")


if __name__ == "__main__":
    main()