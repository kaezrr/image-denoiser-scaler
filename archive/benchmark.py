#!/usr/bin/env python3
"""Benchmark multiple denoising models and produce a Markdown report.

Usage
-----
    python benchmark.py                        # uses models/model_registry.json
    python benchmark.py --registry path/to/registry.json
    python benchmark.py --no-power             # skip power sampling (faster)

Output
------
    benchmark_results/
        report.md                  ← Markdown report with embedded images
        images/<model_name>/       ← noisy / denoised / original PNGs per model

Power sampling (Arch Linux)
---------------------------
Nvidia GPU  →  nvidia-smi  (works out of the box if driver is installed)
CPU / iGPU  →  RAPL via /sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj
               (Intel only; AMD CPUs expose energy via /sys/class/hwmon instead)

If neither source is available the script still runs — power columns just
show "—" in the report.
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless — no display needed
import matplotlib.pyplot as plt

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, _THIS_DIR)

# ---------------------------------------------------------------------------
# Optional deps — warn clearly if missing
# ---------------------------------------------------------------------------
try:
    from skimage.metrics import structural_similarity as ssim
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("[warn] scikit-image not found — SSIM will be skipped. "
          "Install with:  pip install scikit-image")

import tensorflow as tf                          # type: ignore
import keras                                     # type: ignore
from tensorflow.keras.models import load_model  # type: ignore

from archive.dataset_de import prepare_data
from archive.noise import add_gaussian_to_dataset

# ---------------------------------------------------------------------------
# Custom layer stubs — must be registered HERE in benchmark.py so that
# load_model() can find them when deserializing any model that used them.
#
# Why they live here too and not just in model.py:
#   Keras serializes the *class name* into the .keras file.  When you call
#   load_model() in benchmark.py, Python hasn't imported model.py, so the
#   classes don't exist yet and Keras raises "Could not locate class".
#   The @register_keras_serializable decorator adds them to Keras's global
#   registry at import time — so as long as benchmark.py is run, loading
#   works regardless of where the model was originally defined.
#
# keras.saving.register_keras_serializable() is the correct path for
# standalone Keras 2/3.  tf.keras.saving doesn't exist in all TF builds.
# ---------------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class CropToMatch(keras.layers.Layer):
    """Center-crop skip tensor to match target spatial dims."""
    def call(self, inputs):
        skip, target = inputs
        sh = tf.shape(skip)
        th = tf.shape(target)
        dh = sh[1] - th[1]
        dw = sh[2] - th[2]
        return skip[
            :,
            dh // 2 : sh[1] - (dh - dh // 2),
            dw // 2 : sh[2] - (dw - dw // 2),
            :,
        ]
    def compute_output_shape(self, input_shape):
        skip_shape, target_shape = input_shape
        return (skip_shape[0], target_shape[1], target_shape[2], skip_shape[3])
    def get_config(self):
        return super().get_config()


@keras.saving.register_keras_serializable()
class ResizeTo(keras.layers.Layer):
    """Bilinear resize to a fixed (H, W)."""
    def __init__(self, target_h: int, target_w: int, **kwargs):
        super().__init__(**kwargs)
        self.target_h = target_h
        self.target_w = target_w
    def call(self, x):
        return tf.image.resize(x, (self.target_h, self.target_w))
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.target_h, self.target_w, input_shape[3])
    def get_config(self):
        return {**super().get_config(), "target_h": self.target_h, "target_w": self.target_w}

# ---------------------------------------------------------------------------
# Output dirs
# ---------------------------------------------------------------------------
RESULTS_DIR = os.path.join(_PROJECT_ROOT, "benchmark_results")
IMAGES_DIR  = os.path.join(RESULTS_DIR, "images")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR,  exist_ok=True)

N_SAMPLES        = 5     # image triplets saved per model
POWER_SAMPLE_HZ  = 0.1  # seconds between power readings (100 ms)


# ===========================================================================
# Power sampling — detects what is available on this machine automatically
# ===========================================================================

# Intel RAPL energy counter (standard path on Arch Linux)
_RAPL_PATH = "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj"


def _find_amd_hwmon_path() -> str | None:
    """Find AMD CPU power file under /sys/class/hwmon (k10temp driver)."""
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
    """
    Returns one of: "nvidia", "rapl", "amd_hwmon", "none".

    Priority: Nvidia GPU > Intel RAPL > AMD hwmon > nothing.
    This matches the most common Arch Linux setups.
    """
    # Nvidia — try a quick nvidia-smi call
    try:
        subprocess.check_output(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        return "nvidia"
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        pass

    # Intel RAPL
    if os.path.exists(_RAPL_PATH):
        try:
            with open(_RAPL_PATH):
                pass
            return "rapl"
        except PermissionError:
            print(
                "[power] Intel RAPL found but not readable.\n"
                "  Fix with:  sudo chmod a+r /sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj\n"
                "  Or run benchmark.py as root (not recommended)."
            )

    # AMD hwmon
    if _AMD_HWMON_PATH:
        return "amd_hwmon"

    return "none"


POWER_SOURCE = _detect_power_source()
print(f"[power] Source detected: {POWER_SOURCE}")


def _read_watts_nvidia() -> float | None:
    """Read instantaneous GPU power draw via nvidia-smi (Watts)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
        )
        return float(out.strip())
    except Exception:
        return None


def _read_watts_rapl() -> float | None:
    """
    Read Intel RAPL energy counter (µJ) and convert to Watts.

    RAPL gives cumulative energy — not instantaneous power — so we read
    it twice with a short gap and compute:

        watts = (E1 - E0) in joules / time in seconds

    The counter occasionally wraps at 2^32 µJ; we handle that case.
    """
    try:
        def _read():
            with open(_RAPL_PATH) as f:
                return int(f.read().strip())

        e0 = _read()
        time.sleep(POWER_SAMPLE_HZ)
        e1 = _read()
        delta_uj = e1 - e0
        if delta_uj < 0:           # counter wrap-around
            delta_uj += 2 ** 32
        return delta_uj / 1e6 / POWER_SAMPLE_HZ   # µJ → J → W
    except Exception:
        return None


def _read_watts_amd() -> float | None:
    """Read AMD CPU power from hwmon (µW) and convert to Watts."""
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
    """
    Samples power draw in a background thread.

    Start it just before model.predict(), stop it just after.
    Then read .mean_watts for the average power over that window.

    Why a thread?
    -------------
    model.predict() blocks the main thread. We need concurrent sampling
    so the power readings cover exactly the inference window, not the
    entire script. Python threads release the GIL for I/O (subprocess,
    file reads), so the sampler runs freely alongside TF's C++ compute.
    """

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
            # RAPL reader sleeps internally; others need an explicit pause
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

def compute_psnr(original: np.ndarray, denoised: np.ndarray) -> float:
    """PSNR in dB. Both arrays float32 in [0, 1]."""
    mse = float(np.mean((original - denoised) ** 2))
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(1.0 / mse)


def compute_ssim(original: np.ndarray, denoised: np.ndarray) -> float | None:
    """Mean SSIM over a batch. Returns None if scikit-image unavailable."""
    if not HAS_SKIMAGE:
        return None
    scores = []
    for o, d in zip(original, denoised):
        scores.append(ssim(o, d, data_range=1.0, channel_axis=-1))
    return float(np.mean(scores))


def compute_inference_and_power(
    model,
    sample_batch: np.ndarray,
    measure_power: bool = True,
) -> tuple[float, float | None]:
    """
    Run a timed inference pass and sample power concurrently.

    Returns
    -------
    ms_per_img : float        — mean inference time per image in ms
    mean_watts : float | None — mean power draw in W (None if unavailable)

    Why a warm-up pass?
    -------------------
    TensorFlow compiles the compute graph on the very first call.
    That compile step (can be 1-3 s) would massively inflate both time
    and power readings. The warm-up triggers compilation; the measured
    pass captures only steady-state inference.
    """
    # Warm-up — not timed, not power-sampled
    model.predict(sample_batch[:1], verbose=0)

    sampler = PowerSampler()

    if measure_power and sampler.available:
        sampler.start()

    t0 = time.perf_counter()
    model.predict(sample_batch, verbose=0)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    if measure_power and sampler.available:
        sampler.stop()

    ms_per_img = elapsed_ms / len(sample_batch)
    mean_watts = sampler.mean_watts

    return ms_per_img, mean_watts


# ===========================================================================
# Image saving
# ===========================================================================

def _to_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.float16:
        img = img.astype(np.float32)
    return np.clip(img * 255, 0, 255).astype(np.uint8)


def save_comparison_grid(
    model_name: str,
    noisy: np.ndarray,
    denoised: np.ndarray,
    original: np.ndarray,
    n: int = N_SAMPLES,
) -> str:
    """Save side-by-side grid (noisy | denoised | original) and return path."""
    n = min(n, len(noisy))
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    titles = ["Noisy", "Denoised", "Original"]
    for i in range(n):
        for j, img in enumerate([noisy[i], denoised[i], original[i]]):
            rgb = _to_uint8(img)[..., ::-1]   # BGR → RGB for matplotlib
            axes[i, j].imshow(rgb)
            axes[i, j].set_title(titles[j], fontsize=13)
            axes[i, j].axis("off")

    plt.suptitle(model_name, fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()

    model_img_dir = os.path.join(IMAGES_DIR, model_name.replace(" ", "_"))
    os.makedirs(model_img_dir, exist_ok=True)
    save_path = os.path.join(model_img_dir, "comparison.png")
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return save_path


# ===========================================================================
# Markdown report
# ===========================================================================

def _rel(path: str) -> str:
    return os.path.relpath(path, RESULTS_DIR)


def generate_report(results: list[dict], output_path: str) -> None:
    has_ssim  = any(r.get("ssim")       is not None for r in results)
    has_power = any(r.get("mean_watts") is not None for r in results)

    lines = [
        "# Model Benchmark Report",
        "",
        "> Auto-generated by `benchmark.py`",
        "",
        "---",
        "",
        "## Summary Table",
        "",
    ]

    # Build column list based on what data is actually available
    cols = ["Model", "Train Time", "MSE", "PSNR (dB)"]
    if has_ssim:
        cols.append("SSIM")
    cols.append("Inference (ms/img)")
    if has_power:
        cols += ["Avg Power (W)", "PSNR / W", "FPS / W"]

    header = "| " + " | ".join(cols) + " |"
    sep    = "|" + "|".join(["---"] * len(cols)) + "|"
    lines += [header, sep]

    for r in results:
        fps = 1000.0 / r["inference_ms_per_img"]
        row_vals = [
            r["name"],
            r.get("train_time_human", "—"),
            f"{r['mse']:.6f}",
            f"{r['psnr']:.2f}",
        ]
        if has_ssim:
            row_vals.append(f"{r['ssim']:.4f}" if r.get("ssim") is not None else "—")
        row_vals.append(f"{r['inference_ms_per_img']:.1f}")
        if has_power:
            if r.get("mean_watts") is not None:
                row_vals.append(f"{r['mean_watts']:.1f}")
                row_vals.append(f"{r['psnr_per_watt']:.3f}")
                row_vals.append(f"{fps / r['mean_watts']:.3f}")
            else:
                row_vals += ["—", "—", "—"]

        lines.append("| " + " | ".join(row_vals) + " |")

    lines += ["", "---", "", "## Per-Model Results", ""]

    for r in results:
        fps = 1000.0 / r["inference_ms_per_img"]
        lines += [
            f"### {r['name']}",
            "",
            f"- **Train time**      : {r.get('train_time_human', '—')}",
            f"- **MSE**             : {r['mse']:.6f}",
            f"- **PSNR**            : {r['psnr']:.2f} dB",
        ]
        if r.get("ssim") is not None:
            lines.append(f"- **SSIM**            : {r['ssim']:.4f}")
        lines.append(f"- **Inference speed** : {r['inference_ms_per_img']:.1f} ms / image")

        if r.get("mean_watts") is not None:
            lines += [
                f"- **Avg power draw**  : {r['mean_watts']:.1f} W  "
                f"(source: `{POWER_SOURCE}`)",
                f"- **PSNR / W**        : {r['psnr_per_watt']:.3f} dB/W"
                f"  ← accuracy-per-watt",
                f"- **FPS / W**         : {fps / r['mean_watts']:.3f}"
                f"  ← throughput-per-watt",
            ]
        else:
            lines.append(
                f"- **Power draw**      : — (source `{POWER_SOURCE}` unavailable)"
            )

        lines.append("")

        if r.get("description"):
            lines += [f"> {r['description']}", ""]

        img_rel = _rel(r["comparison_image"])
        lines += [
            "#### Sample Results  (Noisy → Denoised → Original)",
            "",
            f"![{r['name']} comparison]({img_rel})",
            "",
            "---",
            "",
        ]

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"[benchmark] Report written → {output_path}")


# ===========================================================================
# Per-model benchmark runner
# ===========================================================================

def benchmark_model(
    entry: dict,
    noisy_test: np.ndarray,
    test_data: np.ndarray,
    measure_power: bool = True,
) -> dict | None:

    name = entry["name"]
    raw_model_path = entry["model_path"]
    model_path = raw_model_path if os.path.isabs(raw_model_path) else os.path.join(_PROJECT_ROOT, raw_model_path)

    print(f"\n{'='*60}")
    print(f"  Benchmarking : {name}")
    print(f"  Path         : {model_path}")
    print(f"{'='*60}")

    if not os.path.exists(model_path):
        print(f"  [skip] File not found: {model_path}")
        return None

    model = load_model(
        model_path,
        custom_objects={"CropToMatch": CropToMatch, "ResizeTo": ResizeTo},
        safe_mode=False,
    )

    # MSE over the full test set
    mse = float(model.evaluate(noisy_test, test_data, verbose=0))

    # Inference time + power measured in the same window
    inf_ms, mean_watts = compute_inference_and_power(
        model, noisy_test[:N_SAMPLES], measure_power=measure_power
    )

    # Denoised images for PSNR / SSIM / visuals
    denoised = model.predict(noisy_test[:N_SAMPLES], verbose=0)

    psnr       = compute_psnr(test_data[:N_SAMPLES], denoised)
    ssim_score = compute_ssim(test_data[:N_SAMPLES], denoised)

    # The accuracy-per-watt number
    psnr_per_watt = (psnr / mean_watts) if mean_watts else None

    img_path = save_comparison_grid(
        name, noisy_test[:N_SAMPLES], denoised, test_data[:N_SAMPLES]
    )

    fps = 1000.0 / inf_ms
    print(f"  MSE          : {mse:.6f}")
    print(f"  PSNR         : {psnr:.2f} dB")
    if ssim_score is not None:
        print(f"  SSIM         : {ssim_score:.4f}")
    print(f"  Inference    : {inf_ms:.1f} ms/img  ({fps:.1f} FPS)")
    if mean_watts is not None:
        print(f"  Power        : {mean_watts:.1f} W  (source: {POWER_SOURCE})")
        print(f"  PSNR / W     : {psnr_per_watt:.3f} dB/W")
        print(f"  FPS  / W     : {fps / mean_watts:.3f}")
    else:
        print(f"  Power        : not available (source: {POWER_SOURCE})")

    return {
        "name"                : name,
        "model_path"          : raw_model_path,
        "train_time_human"    : entry.get("train_time", "—"),
        "description"         : entry.get("description", ""),
        "mse"                 : mse,
        "psnr"                : psnr,
        "ssim"                : ssim_score,
        "inference_ms_per_img": inf_ms,
        "mean_watts"          : mean_watts,
        "psnr_per_watt"       : psnr_per_watt,
        "comparison_image"    : img_path,
    }


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark denoising models")
    parser.add_argument(
        "--registry",
        default=os.path.join(_PROJECT_ROOT, "models", "model_registry.json"),
        help="Path to model_registry.json",
    )
    parser.add_argument(
        "--no-power",
        action="store_true",
        help="Skip power sampling — useful for a quick quality-only run",
    )
    args = parser.parse_args()

    if not os.path.exists(args.registry):
        print(f"[error] Registry not found: {args.registry}")
        print("  Create models/model_registry.json — see README for the format.")
        sys.exit(1)

    with open(args.registry) as f:
        registry = json.load(f)

    print(f"[benchmark] {len(registry['models'])} model(s) in registry")

    print("\n[benchmark] Preparing test dataset...")
    _, test_data = prepare_data()
    noisy_test   = add_gaussian_to_dataset(test_data)

    measure_power = not args.no_power

    results = []
    for entry in registry["models"]:
        result = benchmark_model(entry, noisy_test, test_data, measure_power=measure_power)
        if result:
            results.append(result)

    if not results:
        print("[benchmark] No models were successfully benchmarked.")
        sys.exit(1)

    report_path = os.path.join(RESULTS_DIR, "report.md")
    generate_report(results, report_path)
    print(f"\n[benchmark] Done. Open benchmark_results/report.md to view results.")


if __name__ == "__main__":
    main()