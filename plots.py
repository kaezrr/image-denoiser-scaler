#!/usr/bin/env python3
"""Generate all diagnostic plots for trained SR models.

Reads history JSON files written by train_sr.py and runs live inference
for the visual/robustness plots that need model predictions.

Plots produced
--------------
Training curves (per model, from history JSON)
    1.  loss_curve.png          — train + val combined loss vs epoch
    2.  psnr_curve.png          — train + val PSNR vs epoch
    3.  ssim_curve.png          — train + val SSIM vs epoch
    4.  lr_schedule.png         — learning rate vs epoch (shows ReduceLROnPlateau drops)
    5.  all_curves.png          — all four stacked on shared x-axis, publication-ready

Model comparison (across all models in sr_model_registry.json)
    6.  psnr_comparison.png     — clean vs degraded PSNR grouped bar chart
    7.  ssim_comparison.png     — clean vs degraded SSIM grouped bar chart
    8.  speed_vs_psnr.png       — inference ms/img vs PSNR scatter (pareto front)
    9.  psnr_per_watt.png       — PSNR/W bar chart (skipped if no power data)

Robustness analysis (needs model + test data)
    10. psnr_vs_jpeg.png        — PSNR vs JPEG quality (60→10) per model
    11. psnr_vs_noise.png       — PSNR vs Gaussian noise std (10→80) per model
    12. degradation_breakdown.png — PSNR per degradation type, grouped bar chart
    13. psnr_drop_heatmap.png   — models × degradation types heatmap

Visual / qualitative (needs model + test data)
    14. residual_maps.png       — |SR − HR| × 10 heatmaps for N test images
    15. pixel_histogram.png     — pixel value distributions: HR vs SR (clean) vs SR (degraded)
    16. frequency_analysis.png  — FFT magnitude comparison: HR vs SR outputs

Usage
-----
    python plots.py                          # uses sr_model_registry.json, all plots
    python plots.py --no-model               # skip plots that need live inference
    python plots.py --model robust_sr        # single model training curves only
    python plots.py --n 3                    # use 3 test images for visual plots
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tensorflow.keras.models import load_model  # type: ignore
from model_sr import SubPixelConv2D, combined_loss, PSNRMetric, SSIMMetric
from noise_sr import gaussian_noise, salt_and_pepper_noise, jpeg_compression, random_degrade
from dataset_sr import prepare_sr_data

_BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
HISTORY_DIR = os.path.join(_BASE_DIR, "history")
MODELS_DIR  = os.path.join(_BASE_DIR, "models")
PLOTS_DIR   = os.path.join(_BASE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Consistent colour palette across all plots
_C_TRAIN    = "#378ADD"   # blue  — training curves
_C_VAL      = "#1D9E75"   # teal  — validation curves
_C_CLEAN    = "#378ADD"   # blue  — clean input bars
_C_DEGRADED = "#D85A30"   # coral — degraded input bars
_C_HR       = "#639922"   # green — ground truth
_C_ES       = "#E24B4A"   # red   — early stopping line
_C_LR       = "#BA7517"   # amber — LR drop lines

SPINE_ALPHA = 0.3


def _style(ax, title: str = "", xlabel: str = "", ylabel: str = ""):
    """Apply consistent minimal styling to an axes."""
    ax.set_title(title, fontsize=12, fontweight="normal", pad=8)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.tick_params(labelsize=9)
    for spine in ax.spines.values():
        spine.set_alpha(SPINE_ALPHA)
    ax.grid(axis="y", alpha=0.2, linewidth=0.5)


def _savefig(fig, name: str, dpi: int = 150):
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved → {path}")
    return path


def _load_history(model_name: str) -> dict:
    path = os.path.join(HISTORY_DIR, f"{model_name}_history.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"History file not found: {path}\n"
            f"  Train the model first:  python train_sr.py --name {model_name}"
        )
    with open(path) as f:
        return json.load(f)


def _load_model(model_path: str):
    return load_model(
        model_path,
        custom_objects={
            "SubPixelConv2D": SubPixelConv2D,
            "combined_loss":  combined_loss,
            "PSNRMetric":     PSNRMetric,
            "SSIMMetric":     SSIMMetric,
        },
    )


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    return 10.0 * np.log10(1.0 / mse) if mse > 0 else float("inf")


def _to_rgb(img: np.ndarray) -> np.ndarray:
    """BGR float32 [0,1] → RGB uint8."""
    return cv2.cvtColor(
        np.clip(img * 255, 0, 255).astype(np.uint8),
        cv2.COLOR_BGR2RGB,
    )


# ===========================================================================
# 1–5  Training curves
# ===========================================================================

def plot_training_curves(model_name: str):
    """Plots 1–5: individual and combined training curves."""
    print(f"\n[plots] Training curves — {model_name}")
    data    = _load_history(model_name)
    history = data["history"]
    epochs  = list(range(1, len(history["loss"]) + 1))

    # Detect early stopping epoch and LR drop epochs
    lr_key      = "learning_rate" if "learning_rate" in history else None
    es_epoch    = len(epochs)   # default: no early stopping fired
    lr_drops    = []

    if lr_key:
        lrs = history[lr_key]
        for i in range(1, len(lrs)):
            if lrs[i] < lrs[i - 1] * 0.6:   # halved (factor=0.5 ± float noise)
                lr_drops.append(i + 1)        # 1-indexed epoch

    # ----------------------------------------------------------------
    # Individual plots
    # ----------------------------------------------------------------
    pairs = [
        ("loss",   "val_loss",   "Combined loss",  "loss_curve.png",  "Loss"),
        ("psnr",   "val_psnr",   "PSNR (dB)",      "psnr_curve.png",  "PSNR (dB)"),
        ("ssim",   "val_ssim",   "SSIM",           "ssim_curve.png",  "SSIM"),
    ]

    for train_key, val_key, title, fname, ylabel in pairs:
        if train_key not in history:
            print(f"  [skip] {train_key} not in history")
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(epochs, history[train_key], color=_C_TRAIN, linewidth=1.5, label="train")
        if val_key in history:
            ax.plot(epochs, history[val_key], color=_C_VAL, linewidth=1.5,
                    linestyle="--", label="val")
        for ep in lr_drops:
            ax.axvline(ep, color=_C_LR, linewidth=0.8, linestyle=":", alpha=0.7)
        ax.axvline(es_epoch, color=_C_ES, linewidth=0.8, linestyle=":", alpha=0.7,
                   label=f"early stop (ep {es_epoch})")
        ax.legend(fontsize=9)
        _style(ax, title=f"{model_name} — {title}", xlabel="epoch", ylabel=ylabel)
        _savefig(fig, f"{model_name}_{fname}")

    # LR schedule
    if lr_key:
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.plot(epochs, history[lr_key], color=_C_LR, linewidth=1.5)
        for ep in lr_drops:
            ax.axvline(ep, color=_C_LR, linewidth=0.8, linestyle=":", alpha=0.6)
        ax.set_yscale("log")
        _style(ax, title=f"{model_name} — learning rate schedule",
               xlabel="epoch", ylabel="LR (log scale)")
        _savefig(fig, f"{model_name}_lr_schedule.png")

    # ----------------------------------------------------------------
    # Combined 4-panel figure (publication-ready)
    # ----------------------------------------------------------------
    n_panels = 3 + (1 if lr_key else 0)
    fig = plt.figure(figsize=(9, 3 * n_panels))
    gs  = gridspec.GridSpec(n_panels, 1, hspace=0.45)

    panel_specs = [
        ("loss",   "val_loss",  "Combined loss", "Loss"),
        ("psnr",   "val_psnr",  "PSNR",          "PSNR (dB)"),
        ("ssim",   "val_ssim",  "SSIM",           "SSIM"),
    ]
    if lr_key:
        panel_specs.append((lr_key, None, "Learning rate", "LR"))

    for i, (tk, vk, title, ylabel) in enumerate(panel_specs):
        ax = fig.add_subplot(gs[i])
        if tk not in history:
            continue
        if tk == lr_key:
            ax.plot(epochs, history[tk], color=_C_LR, linewidth=1.5)
            ax.set_yscale("log")
        else:
            ax.plot(epochs, history[tk], color=_C_TRAIN, linewidth=1.5, label="train")
            if vk and vk in history:
                ax.plot(epochs, history[vk], color=_C_VAL, linewidth=1.5,
                        linestyle="--", label="val")
            ax.legend(fontsize=8, loc="upper right" if tk == "loss" else "lower right")
        for ep in lr_drops:
            ax.axvline(ep, color=_C_LR, linewidth=0.7, linestyle=":", alpha=0.6)
        ax.axvline(es_epoch, color=_C_ES, linewidth=0.7, linestyle=":", alpha=0.6)
        _style(ax, title=title,
               xlabel="epoch" if i == n_panels - 1 else "",
               ylabel=ylabel)

    fig.suptitle(f"{model_name} — training curves", fontsize=13, y=1.01)
    _savefig(fig, f"{model_name}_all_curves.png")


# ===========================================================================
# 6–9  Model comparison charts
# ===========================================================================

def plot_model_comparison(registry: list[dict]):
    """Plots 6–9: bar charts and scatter from sr_model_registry eval fields."""
    print("\n[plots] Model comparison charts")

    names, psnr_c, psnr_d, ssim_c, ssim_d, inf_ms, watts = [], [], [], [], [], [], []

    for entry in registry:
        hpath = os.path.join(
            HISTORY_DIR,
            os.path.basename(entry["model_path"]).replace(".keras", "_history.json"),
        )
        if not os.path.exists(hpath):
            print(f"  [skip] no history for {entry['name']}")
            continue
        with open(hpath) as f:
            data = json.load(f)
        ev = data.get("eval")
        if not ev:
            print(f"  [skip] no eval block in history for {entry['name']}")
            continue

        names.append(entry["name"])
        psnr_c.append(ev["clean"]["psnr"])
        psnr_d.append(ev["degraded"]["psnr"])
        ssim_c.append(ev["clean"]["ssim"])
        ssim_d.append(ev["degraded"]["ssim"])
        inf_ms.append(entry.get("inference_ms", None))
        watts.append(entry.get("mean_watts", None))

    if not names:
        print("  [skip] no eval data found in any history file — run train_sr.py first")
        return

    x   = np.arange(len(names))
    w   = 0.35
    fs  = max(8, 10 - len(names))   # shrink font for many models

    # --- PSNR comparison ---
    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.8), 4))
    ax.bar(x - w/2, psnr_c, w, label="clean input",    color=_C_CLEAN,    alpha=0.85)
    ax.bar(x + w/2, psnr_d, w, label="degraded input", color=_C_DEGRADED, alpha=0.85)
    for xi, (c, d) in enumerate(zip(psnr_c, psnr_d)):
        ax.text(xi - w/2, c + 0.1, f"{c:.1f}", ha="center", fontsize=fs - 1)
        ax.text(xi + w/2, d + 0.1, f"{d:.1f}", ha="center", fontsize=fs - 1)
        # Drop annotation
        drop = c - d
        ax.annotate(f"−{drop:.1f}", xy=(xi, min(c, d) - 0.3),
                    ha="center", fontsize=fs - 2, color=_C_DEGRADED, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=fs, rotation=15, ha="right")
    ax.legend(fontsize=9)
    _style(ax, title="PSNR — clean vs degraded input", ylabel="PSNR (dB)")
    _savefig(fig, "comparison_psnr.png")

    # --- SSIM comparison ---
    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.8), 4))
    ax.bar(x - w/2, ssim_c, w, label="clean input",    color=_C_CLEAN,    alpha=0.85)
    ax.bar(x + w/2, ssim_d, w, label="degraded input", color=_C_DEGRADED, alpha=0.85)
    for xi, (c, d) in enumerate(zip(ssim_c, ssim_d)):
        ax.text(xi - w/2, c + 0.002, f"{c:.3f}", ha="center", fontsize=fs - 1)
        ax.text(xi + w/2, d + 0.002, f"{d:.3f}", ha="center", fontsize=fs - 1)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=fs, rotation=15, ha="right")
    ax.legend(fontsize=9)
    _style(ax, title="SSIM — clean vs degraded input", ylabel="SSIM")
    _savefig(fig, "comparison_ssim.png")

    # --- Speed vs PSNR scatter ---
    valid_speed = [(n, ms, pc) for n, ms, pc in zip(names, inf_ms, psnr_c) if ms is not None]
    if valid_speed:
        fig, ax = plt.subplots(figsize=(6, 4))
        for n, ms, pc in valid_speed:
            ax.scatter(ms, pc, s=80, color=_C_CLEAN, zorder=3)
            ax.annotate(n, (ms, pc), textcoords="offset points",
                        xytext=(6, 4), fontsize=8)
        ax.invert_xaxis()   # faster (lower ms) = right = better
        _style(ax, title="Speed vs quality  (top-right = best)",
               xlabel="inference  ms / image  (lower = faster)",
               ylabel="PSNR — clean input (dB)")
        _savefig(fig, "comparison_speed_vs_psnr.png")

    # --- PSNR / watt ---
    valid_power = [(n, pc, w_) for n, pc, w_ in zip(names, psnr_c, watts)
                   if w_ is not None and w_ > 0]
    if valid_power:
        pnw_names = [v[0] for v in valid_power]
        pnw_vals  = [v[1] / v[2] for v in valid_power]
        fig, ax = plt.subplots(figsize=(max(5, len(pnw_names) * 1.5), 4))
        bars = ax.bar(pnw_names, pnw_vals, color=_C_CLEAN, alpha=0.85)
        for bar, val in zip(bars, pnw_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                    f"{val:.2f}", ha="center", fontsize=9)
        ax.set_xticklabels(pnw_names, rotation=15, ha="right", fontsize=9)
        _style(ax, title="PSNR per watt  (higher = more efficient)",
               ylabel="PSNR / W  (dB/W)")
        _savefig(fig, "comparison_psnr_per_watt.png")


# ===========================================================================
# 10–13  Robustness analysis (needs live model + test data)
# ===========================================================================

def plot_robustness(models_info: list[dict], test_lr: np.ndarray, test_hr: np.ndarray):
    """Plots 10–13: PSNR vs degradation severity and type breakdown."""
    print("\n[plots] Robustness analysis")

    loaded = []
    for info in models_info:
        if not os.path.exists(info["model_path"]):
            print(f"  [skip] {info['name']} — model file not found")
            continue
        print(f"  Loading {info['name']}...")
        loaded.append((info["name"], _load_model(info["model_path"])))

    if not loaded:
        print("  [skip] no models available for robustness plots")
        return

    n_imgs = min(20, len(test_lr))   # use up to 20 images for stability
    lr_sub = test_lr[:n_imgs]
    hr_sub = test_hr[:n_imgs]

    # ----------------------------------------------------------------
    # Plot 10 — PSNR vs JPEG quality
    # ----------------------------------------------------------------
    jpeg_qualities = [60, 50, 40, 30, 20, 10]
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = [_C_TRAIN, _C_DEGRADED, _C_HR, "#7F77DD", "#D4537E"]

    for (name, model), col in zip(loaded, colors):
        psnrs = []
        for q in jpeg_qualities:
            degraded = np.array(
                [jpeg_compression(img, quality=q) for img in lr_sub],
                dtype=np.float32,
            )
            sr = model.predict(degraded, verbose=0)
            psnrs.append(_psnr(hr_sub, sr))
        ax.plot(jpeg_qualities, psnrs, marker="o", linewidth=1.8,
                markersize=5, label=name, color=col)

    ax.invert_xaxis()   # left = worse quality
    ax.legend(fontsize=9)
    _style(ax, title="PSNR vs JPEG quality  (left = more degraded)",
           xlabel="JPEG quality", ylabel="PSNR (dB)")
    _savefig(fig, "robustness_psnr_vs_jpeg.png")

    # ----------------------------------------------------------------
    # Plot 11 — PSNR vs Gaussian noise std
    # ----------------------------------------------------------------
    noise_stds = [10, 20, 30, 40, 50, 60, 80]
    fig, ax = plt.subplots(figsize=(7, 4))

    for (name, model), col in zip(loaded, colors):
        psnrs = []
        for std in noise_stds:
            rng = np.random.RandomState(0)
            degraded = []
            for img in lr_sub:
                img_u8 = np.clip(img * 255, 0, 255).astype(np.uint8)
                noise  = rng.normal(0, std, img_u8.shape).astype(np.int16)
                noisy  = np.clip(img_u8.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                degraded.append(noisy.astype(np.float32) / 255.0)
            degraded = np.array(degraded, dtype=np.float32)
            sr = model.predict(degraded, verbose=0)
            psnrs.append(_psnr(hr_sub, sr))
        ax.plot(noise_stds, psnrs, marker="o", linewidth=1.8,
                markersize=5, label=name, color=col)

    ax.legend(fontsize=9)
    _style(ax, title="PSNR vs Gaussian noise std  (right = more degraded)",
           xlabel="noise std", ylabel="PSNR (dB)")
    _savefig(fig, "robustness_psnr_vs_noise.png")

    # ----------------------------------------------------------------
    # Plot 12 — PSNR per degradation type (grouped bar)
    # ----------------------------------------------------------------
    degrad_fns = {
        "clean":         lambda img: img,
        "gaussian":      lambda img: gaussian_noise(img),
        "salt & pepper": lambda img: salt_and_pepper_noise(img, p=0.05),
        "jpeg q=20":     lambda img: jpeg_compression(img, quality=20),
        "combined":      lambda img: random_degrade(img, rng=np.random.RandomState(1)),
    }

    degrad_names = list(degrad_fns.keys())
    x  = np.arange(len(degrad_names))
    bw = 0.7 / max(len(loaded), 1)

    fig, ax = plt.subplots(figsize=(max(8, len(degrad_names) * 1.6), 4))
    for mi, ((name, model), col) in enumerate(zip(loaded, colors)):
        psnrs = []
        for fn in degrad_fns.values():
            deg = np.array([fn(img) for img in lr_sub], dtype=np.float32)
            sr  = model.predict(deg, verbose=0)
            psnrs.append(_psnr(hr_sub, sr))
        offset = (mi - len(loaded) / 2 + 0.5) * bw
        bars = ax.bar(x + offset, psnrs, bw * 0.9, label=name, color=col, alpha=0.85)
        for bar, val in zip(bars, psnrs):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.1,
                    f"{val:.1f}", ha="center", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(degrad_names, fontsize=9)
    ax.legend(fontsize=9)
    _style(ax, title="PSNR by degradation type", ylabel="PSNR (dB)")
    _savefig(fig, "robustness_degradation_breakdown.png")

    # ----------------------------------------------------------------
    # Plot 13 — PSNR drop heatmap
    # ----------------------------------------------------------------
    model_names_list = [n for n, _ in loaded]
    # baseline PSNR (clean) per model
    clean_psnrs = []
    drop_matrix = []

    for name, model in loaded:
        clean_sr   = model.predict(lr_sub, verbose=0)
        base_psnr  = _psnr(hr_sub, clean_sr)
        clean_psnrs.append(base_psnr)
        row = []
        for fn in list(degrad_fns.values())[1:]:   # skip "clean"
            deg = np.array([fn(img) for img in lr_sub], dtype=np.float32)
            sr  = model.predict(deg, verbose=0)
            row.append(base_psnr - _psnr(hr_sub, sr))
        drop_matrix.append(row)

    drop_arr   = np.array(drop_matrix)
    degrad_cols = degrad_names[1:]   # skip "clean"

    fig, ax = plt.subplots(figsize=(max(6, len(degrad_cols) * 1.4), max(3, len(loaded) * 0.9 + 1)))
    cmap = LinearSegmentedColormap.from_list("drop", ["#E1F5EE", "#FAEEDA", "#FAECE7", "#993C1D"])
    im   = ax.imshow(drop_arr, cmap=cmap, aspect="auto", vmin=0)

    ax.set_xticks(range(len(degrad_cols)))
    ax.set_xticklabels(degrad_cols, fontsize=9)
    ax.set_yticks(range(len(model_names_list)))
    ax.set_yticklabels(model_names_list, fontsize=9)

    for i in range(len(model_names_list)):
        for j in range(len(degrad_cols)):
            ax.text(j, i, f"{drop_arr[i, j]:.1f}",
                    ha="center", va="center", fontsize=9, color="black")

    plt.colorbar(im, ax=ax, label="PSNR drop (dB)")
    _style(ax, title="PSNR drop heatmap  (higher = more vulnerable)")
    _savefig(fig, "robustness_heatmap.png")


# ===========================================================================
# 14–16  Visual / qualitative plots (needs model + test data)
# ===========================================================================

def plot_visual_analysis(models_info: list[dict], test_lr: np.ndarray,
                         test_hr: np.ndarray, n: int = 3):
    """Plots 14–16: residual maps, pixel histograms, frequency analysis."""
    print("\n[plots] Visual analysis")

    if not models_info:
        return

    # Use the first available model for visual plots
    model_info = None
    for info in models_info:
        if os.path.exists(info["model_path"]):
            model_info = info
            break
    if model_info is None:
        print("  [skip] no model files found")
        return

    print(f"  Using model: {model_info['name']}")
    model = _load_model(model_info["model_path"])

    lr_sub = test_lr[:n]
    hr_sub = test_hr[:n]

    rng = np.random.RandomState(0)
    lr_degraded = np.array(
        [random_degrade(img, rng=rng) for img in lr_sub], dtype=np.float32
    )

    sr_clean    = model.predict(lr_sub,      verbose=0).astype(np.float32)
    sr_degraded = model.predict(lr_degraded, verbose=0).astype(np.float32)

    # ----------------------------------------------------------------
    # Plot 14 — Residual error maps
    # ----------------------------------------------------------------
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for i in range(n):
        res_clean    = np.abs(sr_clean[i]    - hr_sub[i]) * 10
        res_degraded = np.abs(sr_degraded[i] - hr_sub[i]) * 10

        imgs   = [hr_sub[i], sr_clean[i], res_clean, res_degraded]
        titles = ["HR ground truth", "SR output (clean in)",
                  "residual |SR−HR|×10\n(clean input)",
                  "residual |SR−HR|×10\n(degraded input)"]

        for j, (img, title) in enumerate(zip(imgs, titles)):
            if j >= 2:   # residual maps: grayscale heatmap
                err_gray = np.mean(img, axis=-1)
                axes[i, j].imshow(err_gray, cmap="hot", vmin=0, vmax=1)
            else:
                axes[i, j].imshow(_to_rgb(img))
            axes[i, j].set_title(title, fontsize=9)
            axes[i, j].axis("off")

    plt.suptitle(f"Residual error maps — {model_info['name']}", fontsize=12, y=1.01)
    plt.tight_layout()
    _savefig(fig, "visual_residual_maps.png")

    # ----------------------------------------------------------------
    # Plot 15 — Pixel value distributions
    # ----------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), sharey=True)
    bins = 64

    for ax, (label, imgs, col) in zip(axes, [
        ("HR ground truth",      hr_sub,      _C_HR),
        ("SR clean input",       sr_clean,    _C_CLEAN),
        ("SR degraded input",    sr_degraded, _C_DEGRADED),
    ]):
        vals = imgs.flatten()
        ax.hist(vals, bins=bins, color=col, alpha=0.75, density=True)
        ax.axvline(float(np.mean(vals)), color=col, linewidth=1.2,
                   linestyle="--", alpha=0.9)
        _style(ax, title=label, xlabel="pixel value [0,1]",
               ylabel="density" if ax is axes[0] else "")

    plt.suptitle(f"Pixel value distributions — {model_info['name']}", fontsize=12)
    plt.tight_layout()
    _savefig(fig, "visual_pixel_histogram.png")

    # ----------------------------------------------------------------
    # Plot 16 — Frequency domain analysis (FFT magnitude)
    # ----------------------------------------------------------------
    def _mean_fft(imgs: np.ndarray) -> np.ndarray:
        """Mean FFT magnitude (log scale) across a batch, averaged over channels."""
        mags = []
        for img in imgs:
            gray = np.mean(img, axis=-1)
            f    = np.fft.fft2(gray)
            fs   = np.fft.fftshift(f)
            mags.append(np.log1p(np.abs(fs)))
        return np.mean(mags, axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    titles = ["HR ground truth", "SR clean input", "SR degraded input"]
    imgs_list = [hr_sub, sr_clean, sr_degraded]

    vmax = None
    ffts = [_mean_fft(imgs) for imgs in imgs_list]
    vmax = max(f.max() for f in ffts)

    for ax, fft_mag, title in zip(axes, ffts, titles):
        ax.imshow(fft_mag, cmap="inferno", vmin=0, vmax=vmax)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    plt.suptitle(
        f"FFT magnitude (log scale) — {model_info['name']}\n"
        "Brighter edges = more high-frequency content = sharper output",
        fontsize=11,
    )
    plt.tight_layout()
    _savefig(fig, "visual_frequency_analysis.png")



# ===========================================================================
# Stub history utility
# ===========================================================================

def create_stub_history(model_name: str) -> str:
    """Create a minimal history JSON for a model trained without train_sr.py.

    Writes a stub with no epoch data so robustness/visual/comparison plots
    still work. Training curves (plots 1-5) will be skipped for this model.
    """
    os.makedirs(HISTORY_DIR, exist_ok=True)
    path = os.path.join(HISTORY_DIR, f"{model_name}_history.json")
    if os.path.exists(path):
        print(f"  [stub] History already exists: {path} — not overwriting.")
        return path
    stub = {
        "model_name"    : model_name,
        "num_res_blocks": None,
        "history"       : {},
        "eval"          : {
            "clean"   : {"loss": None, "psnr": None, "ssim": None},
            "degraded": {"loss": None, "psnr": None, "ssim": None},
            "inference_ms": None,
            "mean_watts"  : None,
        },
    }
    with open(path, "w") as f:
        json.dump(stub, f, indent=2)
    print(f"  [stub] Created stub history -> {path}")
    print(f"         Training curves skipped. Robustness + visual plots will run.")
    return path


# ===========================================================================
# Entry point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate SR diagnostic plots — works without retraining.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  Already have .keras files but no history JSON? Create stubs then plot:
    python plots.py --stub sr_edsr_model --stub sr_robust_model

  Skip training curves, run everything else (robustness + visual + comparison):
    python plots.py --skip-curves

  Training curves only, no model loading needed:
    python plots.py --curves-only

  All plots for a single model:
    python plots.py --model sr_robust_model

  Fewer test images (faster):
    python plots.py --n 2
""",
    )
    parser.add_argument(
        "--registry",
        default=os.path.join(MODELS_DIR, "sr_model_registry.json"),
        help="Path to sr_model_registry.json",
    )
    parser.add_argument(
        "--model", default=None,
        help="Run all plots for a single model name (e.g. sr_robust_model)",
    )
    parser.add_argument(
        "--stub", action="append", metavar="MODEL_NAME", default=[],
        help="Create a stub history JSON for a model that has no history file. "
             "Can be passed multiple times: --stub model_a --stub model_b",
    )
    parser.add_argument(
        "--skip-curves", action="store_true",
        help="Skip training curve plots (1-5). Use when no history JSON exists.",
    )
    parser.add_argument(
        "--curves-only", action="store_true",
        help="Only plot training curves. No model loading, no inference.",
    )
    parser.add_argument(
        "--n", type=int, default=3,
        help="Number of test images for visual plots (default: 3)",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Stub creation
    # ------------------------------------------------------------------
    for model_name in args.stub:
        create_stub_history(model_name)

    # ------------------------------------------------------------------
    # Resolve registry
    # ------------------------------------------------------------------
    registry = []
    if os.path.exists(args.registry):
        with open(args.registry) as f:
            registry = json.load(f)["models"]

    if args.model:
        model_stem = args.model.replace(".keras", "")
        matched = [e for e in registry
                   if os.path.basename(e["model_path"]).replace(".keras", "") == model_stem]
        if matched:
            registry = matched
        else:
            registry = [{
                "name"      : model_stem,
                "model_path": os.path.join(MODELS_DIR, f"{model_stem}.keras"),
            }]

    # ------------------------------------------------------------------
    # Training curves (plots 1-5)
    # ------------------------------------------------------------------
    if not args.skip_curves:
        if registry:
            for entry in registry:
                model_name = os.path.basename(entry["model_path"]).replace(".keras", "")
                try:
                    plot_training_curves(model_name)
                except FileNotFoundError:
                    print(f"  [skip curves] No history JSON for {model_name}.")
                    print(f"    Fix: python plots.py --stub {model_name}  then re-run.")
        else:
            print("[plots] No registry and no --model set — skipping training curves.")

    if args.curves_only:
        print(f"\n[plots] --curves-only set. Done. Plots saved to {PLOTS_DIR}/")
        return

    if not registry:
        print("[plots] No registry and no --model set.")
        print("  Create models/sr_model_registry.json or pass --model <name>")
        return

    # ------------------------------------------------------------------
    # Model comparison (plots 6-9)
    # ------------------------------------------------------------------
    plot_model_comparison(registry)

    # ------------------------------------------------------------------
    # Load test dataset (needed for plots 10-16)
    # ------------------------------------------------------------------
    print("\n[plots] Loading test dataset...")
    _, _, test_lr, test_hr = prepare_sr_data()
    test_lr = test_lr[:50]
    test_hr = test_hr[:50]

    # ------------------------------------------------------------------
    # Robustness analysis (plots 10-13)
    # ------------------------------------------------------------------
    plot_robustness(registry, test_lr, test_hr)

    # ------------------------------------------------------------------
    # Visual / qualitative (plots 14-16)
    # ------------------------------------------------------------------
    plot_visual_analysis(registry, test_lr, test_hr, n=args.n)

    print(f"\n[plots] Done. All plots saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()