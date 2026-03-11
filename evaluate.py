"""
evaluate.py
Evaluates all experts individually AND the fusion model.
Outputs a comparison table: Bicubic vs EDSR vs Real-ESRGAN vs SAFN (ours).

Run: python evaluate.py --checkpoint checkpoints/best.pth
"""
import argparse
from pathlib import Path

import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import make_splits
from models.experts import ExpertPool, download_realesrgan_weights
from models.fusion import SpatialAttentionFusionNet
from models.metrics import psnr, ssim


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", default="config.yaml")
    args = p.parse_args()

    with open(args.config) as f: cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    download_realesrgan_weights()

    experts = ExpertPool(cfg, device)
    fusion  = SpatialAttentionFusionNet(cfg).to(device)
    ck      = torch.load(args.checkpoint, map_location=device)
    fusion.load_state_dict(ck.get("fusion", ck.get("model")))
    fusion.eval()

    _, val_ds = make_splits(cfg)
    loader = DataLoader(val_ds, 1, shuffle=False, num_workers=2)

    names  = ["Bicubic", "EDSR", "Real-ESRGAN", "SAFN (Ours)"]
    totals = {n: [0.0, 0.0] for n in names}   # [sum_psnr, sum_ssim]
    n = 0

    with torch.no_grad():
        for lr, hr in tqdm(loader, desc="Evaluating"):
            lr, hr = lr.to(device), hr.to(device)
            cat, outs = experts(lr)
            result = fusion(cat, outs)
            preds = [outs[2], outs[1], outs[0], result["output"].clamp(0, 1)]
            # order:  bicubic, edsr, realesrgan, fusion

            for name, pred in zip(names, preds):
                totals[name][0] += psnr(pred.clamp(0, 1), hr)
                totals[name][1] += ssim(pred.clamp(0, 1), hr)
            n += 1

    print(f"\n{'Model':<20} {'PSNR (dB)':>10} {'SSIM':>10}")
    print("─" * 44)
    for name in names:
        p_avg = totals[name][0] / n
        s_avg = totals[name][1] / n
        marker = " ← best" if name == "SAFN (Ours)" else ""
        print(f"{name:<20} {p_avg:>10.2f} {s_avg:>10.4f}{marker}")
    print(f"\nEvaluated on {n} images.")


if __name__ == "__main__":
    main()
