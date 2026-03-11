"""
train.py
Trains ONLY the SpatialAttentionFusionNet. All experts are frozen.
Expected training time: ~20-30 minutes on any GPU.

Run: python train.py [--config config.yaml] [--resume checkpoints/best.pth]
"""
import argparse, os
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from data.dataset import make_splits
from models.experts import ExpertPool, download_realesrgan_weights
from models.fusion import SpatialAttentionFusionNet
from models.metrics import psnr, ssim


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--resume", default=None)
    return p.parse_args()


def fft_loss(pred, target):
    return F.l1_loss(
        torch.abs(torch.fft.rfft2(pred,   norm="ortho")),
        torch.abs(torch.fft.rfft2(target, norm="ortho")),
    )


def save_ckpt(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    print(f"  ✓ Saved: {path}")


@torch.no_grad()
def validate(fusion, experts, loader, device, lc, max_b=20):
    fusion.eval()
    tp, ts, n = 0.0, 0.0, 0
    for i, (lr, hr) in enumerate(loader):
        if i >= max_b: break
        lr, hr = lr.to(device), hr.to(device)
        cat, outs = experts(lr)
        result = fusion(cat, outs)
        pred = result["output"].clamp(0, 1)
        tp += psnr(pred, hr); ts += ssim(pred, hr); n += 1
    fusion.train()
    return tp / max(n, 1), ts / max(n, 1)


def main():
    args = parse_args()
    with open(args.config) as f: cfg = yaml.safe_load(f)
    tc, dc = cfg["training"], cfg["data"]
    lc = cfg["loss"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Download weights
    download_realesrgan_weights()

    # Data
    train_ds, val_ds = make_splits(cfg)
    pin = device.type == "cuda"
    train_dl = DataLoader(train_ds, tc["batch_size"], shuffle=True,
                          num_workers=dc["num_workers"], pin_memory=pin, drop_last=True)
    val_dl   = DataLoader(val_ds, 1, shuffle=False, num_workers=2)

    # Models
    experts = ExpertPool(cfg, device)       # ALL FROZEN
    fusion  = SpatialAttentionFusionNet(cfg).to(device)

    n_params = sum(p.numel() for p in fusion.parameters()) / 1e6
    print(f"Fusion network trainable parameters: {n_params:.3f}M")
    print(f"Expert parameters (frozen): {sum(p.numel() for p in experts.parameters()) / 1e6:.1f}M")

    optimizer = torch.optim.AdamW(fusion.parameters(), lr=tc["lr"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=tc["total_iters"], eta_min=tc["lr_min"])
    scaler = GradScaler(enabled=tc["amp"] and device.type == "cuda")

    start_iter, best_psnr = 0, 0.0
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu")
        fusion.load_state_dict(ck["fusion"])
        optimizer.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        start_iter = ck["iteration"]
        print(f"Resumed from iter {start_iter}")

    fusion.train()
    data_iter = iter(train_dl)
    pbar = tqdm(range(start_iter, tc["total_iters"]),
                initial=start_iter, total=tc["total_iters"], desc="Training fusion")

    for it in pbar:
        try:
            lr_b, hr_b = next(data_iter)
        except StopIteration:
            data_iter = iter(train_dl)
            lr_b, hr_b = next(data_iter)

        lr_b = lr_b.to(device, non_blocking=True)
        hr_b = hr_b.to(device, non_blocking=True)

        # Warmup
        if it < tc["warmup_iters"]:
            for pg in optimizer.param_groups:
                pg["lr"] = tc["lr"] * (it + 1) / tc["warmup_iters"]

        optimizer.zero_grad(set_to_none=True)

        # Get frozen expert outputs (no grad needed)
        with torch.no_grad():
            cat, outs = experts(lr_b)

        # Train only fusion net
        with autocast(enabled=tc["amp"] and device.type == "cuda"):
            result = fusion(cat, outs)
            pred   = result["output"]
            loss   = lc["l1_weight"]  * F.l1_loss(pred, hr_b) + \
                     lc["fft_weight"] * fft_loss(pred, hr_b)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(fusion.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        if it >= tc["warmup_iters"]: scheduler.step()

        if (it + 1) % tc["log_every"] == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f'{optimizer.param_groups[0]["lr"]:.2e}')

        if (it + 1) % tc["eval_every"] == 0:
            vp, vs = validate(fusion, experts, val_dl, device, lc)
            print(f"\n  [iter {it+1}] Val PSNR: {vp:.2f} dB | SSIM: {vs:.4f}")
            if vp > best_psnr:
                best_psnr = vp
                save_ckpt({"fusion": fusion.state_dict(), "optimizer": optimizer.state_dict(),
                           "scaler": scaler.state_dict(), "iteration": it + 1, "psnr": vp},
                          f"{tc['checkpoint_dir']}/best.pth")

        if (it + 1) % tc["save_every"] == 0:
            save_ckpt({"fusion": fusion.state_dict(), "optimizer": optimizer.state_dict(),
                       "scaler": scaler.state_dict(), "iteration": it + 1},
                      f"{tc['checkpoint_dir']}/iter_{it+1:06d}.pth")

    print(f"\nDone. Best Val PSNR: {best_psnr:.2f} dB")


if __name__ == "__main__":
    main()
