"""
demo.py
CLI demo: restores a single image and shows per-model comparison metrics.

Modes:
  # End-to-end: degrade automatically then restore
  python demo.py --input clean.png --degrade --output restored.png --checkpoint checkpoints/best.pth

  # Pre-degraded input with ground truth for metrics
  python demo.py --input degraded.jpg --gt clean.png --output restored.png --checkpoint checkpoints/best.pth

  # Blind restore, no metrics
  python demo.py --input degraded.jpg --output restored.png --checkpoint checkpoints/best.pth

Also saves side-by-side comparison image: <output>_compare.png
"""
import argparse
from pathlib import Path

import yaml
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFilter

from models.experts import ExpertPool, download_realesrgan_weights
from models.fusion import SpatialAttentionFusionNet
from models.metrics import psnr, ssim


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",      required=True)
    p.add_argument("--output",     required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config",     default="config.yaml")
    p.add_argument("--gt",         default=None)
    p.add_argument("--degrade",    action="store_true")
    return p.parse_args()


def to_tensor(img, device):
    return transforms.ToTensor()(img).unsqueeze(0).to(device)

def to_pil(t):
    return transforms.ToPILImage()(t.clamp(0, 1).squeeze(0).cpu())

def label_image(img: Image.Image, text: str) -> Image.Image:
    """Add a label bar at the bottom of an image."""
    bar_h = 28
    new = Image.new("RGB", (img.width, img.height + bar_h), (30, 30, 30))
    new.paste(img, (0, 0))
    draw = ImageDraw.Draw(new)
    draw.text((6, img.height + 5), text, fill=(220, 220, 220))
    return new

def make_comparison(images_with_labels: list, metrics: dict) -> Image.Image:
    """Stitch labeled images side by side."""
    imgs = [label_image(img, f"{label}\nPSNR:{metrics[label][0]:.2f} SSIM:{metrics[label][1]:.4f}"
                         if label in metrics else label)
            for img, label in images_with_labels]
    w = sum(i.width for i in imgs)
    h = max(i.height for i in imgs)
    out = Image.new("RGB", (w, h), (50, 50, 50))
    x = 0
    for img in imgs:
        out.paste(img, (x, 0)); x += img.width
    return out


def _degrade_image(hr, dcfg, scale):
    """Apply random degradation to an HR image for demo mode."""
    import io, random

    img = hr.copy()

    def noise(x):
        s = random.uniform(*dcfg["noise_sigma_range"])
        a = np.array(x, np.float32)
        return Image.fromarray(np.clip(a + np.random.randn(*a.shape) * s, 0, 255).astype(np.uint8))

    def jpeg(x):
        q = random.randint(*dcfg["jpeg_quality_range"])
        b = io.BytesIO()
        x.save(b, "JPEG", quality=q); b.seek(0)
        return Image.open(b).copy()

    def blur(x):
        k = random.choice([v for v in range(dcfg["blur_kernel_range"][0],
                                            dcfg["blur_kernel_range"][1] + 1, 2)])
        return x.filter(ImageFilter.GaussianBlur(k / 6.0))

    fns = [noise, jpeg, blur]
    for fn in random.sample(fns, random.randint(1, dcfg["max_degradations"])):
        img = fn(img)
    w, h = img.size
    return img.resize((w // scale, h // scale), Image.BICUBIC)


def main():
    args = parse_args()
    with open(args.config) as f: cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    download_realesrgan_weights()

    experts = ExpertPool(cfg, device)
    fusion  = SpatialAttentionFusionNet(cfg).to(device)
    ck      = torch.load(args.checkpoint, map_location=device)
    fusion.load_state_dict(ck.get("fusion", ck.get("model")))
    fusion.eval()

    inp = Image.open(args.input).convert("RGB")
    print(f"Input: {args.input} ({inp.size[0]}x{inp.size[1]})")

    gt_tensor = None
    if args.degrade:
        lr_img    = _degrade_image(inp, cfg["degradation"], cfg["data"]["scale"])
        gt_tensor = to_tensor(inp, device)
        print(f"Degraded to: {lr_img.size[0]}x{lr_img.size[1]}")
    else:
        lr_img = inp
        if args.gt:
            gt_tensor = to_tensor(Image.open(args.gt).convert("RGB"), device)

    lr_t = to_tensor(lr_img, device)

    with torch.no_grad():
        cat, outs = experts(lr_t)
        result    = fusion(cat, outs)

    bicubic_img   = to_pil(outs[2])
    edsr_img      = to_pil(outs[1])
    realesr_img   = to_pil(outs[0])
    fused_img     = to_pil(result["output"])

    # Save primary output
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fused_img.save(args.output)
    print(f"Output: {args.output} ({fused_img.size[0]}x{fused_img.size[1]})")

    # Metrics
    metrics = {}
    if gt_tensor is not None:
        for name, pred_t in [("Bicubic", outs[2]), ("EDSR", outs[1]),
                              ("Real-ESRGAN", outs[0]), ("SAFN (Ours)", result["output"])]:
            if pred_t.shape != gt_tensor.shape:
                pred_t = F.interpolate(pred_t, size=gt_tensor.shape[-2:],
                                       mode="bicubic", align_corners=False)
            metrics[name] = (psnr(pred_t.clamp(0, 1), gt_tensor),
                             ssim(pred_t.clamp(0, 1), gt_tensor))

        print(f"\n{'Model':<20} {'PSNR (dB)':>10} {'SSIM':>10}")
        print("─" * 44)
        for name, (p_val, s_val) in metrics.items():
            marker = " ◄ best" if name == "SAFN (Ours)" else ""
            print(f"{name:<20} {p_val:>10.2f} {s_val:>10.4f}{marker}")

    # Save comparison strip
    compare_path = str(args.output).replace(".", "_compare.")
    compare = make_comparison(
        [(bicubic_img, "Bicubic"), (edsr_img, "EDSR"),
         (realesr_img, "Real-ESRGAN"), (fused_img, "SAFN (Ours)")],
        metrics
    )
    compare.save(compare_path)
    print(f"\nComparison saved: {compare_path}")


if __name__ == "__main__":
    main()
