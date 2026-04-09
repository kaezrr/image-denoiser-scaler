import argparse
import io
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
from PIL import Image
from PIL import ImageEnhance, ImageFilter
from scipy.ndimage import gaussian_filter

AttackSpec = Tuple[str, str, Callable[[Image.Image], Image.Image]]


def _pil_to_rgb_array(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"), dtype=np.uint8)


def _rgb_array_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")


def attack_jpeg(img: Image.Image, quality: int) -> Image.Image:
    bio = io.BytesIO()
    img.convert("RGB").save(bio, format="JPEG", quality=int(quality))
    bio.seek(0)
    out = Image.open(bio).convert("RGB")
    return out


def attack_gaussian_noise(img: Image.Image, sigma: float) -> Image.Image:
    arr = _pil_to_rgb_array(img).astype(np.float64)
    noise = np.random.default_rng(12345).normal(0.0, float(sigma), size=arr.shape)
    arr = np.clip(arr + noise, 0, 255)
    return _rgb_array_to_pil(arr)


def attack_median_filter(img: Image.Image, size: int) -> Image.Image:
    rgb = img.convert("RGB")
    filtered = rgb.filter(ImageFilter.MedianFilter(size=int(size))).convert("RGB")
    if int(size) >= 5:
        return Image.blend(rgb, filtered, alpha=0.7)
    return filtered


def attack_gaussian_blur(img: Image.Image, sigma: float) -> Image.Image:
    arr = _pil_to_rgb_array(img).astype(np.float64)
    out = np.empty_like(arr)
    for ch in range(3):
        out[..., ch] = gaussian_filter(arr[..., ch], sigma=float(sigma), mode="nearest")
    return _rgb_array_to_pil(out)


def attack_brightness(img: Image.Image, factor: float) -> Image.Image:
    arr = _pil_to_rgb_array(img).astype(np.float64)
    out = np.clip(arr * float(factor), 0, 255)
    return _rgb_array_to_pil(out)


def attack_contrast(img: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Contrast(img.convert("RGB")).enhance(float(factor)).convert("RGB")


def attack_crop_and_pad(img: Image.Image, fraction: float) -> Image.Image:
    rgb = img.convert("RGB")
    w, h = rgb.size
    frac = float(fraction)
    keep_w = max(1, int(round(w * (1.0 - frac))))
    keep_h = max(1, int(round(h * (1.0 - frac))))
    left = (w - keep_w) // 2
    top = (h - keep_h) // 2
    crop = rgb.crop((left, top, left + keep_w, top + keep_h))

    canvas = Image.new("RGB", (w, h), (0, 0, 0))
    paste_left = (w - keep_w) // 2
    paste_top = (h - keep_h) // 2
    canvas.paste(crop, (paste_left, paste_top))
    return canvas


def attack_resize(img: Image.Image, scale: float) -> Image.Image:
    rgb = img.convert("RGB")
    w, h = rgb.size
    s = float(scale)
    new_w = max(1, int(round(w * s)))
    new_h = max(1, int(round(h * s)))
    down = rgb.resize((new_w, new_h), Image.Resampling.LANCZOS)
    up = down.resize((w, h), Image.Resampling.LANCZOS)
    return up


def attack_rotation(img: Image.Image, degrees: float) -> Image.Image:
    rgb = img.convert("RGB")
    w, h = rgb.size
    rot = rgb.rotate(float(degrees), resample=Image.Resampling.BICUBIC, expand=False, fillcolor=(0, 0, 0))
    return rot.resize((w, h), Image.Resampling.LANCZOS)


def attack_salt_pepper(img: Image.Image, density: float) -> Image.Image:
    arr = _pil_to_rgb_array(img).copy()
    h, w, _ = arr.shape
    num = int(h * w * float(density))
    rng = np.random.default_rng(54321)
    ys = rng.integers(0, h, size=num)
    xs = rng.integers(0, w, size=num)
    vals = rng.integers(0, 2, size=num) * 255
    arr[ys, xs] = np.stack([vals, vals, vals], axis=1)
    return _rgb_array_to_pil(arr)


def attack_combined_jpeg_noise(img: Image.Image, quality: int, sigma: float) -> Image.Image:
    return attack_gaussian_noise(attack_jpeg(img, quality), sigma)


def attack_screenshot_simulation(img: Image.Image) -> Image.Image:
    out = attack_jpeg(img, quality=85)
    out = attack_gaussian_blur(out, sigma=0.5)
    out = attack_brightness(out, factor=1.05)
    return out


def build_attack_specs() -> List[AttackSpec]:
    return [
        ("jpeg", "quality90", lambda im: attack_jpeg(im, 90)),
        ("jpeg", "quality75", lambda im: attack_jpeg(im, 75)),
        ("jpeg", "quality60", lambda im: attack_jpeg(im, 60)),
        ("jpeg", "quality50", lambda im: attack_jpeg(im, 50)),
        ("jpeg", "quality40", lambda im: attack_jpeg(im, 40)),
        ("gaussian_noise", "sigma2", lambda im: attack_gaussian_noise(im, 2)),
        ("gaussian_noise", "sigma5", lambda im: attack_gaussian_noise(im, 5)),
        ("gaussian_noise", "sigma10", lambda im: attack_gaussian_noise(im, 10)),
        ("gaussian_noise", "sigma15", lambda im: attack_gaussian_noise(im, 15)),
        ("median_filter", "size3", lambda im: attack_median_filter(im, 3)),
        ("median_filter", "size5", lambda im: attack_median_filter(im, 5)),
        ("gaussian_blur", "sigma1", lambda im: attack_gaussian_blur(im, 1.0)),
        ("gaussian_blur", "sigma2", lambda im: attack_gaussian_blur(im, 2.0)),
        ("gaussian_blur", "sigma3", lambda im: attack_gaussian_blur(im, 3.0)),
        ("brightness", "factor07", lambda im: attack_brightness(im, 0.7)),
        ("brightness", "factor085", lambda im: attack_brightness(im, 0.85)),
        ("brightness", "factor115", lambda im: attack_brightness(im, 1.15)),
        ("brightness", "factor13", lambda im: attack_brightness(im, 1.3)),
        ("contrast", "factor07", lambda im: attack_contrast(im, 0.7)),
        ("contrast", "factor085", lambda im: attack_contrast(im, 0.85)),
        ("contrast", "factor115", lambda im: attack_contrast(im, 1.15)),
        ("contrast", "factor13", lambda im: attack_contrast(im, 1.3)),
        ("crop_pad", "fraction005", lambda im: attack_crop_and_pad(im, 0.05)),
        ("crop_pad", "fraction010", lambda im: attack_crop_and_pad(im, 0.10)),
        ("crop_pad", "fraction020", lambda im: attack_crop_and_pad(im, 0.20)),
        ("crop_pad", "fraction030", lambda im: attack_crop_and_pad(im, 0.30)),
        ("crop_pad", "fraction040", lambda im: attack_crop_and_pad(im, 0.40)),
        ("resize", "scale075", lambda im: attack_resize(im, 0.75)),
        ("resize", "scale050", lambda im: attack_resize(im, 0.50)),
        ("salt_pepper", "density001", lambda im: attack_salt_pepper(im, 0.01)),
        ("salt_pepper", "density005", lambda im: attack_salt_pepper(im, 0.05)),
        ("screenshot", "default", lambda im: attack_screenshot_simulation(im)),
        ("combined", "jpeg75_noise5", lambda im: attack_combined_jpeg_noise(im, 75, 5)),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate degraded variants of an input image."
    )
    parser.add_argument("input_image", help="Path to input image")
    parser.add_argument(
        "--output-dir", default="degraded", help="Output directory (default: degraded)"
    )
    parser.add_argument("--prefix", default="", help="Optional filename prefix")
    args = parser.parse_args()

    input_path = Path(args.input_image)
    if not input_path.exists():
        raise FileNotFoundError(f"Input image not found: {input_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    src = Image.open(input_path).convert("RGB")

    original_name = f"{args.prefix}original.png" if args.prefix else "original.png"
    src.save(output_dir / original_name, format="PNG")

    specs = build_attack_specs()
    for attack_name, param_tag, attack_fn in specs:
        degraded = attack_fn(src)
        out_name = (
            f"{args.prefix}{attack_name}_{param_tag}.png"
            if args.prefix
            else f"{attack_name}_{param_tag}.png"
        )
        degraded.save(output_dir / out_name, format="PNG")

    print(f"Saved {len(specs) + 1} images to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
