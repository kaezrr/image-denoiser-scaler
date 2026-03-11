"""
data/download.py
Downloads DIV2K valid_HR (100 images) — used for both training the fusion head
and evaluation. Intentionally uses val set only to keep download small (~430MB).

Run: python data/download.py
Creates: data/DIV2K/valid_HR/  (100 PNG images)
"""
import zipfile, requests
from pathlib import Path
from tqdm import tqdm

URL  = "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip"
DEST = Path("data/DIV2K")


def download(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(desc=dest.name, total=total, unit="B", unit_scale=True) as bar:
        for chunk in r.iter_content(1 << 20):
            f.write(chunk); bar.update(len(chunk))


def main():
    out_dir = DEST / "valid_HR"
    if out_dir.exists() and len(list(out_dir.glob("*.png"))) >= 90:
        print(f"✓ valid_HR already present ({len(list(out_dir.glob('*.png')))} images)")
        return

    zip_path = DEST / "valid_HR.zip"
    print("Downloading DIV2K valid_HR (~430MB)…")
    download(URL, zip_path)

    print("Extracting…")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(DEST)

    extracted = DEST / "DIV2K_valid_HR"
    if extracted.exists():
        extracted.rename(out_dir)

    zip_path.unlink()
    print(f"✓ Done: {len(list(out_dir.glob('*.png')))} images in {out_dir}")


if __name__ == "__main__":
    main()
