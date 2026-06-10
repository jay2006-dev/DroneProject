"""Fetch REAL drone audio (A5) — saraalemadi/DroneAudioDataset (~281 MB).

No credentials needed. Downloads the GitHub zip, then adapts the Binary_Drone_Audio
folders into our layout:  dataset/audio/{drone,noise}/*.wav

    python -m ml.datasets.fetch_audio                 # download + adapt
    python -m ml.datasets.fetch_audio --max 400       # cap files per class

'yes_drone' -> drone,  'unknown' (ESC-50 / silence) -> noise.
Keeps the synthetic 'motor' class from the mock generator if you want 3-class.
"""
from __future__ import annotations
import os, sys, zipfile, shutil, urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from ml.common.config import DATASET_DIR

ZIP_URL = "https://github.com/saraalemadi/DroneAudioDataset/archive/refs/heads/master.zip"
RAW = os.path.join(DATASET_DIR, "_raw", "drone_audio")


def _download(url, out):
    print(f"[audio] downloading {url}")
    print("        (~281 MB, one time; may take a few minutes)")
    def _hook(b, bs, total):
        if total > 0:
            pct = min(100, b * bs * 100 // total)
            print(f"\r        {pct:3d}%", end="", flush=True)
    urllib.request.urlretrieve(url, out, _hook)
    print()


def classify(folder_name: str) -> str | None:
    n = folder_name.lower()
    if "drone" in n:
        return "drone"
    if "unknown" in n or "noise" in n or "silence" in n:
        return "noise"
    return None


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=0, help="cap files per class (0=all)")
    args = ap.parse_args()

    os.makedirs(RAW, exist_ok=True)
    zip_path = os.path.join(RAW, "drone_audio.zip")
    if not os.path.exists(zip_path):
        _download(ZIP_URL, zip_path)
    extract_dir = os.path.join(RAW, "extracted")
    if not os.path.isdir(extract_dir):
        print("[audio] extracting...")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(extract_dir)

    # Walk for Binary_Drone_Audio/<class>/*.wav
    counts = {"drone": 0, "noise": 0}
    for root, _, files in os.walk(extract_dir):
        if "Binary_Drone_Audio" not in root:
            continue
        cls = classify(os.path.basename(root))
        if not cls:
            continue
        out = os.path.join(DATASET_DIR, "audio", cls)
        os.makedirs(out, exist_ok=True)
        for f in files:
            if not f.lower().endswith(".wav"):
                continue
            if args.max and counts[cls] >= args.max:
                break
            shutil.copy(os.path.join(root, f),
                        os.path.join(out, f"real_{cls}_{counts[cls]:05d}.wav"))
            counts[cls] += 1

    print(f"[audio] real WAVs -> drone={counts['drone']}  noise={counts['noise']}")
    if sum(counts.values()) == 0:
        sys.exit("No WAVs found — check the extracted folder structure.")
    print("Now train:  python -m ml.a5_acoustic.train --epochs 8")
    print("(Tip: keep mock 'motor' class for 3-class, or edit ACOUSTIC_CLASSES.)")


if __name__ == "__main__":
    main()
