"""A3 data prep — turn A1 IQ captures into 64x64 spectrogram PNGs.

    python -m ml.a3_spectrogram_cnn.make_spectrograms

Reads dataset/<class>/*.bin -> dataset/spectrograms/{train,val}/<class>/*.png.
Same 4 classes as A1. Run A1's mock generator first.
"""
from __future__ import annotations
import os, sys
import numpy as np
from scipy import signal as sp
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from ml.common.config import DATASET_DIR, SPECTRO_CLASSES, SAMPLE_RATE
from ml.common.iq import load_iq

OUT = os.path.join(DATASET_DIR, "spectrograms")
IMG = 64


def iq_to_spectrogram_img(iq: np.ndarray, size: int = IMG) -> Image.Image:
    f, t, Sxx = sp.spectrogram(iq, fs=SAMPLE_RATE, nperseg=128,
                               noverlap=64, return_onesided=False)
    S = np.fft.fftshift(Sxx, axes=0)
    S = 10 * np.log10(np.abs(S) + 1e-12)
    S = (S - S.min()) / (S.max() - S.min() + 1e-12)
    img = Image.fromarray((S * 255).astype(np.uint8)).resize((size, size))
    return img.convert("RGB")  # 3-channel for MobileNet


def main():
    n = 0
    for cls in SPECTRO_CLASSES:
        src = os.path.join(DATASET_DIR, cls)
        if not os.path.isdir(src):
            continue
        files = sorted(f for f in os.listdir(src) if f.endswith(".bin"))
        split_at = int(len(files) * 0.8)
        for i, f in enumerate(files):
            split = "train" if i < split_at else "val"
            d = os.path.join(OUT, split, cls)
            os.makedirs(d, exist_ok=True)
            img = iq_to_spectrogram_img(load_iq(os.path.join(src, f)))
            img.save(os.path.join(d, f.replace(".bin", ".png")))
            n += 1
    print(f"Wrote {n} spectrograms under {OUT}")


if __name__ == "__main__":
    main()
