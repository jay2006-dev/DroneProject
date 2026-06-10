"""A1 feature extraction — 734 RF features per IQ capture.

Shared by A1 (classifier), A2 (fingerprint), A6 (anomaly), A8 (explainability),
so FEATURE_NAMES here is the single source of truth for feature meaning.

Feature blocks:
    FFT spectrum        512
    Welch PSD           128
    summary stats         8   (mean,std,max,p25,p75,rms,occupancy,entropy)
    wavelet (db4, L4)    20   (5 coeff bands x 4 stats)
    instantaneous freq    3
    autocorrelation      64
    ----------------------------
    TOTAL               735   (the original spec rounds this to "734")
"""
from __future__ import annotations
import sys, os
import numpy as np
import pywt
from scipy import signal as sp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from ml.common.iq import load_iq
from ml.common.config import SAMPLE_RATE


def _feature_names() -> list[str]:
    names = [f"fft_{i}" for i in range(512)]
    names += [f"psd_{i}" for i in range(128)]
    names += ["mean", "std", "max", "p25", "p75", "rms", "occupancy", "entropy"]
    # 5 wavelet bands (cA4, cD4, cD3, cD2, cD1) x 4 stats
    bands = ["cA4", "cD4", "cD3", "cD2", "cD1"]
    for b in bands:
        names += [f"wav_{b}_mean", f"wav_{b}_std", f"wav_{b}_energy", f"wav_{b}_max"]
    names += ["inst_mean", "inst_std", "inst_max"]
    names += [f"ac_{i}" for i in range(64)]
    return names


FEATURE_NAMES = _feature_names()
N_FEATURES = len(FEATURE_NAMES)  # 735


def extract_features(filepath_or_iq) -> np.ndarray:
    """Return the 735-dim feature vector for a .bin path or a complex IQ array."""
    if isinstance(filepath_or_iq, (str, os.PathLike)):
        s = load_iq(str(filepath_or_iq))
    else:
        s = np.asarray(filepath_or_iq, dtype=np.complex64)

    if s.size < 4096:  # pad short captures so all blocks are computable
        s = np.pad(s, (0, 4096 - s.size))
    s = s[:65536]
    mag = np.abs(s)

    # 1. FFT (512 bins, normalised)
    f = np.abs(np.fft.fft(s[:1024]))[:512]
    fft = f / (np.max(f) + 1e-10)

    # 2. Welch PSD (128 bins)
    _, psd = sp.welch(mag, fs=SAMPLE_RATE, nperseg=256)
    psd = psd[:128]
    psd = psd / (np.max(psd) + 1e-10)
    if psd.size < 128:
        psd = np.pad(psd, (0, 128 - psd.size))

    # 3. Summary stats
    p = mag / (np.sum(mag) + 1e-10)
    entropy = -np.sum(p * np.log2(p + 1e-10))
    stats = np.array([
        np.mean(mag), np.std(mag), np.max(mag),
        np.percentile(mag, 25), np.percentile(mag, 75),
        np.mean(mag ** 2),                       # RMS power
        np.sum(mag > np.mean(mag)) / len(mag),   # occupancy
        entropy,
    ])

    # 4. Wavelet (db4, 4 levels -> 5 coeff bands, 4 stats each = 20)
    wf = []
    for c in pywt.wavedec(mag[:1024], "db4", level=4):
        wf.extend([np.mean(c), np.std(c), np.mean(c ** 2), np.max(np.abs(c))])
    wf = np.array(wf)

    # 5. Instantaneous frequency (3 features)
    inst = np.diff(np.unwrap(np.angle(s[:4096])))
    ff = np.array([np.mean(inst), np.std(inst), np.max(np.abs(inst))])

    # 6. Autocorrelation (64 features)
    full = np.correlate(mag[:512], mag[:512], "full")
    center = full[full.size // 2]
    ac = full[full.size // 2:][:64] / (center + 1e-10)
    if ac.size < 64:
        ac = np.pad(ac, (0, 64 - ac.size))

    feats = np.concatenate([fft, psd, stats, wf, ff, ac]).astype(np.float32)
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    assert feats.size == N_FEATURES, f"got {feats.size}, expected {N_FEATURES}"
    return feats


if __name__ == "__main__":
    print(f"Feature vector length: {N_FEATURES}")
    print("First/last names:", FEATURE_NAMES[0], "...", FEATURE_NAMES[-1])
