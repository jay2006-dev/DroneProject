"""A5 data — synthesize mock WAVs into dataset/audio/<class>/.

Classes: drone (rotor broadband + blade-pass harmonics, amplitude-modulated),
motor (clean harmonic whine), noise (pink/white background).

    python -m ml.a5_acoustic.generate_mock_audio --per-class 80
"""
from __future__ import annotations
import os, sys, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from ml.common.config import DATASET_DIR, ACOUSTIC_CLASSES

SR = 16000
DUR = 2.0


def _pink(n, rng):
    white = rng.normal(0, 1, n)
    # simple 1/f shaping
    f = np.fft.rfft(white)
    k = np.arange(1, f.size + 1)
    return np.fft.irfft(f / np.sqrt(k), n)


def synth_audio(kind, rng):
    n = int(SR * DUR)
    t = np.arange(n) / SR
    if kind == "noise":
        x = _pink(n, rng)
    elif kind == "motor":
        f0 = rng.uniform(90, 160)
        x = sum(np.sin(2 * np.pi * f0 * h * t) / h for h in range(1, 6))
        x += 0.05 * rng.normal(0, 1, n)
    elif kind == "drone":
        f0 = rng.uniform(110, 240)                  # blade-pass fundamental
        harm = sum(np.sin(2 * np.pi * f0 * h * t) / h for h in range(1, 8))
        am = 1 + 0.4 * np.sin(2 * np.pi * rng.uniform(8, 20) * t)  # rotor wobble
        x = harm * am + 0.6 * _pink(n, rng)         # broadband rotor wash
    else:
        raise ValueError(kind)
    x = x / (np.max(np.abs(x)) + 1e-9) * 0.9
    return x.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=80)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    import soundfile as sf
    rng = np.random.default_rng(args.seed)
    for cls in ACOUSTIC_CLASSES:
        d = os.path.join(DATASET_DIR, "audio", cls)
        os.makedirs(d, exist_ok=True)
        for i in range(args.per_class):
            sf.write(os.path.join(d, f"{cls}_{i:04d}.wav"), synth_audio(cls, rng), SR)
        print(f"  {cls:8s} -> {args.per_class} wavs")


if __name__ == "__main__":
    main()
