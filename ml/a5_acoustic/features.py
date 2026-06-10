"""A5 — MFCC feature extraction (shared by train + infer)."""
from __future__ import annotations
import numpy as np

SR = 16000
N_MFCC = 40
MAX_FRAMES = 64  # fixed time width so every clip -> 40x64 image


def wav_to_mfcc(path_or_audio, sr=SR) -> np.ndarray:
    """Return a normalised 40x64 MFCC 'image' (float32) for a WAV path or array."""
    import librosa
    if isinstance(path_or_audio, str):
        y, sr = librosa.load(path_or_audio, sr=sr)
    else:
        y = np.asarray(path_or_audio, dtype=np.float32)
    m = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    if m.shape[1] < MAX_FRAMES:
        m = np.pad(m, ((0, 0), (0, MAX_FRAMES - m.shape[1])))
    m = m[:, :MAX_FRAMES]
    m = (m - m.mean()) / (m.std() + 1e-9)
    return m.astype(np.float32)
