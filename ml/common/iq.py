"""IQ sample I/O — one canonical on-disk format shared by mock generators,
feature extraction, and (later) real RTL-SDR captures.

On-disk format: interleaved unsigned 8-bit I,Q  ->  [I0, Q0, I1, Q1, ...]
This is exactly what `rtl_sdr` writes, so real captures drop in unchanged.

    save_iq(path, complex_array)   # complex64/128 in, uint8 .bin out
    load_iq(path)                  # uint8 .bin in, complex64 out (centered)
"""
from __future__ import annotations
import numpy as np


def save_iq(path: str, samples: np.ndarray) -> None:
    """Quantize a complex IQ array to RTL-SDR-style interleaved uint8 and write."""
    s = np.asarray(samples, dtype=np.complex64)
    # Scale real/imag from roughly [-1, 1] into [0, 255] centered at 127.5,
    # the way an 8-bit SDR front end does.
    i = np.clip(np.real(s) * 127.0 + 127.5, 0, 255).astype(np.uint8)
    q = np.clip(np.imag(s) * 127.0 + 127.5, 0, 255).astype(np.uint8)
    interleaved = np.empty(i.size * 2, dtype=np.uint8)
    interleaved[0::2] = i
    interleaved[1::2] = q
    interleaved.tofile(path)


def load_iq(path: str, max_samples: int | None = None) -> np.ndarray:
    """Read interleaved uint8 .bin and return centered complex64 IQ samples."""
    raw = np.fromfile(path, dtype=np.uint8)
    if raw.size % 2:           # guard odd byte count
        raw = raw[:-1]
    iq = raw.astype(np.float32)
    iq = (iq - 127.5) / 127.5  # back to ~[-1, 1]
    s = iq[0::2] + 1j * iq[1::2]
    s = s.astype(np.complex64)
    if max_samples is not None:
        s = s[:max_samples]
    return s
