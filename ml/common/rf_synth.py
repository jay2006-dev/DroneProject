"""Mock RF signal synthesizer.

Produces complex IQ that *looks like* real 2.4 GHz ISM traffic so the whole ML
pipeline trains and evaluates end-to-end before any RTL-SDR data exists. Each
class has a physically-motivated signature, so the extracted features are
genuinely separable (not random):

    noise      : additive white Gaussian noise only
    wifi       : wide (~16/20 MHz) OFDM bursts + periodic beacons
    bluetooth  : narrowband frequency-hopping (1600 hops/s feel), short dwell
    drone      : OFDM video downlink burst + slower hopping control channel

Drone *models* (A2) differ in hop rate, occupied bandwidth and burst duty so the
fingerprint SVM has something real to latch onto.

A per-call `rng` seed keeps datasets reproducible.
"""
from __future__ import annotations
import numpy as np
from ml.common.config import IQ_SAMPLES_PER_FILE, SAMPLE_RATE

N = IQ_SAMPLES_PER_FILE


def _awgn(n, sigma, rng):
    return (rng.normal(0, sigma, n) + 1j * rng.normal(0, sigma, n)).astype(np.complex64)


def _tone(n, f_norm, rng, phase=None):
    t = np.arange(n)
    ph = rng.uniform(0, 2 * np.pi) if phase is None else phase
    return np.exp(1j * (2 * np.pi * f_norm * t + ph)).astype(np.complex64)


def _ofdm_burst(n, bw_norm, rng):
    """Crude multi-carrier burst over a band of width bw_norm (fraction of fs)."""
    sig = np.zeros(n, dtype=np.complex64)
    n_carriers = max(4, int(bw_norm * 64))
    centers = rng.uniform(-bw_norm / 2, bw_norm / 2, n_carriers)
    for fc in centers:
        sig += _tone(n, fc, rng) * rng.uniform(0.3, 1.0)
    return sig / np.sqrt(n_carriers)


def _hopping(n, dwell, bw_norm, rng, hop_span=0.4):
    """Frequency-hopping signal: retune every `dwell` samples."""
    sig = np.zeros(n, dtype=np.complex64)
    i = 0
    while i < n:
        seg = min(dwell, n - i)
        fc = rng.uniform(-hop_span, hop_span)
        t = np.arange(seg)
        sig[i:i + seg] = (_ofdm_burst(seg, bw_norm, rng)
                          * np.exp(1j * 2 * np.pi * fc * t)).astype(np.complex64)
        i += seg
    return sig


def _bursty(sig, period, duty, rng):
    """Gate a continuous signal into bursts (period/duty in samples)."""
    env = np.zeros(len(sig), dtype=np.float32)
    i = 0
    while i < len(sig):
        on = int(period * duty)
        env[i:i + on] = 1.0
        i += period
    return sig * env


# ---- public API ---------------------------------------------------------------

def synth(kind: str, rng: np.random.Generator, snr_db: float = 12.0) -> np.ndarray:
    """Generate one capture of the given class: noise|wifi|bluetooth|drone."""
    noise_sigma = 0.08
    sig = _awgn(N, noise_sigma, rng)

    if kind == "noise":
        return sig

    amp = 10 ** (snr_db / 20.0) * noise_sigma
    if kind == "wifi":
        body = _bursty(_ofdm_burst(N, bw_norm=0.7, rng=rng), period=8000, duty=0.35, rng=rng)
    elif kind == "bluetooth":
        body = _hopping(N, dwell=900, bw_norm=0.04, rng=rng, hop_span=0.45)
    elif kind == "drone":
        video = _bursty(_ofdm_burst(N, bw_norm=0.5, rng=rng), period=6000, duty=0.5, rng=rng)
        control = _hopping(N, dwell=2500, bw_norm=0.06, rng=rng, hop_span=0.3) * 0.4
        body = video + control
    else:
        raise ValueError(f"unknown kind {kind!r}")

    body = body / (np.std(body) + 1e-9)
    return (sig + amp * body).astype(np.complex64)


# Per-model fingerprint parameters for A2 (subtle, learnable differences).
_DRONE_PROFILES = {
    "DJI_Mavic":   dict(video_bw=0.55, vid_period=5200, ctrl_dwell=2600, ctrl_bw=0.05),
    "DJI_Tello":   dict(video_bw=0.35, vid_period=7000, ctrl_dwell=1800, ctrl_bw=0.04),
    "Syma_X5C":    dict(video_bw=0.20, vid_period=9000, ctrl_dwell=1200, ctrl_bw=0.03),
    "Parrot_Anafi":dict(video_bw=0.60, vid_period=4800, ctrl_dwell=3000, ctrl_bw=0.07),
}


def synth_drone_model(model: str, rng: np.random.Generator, snr_db: float = 12.0) -> np.ndarray:
    """Generate a drone capture with a model-specific RF fingerprint (A2)."""
    p = _DRONE_PROFILES[model]
    noise_sigma = 0.08
    sig = _awgn(N, noise_sigma, rng)
    amp = 10 ** (snr_db / 20.0) * noise_sigma
    video = _bursty(_ofdm_burst(N, p["video_bw"], rng), period=p["vid_period"], duty=0.5, rng=rng)
    control = _hopping(N, dwell=p["ctrl_dwell"], bw_norm=p["ctrl_bw"], rng=rng, hop_span=0.3) * 0.4
    body = video + control
    body = body / (np.std(body) + 1e-9)
    return (sig + amp * body).astype(np.complex64)
