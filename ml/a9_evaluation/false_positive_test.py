"""A9b — false-positive analysis. How often does A1 mislabel common 2.4 GHz
interferers as 'drone'? Judges ask for this right after accuracy.

    python -m ml.a9_evaluation.false_positive_test

Mock interferers: wifi, bluetooth, microwave (wide continuous 2.45 GHz), hotspot.
Writes reports/false_positive.md.
"""
from __future__ import annotations
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from ml.common.config import REPORTS_DIR, IQ_SAMPLES_PER_FILE
from ml.common.rf_synth import synth, _ofdm_burst, _awgn
from ml.a1_rf_classifier.infer import RFClassifier

N = 50
ACCEPTABLE = {"wifi_4k": 5, "bluetooth_music": 3, "phone_hotspot": 5,
              "microwave": 8, "laptop_wifi": 5}


def _microwave(rng, snr_db=14):
    """Wide continuous 2.45 GHz emission (no bursting) -> classic FP source."""
    sig = _awgn(IQ_SAMPLES_PER_FILE, 0.08, rng)
    amp = 10 ** (snr_db / 20.0) * 0.08
    body = _ofdm_burst(IQ_SAMPLES_PER_FILE, bw_norm=0.8, rng=rng)
    body = body / (np.std(body) + 1e-9)
    return (sig + amp * body).astype(np.complex64)


def gen(source, rng):
    if source == "wifi_4k":          return synth("wifi", rng, snr_db=16)
    if source == "laptop_wifi":      return synth("wifi", rng, snr_db=11)
    if source == "phone_hotspot":    return synth("wifi", rng, snr_db=13)
    if source == "bluetooth_music":  return synth("bluetooth", rng, snr_db=14)
    if source == "microwave":        return _microwave(rng)
    raise ValueError(source)


def main():
    clf = RFClassifier()
    rng = np.random.default_rng(123)
    md = ["# A9 False-Positive Analysis\n",
          "| Source | Tests | FP=drone | FP rate | Acceptable | Status |",
          "|---|---|---|---|---|---|"]
    print("source            FP%   (accept)")
    for source, acc_thresh in ACCEPTABLE.items():
        fp = sum(clf.predict(gen(source, rng))["label"] == "drone" for _ in range(N))
        rate = fp / N * 100
        status = "OK" if rate <= acc_thresh else "TOO HIGH"
        md.append(f"| {source} | {N} | {fp} | {rate:.1f}% | <{acc_thresh}% | {status} |")
        print(f"  {source:16s} {rate:5.1f}%  (<{acc_thresh}%)  {status}")
    md.append("\n*Mock interferers — re-run with real recordings of each source.*")
    out = os.path.join(REPORTS_DIR, "false_positive.md")
    open(out, "w", encoding="utf-8").write("\n".join(md))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
