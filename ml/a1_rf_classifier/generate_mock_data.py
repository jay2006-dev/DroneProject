"""Generate mock RF .bin captures for A1 (4-class) into dataset/<class>/.

    python -m ml.a1_rf_classifier.generate_mock_data --per-class 120

Swap-in real data later: just drop real .bin captures into the same folders and
skip this script. Format matches rtl_sdr output (interleaved uint8 I/Q).
"""
from __future__ import annotations
import os, sys, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from ml.common.config import DATASET_DIR, RF_CLASSES
from ml.common.iq import save_iq
from ml.common.rf_synth import synth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=120)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    for cls in RF_CLASSES:
        out = os.path.join(DATASET_DIR, cls)
        os.makedirs(out, exist_ok=True)
        for i in range(args.per_class):
            snr = rng.uniform(6, 18)            # vary signal strength = distances
            s = synth(cls, rng, snr_db=snr)
            save_iq(os.path.join(out, f"{cls}_{i:04d}.bin"), s)
        print(f"  {cls:10s} -> {args.per_class} files")
    print(f"Done. dataset/ now holds {len(RF_CLASSES) * args.per_class} captures.")


if __name__ == "__main__":
    main()
