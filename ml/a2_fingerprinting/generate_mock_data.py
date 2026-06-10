"""Generate per-drone-model RF captures for A2 into dataset/fingerprint/<model>/.

    python -m ml.a2_fingerprinting.generate_mock_data --per-model 100
"""
from __future__ import annotations
import os, sys, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from ml.common.config import DATASET_DIR, DRONE_MODELS
from ml.common.iq import save_iq
from ml.common.rf_synth import synth_drone_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-model", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    base = os.path.join(DATASET_DIR, "fingerprint")
    for model in DRONE_MODELS:
        out = os.path.join(base, model)
        os.makedirs(out, exist_ok=True)
        for i in range(args.per_model):
            snr = rng.uniform(8, 18)
            s = synth_drone_model(model, rng, snr_db=snr)
            save_iq(os.path.join(out, f"{model}_{i:04d}.bin"), s)
        print(f"  {model:14s} -> {args.per_model} files")


if __name__ == "__main__":
    main()
