"""A2 inference — identify the exact drone model from an IQ capture."""
from __future__ import annotations
import os, sys
import numpy as np
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from ml.common.config import A2_FP_MODEL, A2_FP_SCALER, DRONE_MODEL_NAMES
from ml.a1_rf_classifier.feature_extraction import extract_features


class FingerprintID:
    def __init__(self, model_path=A2_FP_MODEL, scaler_path=A2_FP_SCALER):
        if not os.path.exists(model_path):
            raise FileNotFoundError("train first: python -m ml.a2_fingerprinting.train")
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

    def predict(self, filepath_or_iq) -> dict:
        f = extract_features(filepath_or_iq).reshape(1, -1)
        proba = self.model.predict_proba(self.scaler.transform(f))[0]
        idx = int(np.argmax(proba))
        return {"model": DRONE_MODEL_NAMES[idx],
                "confidence": round(float(proba[idx]) * 100, 1)}


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python -m ml.a2_fingerprinting.infer <file.bin>")
    r = FingerprintID().predict(sys.argv[1])
    print(f"-> {r['model']}  ({r['confidence']}%)")


if __name__ == "__main__":
    main()
