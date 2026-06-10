"""A1 inference — classify one .bin (or live IQ array) into
noise/drone/wifi/bluetooth with a confidence %.

    python -m ml.a1_rf_classifier.infer path/to/capture.bin

Importable: `from ml.a1_rf_classifier.infer import RFClassifier`
"""
from __future__ import annotations
import os, sys
import numpy as np
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from ml.common.config import A1_RF_MODEL, A1_RF_SCALER, RF_CLASS_NAMES
from ml.a1_rf_classifier.feature_extraction import extract_features


class RFClassifier:
    def __init__(self, model_path=A1_RF_MODEL, scaler_path=A1_RF_SCALER):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"{model_path} missing — train first: "
                f"python -m ml.a1_rf_classifier.train")
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

    def predict(self, filepath_or_iq) -> dict:
        feats = extract_features(filepath_or_iq).reshape(1, -1)
        fs = self.scaler.transform(feats)
        proba = self.model.predict_proba(fs)[0]
        idx = int(np.argmax(proba))
        return {
            "label": RF_CLASS_NAMES[idx],
            "confidence": round(float(proba[idx]) * 100, 1),
            "probabilities": {n: round(float(p) * 100, 1)
                              for n, p in zip(RF_CLASS_NAMES, proba)},
            "features": feats.ravel(),  # reused by A6/A8 to avoid recompute
        }


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python -m ml.a1_rf_classifier.infer <file.bin>")
    clf = RFClassifier()
    r = clf.predict(sys.argv[1])
    print(f"-> {r['label'].upper()}  ({r['confidence']}%)")
    for n, p in sorted(r["probabilities"].items(), key=lambda kv: -kv[1]):
        print(f"     {n:10s} {p:5.1f}%")


if __name__ == "__main__":
    main()
