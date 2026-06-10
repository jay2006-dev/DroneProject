"""A6 inference — combine A1 classifier + Isolation Forest into a
KNOWN / UNKNOWN / SUSPICIOUS verdict.
"""
from __future__ import annotations
import os, sys
import numpy as np
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from ml.common.config import (A1_RF_MODEL, A1_RF_SCALER, A6_ISO_MODEL,
                              RF_CLASS_NAMES)
from ml.a1_rf_classifier.feature_extraction import extract_features


class AnomalyDetector:
    def __init__(self, anomaly_thresh: float = -0.05):
        self.rf = joblib.load(A1_RF_MODEL)
        self.sc = joblib.load(A1_RF_SCALER)
        self.iso = joblib.load(A6_ISO_MODEL)
        self.thresh = anomaly_thresh

    def classify(self, filepath_or_iq) -> dict:
        f = extract_features(filepath_or_iq).reshape(1, -1)
        fs = self.sc.transform(f)
        proba = self.rf.predict_proba(fs)[0]
        iso_score = float(self.iso.decision_function(fs)[0])
        top = int(np.argmax(proba))
        label, conf = RF_CLASS_NAMES[top], float(proba[top])

        if label == "drone" and conf > 0.7:
            verdict = f"KNOWN: drone ({conf:.0%})"
            kind = "known"
        elif iso_score < self.thresh:
            verdict = "UNKNOWN DRONE — possible new model"
            kind = "unknown"
        else:
            verdict = "SUSPICIOUS — monitoring"
            kind = "suspicious"
        return {"verdict": verdict, "kind": kind, "rf_label": label,
                "rf_confidence": round(conf * 100, 1),
                "anomaly_score": round(iso_score, 4)}


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python -m ml.a6_anomaly.infer <file.bin>")
    print(AnomalyDetector().classify(sys.argv[1]))


if __name__ == "__main__":
    main()
