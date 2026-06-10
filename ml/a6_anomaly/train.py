"""A6 — unknown-drone detection via Isolation Forest.

Trains ONLY on known-drone captures, learning what a 'normal known drone' looks
like in feature space. At inference an unusually-placed sample is flagged as a
possible new/unknown drone model.

    python -m ml.a6_anomaly.train
"""
from __future__ import annotations
import os, sys
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from ml.common.config import DATASET_DIR, A1_RF_SCALER, A6_ISO_MODEL
from ml.a1_rf_classifier.feature_extraction import extract_features


def main():
    if not os.path.exists(A1_RF_SCALER):
        sys.exit("Train A1 first (shares its scaler): python -m ml.a1_rf_classifier.train")
    sc = joblib.load(A1_RF_SCALER)

    drone_dir = os.path.join(DATASET_DIR, "drone")
    files = [f for f in os.listdir(drone_dir) if f.endswith(".bin")]
    if not files:
        sys.exit("No drone captures. Run A1 mock generator first.")
    X = np.array([extract_features(os.path.join(drone_dir, f)) for f in files])
    Xs = sc.transform(X)

    iso = IsolationForest(contamination=0.1, random_state=42, n_jobs=-1)
    iso.fit(Xs)
    joblib.dump(iso, A6_ISO_MODEL)

    scores = iso.decision_function(Xs)
    print(f"Trained on {len(files)} known-drone captures.")
    print(f"  in-distribution decision_function: "
          f"mean={scores.mean():.3f}  min={scores.min():.3f}  max={scores.max():.3f}")
    print(f"  anomaly threshold suggestion: < {np.percentile(scores, 5):.3f}")
    print(f"Saved: {A6_ISO_MODEL}")


if __name__ == "__main__":
    main()
