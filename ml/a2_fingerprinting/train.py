"""A2 RF fingerprinting — SVM (RBF) to identify the exact drone model.

    python -m ml.a2_fingerprinting.train

Uses the same 735-dim features as A1 but trains only on drone-model captures.
"""
from __future__ import annotations
import os, sys
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from ml.common.config import (DATASET_DIR, DRONE_MODELS, DRONE_MODEL_NAMES,
                              A2_FP_MODEL, A2_FP_SCALER)
from ml.a1_rf_classifier.feature_extraction import extract_features


def load():
    X, y = [], []
    base = os.path.join(DATASET_DIR, "fingerprint")
    for model, idx in DRONE_MODELS.items():
        d = os.path.join(base, model)
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if f.endswith(".bin"):
                X.append(extract_features(os.path.join(d, f)))
                y.append(idx)
    return np.array(X), np.array(y)


def main():
    print("Loading fingerprint dataset...")
    X, y = load()
    if len(X) == 0:
        sys.exit("No data. Run: python -m ml.a2_fingerprinting.generate_mock_data")
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.2, stratify=y,
                                          random_state=42)
    svm = SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42)
    svm.fit(Xtr, ytr)
    cv = cross_val_score(svm, Xs, y, cv=5, n_jobs=-1)
    print(f"SVM(RBF) test={svm.score(Xte, yte)*100:.1f}%  "
          f"CV={cv.mean()*100:.1f}+/-{cv.std()*100:.1f}%\n")
    print(classification_report(yte, svm.predict(Xte), target_names=DRONE_MODEL_NAMES))
    print(confusion_matrix(yte, svm.predict(Xte)))
    joblib.dump(svm, A2_FP_MODEL)
    joblib.dump(sc, A2_FP_SCALER)
    print(f"\nSaved: {A2_FP_MODEL}")


if __name__ == "__main__":
    main()
