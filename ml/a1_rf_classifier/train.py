"""A1 training — compare RandomForest / XGBoost / SVM, save the best by CV.

    python -m ml.a1_rf_classifier.train

Reads dataset/<class>/*.bin, extracts 735 features, trains 3 models, prints a
comparison + per-class report, and persists the winner + scaler + metadata.
"""
from __future__ import annotations
import os, sys, json, time
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from ml.common.config import (DATASET_DIR, RF_CLASSES, RF_CLASS_NAMES,
                              A1_RF_MODEL, A1_RF_SCALER, A1_MODEL_META)
from ml.a1_rf_classifier.feature_extraction import extract_features, N_FEATURES


def load_dataset():
    X, y = [], []
    for label, idx in RF_CLASSES.items():
        d = os.path.join(DATASET_DIR, label)
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if f.endswith(".bin"):
                try:
                    X.append(extract_features(os.path.join(d, f)))
                    y.append(idx)
                except Exception as e:
                    print(f"  skip {f}: {e}")
    return np.array(X), np.array(y)


def main():
    print("Loading dataset + extracting features...")
    X, y = load_dataset()
    if len(X) == 0:
        sys.exit("No data. Run: python -m ml.a1_rf_classifier.generate_mock_data")
    print(f"  {X.shape[0]} samples x {X.shape[1]} features "
          f"({'OK' if X.shape[1] == N_FEATURES else 'MISMATCH'})")

    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.2, stratify=y,
                                          random_state=42)

    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=42),
        "SVM": SVC(kernel="rbf", C=10, gamma="scale", probability=True,
                   random_state=42),
    }
    try:
        import xgboost as xgb
        models["XGBoost"] = xgb.XGBClassifier(
            n_estimators=200, eval_metric="mlogloss", n_jobs=-1, random_state=42)
    except ImportError:
        print("  (xgboost not installed; comparing RF + SVM only)")

    results = {}
    for name, m in models.items():
        t0 = time.time()
        m.fit(Xtr, ytr)
        test_acc = m.score(Xte, yte)
        cv = cross_val_score(m, Xs, y, cv=5, n_jobs=-1)
        results[name] = dict(test=test_acc, cv_mean=cv.mean(), cv_std=cv.std())
        print(f"{name:14s} test={test_acc*100:5.1f}%  "
              f"CV={cv.mean()*100:5.1f}+/-{cv.std()*100:.1f}%  ({time.time()-t0:.1f}s)")

    print("\nBest model per-class report:")
    best_name = max(results, key=lambda n: results[n]["cv_mean"])
    best = models[best_name]
    print(f"  winner by CV: {best_name}\n")
    print(classification_report(yte, best.predict(Xte), target_names=RF_CLASS_NAMES))
    print("Confusion matrix (rows=true):")
    print(confusion_matrix(yte, best.predict(Xte)))

    joblib.dump(best, A1_RF_MODEL)
    joblib.dump(sc, A1_RF_SCALER)
    json.dump({"best_model": best_name, "results": results,
               "classes": RF_CLASS_NAMES, "n_features": int(X.shape[1])},
              open(A1_MODEL_META, "w"), indent=2)
    print(f"\nSaved: {A1_RF_MODEL}\n       {A1_RF_SCALER}")


if __name__ == "__main__":
    main()
