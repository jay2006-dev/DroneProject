"""A8 — explainable AI for the A1 RF classifier.

Two layers:
  1. Global feature importance (fast, no extra deps) -> top drivers overall.
  2. Per-prediction SHAP -> 'Detected as DRONE because: entropy 31%, ...'.

    python -m ml.a8_explainability.explain                 # global top-10
    python -m ml.a8_explainability.explain path/to.bin     # explain one capture
"""
from __future__ import annotations
import os, sys
import numpy as np
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from ml.common.config import A1_RF_MODEL, A1_RF_SCALER, RF_CLASS_NAMES, REPORTS_DIR
from ml.a1_rf_classifier.feature_extraction import (extract_features,
                                                    FEATURE_NAMES)

# Friendly labels for the feature blocks (for nicer dashboard text).
def _pretty(name: str) -> str:
    if name.startswith("fft_"):  return f"FFT bin {name[4:]}"
    if name.startswith("psd_"):  return f"PSD bin {name[4:]}"
    if name.startswith("wav_"):  return f"Wavelet {name[4:]}"
    if name.startswith("ac_"):   return f"Autocorr lag {name[3:]}"
    return {"entropy": "Spectral entropy", "occupancy": "Spectral occupancy",
            "rms": "RMS power", "inst_std": "Inst-freq spread"}.get(name, name)


def global_importance(top_k: int = 10):
    rf = joblib.load(A1_RF_MODEL)
    if not hasattr(rf, "feature_importances_"):
        print(f"Model {type(rf).__name__} has no feature_importances_ "
              f"(winner was not a tree model). Use SHAP path instead.")
        return []
    imp = rf.feature_importances_
    order = np.argsort(imp)[::-1][:top_k]
    print(f"Top {top_k} features driving RF classification:")
    out = []
    for rank, idx in enumerate(order, 1):
        print(f"  {rank:2d}. {_pretty(FEATURE_NAMES[idx]):22s} {imp[idx]*100:4.1f}%")
        out.append((FEATURE_NAMES[idx], float(imp[idx])))
    return out


def explain_prediction(filepath_or_iq, top_k: int = 5) -> dict:
    rf = joblib.load(A1_RF_MODEL)
    sc = joblib.load(A1_RF_SCALER)
    feats = extract_features(filepath_or_iq).reshape(1, -1)
    fs = sc.transform(feats)
    proba = rf.predict_proba(fs)[0]
    cls = int(np.argmax(proba))
    label = RF_CLASS_NAMES[cls]

    try:
        import shap
        explainer = shap.TreeExplainer(rf)
        sv = explainer.shap_values(fs)
        # multiclass -> list per class (or 3D array in newer shap)
        contrib = sv[cls][0] if isinstance(sv, list) else np.asarray(sv)[0, :, cls]
    except Exception as e:
        print(f"  (SHAP unavailable: {type(e).__name__}; falling back to importance)")
        contrib = getattr(rf, "feature_importances_", np.zeros(len(FEATURE_NAMES)))

    order = np.argsort(np.abs(contrib))[::-1][:top_k]
    total = np.sum(np.abs(contrib[order])) + 1e-12
    drivers = [{"feature": _pretty(FEATURE_NAMES[i]),
                "share": round(float(abs(contrib[i]) / total) * 100, 1)}
               for i in order]

    print(f"Detected as {label.upper()} ({proba[cls]*100:.0f}%) because:")
    for d in drivers:
        print(f"    {d['feature']:22s} {d['share']:4.1f}%")
    return {"label": label, "confidence": round(float(proba[cls]) * 100, 1),
            "drivers": drivers}


def main():
    if not os.path.exists(A1_RF_MODEL):
        sys.exit("Train A1 first: python -m ml.a1_rf_classifier.train")
    if len(sys.argv) >= 2:
        explain_prediction(sys.argv[1])
    else:
        global_importance()


if __name__ == "__main__":
    main()
