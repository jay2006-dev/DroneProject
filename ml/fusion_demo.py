"""End-to-end fusion demo: run every available ML head on one mock event and
fold the results into an A7 threat score with an A8 explanation.

    python -m ml.fusion_demo

Shows how A1 (RF) + A2 (fingerprint) + A6 (anomaly) + A5 (acoustic) feed A7.
Deep heads (A3/A5) are used if their models exist, otherwise skipped cleanly.
"""
from __future__ import annotations
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from ml.common.rf_synth import synth, synth_drone_model
from ml.a1_rf_classifier.infer import RFClassifier
from ml.a7_threat_scoring.threat_score import compute_threat_score, ThreatInputs


def main():
    rng = np.random.default_rng(2026)
    # Simulate a real DJI Mavic present for 45s at medium range.
    iq = synth_drone_model("DJI_Mavic", rng, snr_db=13)

    rf = RFClassifier().predict(iq)
    print(f"A1 RF        : {rf['label']} ({rf['confidence']}%)")
    rf_conf = rf["probabilities"].get("drone", 0) / 100

    # A2 fingerprint (optional)
    fp_conf = 0.0
    try:
        from ml.a2_fingerprinting.infer import FingerprintID
        fp = FingerprintID().predict(iq)
        fp_conf = fp["confidence"] / 100
        print(f"A2 fingerprint: {fp['model']} ({fp['confidence']}%)")
    except Exception as e:
        print(f"A2 fingerprint: skipped ({type(e).__name__})")

    # A6 anomaly (optional)
    is_unknown = False
    try:
        from ml.a6_anomaly.infer import AnomalyDetector
        an = AnomalyDetector().classify(iq)
        is_unknown = an["kind"] == "unknown"
        print(f"A6 anomaly   : {an['verdict']}")
    except Exception as e:
        print(f"A6 anomaly   : skipped ({type(e).__name__})")

    # A5 acoustic (optional, deep)
    acoustic_conf = 0.0
    try:
        from ml.a5_acoustic.infer import AcousticCNN
        from ml.a5_acoustic.generate_mock_audio import synth_audio
        au = AcousticCNN().predict(synth_audio("drone", rng))
        acoustic_conf = au["confidence"] / 100 if au["label"] == "drone" else 0.0
        print(f"A5 acoustic  : {au['label']} ({au['confidence']}%)")
    except Exception as e:
        print(f"A5 acoustic  : skipped ({type(e).__name__})")

    inp = ThreatInputs(rf_conf=rf_conf, visual_conf=0.0, acoustic_conf=acoustic_conf,
                       fingerprint_conf=fp_conf, bluetooth_conf=0.0,
                       duration_sec=45, is_unknown=is_unknown)
    score = compute_threat_score(inp)
    print("\n=== A7 THREAT SCORE ===")
    print(f"{score['icon']} {score['level']}  {score['score']}/100")
    print("  modifiers:", ", ".join(score["modifiers"]) or "none")

    print("\n=== A8 EXPLANATION ===")
    try:
        from ml.a8_explainability.explain import explain_prediction
        explain_prediction(iq, top_k=5)
    except Exception as e:
        print(f"  skipped ({type(e).__name__})")


if __name__ == "__main__":
    main()
