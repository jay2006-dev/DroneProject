"""A9a — distance test. Measures A1 drone-vs-noise accuracy as range increases.

Mock mode maps each distance to a lower SNR (farther = weaker signal). Replace
`synth` calls with real captures recorded at each distance to get real numbers.

    python -m ml.a9_evaluation.distance_test

Writes reports/distance_test.md (+ .png if matplotlib available).
"""
from __future__ import annotations
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from ml.common.config import REPORTS_DIR
from ml.common.rf_synth import synth
from ml.a1_rf_classifier.infer import RFClassifier

# distance (m) -> approximate received SNR (dB). Tune to your LNA/antenna.
DISTANCE_SNR = {5: 18, 10: 15, 20: 12, 30: 9, 50: 6}
N_DRONE, N_NOISE = 30, 15


def main():
    clf = RFClassifier()
    rng = np.random.default_rng(99)
    rows = []
    for dist, snr in DISTANCE_SNR.items():
        correct = 0
        for _ in range(N_DRONE):
            pred = clf.predict(synth("drone", rng, snr_db=snr))["label"]
            correct += (pred == "drone")
        for _ in range(N_NOISE):
            pred = clf.predict(synth("noise", rng, snr_db=snr))["label"]
            correct += (pred == "noise")
        acc = correct / (N_DRONE + N_NOISE) * 100
        rows.append((dist, snr, acc))
        print(f"  {dist:3d} m  (SNR {snr:2d} dB)  acc={acc:5.1f}%")

    md = ["# A9 Distance Test\n",
          "| Distance | Sim SNR | Files | Accuracy |",
          "|---|---|---|---|"]
    for dist, snr, acc in rows:
        md.append(f"| {dist} m | {snr} dB | {N_DRONE}+{N_NOISE} | {acc:.1f}% |")
    md.append("\n*Mock data — replace synth() with real captures per distance.*")
    out = os.path.join(REPORTS_DIR, "distance_test.md")
    open(out, "w", encoding="utf-8").write("\n".join(md))
    print(f"Wrote {out}")

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        d = [r[0] for r in rows]; a = [r[2] for r in rows]
        plt.figure(figsize=(6, 4)); plt.plot(d, a, "o-")
        plt.xlabel("Distance (m)"); plt.ylabel("Accuracy (%)")
        plt.title("A1 accuracy vs distance"); plt.grid(True); plt.ylim(0, 105)
        png = os.path.join(REPORTS_DIR, "distance_test.png")
        plt.savefig(png, dpi=120, bbox_inches="tight")
        print(f"Wrote {png}")
    except Exception as e:
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
