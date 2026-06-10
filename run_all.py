"""One-shot pipeline runner for the ML side (Part A).

    python run_all.py                # core only (A1,A2,A6,A7,A8,A9,A10) - fast
    python run_all.py --deep         # also A3 spectrogram CNN + A5 acoustic CNN
    python run_all.py --skip-gen     # reuse existing dataset/

Steps: generate mock data -> train models -> run evaluations -> fusion demo.
"""
from __future__ import annotations
import os, sys, subprocess, argparse

PY = sys.executable
ROOT = os.path.dirname(os.path.abspath(__file__))


def run(mod, *args):
    print(f"\n{'='*60}\n>>> {mod} {' '.join(args)}\n{'='*60}")
    r = subprocess.run([PY, "-m", mod, *args], cwd=ROOT)
    if r.returncode != 0:
        print(f"!! {mod} exited {r.returncode}")
    return r.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deep", action="store_true")
    ap.add_argument("--skip-gen", action="store_true")
    a = ap.parse_args()

    if not a.skip_gen:
        run("ml.a1_rf_classifier.generate_mock_data", "--per-class", "80")
        run("ml.a2_fingerprinting.generate_mock_data", "--per-model", "80")

    # Core training
    run("ml.a1_rf_classifier.train")
    run("ml.a2_fingerprinting.train")
    run("ml.a6_anomaly.train")

    # Evaluation (A9) + registry demo (A10)
    run("ml.a9_evaluation.distance_test")
    run("ml.a9_evaluation.false_positive_test")
    run("ml.a10_registry.registry", "--drone", "DJI_Tello", "--distance", "20",
        "--location", "Lab_Room3", "--weather", "indoor_clear", "--files", "80",
        "--notes", "mock bootstrap session")

    if a.deep:
        if not a.skip_gen:
            run("ml.a5_acoustic.generate_mock_audio", "--per-class", "80")
        run("ml.a3_spectrogram_cnn.make_spectrograms")
        run("ml.a3_spectrogram_cnn.train", "--epochs", "4")
        run("ml.a5_acoustic.train", "--epochs", "8")
        run("ml.a4_yolo_vision.make_synthetic_images", "--per-class", "40")
        # YOLO training is the slowest; left as an explicit manual step.
        print("\n(YOLO A4: run `python -m ml.a4_yolo_vision.train --epochs 30` when ready)")

    run("ml.fusion_demo")
    print("\nALL DONE. See reports/ for A9 tables, models/ for artifacts.")


if __name__ == "__main__":
    main()
