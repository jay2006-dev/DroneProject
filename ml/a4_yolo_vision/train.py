"""A4 — fine-tune YOLOv8n on the drone/bird/plane dataset.

    python -m ml.a4_yolo_vision.train --epochs 30

Reads dataset/drone_visual/data.yaml (run make_synthetic_images first, or point
it at a real Roboflow/VisDrone export). Saves best weights to models/a4_best_drone.pt.
"""
from __future__ import annotations
import os, sys, argparse, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from ml.common.config import DATASET_DIR, A4_YOLO_MODEL, MODELS_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=320)
    args = ap.parse_args()
    from ultralytics import YOLO

    data = os.path.join(DATASET_DIR, "drone_visual", "data.yaml")
    if not os.path.exists(data):
        sys.exit("No data.yaml. Run: python -m ml.a4_yolo_vision.make_synthetic_images")

    model = YOLO("yolov8n.pt")  # pretrained -> transfer learning
    res = model.train(data=data, epochs=args.epochs, imgsz=args.imgsz,
                      project=os.path.join(MODELS_DIR, "yolo_runs"), name="drone",
                      exist_ok=True, verbose=True)
    best = os.path.join(res.save_dir, "weights", "best.pt")
    if os.path.exists(best):
        shutil.copy(best, A4_YOLO_MODEL)
        print(f"Saved best weights -> {A4_YOLO_MODEL}")


if __name__ == "__main__":
    main()
