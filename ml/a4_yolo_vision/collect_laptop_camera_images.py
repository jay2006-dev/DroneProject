"""A4 data collection from the laptop webcam.

    python -m ml.a4_yolo_vision.collect_laptop_camera_images

S = save current frame as 'drone', N = save as 'background', Q = quit.
Saves to dataset/drone_visual/raw/{drone,background}/. Annotate later with
LabelImg or Roboflow to produce YOLO labels.
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from ml.common.config import DATASET_DIR


def main():
    import cv2
    base = os.path.join(DATASET_DIR, "drone_visual", "raw")
    for sub in ("drone", "background"):
        os.makedirs(os.path.join(base, sub), exist_ok=True)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("Could not open webcam (index 0).")
    count = 0
    print("S=save drone  N=save background  Q=quit")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imshow("Collect — Laptop Camera", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("s"):
            cv2.imwrite(os.path.join(base, "drone", f"frame_{count:04d}.jpg"), frame)
            count += 1; print(f"saved drone {count}")
        elif k == ord("n"):
            cv2.imwrite(os.path.join(base, "background", f"frame_{count:04d}.jpg"), frame)
            count += 1; print(f"saved background {count}")
        elif k == ord("q"):
            break
    cap.release(); cv2.destroyAllWindows()
    print(f"Collected {count} images. Annotate, then train.")


if __name__ == "__main__":
    main()
