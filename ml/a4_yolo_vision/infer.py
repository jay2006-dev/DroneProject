"""A4 live inference — laptop webcam OR ESP-EYE MJPEG stream.

    python -m ml.a4_yolo_vision.infer                 # laptop webcam (index 0)
    python -m ml.a4_yolo_vision.infer --source http://ESP_EYE_IP:81/stream
    python -m ml.a4_yolo_vision.infer --source path/to/video.mp4 --no-show

Publishes {sensor, detected, confidence, n} to MQTT topic
campusshield/sensor/visual (no-ops if no broker). A4 and A4b (ESP-EYE) share
these exact weights — only --source differs.
"""
from __future__ import annotations
import os, sys, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from ml.common.config import A4_YOLO_MODEL
from ml.common.bus import publish


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="0=webcam | URL | video path")
    ap.add_argument("--weights", default=A4_YOLO_MODEL)
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()

    import cv2
    from ultralytics import YOLO
    weights = args.weights if os.path.exists(args.weights) else "yolov8n.pt"
    if weights != args.weights:
        print(f"(trained weights not found; falling back to pretrained {weights})")
    model = YOLO(weights)

    src = 0 if args.source == "0" else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        sys.exit(f"Could not open source {src!r}")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = model.predict(frame, conf=args.conf, verbose=False)
        names = res[0].names
        drones = [b for b in res[0].boxes if names[int(b.cls)] == "drone"]
        conf = float(max([float(b.conf) for b in drones], default=0.0))
        publish("campusshield/sensor/visual",
                {"sensor": "visual", "detected": len(drones) > 0,
                 "confidence": round(conf, 3), "n": len(drones)}, verbose=False)
        if not args.no_show:
            cv2.imshow("CampusShield — A4", res[0].plot())
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release(); cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
