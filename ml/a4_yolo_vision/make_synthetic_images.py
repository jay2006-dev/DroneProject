"""A4 synthetic vision dataset — generate drone/bird/plane images with YOLO
labels so YOLOv8 training runs before any real footage exists.

    python -m ml.a4_yolo_vision.make_synthetic_images --per-class 60

Produces dataset/drone_visual/{images,labels}/{train,val}/ + data.yaml.
Each image: a simple procedural shape (quad-rotor / bird / plane silhouette) on
a sky/ground gradient. Swap for real annotated data (Roboflow/VisDrone) anytime.
"""
from __future__ import annotations
import os, sys, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from ml.common.config import DATASET_DIR, VISION_CLASSES

IMG = 320
ROOT = os.path.join(DATASET_DIR, "drone_visual")


def _bg(rng):
    from PIL import Image
    top = np.array([rng.integers(120, 180), rng.integers(160, 210), rng.integers(200, 255)])
    bot = np.array([rng.integers(200, 240)] * 3)
    grad = np.linspace(0, 1, IMG)[:, None, None]
    arr = (top * (1 - grad) + bot * grad).astype(np.uint8)
    arr = np.repeat(arr, IMG, axis=1)
    return Image.fromarray(arr)


def _draw(cls, img, rng):
    """Draw the object, return YOLO bbox (cx,cy,w,h) normalised."""
    from PIL import ImageDraw
    d = ImageDraw.Draw(img)
    w = rng.integers(40, 90); h = rng.integers(30, 70)
    cx = rng.integers(w, IMG - w); cy = rng.integers(h, IMG - h)
    x0, y0, x1, y1 = cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2
    col = (40, 40, 40)
    if cls == "drone":          # body + 4 rotor circles
        d.rectangle([cx - 10, cy - 6, cx + 10, cy + 6], fill=col)
        for dx, dy in [(-w//2, -h//2), (w//2, -h//2), (-w//2, h//2), (w//2, h//2)]:
            d.ellipse([cx+dx-8, cy+dy-8, cx+dx+8, cy+dy+8], outline=col, width=3)
    elif cls == "bird":         # two arcs (wings)
        d.arc([x0, y0, cx, y1], 200, 340, fill=col, width=4)
        d.arc([cx, y0, x1, y1], 200, 340, fill=col, width=4)
    else:                       # plane: fuselage + wings
        d.line([x0, cy, x1, cy], fill=col, width=5)
        d.line([cx, y0, cx, y1], fill=col, width=4)
    return (cx / IMG, cy / IMG, (w + 10) / IMG, (h + 10) / IMG)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=60)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    for split in ("train", "val"):
        for sub in ("images", "labels"):
            os.makedirs(os.path.join(ROOT, sub, split), exist_ok=True)

    n = 0
    for ci, cls in enumerate(VISION_CLASSES):
        for i in range(args.per_class):
            split = "train" if i < int(args.per_class * 0.8) else "val"
            img = _bg(rng)
            cx, cy, bw, bh = _draw(cls, img, rng)
            stem = f"{cls}_{i:04d}"
            img.save(os.path.join(ROOT, "images", split, stem + ".jpg"))
            with open(os.path.join(ROOT, "labels", split, stem + ".txt"), "w") as f:
                f.write(f"{ci} {cx:.5f} {cy:.5f} {bw:.5f} {bh:.5f}\n")
            n += 1

    yaml = os.path.join(ROOT, "data.yaml")
    with open(yaml, "w") as f:
        f.write(f"path: {ROOT}\ntrain: images/train\nval: images/val\n")
        f.write(f"nc: {len(VISION_CLASSES)}\n")
        f.write(f"names: {VISION_CLASSES}\n")
    print(f"Wrote {n} images + labels. data.yaml -> {yaml}")


if __name__ == "__main__":
    main()
