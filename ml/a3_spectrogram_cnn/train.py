"""A3 — spectrogram CNN via MobileNetV2 transfer learning (PyTorch).

    python -m ml.a3_spectrogram_cnn.train --epochs 5

Trains on dataset/spectrograms/{train,val}. Run make_spectrograms first.
(Uses torchvision MobileNetV2 — same architecture the spec calls for, one
framework shared with A4/A5.)
"""
from __future__ import annotations
import os, sys, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from ml.common.config import DATASET_DIR, A3_CNN_MODEL, SPECTRO_CLASSES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms, models

    root = os.path.join(DATASET_DIR, "spectrograms")
    if not os.path.isdir(os.path.join(root, "train")):
        sys.exit("No spectrograms. Run: python -m ml.a3_spectrogram_cnn.make_spectrograms")

    tf = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    train_ds = datasets.ImageFolder(os.path.join(root, "train"), tf)
    val_ds = datasets.ImageFolder(os.path.join(root, "val"), tf)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch)
    classes = train_ds.classes
    print(f"classes={classes}  train={len(train_ds)} val={len(val_ds)}")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    net.classifier[1] = nn.Linear(net.last_channel, len(classes))
    net = net.to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    for ep in range(args.epochs):
        net.train()
        for x, y in train_dl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(); loss = crit(net(x), y); loss.backward(); opt.step()
        net.eval(); correct = total = 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(dev), y.to(dev)
                correct += (net(x).argmax(1) == y).sum().item(); total += y.numel()
        print(f"epoch {ep+1}/{args.epochs}  val_acc={correct/max(total,1)*100:.1f}%")

    torch.save({"state_dict": net.state_dict(), "classes": classes}, A3_CNN_MODEL)
    print(f"Saved: {A3_CNN_MODEL}")


if __name__ == "__main__":
    main()
