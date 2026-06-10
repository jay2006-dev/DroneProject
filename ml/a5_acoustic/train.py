"""A5 — train acoustic CNN on MFCC images.

    python -m ml.a5_acoustic.train --epochs 8
"""
from __future__ import annotations
import os, sys, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from ml.common.config import DATASET_DIR, ACOUSTIC_CLASSES, A5_CNN_MODEL
from ml.a5_acoustic.features import wav_to_mfcc
from ml.a5_acoustic.model import build_cnn


def load():
    X, y = [], []
    for cls, idx in ACOUSTIC_CLASSES.items():
        d = os.path.join(DATASET_DIR, "audio", cls)
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if f.endswith(".wav"):
                X.append(wav_to_mfcc(os.path.join(d, f))); y.append(idx)
    return np.array(X), np.array(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader, random_split

    X, y = load()
    if len(X) == 0:
        sys.exit("No audio. Run: python -m ml.a5_acoustic.generate_mock_audio")
    X = torch.tensor(X).unsqueeze(1)            # (N,1,40,64)
    y = torch.tensor(y, dtype=torch.long)
    ds = TensorDataset(X, y)
    n_val = max(1, int(len(ds) * 0.2))
    tr, va = random_split(ds, [len(ds) - n_val, n_val],
                          generator=torch.Generator().manual_seed(42))
    tdl = DataLoader(tr, batch_size=args.batch, shuffle=True)
    vdl = DataLoader(va, batch_size=args.batch)

    net = build_cnn(len(ACOUSTIC_CLASSES))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    for ep in range(args.epochs):
        net.train()
        for xb, yb in tdl:
            opt.zero_grad(); loss = crit(net(xb), yb); loss.backward(); opt.step()
        net.eval(); c = t = 0
        with torch.no_grad():
            for xb, yb in vdl:
                c += (net(xb).argmax(1) == yb).sum().item(); t += yb.numel()
        print(f"epoch {ep+1}/{args.epochs}  val_acc={c/max(t,1)*100:.1f}%")
    torch.save({"state_dict": net.state_dict(),
                "classes": list(ACOUSTIC_CLASSES)}, A5_CNN_MODEL)
    print(f"Saved: {A5_CNN_MODEL}")


if __name__ == "__main__":
    main()
