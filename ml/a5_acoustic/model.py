"""A5 — small CNN over the 40x64 MFCC image (drone/noise/motor)."""
from __future__ import annotations


def build_cnn(n_classes: int):
    import torch.nn as nn
    return nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        nn.Flatten(), nn.Dropout(0.3), nn.Linear(64, n_classes),
    )
