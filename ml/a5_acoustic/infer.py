"""A5 inference — classify a WAV (or audio array) as drone/noise/motor."""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from ml.common.config import A5_CNN_MODEL
from ml.a5_acoustic.features import wav_to_mfcc
from ml.a5_acoustic.model import build_cnn


class AcousticCNN:
    def __init__(self, model_path=A5_CNN_MODEL):
        import torch
        if not os.path.exists(model_path):
            raise FileNotFoundError("train first: python -m ml.a5_acoustic.train")
        ckpt = torch.load(model_path, map_location="cpu")
        self.classes = ckpt["classes"]
        self.net = build_cnn(len(self.classes))
        self.net.load_state_dict(ckpt["state_dict"]); self.net.eval()
        self.torch = torch

    def predict(self, path_or_audio) -> dict:
        m = wav_to_mfcc(path_or_audio)
        x = self.torch.tensor(m).unsqueeze(0).unsqueeze(0)
        with self.torch.no_grad():
            p = self.torch.softmax(self.net(x), 1)[0]
        i = int(p.argmax())
        return {"label": self.classes[i], "confidence": round(float(p[i]) * 100, 1)}


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python -m ml.a5_acoustic.infer <file.wav>")
    print(AcousticCNN().predict(sys.argv[1]))


if __name__ == "__main__":
    main()
