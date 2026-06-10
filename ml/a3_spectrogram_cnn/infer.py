"""A3 inference — classify an IQ capture (or spectrogram PNG) with the CNN."""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from ml.common.config import A3_CNN_MODEL
from ml.common.iq import load_iq
from ml.a3_spectrogram_cnn.make_spectrograms import iq_to_spectrogram_img


class SpectrogramCNN:
    def __init__(self, model_path=A3_CNN_MODEL):
        import torch
        import torch.nn as nn
        from torchvision import models, transforms
        if not os.path.exists(model_path):
            raise FileNotFoundError("train first: python -m ml.a3_spectrogram_cnn.train")
        ckpt = torch.load(model_path, map_location="cpu")
        self.classes = ckpt["classes"]
        net = models.mobilenet_v2()
        net.classifier[1] = nn.Linear(net.last_channel, len(self.classes))
        net.load_state_dict(ckpt["state_dict"]); net.eval()
        self.net, self.torch = net, torch
        self.tf = transforms.Compose([
            transforms.Resize((64, 64)), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def predict(self, path_or_iq) -> dict:
        from PIL import Image
        if isinstance(path_or_iq, str) and path_or_iq.endswith(".png"):
            img = Image.open(path_or_iq).convert("RGB")
        elif isinstance(path_or_iq, str):
            img = iq_to_spectrogram_img(load_iq(path_or_iq))
        else:
            img = iq_to_spectrogram_img(path_or_iq)
        x = self.tf(img).unsqueeze(0)
        with self.torch.no_grad():
            p = self.torch.softmax(self.net(x), 1)[0]
        i = int(p.argmax())
        return {"label": self.classes[i], "confidence": round(float(p[i]) * 100, 1)}


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python -m ml.a3_spectrogram_cnn.infer <file.bin|png>")
    print(SpectrogramCNN().predict(sys.argv[1]))


if __name__ == "__main__":
    main()
