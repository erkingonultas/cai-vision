# Mirrors test_tflite_inference.py: single-image Top-5 prediction.

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

import timm

from pathlib import Path

IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load labels
labels = []
with open("labels.txt", encoding="utf-8") as f:
    labels = [l.strip() for l in f]

# Load scripted model (choose one)
MODEL_PATH = Path("./torch_runs/outputs/ts_20250924_142401/model_lite0_fp32.ts")  # or model_lite0_int8_head.ts
model = torch.jit.load(str(MODEL_PATH), map_location=DEVICE)
model.eval()

tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.ToTensor(),      # -> [0,1] (MUST match training/export)
])

def preprocess(img_path):
    x = tfms(Image.open(img_path))
    return x.unsqueeze(0).to(DEVICE)

def predict(img_path):
    x = preprocess(img_path)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        topk = np.argsort(probs)[-5:][::-1]
        return [(int(i), labels[i], float(probs[i])) for i in topk]

# Example:
print(predict("../datasets/lahmacun-yemekcom-1.jpg"))