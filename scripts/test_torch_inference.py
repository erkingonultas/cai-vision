"""Single-image Top-5 prediction using a TorchScript-exported model.

Mirrors the original test_tflite_inference.py: opens an image, runs it
through the model, and prints the top-5 predicted classes with their
softmax probabilities.

Usage:
    python scripts/test_torch_inference.py path/to/image.jpg
    python scripts/test_torch_inference.py path/to/image.jpg --model path/to/model.ts
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from cai.constants import IMG_SIZE, LABELS_FILE
from cai.data import eval_transforms
from cai.device import get_device
from cai.labels import load_labels


def _build_transform() -> transforms.Compose:
    """Preprocessing that matches the training pipeline (resize + [0,1]).

    The training CSVDataset converts to RGB before the transform compose,
    so we replicate that here for single-image inference.
    """
    return transforms.Compose(
        [
            transforms.Lambda(lambda img: img.convert("RGB")),
            eval_transforms(),
        ]
    )


def _load_scripted_model(model_path: Path, device: torch.device) -> torch.jit.ScriptModule:
    if not model_path.exists():
        raise FileNotFoundError(
            f"TorchScript model not found: {model_path}. "
            f"Generate it with `python scripts/export_torchscript.py`."
        )
    model = torch.jit.load(str(model_path), map_location=device)
    model.eval()
    return model


def _preprocess(image_path: Path, tfms: transforms.Compose, device: torch.device) -> torch.Tensor:
    img = Image.open(image_path)
    return tfms(img).unsqueeze(0).to(device)


def predict(
    image_path: Path,
    model: torch.jit.ScriptModule,
    labels: list[str],
    device: torch.device,
    tfms: transforms.Compose,
    top_k: int = 5,
) -> list[tuple[int, str, float]]:
    """Return the top-k (class_index, label, probability) tuples for an image."""
    x = _preprocess(image_path, tfms, device)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    topk_idx = np.argsort(probs)[-top_k:][::-1]
    return [(int(i), labels[i], float(probs[i])) for i in topk_idx]


def main() -> None:
    ap = argparse.ArgumentParser(description="Single-image Top-k prediction (TorchScript).")
    ap.add_argument("image", type=Path, help="Path to the input image.")
    ap.add_argument(
        "--model",
        type=Path,
        default=Path("./torch_runs/outputs/ts_latest/model_lite0_fp32.ts"),
        help="Path to a TorchScript .ts model file.",
    )
    ap.add_argument("--top-k", type=int, default=5, help="How many top predictions to print.")
    ap.add_argument(
        "--labels",
        type=Path,
        default=LABELS_FILE,
        help="Path to a labels.txt file (one label per line).",
    )
    args = ap.parse_args()

    device = get_device()
    labels = load_labels(args.labels)
    model = _load_scripted_model(args.model, device)
    tfms = _build_transform()

    top_k = predict(args.image, model, labels, device, tfms, top_k=args.top_k)
    print(f"Top-{args.top_k} for {args.image}:")
    for rank, (idx, label, prob) in enumerate(top_k, 1):
        print(f"  {rank}. {label:<40} {prob:.4f}  (idx={idx})")


if __name__ == "__main__":
    main()