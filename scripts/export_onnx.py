"""Export a trained CAI-Vision checkpoint to ONNX (FP32).

Loads a final FP32 PyTorch checkpoint, then writes two ONNX artefacts to
a timestamped output directory:
  - efficientnet_lite0.onnx  (the raw torch.export artefact)
  - cai_vision.onnx          (the .save()'d version, ready to ship)

Usage:
    python scripts/export_onnx.py
    python scripts/export_onnx.py --ckpt path/to/model_final_fp32.pt
    python scripts/export_onnx.py --out-dir /tmp/exports
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from cai.constants import IMG_SIZE, MODEL_FINAL_FP32, TORCH_RUNS_DIR
from cai.model import build_model


def _resolve_output_dir(out_dir: Path) -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    target = out_dir / f"outputs/onnx_{ts}"
    target.mkdir(parents=True, exist_ok=True)
    return target


def export_onnx(checkpoint: Path, exp_dir: Path) -> tuple[Path, Path]:
    """Load `checkpoint` and write ONNX artefacts into `exp_dir`.

    Returns the (onnx_fp32_path, cai_vision_path) pair.
    """
    onnx_fp32 = exp_dir / "efficientnet_lite0.onnx"
    onnx_caiv = exp_dir / "cai_vision.onnx"

    ck = torch.load(checkpoint, map_location="cpu")
    num_classes = ck["num_classes"]
    model = build_model(num_classes, pretrained=False)
    model.load_state_dict(ck["model"])
    model.eval()

    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    onnx_program = torch.onnx.export(
        model,
        dummy,
        str(onnx_fp32),
        input_names=["input"],
        output_names=["logits"],
        dynamo=True,
        opset_version=18,
    )
    onnx_program.save(str(onnx_caiv))
    return onnx_fp32, onnx_caiv


def main() -> None:
    ap = argparse.ArgumentParser(description="Export a CAI-Vision checkpoint to ONNX.")
    ap.add_argument(
        "--ckpt",
        type=Path,
        default=MODEL_FINAL_FP32,
        help="Path to a final FP32 PyTorch checkpoint (default: scripts/torch_runs/model_final_fp32.pt).",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=TORCH_RUNS_DIR,
        help="Directory under which a timestamped outputs/onnx_<ts>/ folder is created.",
    )
    args = ap.parse_args()

    if not args.ckpt.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.ckpt}. "
            f"Train a model first (python scripts/train_torch_lite0.py) "
            f"or pass --ckpt to point at an existing one."
        )

    exp_dir = _resolve_output_dir(args.out_dir)
    fp32_path, caiv_path = export_onnx(args.ckpt, exp_dir)
    print(f"Saved {fp32_path}")
    print(f"Saved {caiv_path}")


if __name__ == "__main__":
    main()