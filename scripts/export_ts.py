"""Export a trained CAI-Vision checkpoint to TorchScript (FP32 + INT8-head).

Produces two artefacts inside a timestamped `outputs/ts_<ts>/` folder:
  - model_lite0_fp32.ts         - TorchScript FP32 (recommended baseline)
  - model_lite0_int8_head.ts    - TorchScript with dynamically-quantised
                                  Linear layers (smaller, CPU-friendly,
                                  convs remain FP32)

Usage:
    python scripts/export_torchscript_int8.py
    python scripts/export_torchscript_int8.py --ckpt path/to/model_final_fp32.pt
    python scripts/export_torchscript_int8.py --out-dir /tmp/exports

Notes for full static INT8 (optional, not produced by this script):
  1) torch.ao.quantization with prepare/convert + a calibration DataLoader.
  2) Not all timm models have built-in fusion patterns; EfficientNet often
     needs custom fuse rules.
  3) If you require full-conv INT8 (TFLite-style), consider:
       - Quantization Aware Training (QAT) in torch.ao.quantization
       - ExecuTorch (for mobile) or ONNX -> onnxruntime.quantization
     Both routes need a small calibration set representative of your
     training preprocessing ([0, 1] scale).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from cai.constants import IMG_SIZE, LABELS_FILE, MODEL_FINAL_FP32, TORCH_RUNS_DIR
from cai.labels import load_labels
from cai.model import build_model


def _resolve_output_dir(out_dir: Path) -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    target = out_dir / f"outputs/ts_{ts}"
    target.mkdir(parents=True, exist_ok=True)
    return target


def export_torchscript(checkpoint: Path, exp_dir: Path) -> tuple[Path, Path]:
    """Load `checkpoint`, then write FP32 + INT8-head TorchScript modules.

    Returns the (fp32_path, int8_head_path) pair.
    """
    fp32_path = exp_dir / "model_lite0_fp32.ts"
    int8_path = exp_dir / "model_lite0_int8_head.ts"

    ckpt = torch.load(checkpoint, map_location="cpu")
    num_classes = ckpt["num_classes"]

    # Validate labels match the checkpoint head size before exporting.
    labels = load_labels(LABELS_FILE)
    if len(labels) != num_classes:
        raise ValueError(
            f"labels.txt has {len(labels)} entries but the checkpoint expects "
            f"{num_classes} classes. Regenerate labels (python scripts/write_labels.py) "
            f"or retrain the model."
        )

    model = build_model(num_classes, pretrained=False)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    example = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)

    # FP32 baseline (recommended).
    ts_fp32 = torch.jit.trace(model, example)
    ts_fp32.save(str(fp32_path))
    print(f"Saved TorchScript FP32 → {fp32_path}")

    # Dynamic INT8 (Linear-only). Good for size/CPU; minimal accuracy drop;
    # does not quantise convolutions.
    dq_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    ts_dq = torch.jit.trace(dq_model, example)
    ts_dq.save(str(int8_path))
    print(f"Saved TorchScript dynamic INT8(head) → {int8_path}")

    return fp32_path, int8_path


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Export a CAI-Vision checkpoint to TorchScript (FP32 + INT8-head)."
    )
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
        help="Directory under which a timestamped outputs/ts_<ts>/ folder is created.",
    )
    args = ap.parse_args()

    if not args.ckpt.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.ckpt}. "
            f"Train a model first (python scripts/train_torch_lite0.py) "
            f"or pass --ckpt to point at an existing one."
        )

    exp_dir = _resolve_output_dir(args.out_dir)
    export_torchscript(args.ckpt, exp_dir)


if __name__ == "__main__":
    main()
