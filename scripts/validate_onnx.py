"""Structural + smoke-test validation for an exported ONNX model.

Performs two checks:
  1. onnx.checker: confirms the file is a well-formed ONNX model and
     prints its opset version and IO tensor metadata.
  2. onnxruntime:  loads the model with the requested execution
     providers, runs a dummy tensor through it, and prints per-output
     statistics (shape, dtype, min/max/mean, NaN/Inf flags).

Usage examples:
    # CPU only (default)
    python scripts/validate_onnx.py path/to/model.onnx

    # Custom shape / dtype
    python scripts/validate_onnx.py path/to/model.onnx --shape 1,3,224,224 --dtype float32

    # Multi-provider fallback chain (e.g. macOS CoreML, Windows DirectML)
    python scripts/validate_onnx.py path/to/model.onnx \\
        --providers "CoreMLExecutionProvider,CPUExecutionProvider"
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import onnx
import onnxruntime as ort


def _io_info(value_info) -> dict:
    """Return a JSON-serialisable summary of an ONNX IO tensor's metadata."""
    shape: list[str | int] = []
    try:
        for d in value_info.type.tensor_type.shape.dim:
            shape.append(d.dim_value if d.HasField("dim_value") else f"?{d.dim_param or ''}")
    except Exception:
        shape = ["?"]
    return {
        "name": value_info.name,
        "elem_type": int(value_info.type.tensor_type.elem_type),
        "shape": shape,
    }


def _make_dummy(shape: tuple[int, ...], dtype: str) -> np.ndarray:
    """Build a deterministic-ish random tensor of the requested shape/dtype."""
    if dtype == "float32":
        return (np.random.rand(*shape).astype(np.float32) - 0.5) * 2  # [-1, 1]
    if dtype == "float16":
        return (np.random.rand(*shape).astype(np.float16) - np.float16(0.5)) * np.float16(2)
    return np.random.randint(0, 256, size=shape, dtype=np.uint8)


def _parse_shape(s: str) -> tuple[int, ...]:
    return tuple(int(x) for x in s.split(","))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Structural + smoke-test validation for an exported ONNX model."
    )
    ap.add_argument("model", help="Path to .onnx model")
    ap.add_argument("--input-name", default="input", help="Model input name (default: input)")
    ap.add_argument(
        "--shape",
        default="1,3,224,224",
        help="Input shape as comma-separated ints (default: 1,3,224,224)",
    )
    ap.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "float16", "uint8"],
        help="Input dtype (default: float32)",
    )
    ap.add_argument(
        "--providers",
        default="CPUExecutionProvider",
        help="Comma-separated ORT providers (e.g. 'CPUExecutionProvider' or "
        "'CoreMLExecutionProvider,CPUExecutionProvider')",
    )
    args = ap.parse_args()

    # --- 1) ONNX structural check ---
    model = onnx.load(args.model)
    onnx.checker.check_model(model)
    opset = [imp.version for imp in model.opset_import if imp.domain in ("", "ai.onnx")][0]
    print(f"[OK] Model loaded & checked. Opset: {opset}")

    info = {
        "inputs": [_io_info(v) for v in model.graph.input],
        "outputs": [_io_info(v) for v in model.graph.output],
    }
    print("[Info] IO summary:", json.dumps(info, indent=2))

    # --- 2) ORT session & dummy inference ---
    providers = [prov.strip() for prov in args.providers.split(",") if prov.strip()]
    session = ort.InferenceSession(args.model, providers=providers)
    print(f"[OK] ORT session created with providers: {session.get_providers()}")

    shape = _parse_shape(args.shape)
    x = _make_dummy(shape, args.dtype)

    # Resolve the input name: prefer user override, else fall back to the
    # session's first declared input.
    declared_input_names = [i.name for i in session.get_inputs()]
    if args.input_name in declared_input_names:
        input_name = args.input_name
    else:
        input_name = declared_input_names[0]
        if args.input_name != "input":  # don't warn on default
            print(f"[Warn] '{args.input_name}' not found. Using first input name: '{input_name}'")

    outs = session.run(None, {input_name: x})

    for i, y in enumerate(outs):
        isnan = bool(np.isnan(y).any())
        isinf = bool(np.isinf(y).any())
        print(
            f"[Output {i}] shape={y.shape} dtype={y.dtype} "
            f"min={np.min(y):.6f} max={np.max(y):.6f} mean={np.mean(y):.6f} "
            f"nan={isnan} inf={isinf}"
        )

    print("[DONE] Validation succeeded.")


if __name__ == "__main__":
    main()
