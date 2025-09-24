"""
 USAGE:
    # Run (CPU)
    python validate_onnx.py path/to/your_model.onnx --shape 1,3,224,224 --dtype float32

    # If you want to test other providers on desktop (optional, if available)
    python validate_onnx.py path/to/your_model.onnx --providers "CPUExecutionProvider"
    # macOS (if CoreML EP installed): --providers "CoreMLExecutionProvider,CPUExecutionProvider"
    # Windows (if DirectML EP installed): --providers "DmlExecutionProvider,CPUExecutionProvider"
"""

import argparse, json, numpy as np, onnx, onnxruntime as ort

def main():
    p = argparse.ArgumentParser()
    p.add_argument("model", help="Path to .onnx model")
    p.add_argument("--input-name", default="input", help="Model input name (default: input)")
    p.add_argument("--shape", default="1,3,224,224",
                   help="Input shape as comma-separated ints (default: 1,3,224,224)")
    p.add_argument("--dtype", default="float32", choices=["float32","float16","uint8"],
                   help="Input dtype (default: float32)")
    p.add_argument("--providers", default="CPUExecutionProvider",
                   help="Comma-separated ORT providers (e.g. 'CPUExecutionProvider' or "
                        "'CoreMLExecutionProvider,CPUExecutionProvider')")
    args = p.parse_args()

    # --- 1) ONNX structural check ---
    m = onnx.load(args.model)
    onnx.checker.check_model(m)
    opset = [imp.version for imp in m.opset_import if imp.domain in ("", "ai.onnx")][0]
    print(f"[OK] Model loaded & checked. Opset: {opset}")

    # Print IO info (helpful for matching names/shapes)
    def io_info(v):
        # Some models don't have static shapes everywhere; handle gently.
        shape = []
        try:
            dims = v.type.tensor_type.shape.dim
            for d in dims:
                shape.append(d.dim_value if d.HasField("dim_value") else f"?{d.dim_param or ''}")
        except Exception:
            shape = ["?"]
        elem = v.type.tensor_type.elem_type
        return {"name": v.name, "elem_type": int(elem), "shape": shape}

    info = {
        "inputs": [io_info(v) for v in m.graph.input],
        "outputs": [io_info(v) for v in m.graph.output],
    }
    print("[Info] IO summary:", json.dumps(info, indent=2))

    # --- 2) ORT session & dummy inference ---
    providers = [p.strip() for p in args.providers.split(",") if p.strip()]
    sess = ort.InferenceSession(args.model, providers=providers)
    print(f"[OK] ORT session created with providers: {sess.get_providers()}")

    # Prepare dummy input
    shape = tuple(int(x) for x in args.shape.split(","))
    if args.dtype == "float32":
        x = (np.random.rand(*shape).astype(np.float32) - 0.5) * 2  # [-1,1]
    elif args.dtype == "float16":
        x = (np.random.rand(*shape).astype(np.float16) - np.float16(0.5)) * np.float16(2)
    else:  # uint8
        x = np.random.randint(0, 256, size=shape, dtype=np.uint8)

    # Inference
    ort_inputs = {args.input_name if args.input_name else "input": x}
    # Fallback if user passed wrong name: map first input name from session
    if list(ort_inputs.keys())[0] not in [i.name for i in sess.get_inputs()]:
        first_name = sess.get_inputs()[0].name
        ort_inputs = {first_name: x}
        print(f"[Warn] '{args.input_name}' not found. Using first input name: '{first_name}'")

    outs = sess.run(None, ort_inputs)

    # Report stats
    for i, y in enumerate(outs):
        isnan = np.isnan(y).any()
        isinf = np.isinf(y).any()
        print(f"[Output {i}] shape={y.shape} dtype={y.dtype} "
              f"min={np.min(y):.6f} max={np.max(y):.6f} mean={np.mean(y):.6f} "
              f"nan={isnan} inf={isinf}")

    print("[DONE] Validation succeeded.")

if __name__ == "__main__":
    main()
