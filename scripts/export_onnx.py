import torch, timm
from pathlib import Path

IMG_SIZE = 224
CKPT = Path("torch_runs/ckpt_best.pt")
ONNX_FP32 = Path("torch_runs/outputs/efficientnet_lite0.onnx")
ONNX_INT8 = Path("torch_runs/outputs/efficientnet_lite0.int8.onnx")  # optional

# Load best state
ck = torch.load(CKPT, map_location="cpu")
num_classes = ck["num_classes"]
model = timm.create_model("efficientnet_lite0", pretrained=False, num_classes=num_classes)
model.load_state_dict(ck["model"])
model.eval()

# Export FP32 ONNX
dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
torch.onnx.export(
    model, dummy, str(ONNX_FP32),
    input_names=["input"], output_names=["logits"],
    dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    opset_version=17
)
print(f"Saved {ONNX_FP32}")

# (Optional) Dynamic INT8 quantization for smaller/faster CPU model
try:
    from onnxruntime.quantization import quantize_dynamic, QuantType
    quantize_dynamic(str(ONNX_FP32), str(ONNX_INT8), weight_type=QuantType.QUInt8)
    print(f"Saved {ONNX_INT8}")
except Exception as e:
    print(f"Quantization skipped: {e}")