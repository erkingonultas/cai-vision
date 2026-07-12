import time
import torch, timm
from pathlib import Path

IMG_SIZE = 224
CKPT = Path("torch_runs/model_final_fp32.pt")
ts = time.strftime("%Y%m%d_%H%M%S")
OUT_DIR = Path(f"./torch_runs")
EXP_DIR = OUT_DIR / f"outputs/onnx_{ts}"
EXP_DIR.mkdir(parents=True, exist_ok=True)
ONNX_FP32 = EXP_DIR / "efficientnet_lite0.onnx"
ONNX_CAIV = EXP_DIR / "cai_vision.onnx"

# Load best state
ck = torch.load(CKPT, map_location="cpu")
num_classes = ck["num_classes"]
model = timm.create_model("efficientnet_lite0", pretrained=False, num_classes=num_classes)
model.load_state_dict(ck["model"])
model.eval()

# Export FP32 ONNX
dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
onnx_program = torch.onnx.export(
    model, dummy, str(ONNX_FP32),
    input_names=["input"], output_names=["logits"],
    dynamo=True,
    opset_version=18
)
onnx_program.save(str(ONNX_CAIV))
print(f"Saved {ONNX_FP32}")
print(f"Saved {ONNX_CAIV}")