# Exports TorchScript FP32 and a HEAD-dynamically-quantized version.

import csv, random, time
from pathlib import Path
import torch
import timm
from torchvision import transforms
from PIL import Image

IMG_SIZE = 224
SEED = 42
DEVICE = "cpu"   # export/calibration typically on CPU
ts = time.strftime("%Y%m%d_%H%M%S")
OUT_DIR = Path(f"./torch_runs")
CKPT = OUT_DIR / "model_final_fp32.pt"
EXP_DIR = OUT_DIR / f"outputs/ts_{ts}"
EXP_DIR.mkdir(parents=True, exist_ok=True)
TS_FP32 = EXP_DIR / "model_lite0_fp32.ts"
TS_DQ   = EXP_DIR / "model_lite0_int8_head.ts"

# Load best ckpt
ckpt = torch.load(CKPT, map_location="cpu")

# --- Validate labels vs checkpoint head size ---
labels_path = Path("./labels.txt")
if not labels_path.exists():
    raise FileNotFoundError(f"labels.txt not found at {labels_path}. Make sure it matches the training classes.")

with open(labels_path, encoding="utf-8") as f:
    labels = [l.strip() for l in f if l.strip()]

num_classes = ckpt["num_classes"]

model = timm.create_model("efficientnet_lite0", pretrained=False, num_classes=num_classes)
model.load_state_dict(ckpt["model"], strict=True)
model.eval()

# Example input
example = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)

# TorchScript FP32 (recommended baseline)
ts_fp32 = torch.jit.trace(model, example)
ts_fp32.save(str(TS_FP32))
print(f"Saved TorchScript FP32 → {TS_FP32}")

# Lightweight INT8 (dynamic quantization) — affects Linear layers only
# Good for size/CPU; minimal accuracy drop; does not quantize convs.
dq_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
ts_dq = torch.jit.trace(dq_model, example)
ts_dq.save(str(TS_DQ))
print(f"Saved TorchScript dynamic INT8(head) → {TS_DQ}")

# -------- Notes for full static INT8 (optional) --------
# 1) Use torch.ao.quantization with prepare/convert and a calibration DataLoader
# 2) Not all timm models have built-in fusion patterns; EfficientNet often needs custom fuse rules.
# 3) If you require full-conv INT8 like TFLite, consider:
#    - Quantization Aware Training (QAT) in torch.ao.quantization
#    - ExecuTorch (for mobile) or exporting ONNX → onnxruntime.quantization
#    Both routes need a small calibration set representative of your training preprocessing ([0,1] scale).
