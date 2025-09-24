# next actions
# Run train_keras_lite0.py to fine-tune Lite0.
# Run export_tflite_int8.py → model_lite0_int8.tflite + labels.txt.
# Sanity-check with test_tflite_inference.py.
# Ship to Flutter using tflite_flutter, matching preprocessing and enabling NNAPI/Core ML/XNNPACK.

import sys, csv, time, copy, random
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import timm

# -----------------------------
# Config
# -----------------------------
IMG_SIZE = 224
BATCH = 64
EPOCHS = 16
LR = 1e-3
LABEL_SMOOTH = 0.1
SEED = 42
NUM_WORKERS = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_ROOT = Path("../datasets/cai-vision-dataset")
TRAIN_CSV, VAL_CSV, TEST_CSV = DATA_ROOT / "train.csv", DATA_ROOT / "val.csv", DATA_ROOT / "test.csv"
OUT_DIR = Path("./torch_runs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
BEST_CKPT = OUT_DIR / "ckpt_best.pt"

class ProgressPrinter:
    def __init__(self, interval=0.5):
        self.interval = interval
        self._last = 0.0

    def update(self, epoch, total_epochs, i, total_batches, phase):
        now = time.time()
        if (now - self._last) >= self.interval or i == total_batches:
            pct = 100.0 * i / max(1, total_batches)
            sys.stdout.write(f"\rEpoch {epoch}/{total_epochs} [{phase}] {pct:5.1f}%")
            sys.stdout.flush()
            self._last = now

    def newline(self):
        sys.stdout.write("\n")
        sys.stdout.flush()

# -----------------------------
# Repro
# -----------------------------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if DEVICE == "cuda":
    torch.cuda.manual_seed_all(SEED)

# -----------------------------
# Data utils
# -----------------------------
def load_csv(path: Path) -> List[Tuple[str, int]]:
    with open(path, newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        rows = [(r["filepath"], int(r["class_id"])) for r in rdr]
    # Detect 1-based scheme and normalize to 0-based
    min_id = min(cid for _, cid in rows)
    shift = 1 if min_id == 1 else 0
    rows_0 = [(fp, cid - shift) for fp, cid in rows]
    return rows_0

train_items = load_csv(TRAIN_CSV)
val_items   = load_csv(VAL_CSV)
test_items  = load_csv(TEST_CSV)

all_ids = [cid for _, cid in (train_items + val_items + test_items)]
NUM_CLASSES = max(all_ids) + 1

# 0-based, contiguous ids?
assert min(all_ids) == 0, f"Labels must start at 0, got {min(all_ids)}"
assert set(all_ids) == set(range(NUM_CLASSES)), \
       f"Non-contiguous class ids. Seen {len(set(all_ids))} classes, expected 0..{NUM_CLASSES-1}"
# -----------------------------
# Transforms (keep [0,1])
# -----------------------------
from torchvision import transforms

train_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    # transforms.Lambda(lambda img: img.convert("RGB")), # It is inside the Dataset class
    transforms.ToTensor(),                         # -> [0,1]
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
])

eval_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    # transforms.Lambda(lambda img: img.convert("RGB")), # It is inside the Dataset class
    transforms.ToTensor(),                         # -> [0,1]
])

class CSVDataset(Dataset):
    def __init__(self, items, training=False):
        self.items = items
        self.training = training
        self.tfms = train_tfms if training else eval_tfms

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        img = Image.open(path).convert("RGB")  # <- convert here (picklable)
        x = self.tfms(img)
        return x, label

def make_loader(items, training=False):
    return DataLoader(
        CSVDataset(items, training=training),
        batch_size=BATCH,
        shuffle=training,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE=="cuda"),
        drop_last=training
    )
def main():
    train_loader = make_loader(train_items, training=True)
    val_loader   = make_loader(val_items, training=False)
    test_loader  = make_loader(test_items, training=False)
    print("Make Loader Completed")
    # -----------------------------
    # Model: EfficientNet-Lite0 (timm)
    # -----------------------------
    # Keep preprocessing in data loader; model expects float32 [0,1].
    model = timm.create_model("efficientnet_lite0", pretrained=True, num_classes=NUM_CLASSES)
    model = model.to(DEVICE)
    print("Model Loaded")
    # -----------------------------
    # Loss / Optim / Sched / Metrics
    # -----------------------------
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    # Use warm restarts roughly like your TF CosineDecayRestarts
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=max(1, len(train_loader)*5), T_mult=2, eta_min=1e-5)

    def topk_correct(logits, target, k=1):
        with torch.no_grad():
            _, pred = logits.topk(k, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1))
            return correct[:k].reshape(-1).float().sum().item()
    print("Loss / Optim / Sched / Metrics Loaded")
    # -----------------------------
    # Train / Eval loops with EarlyStopping on val_top1
    # -----------------------------
    PATIENCE = 4
    best_top1 = -1.0
    epochs_no_improve = 0
    best_state = None
    print(f"Starting Epochs with {DEVICE.title()}...")
    
    pp_train = ProgressPrinter(interval=0.5)
    pp_val   = ProgressPrinter(interval=0.5)

    for epoch in range(1, EPOCHS+1):
        # ===== TRAIN =====
        model.train()
        tr_loss, tr_top1, tr_top5, n_train = 0.0, 0.0, 0.0, 0
        n_batches = len(train_loader)
        for i, (xb, yb) in enumerate(train_loader, 1):
            xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            scheduler.step(epoch + n_train/len(train_loader))  # per-iteration step works with WarmRestarts

            bs = xb.size(0)
            tr_loss += loss.item() * bs
            tr_top1 += topk_correct(logits, yb, k=1)
            tr_top5 += topk_correct(logits, yb, k=5)
            n_train += bs
            pp_train.update(epoch, EPOCHS, i, n_batches, "train")
        pp_train.newline()

        # ===== VAL =====
        model.eval()
        n_val_batches = len(val_loader)
        va_loss, va_top1, va_top5, n_val = 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for j, (xb, yb) in enumerate(val_loader, 1):
                xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
                logits = model(xb)
                loss = criterion(logits, yb)
                bs = xb.size(0)
                va_loss += loss.item() * bs
                va_top1 += topk_correct(logits, yb, k=1)
                va_top5 += topk_correct(logits, yb, k=5)
                n_val += bs
                pp_val.update(epoch, EPOCHS, j, n_val_batches, "val")
        pp_val.newline()

        tr_loss /= max(1, n_train)
        tr_top1 /= max(1, n_train)
        tr_top5 /= max(1, n_train)
        va_loss /= max(1, n_val)
        va_top1 /= max(1, n_val)
        va_top5 /= max(1, n_val)

        print(f"Epoch {epoch:02d} | "
            f"train loss {tr_loss:.4f} top1 {tr_top1:.4f} top5 {tr_top5:.4f} | "
            f"val loss {va_loss:.4f} top1 {va_top1:.4f} top5 {va_top5:.4f}")

        # Early stopping on val_top1 (restore best like TF)
        if va_top1 > best_top1:
            best_top1 = va_top1
            best_state = copy.deepcopy(model.state_dict())
            torch.save({"model": best_state, "num_classes": NUM_CLASSES, "epoch": epoch}, BEST_CKPT)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch}. Best val_top1={best_top1:.4f}")
                break

    # Load best
    print("Loading the best...")
    if best_state is not None:
        model.load_state_dict(best_state)
    # -----------------------------
    # Quick test eval (mirrors TF "model.evaluate")
    # -----------------------------
    print("Starting Model Eval...")
    model.eval()
    te_loss, te_top1, te_top5, n_test = 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            bs = xb.size(0)
            te_loss += loss.item() * bs
            te_top1 += topk_correct(logits, yb, k=1)
            te_top5 += topk_correct(logits, yb, k=5)
            n_test += bs

    te_loss /= max(1, n_test)
    te_top1 /= max(1, n_test)
    te_top5 /= max(1, n_test)
    print({"TEST_loss": te_loss, "TEST_top1": te_top1, "TEST_top5": te_top5})

    # Save a plain FP32 final for export step
    FINAL_PATH = OUT_DIR / "model_final_fp32.pt"
    torch.save({"model": model.state_dict(), "num_classes": NUM_CLASSES}, FINAL_PATH)
    print(f"Saved: {FINAL_PATH}")

if __name__ == "__main__":
    # Optional: helps when packaging as an .exe; harmless otherwise
    from multiprocessing import freeze_support
    freeze_support()
    main()