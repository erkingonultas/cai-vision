# Minimal, efficient evaluation on a CSV: "filepath,class_id,sha1"


'''
    # Default (auto-detect class_id base)
    python eval_from_csv.py --csv /path/to/test.csv --out runs/eval --save-preds

    # If you *know* your CSV class_ids are 1..N (common in some pipelines):
    python eval_from_csv.py --csv /path/to/test.csv

    # Increase throughput on fast disks/GPUs:
    python eval_from_csv.py --csv /path/to/test.csv --batch-size 64 --workers 4

    Example: python eval_from_csv.py --csv ../datasets/cai-vision-dataset/test.csv --batch-size 64 --workers 8
'''


import csv, json, time
from pathlib import Path
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load labels
with open("labels.txt", encoding="utf-8") as f:
    LABELS = [l.strip() for l in f]
NUM_CLASSES = len(LABELS)

# Load scripted model (choose one)
MODEL_PATH = Path("./torch_runs/model_lite0_int8_head.ts")  # or model_lite0_int8_head.ts
model = torch.jit.load(str(MODEL_PATH), map_location=DEVICE)
model.eval()

tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.ToTensor(),  # must match training/export
])

# ----------------------------
# --- Evaluation Dataset -----
# ----------------------------
class CSVDataset(Dataset):
    def __init__(self, csv_path: str):
        self.items = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            r = csv.reader(f)
            header = next(r, None)  # skip header
            for row in r:
                if not row:
                    continue
                p, cid, *_ = row
                p = p.strip()
                cid = int(cid.strip()) - 1   # 1-based → 0-based
                self.items.append((p, cid))

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        p, target = self.items[idx]
        img = tfms(Image.open(p))
        return img, target, p  # keep path for optional per-sample logging


# ----------------------------
# --- Evaluation Function ----
# ----------------------------
@torch.no_grad()
def evaluate(csv_path: str, out_dir: str, batch_size: int = 32, workers: int = 2, save_preds: bool = False):
    ds = CSVDataset(csv_path)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    num_workers=workers, pin_memory=True)

    print("----- Setup complete. -----")
    print("----- Evaluation started... -----")
    top1_correct = 0
    top5_correct = 0
    total = 0
    conf_mat = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int32)
    pred_rows = []  # optional

    for xb, yb, paths in dl:
        xb = xb.to(DEVICE, non_blocking=True)
        yb = yb.to(DEVICE, non_blocking=True)

        logits = model(xb)
        probs = F.softmax(logits, dim=1)

        pred1 = probs.argmax(dim=1)
        top1_correct += (pred1 == yb).sum().item()

        # confusion
        t_cpu = yb.cpu().numpy()
        p_cpu = pred1.cpu().numpy()
        for t, p in zip(t_cpu, p_cpu):
            conf_mat[t, p] += 1

        # top-5
        top5 = torch.topk(probs, k=min(5, NUM_CLASSES), dim=1).indices
        top5_correct += (top5 == yb.unsqueeze(1)).any(dim=1).sum().item()

        total += yb.size(0)

        if save_preds:
            probs_cpu = probs.cpu().numpy()
            for i, (path, t, p) in enumerate(zip(paths, t_cpu, p_cpu)):
                prob_p = float(probs_cpu[i, p])
                pred_rows.append([
                    path,
                    LABELS[t], int(t),
                    LABELS[p], int(p),
                    f"{prob_p:.6f}"
                ])

    # summary numbers
    top1 = 100.0 * top1_correct / max(1, total)
    top5 = 100.0 * top5_correct / max(1, total)

    # ensure output dir
    print("----- Preparing output... -----")
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = out_dir / f"eval_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # write summary.json
    print("----- Writing summary... -----")
    summary = {
        "csv": str(csv_path),
        "images_evaluated": total,
        "label_indexing": "CSV 1-based → normalized to 0-based",
        "top1_acc": round(top1, 4),
        "top5_acc": round(top5, 4),
        "num_classes": NUM_CLASSES,
        "model_path": str(MODEL_PATH),
        "device": DEVICE,
        "timestamp": ts
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # write per_class.csv
    with open(run_dir / "per_class.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label", "class_index", "support", "correct", "accuracy_percent"])
        for i, label in enumerate(LABELS):
            support = int(conf_mat[i, :].sum())
            correct = int(conf_mat[i, i])
            acc = 100.0 * correct / support if support > 0 else 0.0
            w.writerow([label, i, support, correct, f"{acc:.4f}"])

    # write confusion_matrix.csv (header row/col = labels)
    with open(run_dir / "confusion_matrix.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["true\\pred"] + LABELS)
        for i, label in enumerate(LABELS):
            w.writerow([label] + list(map(int, conf_mat[i, :])))

    # optional predictions.csv
    if save_preds:
        with open(run_dir / "predictions.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["filepath", "true_label", "true_idx", "pred_label", "pred_idx", "pred_confidence"])
            w.writerows(pred_rows)

    # console echo (unchanged)
    print("----- Evaluation Summary -----")
    print(f"CSV: {csv_path}")
    print(f"Images evaluated: {total}")
    print("Label indexing assumed: 1-based in CSV → normalized to 0-based")
    print(f"Top-1 Accuracy: {top1:.2f}%")
    print(f"Top-5 Accuracy: {top5:.2f}%")
    print(f"\nResults written to: {run_dir.resolve()}")

    return {"summary": summary, "out_dir": str(run_dir)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to test CSV (filepath,class_id,sha1)") # ../datasets/cai-vision-dataset/test.csv
    ap.add_argument("--out", default="torch_runs", help="Output directory (will create a timestamped subfolder)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--save-preds", action="store_true", help="Also write per-sample predictions.csv")
    args = ap.parse_args()
    evaluate(args.csv, args.out, args.batch_size, args.workers, args.save_preds)

if __name__ == "__main__":
    main()