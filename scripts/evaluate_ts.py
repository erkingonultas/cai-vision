"""Evaluate a TorchScript-exported model on a labelled CSV.

The CSV is expected to have a header row plus rows of
``filepath,class_id,sha1`` (extra trailing columns are ignored). Class ids
may be 0-based or 1-based; they are auto-detected and normalised to
0-based before evaluation (see ``cai.data.normalize_class_ids``).

For each run the script writes, into a timestamped subdirectory of
``--out``:

* ``summary.json`` — top-1/top-3/top-5 accuracy, device, model path, etc.
* ``per_class.csv`` — per-class recall, precision, F1, and top confusion.
* ``confusion_matrix.csv`` — full ``NUM_CLASSES x NUM_CLASSES`` matrix.
* ``predictions.csv`` — per-sample predictions and predicted probabilities.

Usage:
    # Default paths, auto-detect class_id base:
    python scripts/eval_from_csv.py --csv /path/to/test.csv

    # Pin to a specific TorchScript model, increase throughput:
    python scripts/eval_from_csv.py --csv /path/to/test.csv \\
        --model-path scripts/torch_runs/outputs/ts_latest/model_lite0_fp32.ts \\
        --batch-size 64 --workers 4

"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cai.constants import LABELS_FILE, TORCH_RUNS_DIR
from cai.data import CSVDataset, eval_transforms, normalize_class_ids
from cai.device import get_device
from cai.labels import load_labels
from cai.metrics import topk_correct


# An "item" is a (filepath, class_id) pair. Class ids will be normalised
# to 0-based by normalize_class_ids().
Item = Tuple[str, int]


def _read_csv_items(csv_path: Path) -> List[Item]:
    """Parse a (filepath, class_id, ...) CSV into a list of items.

    Raises:
        FileNotFoundError: If ``csv_path`` does not exist.
        ValueError: If the class ids are not contiguous from 0 after
            normalisation (raised by ``normalize_class_ids``).
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV not found: {csv_path}. "
            f"Pass --csv /path/to/test.csv."
        )

    items: List[Item] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            if not row:
                continue
            path, cid_raw, *_ = row
            items.append((path.strip(), int(cid_raw.strip())))

    return normalize_class_ids(items)


def _load_scripted_model(model_path: Path, device: torch.device) -> torch.jit.ScriptModule:
    if not model_path.exists():
        raise FileNotFoundError(
            f"TorchScript model not found: {model_path}. "
            f"Generate it with `python scripts/export_torchscript.py` "
            f"and pass the resulting .ts path via --model."
        )
    model = torch.jit.load(str(model_path), map_location=device)
    model.eval()
    return model


@torch.no_grad()
def evaluate(
    csv_path: Path,
    model_path: Path,
    out_dir: Path,
    labels: List[str],
    *,
    batch_size: int = 32,
    num_workers: int = 2,
) -> dict:
    """Run the evaluation and write all result files. Returns a summary dict."""
    device = get_device()
    num_classes = len(labels)

    model = _load_scripted_model(model_path, get_device())

    items = _read_csv_items(csv_path)
    dataset = CSVDataset(items, transform=eval_transforms(), return_path=True)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    print("----- Setup complete. -----")
    print("----- Evaluation started... -----")
    top1_correct = 0
    top3_correct = 0
    top5_correct = 0
    total = 0
    conf_mat = np.zeros((num_classes, num_classes), dtype=np.int32)
    pred_rows: List[list] = []

    for xb, yb, paths in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        logits = model(xb)
        probs = F.softmax(logits, dim=1)

        top1_correct += topk_correct(logits, yb, k=1)
        top3_correct += topk_correct(logits, yb, k=min(3, num_classes))
        top5_correct += topk_correct(logits, yb, k=min(5, num_classes))

        pred1 = probs.argmax(dim=1)
        t_cpu = yb.cpu().numpy()
        p_cpu = pred1.cpu().numpy()
        for t, p in zip(t_cpu, p_cpu):
            conf_mat[t, p] += 1

        total += yb.size(0)

        probs_cpu = probs.cpu().numpy()
        for i, (path, t, p) in enumerate(zip(paths, t_cpu, p_cpu)):
            pred_probability = float(probs_cpu[i, p])
            pred_rows.append(
                [
                    path,
                    labels[t], int(t),
                    labels[p], int(p),
                    f"{pred_probability:.6f}",
                ]
            )

    top1 = 100.0 * top1_correct / max(1, total)
    top3 = 100.0 * top3_correct / max(1, total)
    top5 = 100.0 * top5_correct / max(1, total)

    print("----- Validating results... -----")
    confusion_total = int(conf_mat.sum())
    confusion_correct = int(np.trace(conf_mat))
    support_total = int(conf_mat.sum(axis=1).sum())

    assert conf_mat.shape == (num_classes, num_classes)
    assert confusion_total == total, (
        f"Confusion matrix total ({confusion_total}) != evaluated samples ({total})"
    )
    assert confusion_correct == top1_correct, (
        f"Confusion matrix diagonal ({confusion_correct}) != top-1 correct ({top1_correct})"
    )
    assert support_total == total, (
        f"Per-class support total ({support_total}) != evaluated samples ({total})"
    )

    evaluation_integrity = True

    # --- Persist outputs ---
    print("----- Preparing output... -----")
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = out_dir / f"eval_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Label-indexing note: auto-detected by normalize_class_ids.
    min_id = min(cid for _, cid in items)
    indexing = "CSV 0-based" if min_id == 0 else "CSV 1-based → normalised to 0-based"

    print("----- Writing summary... -----")
    summary = {
        "csv": str(csv_path),
        "images_evaluated": total,
        "label_indexing": indexing,
        "top1_acc": round(top1, 4),
        "top3_acc": round(top3, 4),
        "top5_acc": round(top5, 4),
        "num_classes": num_classes,
        "evaluation_integrity": evaluation_integrity,
        "model_path": str(model_path),
        "device": str(device),
        "timestamp": ts,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    with open(run_dir / "per_class.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "label",
                "class_index",
                "support",
                "correct",
                "recall_percent",
                "predicted_count",
                "precision_percent",
                "f1_percent",
                "top_confusion_label",
                "top_confusion_count",
            ]
        )

        for i, label in enumerate(labels):
            support = int(conf_mat[i, :].sum())
            correct = int(conf_mat[i, i])
            predicted_count = int(conf_mat[:, i].sum())

            recall = 100.0 * correct / support if support > 0 else 0.0
            precision = (
                100.0 * correct / predicted_count
                if predicted_count > 0
                else 0.0
            )

            if precision + recall > 0.0:
                f1 = 2.0 * precision * recall / (precision + recall)
            else:
                f1 = 0.0

            # Find the most common incorrect prediction for this true class.
            # Exclude the diagonal because it represents correct predictions.
            confusion_row = conf_mat[i, :].copy()
            confusion_row[i] = 0
            top_confusion_count = int(confusion_row.max())

            if top_confusion_count > 0:
                top_confusion_idx = int(confusion_row.argmax())
                top_confusion_label = labels[top_confusion_idx]
            else:
                top_confusion_label = ""

            w.writerow(
                [
                    label,
                    i,
                    support,
                    correct,
                    f"{recall:.4f}",
                    predicted_count,
                    f"{precision:.4f}",
                    f"{f1:.4f}",
                    top_confusion_label,
                    top_confusion_count,
                ]
            )


    with open(run_dir / "confusion_matrix.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["true\\pred"] + labels)
        for i, label in enumerate(labels):
            w.writerow([label] + list(map(int, conf_mat[i, :])))

    with open(run_dir / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "filepath",
                "true_label",
                "true_idx",
                "pred_label",
                "pred_idx",
                "pred_probability",
            ]
        )
        w.writerows(pred_rows)

    print("----- Evaluation Summary -----")
    print(f"CSV: {csv_path}")
    print(f"Images evaluated: {total}")
    print(f"Label indexing: {indexing}")
    print(f"Top-1 Accuracy: {top1:.2f}%")
    print(f"Top-3 Accuracy: {top3:.2f}%")
    print(f"Top-5 Accuracy: {top5:.2f}%")
    print(f"\nResults written to: {run_dir.resolve()}")

    return {"summary": summary, "out_dir": str(run_dir)}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate a TorchScript model on a labelled CSV "
        "(filepath,class_id,sha1).",
    )
    ap.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Path to test CSV (header + rows of filepath,class_id,sha1).",
    )
    ap.add_argument(
        "--model-path",
        type=Path,
        default=TORCH_RUNS_DIR / "model_lite0_fp32.ts",
        help="Path to a TorchScript .ts model file. "
        "Typically scripts/torch_runs/outputs/ts_<timestamp>/model_lite0_fp32.ts.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=TORCH_RUNS_DIR,
        help="Output directory; a timestamped eval_<ts> subfolder is created inside.",
    )
    ap.add_argument("--labels", type=Path, default=LABELS_FILE, help="Path to labels.txt.")
    ap.add_argument("--batch-size", type=int, default=32, help="Eval batch size.")
    ap.add_argument("--workers", type=int, default=2, help="DataLoader num_workers.")
    args = ap.parse_args()

    labels = load_labels(args.labels)
    
    evaluate(
        csv_path=args.csv,
        model_path=args.model_path,
        out_dir=args.out,
        labels=labels,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )


if __name__ == "__main__":
    main()