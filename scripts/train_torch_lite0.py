"""Train EfficientNet-Lite0 (via timm) on the CAI-Vision dataset.

Reads train/val/test CSVs (filepath,class_id,...), builds the
``efficientnet_lite0`` backbone with ImageNet pretraining, trains with
Adam + cosine-annealing-warm-restarts + label smoothing + early stopping,
evaluates on the test split, and writes:

* ``scripts/torch_runs/ckpt_best.pt`` — best (by val top-1) weights,
  overwritten each time a new best is found.
* ``scripts/torch_runs/model_final_fp32.pt`` — final weights (best
  restored) at end of training, in the format consumed by the export
  scripts.

Both checkpoints are dicts of the form
``{"model": state_dict, "num_classes": N, ...}``.

Usage:
    # Defaults (matches original hardcoded values):
    python scripts/train_torch_lite0.py

    # Hyperparameter sweep:
    python scripts/train_torch_lite0.py --epochs 20 --lr 5e-4 --batch-size 32

    # Point at a different dataset root:
    python scripts/train_torch_lite0.py --data-root /path/to/dataset
"""
from __future__ import annotations

import argparse
import copy
import csv
import random
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

from cai.constants import (
    CKPT_BEST,
    DATASET_ROOT,
    MODEL_FINAL_FP32,
    TEST_CSV,
    TRAIN_CSV,
    TORCH_RUNS_DIR,
    VAL_CSV,
)
from cai.data import CSVDataset, make_loader, normalize_class_ids
from cai.device import get_device
from cai.metrics import topk_correct
from cai.model import build_model

# An "item" is a (filepath, class_id) pair. Class ids are 0-based.
Item = Tuple[str, int]


# -----------------------------
# Progress printer
# -----------------------------
class ProgressPrinter:
    """Throttled, carriage-return progress line for stdout."""

    def __init__(self, interval: float = 0.5) -> None:
        self.interval = interval
        self._last = 0.0

    def update(
        self,
        epoch: int,
        total_epochs: int,
        i: int,
        total_batches: int,
        phase: str,
    ) -> None:
        now = time.time()
        if (now - self._last) >= self.interval or i == total_batches:
            pct = 100.0 * i / max(1, total_batches)
            sys.stdout.write(f"\rEpoch {epoch}/{total_epochs} [{phase}] {pct:5.1f}%")
            sys.stdout.flush()
            self._last = now

    def newline(self) -> None:
        sys.stdout.write("\n")
        sys.stdout.flush()


# -----------------------------
# Data loading
# -----------------------------
def _load_csv_items(csv_path: Path) -> List[Item]:
    """Read a CSV and return (filepath, class_id) items normalised to 0-based.

    The CSV must have at least the columns ``filepath`` and ``class_id``.
    Class ids may be 0-based or 1-based; they are auto-detected by
    ``normalize_class_ids``.

    Raises:
        FileNotFoundError: If ``csv_path`` does not exist.
        ValueError: If the class ids are not contiguous from 0 after
            normalisation.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV not found: {csv_path}. "
            f"Run `python scripts/data-prep/dataset_schema_tool.py` to generate it."
        )

    items: List[Item] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            items.append((row["filepath"].strip(), int(row["class_id"])))
    return normalize_class_ids(items)


def _seed_everything(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _build_loaders(
    train_items: List[Item],
    val_items: List[Item],
    test_items: List[Item],
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Construct train/val/test DataLoaders with the right per-phase behaviour."""
    return (
        make_loader(
            train_items,
            training=True,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        ),
        make_loader(
            val_items,
            training=False,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        ),
        make_loader(
            test_items,
            training=False,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        ),
    )


# -----------------------------
# Train / eval helpers
# -----------------------------
def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingWarmRestarts,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    pp: ProgressPrinter,
) -> Tuple[float, float, float]:
    """Run one training epoch and return (loss, top1, top5) averaged over samples."""
    model.train()
    n_batches = len(loader)
    total_loss, top1_correct, top5_correct, n = 0.0, 0.0, 0.0, 0
    for i, (xb, yb) in enumerate(loader, 1):
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        # Per-iteration step works with CosineAnnealingWarmRestarts.
        scheduler.step(epoch + n / len(loader))

        bs = xb.size(0)
        total_loss += loss.item() * bs
        top1_correct += topk_correct(logits, yb, k=1)
        top5_correct += topk_correct(logits, yb, k=5)
        n += bs
        pp.update(epoch, total_epochs, i, n_batches, "train")
    pp.newline()
    return (
        total_loss / max(1, n),
        top1_correct / max(1, n),
        top5_correct / max(1, n),
    )


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    pp: ProgressPrinter,
) -> Tuple[float, float, float]:
    """Run one evaluation pass and return (loss, top1, top5) averaged over samples."""
    model.eval()
    n_batches = len(loader)
    total_loss, top1_correct, top5_correct, n = 0.0, 0.0, 0.0, 0
    for j, (xb, yb) in enumerate(loader, 1):
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        bs = xb.size(0)
        total_loss += loss.item() * bs
        top1_correct += topk_correct(logits, yb, k=1)
        top5_correct += topk_correct(logits, yb, k=5)
        n += bs
        pp.update(epoch, total_epochs, j, n_batches, "val")
    pp.newline()
    return (
        total_loss / max(1, n),
        top1_correct / max(1, n),
        top5_correct / max(1, n),
    )


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Train EfficientNet-Lite0 on the CAI-Vision dataset."
    )
    ap.add_argument("--data-root", type=Path, default=DATASET_ROOT,
                    help="Directory containing train.csv/val.csv/test.csv.")
    ap.add_argument("--train-csv", type=Path, default=TRAIN_CSV,
                    help="Path to the train CSV.")
    ap.add_argument("--val-csv",   type=Path, default=VAL_CSV,
                    help="Path to the val CSV.")
    ap.add_argument("--test-csv",  type=Path, default=TEST_CSV,
                    help="Path to the test CSV.")
    ap.add_argument("--out-dir",   type=Path, default=TORCH_RUNS_DIR,
                    help="Output directory; ckpt_best.pt and model_final_fp32.pt go here.")
    ap.add_argument("--epochs",        type=int,   default=16)
    ap.add_argument("--batch-size",    type=int,   default=64)
    ap.add_argument("--lr",            type=float, default=1e-3)
    ap.add_argument("--label-smooth",  type=float, default=0.1)
    ap.add_argument("--patience",      type=int,   default=4,
                    help="Early-stopping patience in epochs (on val top-1).")
    ap.add_argument("--num-workers",   type=int,   default=8)
    ap.add_argument("--seed",          type=int,   default=42)
    args = ap.parse_args()

    # Resolve the canonical output paths under the (possibly overridden) out-dir.
    args.out_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = args.out_dir / CKPT_BEST.name
    final_ckpt = args.out_dir / MODEL_FINAL_FP32.name

    # --- Repro ---
    device = get_device()
    _seed_everything(args.seed, device)

    # --- Data ---
    print("Loading CSVs...")
    train_items = _load_csv_items(args.train_csv)
    val_items   = _load_csv_items(args.val_csv)
    test_items  = _load_csv_items(args.test_csv)

    # _load_csv_items already normalises each list to contiguous 0-based ids;
    # the original asserted contiguity across the union, so we keep the same
    # invariant here.
    all_ids = {cid for _, cid in (train_items + val_items + test_items)}
    if min(all_ids) != 0 or all_ids != set(range(len(all_ids))):
        raise ValueError(
            f"Non-contiguous class ids across train+val+test. "
            f"Got {sorted(all_ids)[:5]}... (min={min(all_ids)}, "
            f"max={max(all_ids)}, count={len(all_ids)})."
        )
    num_classes = max(all_ids) + 1
    print(
        f"Loaded {len(train_items)} train, {len(val_items)} val, "
        f"{len(test_items)} test items. num_classes={num_classes}"
    )

    pin_memory = (device.type == "cuda")
    train_loader, val_loader, test_loader = _build_loaders(
        train_items, val_items, test_items,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    print("Make Loader Completed")

    # --- Model ---
    # Keep preprocessing in the data loader; the model expects float32 [0,1].
    model = build_model(num_classes, pretrained=True).to(device)
    print("Model Loaded")

    # --- Loss / Optim / Sched ---
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smooth)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # Warm restarts roughly like TF CosineDecayRestarts.
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=max(1, len(train_loader) * 5), T_mult=2, eta_min=1e-5
    )
    print("Loss / Optim / Sched / Metrics Loaded")

    # --- Train / Val loop with early stopping on val top-1 ---
    best_top1 = -1.0
    epochs_no_improve = 0
    best_state: dict | None = None
    print(f"Starting Epochs with {device.type.title()}...")

    pp_train = ProgressPrinter(interval=0.5)
    pp_val = ProgressPrinter(interval=0.5)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_top1, tr_top5 = _train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            device, epoch, args.epochs, pp_train,
        )
        va_loss, va_top1, va_top5 = _evaluate(
            model, val_loader, criterion, device, epoch, args.epochs, pp_val,
        )

        print(
            f"Epoch {epoch:02d} | "
            f"train loss {tr_loss:.4f} top1 {tr_top1:.4f} top5 {tr_top5:.4f} | "
            f"val loss {va_loss:.4f} top1 {va_top1:.4f} top5 {va_top5:.4f}"
        )

        if va_top1 > best_top1:
            best_top1 = va_top1
            best_state = copy.deepcopy(model.state_dict())
            torch.save(
                {"model": best_state, "num_classes": num_classes, "epoch": epoch},
                best_ckpt,
            )
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(
                    f"Early stopping at epoch {epoch}. "
                    f"Best val_top1={best_top1:.4f}"
                )
                break

    # --- Restore best & run test eval ---
    print("Loading the best...")
    if best_state is not None:
        model.load_state_dict(best_state)

    print("Starting Model Eval...")
    # Reuse the per-epoch eval helper with a silent printer, since for the
    # final test pass we only need the aggregate numbers, not per-batch lines.
    class _SilentPrinter:
        def update(self, *a, **kw): pass
        def newline(self): pass
    te_loss, te_top1, te_top5 = _evaluate(
        model, test_loader, criterion, device,
        epoch=args.epochs, total_epochs=args.epochs, pp=_SilentPrinter(),
    )
    print({"TEST_loss": te_loss, "TEST_top1": te_top1, "TEST_top5": te_top5})

    # --- Save plain FP32 final for export ---
    torch.save({"model": model.state_dict(), "num_classes": num_classes}, final_ckpt)
    print(f"Saved: {final_ckpt}")


if __name__ == "__main__":
    # Optional: helps when packaging as an .exe; harmless otherwise.
    from multiprocessing import freeze_support
    freeze_support()
    main()