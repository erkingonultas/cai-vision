"""Shared constants used across training, evaluation, export, and inference.

Centralising these here removes the silent-drift risk of having the same
constant defined (and possibly edited) in multiple scripts.
"""
from __future__ import annotations

from pathlib import Path

# --- Model architecture ---
MODEL_NAME: str = "efficientnet_lite0"
IMG_SIZE: int = 224
IMG_NORM_MEAN = [0.5, 0.5, 0.5]
IMG_NORM_STD = [0.5, 0.5, 0.5]

# --- Filesystem layout ---
# Project root is two levels up from this file (cai/constants.py -> root).
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

SCRIPTS_DIR: Path = PROJECT_ROOT / "scripts"
DATASET_ROOT: Path = PROJECT_ROOT / "datasets" / "cai-vision-dataset"

# Files inside scripts/ that are referenced by name in the pipeline.
LABELS_FILE: Path = SCRIPTS_DIR / "labels.txt"

# Torch training output directory (per-run subdirs are created by scripts).
TORCH_RUNS_DIR: Path = PROJECT_ROOT / "torch_runs"
CKPT_BEST: Path = TORCH_RUNS_DIR / "ckpt_best.pt"
MODEL_FINAL_FP32: Path = TORCH_RUNS_DIR / "model_final_fp32.pt"

# Dataset manifests (written by data-prep/dataset_schema_tool.py, read by train).
TRAIN_CSV: Path = DATASET_ROOT / "train.csv"
VAL_CSV: Path = DATASET_ROOT / "val.csv"
TEST_CSV: Path = DATASET_ROOT / "test.csv"

# Optional dataset prep subcommand (used by dataset_schema_tool.py).
CATEGORY_TXT: Path = DATASET_ROOT / "category.txt"
CLASSES_CSV: Path = DATASET_ROOT / "classes.csv"
MULTILABEL_OVERRIDES_CSV: Path = DATASET_ROOT / "multilabel_overrides.csv"
DATASET_IMG_DIR: Path = DATASET_ROOT / "images"
DATASET_NEW_DATA_DIR: Path = DATASET_ROOT / "new-data"
