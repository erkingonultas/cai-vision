# ============================================================================
# CAI-Vision Makefile
# ----------------------------------------------------------------------------
# A single entry point for the whole pipeline: env setup, dataset prep,
# training, export (ONNX + TorchScript), validation, evaluation, and
# single-image inference.
#
# Quick start (Windows CMD / PowerShell / Git Bash / WSL):
#   make help
#   make install
#   make check-torch
#   make dataset-init
#   make dataset-validate
#   make dataset-build-manifests
#   make labels
#   make train
#   make export-ts
#   make eval
#   make predict IMAGE=path/to/img.jpg
#
# All targets are phony; you can override the most common knobs:
#   make train CONFIG=configs/example_config.json
#   make eval CSV=datasets/cai-vision-dataset/val.csv
#   make export-onnx CKPT=path/to/model_final_fp32.pt
#
# NOTE: scripts/ must be a Python package (i.e. scripts/__init__.py must
#       exist) for the "python -m scripts.<module>" invocations to resolve.
# ============================================================================

# --- Tooling -----------------------------------------------------------------
PY            ?= python
VENV          ?= .venv

# Use Python to resolve virtualenv executables so this works on Windows,
# POSIX shells, Git Bash, and WSL.
PIP           ?= $(PY) -m pip
PY_IN_VENV    ?= $(PY)

# --- Project layout ----------------------------------------------------------
SCRIPTS       := scripts
DATASET_ROOT  := datasets/cai-vision-dataset
TORCH_RUNS    := torch_runs
LABELS_FILE   := $(SCRIPTS)/labels.txt
CONFIGS       := configs
CKPT_BEST     := $(TORCH_RUNS)/ckpt_best.pt
CKPT_FINAL    := $(TORCH_RUNS)/model_final_fp32.pt

# --- Training knobs (overridable: make train CONFIG=configs/example_config.json)
CONFIG        ?= $(CONFIGS)/example_config.json

# --- Evaluation knobs --------------------------------------------------------
CSV           ?= $(DATASET_ROOT)/test.csv
TS_MODEL      ?= $(TORCH_RUNS)/model_lite0_fp32.ts
EVAL_BATCH    ?= 64
EVAL_WORKERS  ?= 6

# --- Inference knobs ---------------------------------------------------------
IMAGE         ?=
TOPK          ?= 5

# --- Dataset knobs -----------------------------------------------------------
VAL_RATIO     ?= 0.10
TEST_RATIO    ?= 0.15
SEED          ?= 42

# ============================================================================
# Default goal
# ============================================================================
.DEFAULT_GOAL := help

.PHONY: help
help:                          ## Show this help message
	@echo "CAI-Vision - available targets:"
	@echo "A single entry point for the whole pipeline: env setup, dataset prep, training, export (ONNX + TorchScript), validation evaluation, and single-image inference."
	@echo "Common overrides:"
	@echo "  make train CONFIG=configs/example_config.json"
	@echo "  make eval CSV=$(DATASET_ROOT)/val.csv TS_MODEL=$(TS_MODEL)"
	@echo "  make predict IMAGE=path/to/image.jpg"

# ============================================================================
# Environment
# ============================================================================
.PHONY: install
install:                       ## Install full CUDA 13.0 + project dependencies
	$(PIP) install -r requirements.txt

.PHONY: install-stable
install-stable:                ## Install CPU-only deps (inference / eval / export)
	$(PIP) install -r requirements-stable.txt

.PHONY: install-cuda
install-cuda:                  ## Install only the nightly torch/torchvision (CUDA 13.0)
	$(PIP) install -r requirements-cuda130.txt

.PHONY: venv
venv:                          ## Create the local virtualenv
	$(PY) -m venv $(VENV)
	$(PIP) install --upgrade pip

.PHONY: check-torch
check-torch:                   ## Verify CUDA wiring before a long training run
	$(PY) -m $(SCRIPTS).torch_check

# ============================================================================
# Dataset preparation (scripts/manage_dataset.py)
# ============================================================================
.PHONY: dataset-init
dataset-init:                  ## Generate classes.csv from category.txt
	$(PY) -m $(SCRIPTS).manage_dataset init-from-category

.PHONY: dataset-validate
dataset-validate:              ## Sanity-check schema vs images/; detect cross-label dupes
	$(PY) -m $(SCRIPTS).manage_dataset validate

.PHONY: dataset-add-class
dataset-add-class:             ## Add a new class (usage: make dataset-add-class NAME="...")
	@$(PY) -c "import sys; sys.exit(0 if '$(NAME)' else 1)" || (echo ERROR: set NAME="..." && exit 1)
	$(PY) -m $(SCRIPTS).manage_dataset add-class "$(NAME)"

.PHONY: dataset-ingest
dataset-ingest:                ## Move new-data/<class>/* into images/<class_id>/ (auto-create missing)
	$(PY) -m $(SCRIPTS).manage_dataset ingest-new-data --create-missing

.PHONY: dataset-build-manifests
dataset-build-manifests:       ## Build train/val/test CSVs (overrides: VAL_RATIO TEST_RATIO SEED)
	$(PY) -m $(SCRIPTS).manage_dataset build-manifests --val $(VAL_RATIO) --test $(TEST_RATIO) --seed $(SEED) --dedup-in-class

.PHONY: labels
labels:                        ## Regenerate scripts/labels.txt from classes.csv
	$(PY) -m $(SCRIPTS).export_labels

# ============================================================================
# Training
# ============================================================================
.PHONY: train
train:                         ## Train EfficientNet-Lite0 (overrides: EPOCHS BATCH LR CONFIG)
	@$(PY) -c "import sys; sys.exit(0 if '$(CONFIG)' else 1)" || (echo ERROR: CONFIG is not set && exit 1)
	@$(PY) -c "from pathlib import Path; import sys; sys.exit(0 if Path(r'$(CONFIG)').is_file() else 1)" || (echo ERROR: config not found: $(CONFIG) && exit 1)
	@echo Training with CONFIG=$(CONFIG)
	$(PY) -m $(SCRIPTS).train_torch_lite0 --config $(CONFIG)

.PHONY: train-default
train-default:                 ## Train with hardcoded defaults (no JSON config)
	$(PY) -m $(SCRIPTS).train_torch_lite0

# ============================================================================
# Export — TorchScript
# ============================================================================
.PHONY: export-ts
export-ts:                     ## Export checkpoint -> FP32 + INT8-head TorchScript
	@$(PY) -c "from pathlib import Path; import sys; sys.exit(0 if Path(r'$(CKPT_FINAL)').is_file() else 1)" || (echo ERROR: $(CKPT_FINAL) not found. Run "make train" first. && exit 1)
	$(PY) -m $(SCRIPTS).export_ts --ckpt $(CKPT_FINAL)

# ============================================================================
# Export — ONNX
# ============================================================================
.PHONY: export-onnx
export-onnx:                   ## Export checkpoint -> ONNX (FP32 + cai_vision.onnx)
	@$(PY) -c "from pathlib import Path; import sys; sys.exit(0 if Path(r'$(CKPT_FINAL)').is_file() else 1)" || (echo ERROR: $(CKPT_FINAL) not found. Run "make train" first. && exit 1)
	$(PY) -m $(SCRIPTS).export_onnx --ckpt $(CKPT_FINAL)

.PHONY: validate-onnx
validate-onnx:                 ## Structural + smoke-test validate an ONNX model (override ONNX=path)
	@$(PY) -c "import sys; sys.exit(0 if '$(ONNX)' else 1)" || (echo ERROR: set ONNX=path/to/model.onnx && exit 1)
	@$(PY) -c "from pathlib import Path; import sys; sys.exit(0 if Path(r'$(ONNX)').is_file() else 1)" || (echo ERROR: file not found: $(ONNX) && exit 1)
	$(PY) -m $(SCRIPTS).validate_onnx $(ONNX)

# ============================================================================
# Evaluation (TorchScript model on a labelled CSV)
# ============================================================================
.PHONY: eval
eval:                          ## Evaluate TS model on CSV (overrides: CSV TS_MODEL EVAL_BATCH)
	@$(PY) -c "from pathlib import Path; import sys; sys.exit(0 if Path(r'$(CSV)').is_file() else 1)" || (echo ERROR: CSV not found: $(CSV). Run "make dataset-build-manifests". && exit 1)
	@$(PY) -c "from pathlib import Path; import sys; sys.exit(0 if Path(r'$(TS_MODEL)').is_file() else 1)" || (echo ERROR: TS model not found: $(TS_MODEL). Run "make export-ts" or specify model path with "TS_MODEL". && exit 1)
	$(PY) -m $(SCRIPTS).evaluate_ts --csv $(CSV) --model-path $(TS_MODEL) --batch-size $(EVAL_BATCH) --workers $(EVAL_WORKERS)

.PHONY: eval-val
eval-val:                      ## Convenience: evaluate on val.csv
	$(MAKE) eval CSV=$(DATASET_ROOT)/val.csv

.PHONY: eval-test
eval-test:                     ## Convenience: evaluate on test.csv
	$(MAKE) eval CSV=$(DATASET_ROOT)/test.csv

# ============================================================================
# Single-image inference
# ============================================================================
.PHONY: predict
predict:                       ## Top-K prediction on a single image (IMAGE=path TOPK=5)
	@$(PY) -c "import sys; sys.exit(0 if '$(IMAGE)' else 1)" || (echo ERROR: set IMAGE=path/to/image.jpg && exit 1)
	@$(PY) -c "from pathlib import Path; import sys; sys.exit(0 if Path(r'$(IMAGE)').is_file() else 1)" || (echo ERROR: file not found: $(IMAGE) && exit 1)
	@$(PY) -c "from pathlib import Path; import sys; sys.exit(0 if Path(r'$(TS_MODEL)').is_file() else 1)" || (echo ERROR: TS model not found: $(TS_MODEL). Run "make export-ts". && exit 1)
	$(PY) -m $(SCRIPTS).test_torch_inference $(IMAGE) --model $(TS_MODEL) --top-k $(TOPK)

# ============================================================================
# Full pipeline
# ============================================================================
.PHONY: all
all: install check-torch dataset-init dataset-validate labels dataset-build-manifests train export-ts ## Install -> dataset -> train -> export-TS

.PHONY: pipeline-eval
pipeline-eval: train export-ts eval-test ## Train -> export TS -> evaluate on test split

# ============================================================================
# Cleanup
# ============================================================================
.PHONY: clean
clean:                         ## Remove training outputs (checkpoints, exports, eval reports)
	$(PY) -c "import shutil; shutil.rmtree(r'$(TORCH_RUNS)', ignore_errors=True)"
	$(PY) -c "from pathlib import Path; [Path(p).unlink(missing_ok=True) for p in [r'$(DATASET_ROOT)/train.csv', r'$(DATASET_ROOT)/val.csv', r'$(DATASET_ROOT)/test.csv']]"

.PHONY: clean-runs
clean-runs:                    ## Remove only torch_runs/ (keep dataset CSVs)
	$(PY) -c "import shutil; shutil.rmtree(r'$(TORCH_RUNS)', ignore_errors=True)"

.PHONY: clean-all
clean-all: clean               ## Alias for clean (dataset CSVs are derived; safe to remove)