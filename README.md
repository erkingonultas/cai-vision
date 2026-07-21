# 🍽️ CAI-VISION

> **A compact, mobile-first food image classifier.**
> Fine-tuned **EfficientNet-Lite0** backbone, exported as **TorchScript (FP32 + INT8-head)** and **ONNX**, ready to ship to phones.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg)](#-requirements)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](#-requirements)
[![Platform](https://img.shields.io/badge/Target-Android%20%2F%20iOS-3DDC84.svg)](#-export-for-mobile)

CAI-Vision is a small-footprint vision model aimed at identifying foods from around the world, with a special focus on a few Turkish dishes. It ships with a complete, reproducible pipeline: **dataset preparation → training → quantization-aware export → evaluation → mobile-ready ONNX validation**.

*Released under the [MIT License](LICENSE) — Copyright © 2025 Erkin Gönültaş.*

---

## ✨ Highlights

- 🧠 **EfficientNet-Lite0** backbone — tiny, fast, mobile-friendly.
- ⚙️ **JSON-driven training config** (`configs/example_config.json`) with full override support.
- 🧪 **Quantization-aware TorchScript export** (FP32 + INT8-head) for size/throughput wins.
- 📦 **ONNX export** suitable for Android (TFLite/ORT) and iOS (Core ML via onnx-coreml).
- 🧹 **Deduped, schema-validated dataset** with train/val/test manifest CSVs.
- 🛠️ **Two ways to drive the pipeline:** a friendly **Makefile** *or* raw **`python -m scripts.<module>`** calls.

---

## 📁 Project Layout

```
.
├── Makefile                  ← Single entry point for the whole pipeline
├── configs/                  ← Training configs (JSON)
│   └── example_config.json
├── scripts/                  ← All runnable modules (invoked as `python -m scripts.<name>`)
│   ├── manage_dataset.py     ← Dataset schema, ingestion, manifests
│   ├── export_labels.py      ← Regenerate scripts/labels.txt
│   ├── torch_check.py        ← Verify CUDA wiring
│   ├── train_torch_lite0.py  ← Training entry point
│   ├── export_ts.py          ← TorchScript export (FP32 + INT8-head)
│   ├── evaluate_ts.py        ← Evaluate a TS model on a labelled CSV
│   ├── test_torch_inference.py ← Single-image Top-K prediction
│   ├── export_onnx.py        ← ONNX export
│   └── validate_onnx.py      ← Structural + smoke-test for ONNX
├── cai/                      ← Reusable library code (model, data, metrics, …)
├── datasets/cai-vision-dataset/
│   ├── category.txt          ← `<id> <name>` lines, source of truth
│   ├── classes.csv           ← Generated schema (id, slug, display_name_en, …)
│   ├── images/<class_id>/    ← Renumbered, deduped training images
│   ├── new-data/<class>/     ← Drop new images here before ingest
│   └── train.csv / val.csv / test.csv  ← Built by `build-manifests`
├── torch_runs/               ← All training / export / eval outputs land here
└── requirements*.txt         ← CUDA 13.0 / stable / full stacks
```

---

## 📦 Requirements

- **Python** ≥ 3.10
- One of:
  - `requirements.txt` — full CUDA 13.0 + project deps
  - `requirements-stable.txt` — CPU-only, fine for inference / eval / export
  - `requirements-cuda130.txt` — nightly torch/torchvision (CUDA 13.0) only
- A CUDA-capable GPU is **strongly recommended** for training, but **not required** for export, evaluation, or ONNX validation.

> Tip: the Makefile wires `python -m pip` so it works identically on Windows CMD, PowerShell, Git Bash, and WSL.

---

## 🚀 How to Use

You can drive the entire pipeline in **two equivalent ways** — pick whichever fits your shell habits.

- **A) Makefile** — recommended. Single commands, sensible defaults, all overridable.
- **B) Modules** — run the underlying scripts directly as `python -m scripts.<name>` (great for debugging, IDEs, and CI).

> The rest of this section is organized as **a recipe** — follow it top-to-bottom on a fresh clone.

### 0. Install & sanity-check

**Makefile**

```bash
make install           # full CUDA 13.0 + project deps
# or:
make install-stable    # CPU-only
make install-cuda      # torch CUDA 13.0 only

make check-torch       # verify CUDA is wired correctly before training
```

**Modules**

```bash
python -m pip install -r requirements.txt        # or -stable / -cuda130
python -m scripts.torch_check
```

---

### 1. Prepare the dataset

The dataset lives under `./datasets/cai-vision-dataset/`. The pipeline assumes a `<id> <name>`-style `category.txt` as the source of truth and builds the rest from it.

**A. Drop new images into `new-data/<class_name>/`**

Place your incoming images like this — class folders are matched **case/space/underscore-insensitively** against `classes.csv`:

```
datasets/cai-vision-dataset/new-data/<class_name>/*.jpg
```

> If a `<class_name>` doesn't exist yet, you can either create it manually (`add-class`) or let the ingestion step auto-create it (`--create-missing`).

**B. Sync schema, ingest images, and build manifests**

**Makefile**

```bash
make dataset-init               # generate classes.csv from category.txt
make dataset-validate           # schema check + cross-label duplicate detection
make dataset-ingest             # move new-data/<class>/* -> images/<class_id>/
# (add NAME="New Dish" to create a new class manually)
make dataset-build-manifests    # build train/val/test CSVs (10/15% split by default)
make labels                     # regenerate scripts/labels.txt from classes.csv
```

**Modules**

```bash
python -m scripts.manage_dataset init-from-category
python -m scripts.manage_dataset validate
python -m scripts.manage_dataset ingest-new-data --create-missing
# or, manually:
# python -m scripts.manage_dataset add-class "New Dish Name"
python -m scripts.manage_dataset build-manifests --val 0.10 --test 0.15 --seed 42 --dedup-in-class
python -m scripts.export_labels
```

Useful overrides for `build-manifests`:

```bash
make dataset-build-manifests VAL_RATIO=0.10 TEST_RATIO=0.15 SEED=42
```

📖 See the dedicated [`scripts/manage_dataset.README.md`](scripts/manage_dataset.README.md) for the full subcommand reference, schema details, and edge cases.

---

### 2. Train

Training reads the JSON config, fetches the **EfficientNet-Lite0** backbone, fine-tunes it on `train.csv`, and validates on `val.csv`. Checkpoints land in `torch_runs/`.

**Makefile**

```bash
make train                  # uses configs/example_config.json
# or with a custom config:
make train CONFIG=configs/epoch_fraction_exp_config.json
```

**Modules**

```bash
python -m scripts.train_torch_lite0 --config configs/example_config.json
# or use hardcoded defaults (no JSON):
python -m scripts.train_torch_lite0
```

> Outputs you'll see in `torch_runs/`: `ckpt_best.pt`, `model_final_fp32.pt`, plus the `train_config.json` snapshot of the run.

---

### 3. Export to TorchScript (FP32 + INT8-head)

The exporter writes a quantized **INT8-head** variant (head-only dynamic quantization — backbone stays FP32) alongside a clean FP32 copy.

**Makefile**

```bash
make export-ts              # reads torch_runs/model_final_fp32.pt
```

**Modules**

```bash
python -m scripts.export_ts --ckpt torch_runs/model_final_fp32.pt
```

---

### 4. Evaluate

Run the exported TorchScript model over a labelled CSV (`filepath,class_id,sha1`).

**Makefile**

```bash
make eval                   # default: datasets/cai-vision-dataset/test.csv
# convenience targets:
make eval-val
make eval-test

# overrides:
make eval CSV=datasets/cai-vision-dataset/val.csv \
            TS_MODEL=torch_runs/outputs/ts_*/model_lite0_fp32.ts \
            EVAL_BATCH=64 EVAL_WORKERS=6
```

**Modules**

```bash
python -m scripts.evaluate_ts \
    --csv datasets/cai-vision-dataset/test.csv \
    --model-path torch_runs/outputs/ts_*/model_lite0_fp32.ts \
    --batch-size 64 \
    --workers 6
```

Want a quick eyeball check on a single image? Use the inference helper:

**Makefile**

```bash
make predict IMAGE=path/to/image.jpg TOPK=5
```

**Modules**

```bash
python -m scripts.test_torch_inference path/to/image.jpg \
    --model torch_runs/outputs/ts_*/model_lite0_fp32.ts \
    --top-k 5
```

---

### 5. Export for Mobile (ONNX)

ONNX is the bridge to **Android** (ORT/TFLite) and **iOS** (Core ML via `onnx-coreml`).

**Makefile**

```bash
make export-onnx            # writes torch_runs/outputs/onnx_*/cai_vision.onnx
```

**Modules**

```bash
python -m scripts.export_onnx --ckpt torch_runs/model_final_fp32.pt
```

**Always validate the ONNX** before shipping it to a device — this catches shape/IO-name mismatches that would otherwise blow up at runtime on the phone.

**Makefile**

```bash
make validate-onnx ONNX=torch_runs/outputs/onnx_*/cai_vision.onnx
```

**Modules**

```bash
python -m scripts.validate_onnx torch_runs/outputs/onnx_*/cai_vision.onnx
```

---

## 🔁 End-to-End at a glance

**Makefile**

```bash
# Install -> dataset -> train -> export-TS
make all

# Train -> export TS -> evaluate on test split
make pipeline-eval
```

**Modules**

```bash
python -m pip install -r requirements.txt
python -m scripts.torch_check
python -m scripts.manage_dataset init-from-category
python -m scripts.manage_dataset validate
python -m scripts.manage_dataset ingest-new-data --create-missing
python -m scripts.manage_dataset build-manifests
python -m scripts.export_labels
python -m scripts.train_torch_lite0 --config configs/example_config.json
python -m scripts.export_ts --ckpt torch_runs/model_final_fp32.pt
python -m scripts.evaluate_ts --csv datasets/cai-vision-dataset/test.csv --model-path torch_runs/outputs/ts_*/model_lite0_fp32.ts
python -m scripts.export_onnx --ckpt torch_runs/model_final_fp32.pt
python -m scripts.validate_onnx torch_runs/outputs/onnx_*/cai_vision.onnx
```

---

## 🧹 Cleanup

**Makefile**

```bash
make clean          # remove torch_runs/ AND derived dataset CSVs
make clean-runs     # remove only torch_runs/ (keep train/val/test.csv)
```

---

## 🛠️ Troubleshooting

- **`scripts/__init__.py` must exist** — `python -m scripts.<name>` won't resolve otherwise. (The Makefile assumes it does.)
- **`make train` complains about `CONFIG`** — either point to a valid JSON file or use `make train-default` for hardcoded defaults.
- **ONNX runtime errors on device** — always run `make validate-onnx` locally first; the smoke test catches most shape/IO issues.
- **Slow training** — verify CUDA with `make check-torch`. If you're on CPU, switch to `make install-stable` and expect training to be slow.

---

## 📜 License

Released under the [MIT License](LICENSE) — Copyright © 2025 Erkin Gönültaş.

</body>
</html>