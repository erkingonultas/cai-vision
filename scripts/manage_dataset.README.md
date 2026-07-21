# 🗂️ `scripts/manage_dataset.py`

> **Single-source-of-truth tool for the CAI-Vision dataset.**
> Schema generation, ingestion, deduplication, and train/val/test manifest building — driven by `category.txt` and the on-disk `images/` tree.

The rest of the pipeline (training, export, evaluation, ONNX validation) all assumes the on-disk layout this script produces. If your data isn't shaped correctly, nothing else will work cleanly. **Run this first, run it often, and let it be the only thing that mutates `images/`.**

---

## 📍 Scope

This tool is responsible for **everything between "I have a pile of class folders" and "I have `train.csv`, `val.csv`, `test.csv` ready for training"**:

1. Generate / merge the canonical class schema (`classes.csv`) from a human-readable `category.txt`.
2. Keep the schema in sync with the on-disk `images/<class_id>/` folders.
3. Ingest freshly-dropped images from `new-data/<class_name>/` into the right `images/<class_id>/` folder.
4. **Deduplicate** via SHA1 (skip identical re-ingests, flag cross-class duplicates).
5. **Renumber** targets to `1.jpg..N.jpg` for a clean, deterministic layout.
6. Emit deterministic, dedup-aware `train.csv / val.csv / test.csv` manifests.

It does **not** train, export, or evaluate — that's what the other `scripts/*` modules are for.

---

## 🗺️ Directory Contract

The script expects (and produces) this layout under `./datasets/cai-vision-dataset/`:

```
datasets/cai-vision-dataset/
├── category.txt                    # source of truth: "<id>   <display_name_en>" per line
├── classes.csv                     # generated: full schema (see "Schema" below)
├── multilabel_overrides.csv        # optional: excludes from build-manifests
├── images/
│   └── <class_id>/                 # one folder per class, numbered 1.jpg..N.jpg
└── new-data/
    └── <class_name>/               # DROP ZONE for incoming images
        └── *.jpg / *.png / *.webp …
```

`category.txt` is the **only** file you should ever hand-edit for class definitions. Everything else is derived.

---

## 🧬 Schema (`classes.csv`)

Columns written by `init-from-category` / `add-class`:

| Column              | Meaning                                                                 |
|---------------------|-------------------------------------------------------------------------|
| `class_id`          | Stable integer ID. Used as the on-disk folder name.                     |
| `slug`              | URL/file-safe variant of the display name.                              |
| `display_name_en`   | Canonical English name shown in the UI and written to `labels.txt`.     |
| `synonyms_en`       | Pipe-separated aliases accepted by `ingest-new-data` (case-insensitive).|
| `status`            | `active` by default; `disabled` rows are kept but excluded from splits. |
| `parent_id`         | Reserved for future hierarchical labels. Leave blank for now.           |
| `notes`             | Free-form, human notes.                                                 |

---

## ⚙️ Invocation

```bash
python -m scripts.manage_dataset <subcommand> [args]
```

Run with no arguments (or `--help`) to see the live subcommand list — the CLI is the source of truth if this README drifts.

---

## 🧩 Subcommands

### `init-from-category`

**Generate / merge `classes.csv` from `category.txt`.**

- Reads `<id> <name>` pairs from `category.txt`.
- Merges with any existing `classes.csv` (preserves `synonyms_en`, `status`, `parent_id`, `notes` for IDs already present).
- Fills in `slug` and defaults for newly seen classes.
- Writes the result back to `classes.csv`.

```bash
python -m scripts.manage_dataset init-from-category
```

> **Run this once** when you first set up the dataset, and any time you change `category.txt`.

---

### `validate`

**Schema sanity check + cross-class duplicate detection.**

Checks performed:

- `classes.csv` exists and is well-formed.
- Every `images/<class_id>/` folder has a matching entry in `classes.csv` (and vice-versa).
- Folders are non-empty and contain only supported image extensions.
- Every image is a valid file (size > 0, readable).
- **Cross-class duplicates:** same SHA1 appearing under two different `class_id`s are reported (usually means someone misfiled an image).

```bash
python -m scripts.manage_dataset validate
```

> Non-zero exit on any failure. Safe to wire into CI.

---

### `add-class`

**Manually register a new class and create its empty `images/<id>/` folder.**

```bash
python -m scripts.manage_dataset add-class "New Dish Name"
```

- Auto-assigns the next free integer ID (`max(existing) + 1`).
- Appends to `category.txt` for traceability.
- Creates `images/<id>/` (no-op if it already exists).
- Marks the row as `active` with empty `synonyms_en`.

Use this when you know a class is coming and want to seed it cleanly. For ad-hoc additions during ingestion, prefer `ingest-new-data --create-missing`.

---

### `ingest-new-data`

**Move images from `new-data/<class_name>/*` into `images/<class_id>/*`.**

This is the **drop-zone workflow**:

1. You drop images into `datasets/cai-vision-dataset/new-data/<class_name>/*.jpg`.
2. You run `ingest-new-data`.
3. The script:
   - Resolves `<class_name>` against `classes.csv` (case-, space-, and underscore-insensitive; also matches `slug` and `synonyms_en`).
   - Computes SHA1 of every new file.
   - **Skips exact duplicates** already present in `images/`.
   - **Logs cross-class duplicates** (same SHA1 in another class) to a report.
   - Moves accepted files into `images/<class_id>/`.
   - **Renumbers** the destination folder to `1.jpg..N.jpg` for deterministic ordering.
   - Cleans up empty subfolders under `new-data/`.

```bash
# Strict: unknown <class_name> dirs are an error
python -m scripts.manage_dataset ingest-new-data

# Permissive: auto-create classes for unknown <class_name> dirs
python -m scripts.manage_dataset ingest-new-data --create-missing
```

> Always run `validate` after ingestion to catch any cross-class dupes the ingest surfaced.

---

### `build-manifests`

**Emit `train.csv`, `val.csv`, `test.csv` from the on-disk `images/` tree.**

Each manifest is a CSV with columns:

| Column      | Meaning                                                          |
|-------------|------------------------------------------------------------------|
| `filepath`  | Repo-relative path to the image, e.g. `datasets/cai-vision-dataset/images/3/42.jpg`. |
| `class_id`  | Integer label.                                                   |
| `sha1`      | SHA1 of the file bytes. Used downstream for dedup-aware eval.    |

Behavior:

- Walks `images/<class_id>/*`.
- Skips rows listed in `multilabel_overrides.csv` (if present).
- Optionally applies **in-class dedup** (`--dedup-in-class`) so no class is over-represented by a near-duplicate.
- Stratified split per class using `--seed`.
- Ratios are class-stratified; missing classes are tolerated (warned).

```bash
python -m scripts.manage_dataset build-manifests \
    --val 0.10 \
    --test 0.15 \
    --seed 42 \
    --dedup-in-class
```

Common overrides:

```bash
# 70/15/15 split
python -m scripts.manage_dataset build-manifests --val 0.15 --test 0.15

# Reproducible rerun
python -m scripts.manage_dataset build-manifests --seed 1234
```

> **Re-run this any time the `images/` tree changes.** It is fully deterministic for a given `--seed`, so you can commit the resulting CSVs and reproduce training exactly.

---

## 🔁 The Canonical Workflow

```bash
# 0. (one-time, when first setting up) seed the schema
python -m scripts.manage_dataset init-from-category

# 1. Drop images into new-data/<class_name>/
#    …or add a brand new class first:
python -m scripts.manage_dataset add-class "İmam Bayıldı"

# 2. Ingest them
python -m scripts.manage_dataset ingest-new-data --create-missing

# 3. Sanity-check
python -m scripts.manage_dataset validate

# 4. Build train/val/test splits
python -m scripts.manage_dataset build-manifests --val 0.10 --test 0.15 --seed 42 --dedup-in-class

# 5. (optional but recommended) regenerate labels.txt
python -m scripts.export_labels
```

Equivalent via the Makefile (from the project root):

```bash
make dataset-init
make dataset-add-class NAME="İmam Bayıldı"   # if needed
make dataset-ingest
make dataset-validate
make dataset-build-manifests                  # VAL_RATIO=0.10 TEST_RATIO=0.15 SEED=42
make labels
```

---

## 🧠 Design Notes & Edge Cases

- **Class-name matching is forgiving.** `ingest-new-data` matches `<class_name>` against `display_name_en`, `slug`, and `synonyms_en`, normalizing case, whitespace, and underscores. This means sloppy folder names like `Imam_Bayildi` or `imam bayildi` will still land in the right class.
- **SHA1 dedup is content-based, not filename-based.** Re-dropping the same file under any name is a no-op.
- **Cross-class duplicates are flagged, not auto-merged.** Look at the ingest report and decide manually which class is correct.
- **`status: disabled` classes** are kept on disk (so IDs remain stable) but excluded from `build-manifests`. Use this for classes you're temporarily not training on.
- **Multilabel samples** belong in `multilabel_overrides.csv` (columns: `sha1`, `class_ids`, `notes`). Rows listed there are skipped by `build-manifests` so they don't pollute single-label training. (Single-label training is what the current model head expects.)
- **Renumbering is destructive-but-safe.** `1.jpg..N.jpg` is the canonical layout; the script uses a temp directory to avoid mid-rename collisions. Anything **not** matching the schema is still moved over, but only `.jpg/.jpeg/.png/.webp` are accepted.
- **The script is idempotent for `init-from-category` and `add-class`.** You can re-run them safely; existing metadata is preserved.

---

## 🚨 Troubleshooting

| Symptom                                                | Likely cause                                                                 | Fix                                                                                  |
|--------------------------------------------------------|------------------------------------------------------------------------------|--------------------------------------------------------------------------------------|
| `validate` complains about a missing `images/<id>/`    | Class is in `classes.csv` but no folder exists yet.                          | `python -m scripts.manage_dataset add-class "Exact Name"` or create the folder.       |
| `ingest-new-data` errors on an unknown class           | Class folder in `new-data/` doesn't match any row.                           | Re-run with `--create-missing`, or `add-class` first.                                |
| `build-manifests` reports very uneven class sizes      | Class imbalance in `images/`.                                                | Drop more data, or use a config that supports class weighting at training time.       |
| `validate` flags a cross-class duplicate                | Same image filed under two class folders.                                    | Inspect the report; delete from the wrong class and re-run `ingest-new-data`.        |
| Splits change between runs                             | You forgot to pin `--seed`, or `images/` changed.                            | Always pass `--seed`, commit the resulting CSVs.                                     |
| `labels.txt` is out of sync with `classes.csv`         | You added/renamed a class but didn't regenerate labels.                      | `python -m scripts.export_labels` (or `make labels`).                                |

---

## 🔗 See Also

- `Makefile` — wraps all of the above as `make dataset-*` targets.
- `scripts/export_labels.py` — turns `classes.csv` into the `labels.txt` consumed at inference time.
- `scripts/train_torch_lite0.py` — consumes the manifests produced here.
- Top-level [`README.md`](../../README.md) — full pipeline walkthrough.