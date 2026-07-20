"""Generate scripts/labels.txt from the dataset's classes.csv.

Reads `class_id` and `display_name_en` columns from
`<DATASET_ROOT>/classes.csv`, sorts by `class_id` (0-based), and writes
one label per line to the configured `LABELS_FILE`.

Usage:
    python scripts/write_labels.py
    python scripts/write_labels.py --classes path/to/classes.csv --out path/to/labels.txt
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from cai.constants import CLASSES_CSV, LABELS_FILE


def build_labels(classes_csv: Path) -> list[tuple[int, str]]:
    """Return a list of (class_id, display_name_en) pairs, sorted by class_id."""
    with open(classes_csv, newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        rows = sorted(rdr, key=lambda r: int(r["class_id"]))
    return [(int(r["class_id"]), r["display_name_en"]) for r in rows]


def write_labels(classes_csv: Path, out_path: Path) -> int:
    """Write sorted labels to `out_path`. Returns the number of labels written."""
    if not classes_csv.exists():
        raise FileNotFoundError(
            f"classes.csv not found: {classes_csv}. "
            f"Run `python data-prep/dataset_schema_tool.py init-from-category` first."
        )
    pairs = build_labels(classes_csv)
    with open(out_path, "w", encoding="utf-8") as out:
        for _cid, name in pairs:
            out.write(f"{name}\n")
    return len(pairs)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate scripts/labels.txt from the dataset's classes.csv."
    )
    ap.add_argument(
        "--classes",
        type=Path,
        default=CLASSES_CSV,
        help="Path to classes.csv (default: <project>/datasets/cai-vision-dataset/classes.csv).",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=LABELS_FILE,
        help="Output labels file (default: <project>/scripts/labels.txt).",
    )
    args = ap.parse_args()

    n = write_labels(args.classes, args.out)
    print(f"Wrote {n} labels to {args.out}")


if __name__ == "__main__":
    main()
