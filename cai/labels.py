"""Label list loading.

Each entry of `labels.txt` is the display name for one class, in the same
order as the model's class index (0-based).
"""
from __future__ import annotations

from pathlib import Path
from typing import List


def load_labels(path: Path) -> List[str]:
    """Read a labels file and return a list of non-empty stripped names.

    Args:
        path: Path to a labels.txt-style file (one label per line).

    Raises:
        FileNotFoundError: If `path` does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Labels file not found: {path}. "
            f"Generate it with `python scripts/write_labels.py`."
        )
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def num_classes(path: Path) -> int:
    """Convenience: number of classes implied by a labels file."""
    return len(load_labels(path))
