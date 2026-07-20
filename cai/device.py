"""Device selection helpers.

Single source of truth for the `cuda if available else cpu` choice that
is currently repeated in every script.
"""
from __future__ import annotations

import torch


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Return a torch.device, preferring CUDA when available and allowed.

    Args:
        prefer_cuda: If False, always return CPU (useful for export scripts
            that should run on CPU regardless of available hardware).
    """
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def is_cuda_available() -> bool:
    """True if a CUDA-capable PyTorch build is installed and a GPU is visible."""
    return torch.cuda.is_available()
