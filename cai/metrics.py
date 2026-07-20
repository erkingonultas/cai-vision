"""Training and evaluation metrics.
"""
from __future__ import annotations

import torch


def topk_correct(logits: torch.Tensor, target: torch.Tensor, k: int = 1) -> float:
    """Count the number of samples whose ground-truth label is in the top-k predictions.

    Args:
        logits: Model output of shape (batch, num_classes).
        target: Ground-truth class indices of shape (batch,).
        k: How many top predictions to consider.

    Returns:
        Number of correct predictions in the batch (as a Python float).
    """
    with torch.no_grad():
        _, pred = logits.topk(k, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1))
        return correct[:k].reshape(-1).float().sum().item()
