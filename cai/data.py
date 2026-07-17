"""Dataset and image-transform definitions.

Centralised here so training, evaluation, and inference scripts share the
exact same preprocessing pipeline.
"""
from __future__ import annotations

from typing import Callable, List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from cai.constants import IMG_SIZE

# An "item" is a (filepath, class_id) pair. Class ids are 0-based.
Item = Tuple[str, int]


# --- Transforms ---
# Note: image is converted to RGB inside CSVDataset.__getitem__, so the
# transforms do not need a separate Lambda(convert) step.
def _to_tensor() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),  # -> [0, 1] float32
        ]
    )


def train_transforms() -> transforms.Compose:
    """Transforms used during training: resize, light augmentation."""
    return transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
        ]
    )


def eval_transforms() -> transforms.Compose:
    """Deterministic transforms used at val/test/inference time."""
    return _to_tensor()


# --- Dataset ---
class CSVDataset(Dataset):
    """A torch Dataset that reads (filepath, class_id) pairs from a list.

    The class-id list is expected to be 0-based. If you have a 1-based CSV,
    normalise it first (see `cai.data.normalize_class_ids`).
    """

    def __init__(
        self,
        items: List[Item],
        transform: Callable | None = None,
        return_path: bool = False,
    ) -> None:
        """
        Args:
            items: List of (filepath, class_id) pairs.
            transform: Optional torchvision transform. Defaults to eval_transforms().
            return_path: If True, __getitem__ also returns the source filepath
                (useful for per-sample logging in evaluation).
        """
        self.items = items
        self.transform = transform if transform is not None else eval_transforms()
        self.return_path = return_path

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        path, label = self.items[idx]
        img = Image.open(path).convert("RGB")  # always 3-channel
        x = self.transform(img)
        if self.return_path:
            return x, label, path
        return x, label


def normalize_class_ids(items: List[Item]) -> List[Item]:
    """Shift a list of (path, class_id) pairs to 0-based if they are 1-based.

    Auto-detects: if the minimum class id is 1, subtracts 1 from every id.
    Validates that the resulting ids are contiguous from 0.

    Raises:
        ValueError: If the ids are not contiguous from 0 after normalisation.
    """
    if not items:
        return items
    min_id = min(cid for _, cid in items)
    if min_id == 0:
        normalised = items
    elif min_id == 1:
        normalised = [(p, cid - 1) for p, cid in items]
    else:
        raise ValueError(
            f"Unexpected minimum class id {min_id}; expected 0 (0-based) or 1 (1-based)."
        )

    ids = sorted({cid for _, cid in normalised})
    expected = list(range(len(ids)))
    if ids != expected:
        raise ValueError(
            f"Class ids are not contiguous from 0. Got {ids}, expected {expected}."
        )
    return normalised


def make_loader(
    items: List[Item],
    *,
    training: bool = False,
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
) -> torch.utils.data.DataLoader:
    """Build a DataLoader with the right transforms for the phase."""
    transform = train_transforms() if training else eval_transforms()
    dataset = CSVDataset(items, transform=transform)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
