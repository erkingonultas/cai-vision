"""Model construction helper.

Wraps `timm.create_model` so every script constructs the architecture the
same way. The architecture name itself lives in `cai.constants` so the
rest of the project refers to it symbolically.
"""
from __future__ import annotations

import timm
import torch.nn as nn

from cai.constants import MODEL_NAME


def build_model(num_classes: int, *, pretrained: bool = True) -> nn.Module:
    """Construct the CAI-Vision classifier.

    Args:
        num_classes: Size of the classification head.
        pretrained: If True, load ImageNet weights for the backbone
            (use True for training, False when loading a checkpoint for
            inference / export).
    """
    return timm.create_model(MODEL_NAME, pretrained=pretrained, num_classes=num_classes)
