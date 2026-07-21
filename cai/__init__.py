"""CAI-Vision shared library.

This package holds the small amount of code that is currently duplicated
across the training / evaluation / export scripts. The goal is to keep
each top-level script thin (argument parsing + orchestration) while the
domain logic lives here.

Modules:
    constants    - Shared constants (image size, model name, paths)
    device       - Device selection helpers
    data         - Dataset class and image transforms
    labels       - Label list loading
    metrics      - Top-k accuracy helper
    model        - Model construction wrapper around timm
    train_config - Training-config dataclass with JSON (de)serialization
"""
