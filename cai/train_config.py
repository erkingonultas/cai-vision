"""Training configuration for ``scripts/train_torch_lite0.py``.

This module exposes a single :class:`TrainConfig` dataclass that gathers
every training-time knob in one place. Scripts build a ``TrainConfig``
from defaults, optionally overlay a JSON file (``--config path.json``),
and finally apply any CLI overrides on top.

Why a config object (instead of a long ``argparse`` block)?

* Single source of truth: the JSON file you used for a run is dumped
  back to disk under the run's output directory, so the run is fully
  reproducible.
* Cross-knob validation: e.g. the scheduler's first-cycle length is
  derived from ``epochs`` and the configured cycle fraction, with a
  guard for degenerate cases.
* Easier to study: every tunable lives under a named, typed field.

Example JSON file (``experiments/cosine_long.json``):

    {
      "optim": {
        "lr": 5e-4,
        "label_smoothing": 0.05,
        "t0_fraction_of_epochs": 0.25,
        "t_mult": 2,
        "eta_min": 1e-6
      },
      "runtime": {
        "epochs": 24,
        "batch_size": 32,
        "patience": 6
      }
    }

Run it with::

    python scripts/train_torch_lite0.py --config experiments/cosine_long.json

Note: dataset paths (Group A) deliberately stay as CLI-only args; they
are infrastructure, not hyperparameters.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class OptimConfig:
    """Optimizer and LR-scheduler hyperparameters.

    Attributes:
        lr: Peak learning rate for the Adam optimizer.
        label_smoothing: CrossEntropy label-smoothing factor in [0, 1).
        t0_fraction_of_epochs: Length of the *first* cosine-warm-restart
            cycle, expressed as a fraction of the total epoch budget.
            ``T_0 = int(len(train_loader) * t0_fraction_of_epochs * epochs)``.
            Subsequent cycles multiply by ``t_mult`` (TF-style
            ``CosineDecayRestarts`` analogue).
        t_mult: Multiplicative factor applied to cycle length after each
            restart. ``2`` means cycles grow 5 -> 10 -> 20 -> ...
        eta_min: Minimum LR the scheduler decays down to.
    """

    lr: float = 1e-3
    label_smoothing: float = 0.1
    t0_fraction_of_epochs: float = 0.3
    t_mult: int = 2
    eta_min: float = 1e-5


@dataclass
class RuntimeConfig:
    """Per-run practical knobs (training loop, hardware, reproducibility).

    Attributes:
        epochs: Total epoch budget.
        batch_size: Mini-batch size for train/val/test loaders.
        num_workers: DataLoader worker processes.
        seed: RNG seed for Python / NumPy / PyTorch (+ CUDA).
        patience: Early-stopping patience in epochs (on val top-1).
        pretrained: If True, initialise the backbone with ImageNet
            weights via timm.
    """

    epochs: int = 16
    batch_size: int = 64
    num_workers: int = 6
    seed: int = 42
    patience: int = 4
    pretrained: bool = True


@dataclass
class TrainConfig:
    """Top-level training config.

    Top-level groups are kept flat (only two: optim, runtime) so the
    JSON shape stays obvious. If a third group is ever needed, add it
    here and update :meth:`from_dict`.
    """

    optim: OptimConfig = field(default_factory=OptimConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=False)
            f.write("\n")

    @classmethod
    def load(cls, path: Path) -> "TrainConfig":
        path = Path(path)
        # ``utf-8-sig`` transparently strips a BOM if present, so configs
        # saved by Notepad on Windows or by ``Set-Content -Encoding utf8``
        # in PowerShell still load.
        with open(path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainConfig":
        optim = OptimConfig(**(data.get("optim") or {}))
        runtime = RuntimeConfig(**(data.get("runtime") or {}))
        return cls(optim=optim, runtime=runtime)

    # ------------------------------------------------------------------ #
    # Cross-knob validation
    # ------------------------------------------------------------------ #
    def warnings(self) -> List[str]:
        """Return a list of human-readable warnings about misaligned knobs.

        These are non-fatal: training still proceeds, but the user is
        informed that they're likely leaving performance on the table.
        """
        msgs: List[str] = []
        epochs = self.runtime.epochs
        first_cycle_epochs = epochs * self.optim.t0_fraction_of_epochs
        if first_cycle_epochs < 1.0:
            msgs.append(
                f"t0_fraction_of_epochs={self.optim.t0_fraction_of_epochs} "
                f"with epochs={epochs} gives a first cycle of "
                f"{first_cycle_epochs:.2f} epochs (<1). "
                f"CosineAnnealingWarmRestarts will be degenerate."
            )
        if self.runtime.patience >= epochs:
            msgs.append(
                f"patience={self.runtime.patience} >= epochs={epochs}: "
                f"early stopping will never trigger before the loop ends."
            )
        if self.optim.eta_min >= self.optim.lr:
            msgs.append(
                f"eta_min={self.optim.eta_min} >= lr={self.optim.lr}: "
                f"the scheduler will never decay the learning rate."
            )
        return msgs