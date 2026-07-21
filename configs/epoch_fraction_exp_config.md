**`lr: 1e-4` (yours: 3e-4)** — More conservative. The pretrained features are valuable; a lower LR preserves them while still adapting the head. If your dataset is close to ImageNet in domain, this should outperform the higher LR.

**`label_smoothing: 0.05` (yours: 0.1)** — Lighter regularization on the loss. If your labels are clean, `0.1` may be over-softening the targets and slowing convergence. This gives the model a crisper gradient signal.

**`t0_fraction_of_epochs: 0.333` (yours: 0.25)** — Fixes the cycle alignment issue we discussed. This gives you 8 → 16 epoch cycles, landing exactly at epoch 24 with LR near `eta_min`. Cleaner final weights.

**`eta_min: 1e-7` (yours: 1e-6)** — Wider decay floor. With `lr = 1e-4`, this gives a 1000× decay ratio vs. your 300×. The cosine has more room to anneal thoroughly at the bottom of each cycle.

**`patience: 8` (yours: 6)** — One full `T_0` cycle worth of patience. This is intentional: with a restart at epoch 8, the LR spikes and val top-1 may temporarily dip. Patience of 8 ensures you won't stop mid-recovery during cycle 2.

**`seed: 7` (yours: 42)** — Different seed deliberately. Running two configs on different seeds means you're also sampling variance, which will tell you whether a result difference is real or just lucky initialization.