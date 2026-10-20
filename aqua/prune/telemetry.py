"""Instrumentation: how big are the steps a revived connection takes?

For a correctly implemented Adam, a fresh parameter's normalized step
|dw| / lr = |m_hat| / (sqrt(v_hat) + eps) is <= ~1 and exactly 1 on its first
step.  ``RevivalMonitor`` records that ratio as a function of the connection's
age since regrowth, which makes the global-step-counter bug directly visible
in a real training run (not just a unit test).
"""

from __future__ import annotations

import numpy as np

from .pruner import Pruner


class RevivalMonitor:
    def __init__(self, pruner: Pruner, horizon: int = 60) -> None:
        self.pruner = pruner
        self.horizon = horizon
        self.cohorts: list[tuple[int, np.ndarray, int]] = []  # (layer index, flat idx, birth step)
        self.sum_ratio = np.zeros(horizon)
        self.max_ratio = np.zeros(horizon)
        self.count = np.zeros(horizon, dtype=np.int64)

    def on_train_begin(self, trainer) -> None:
        if not trainer.optimizer.track_updates:
            raise RuntimeError("RevivalMonitor needs an optimizer built with track_updates=True")

    def after_backward(self, trainer) -> None:
        pass

    def after_step(self, trainer) -> None:
        opt = trainer.optimizer
        step = trainer.step
        alive_cohorts = []
        for layer_i, idx, birth in self.cohorts:
            age = step - birth  # 1 on the first optimizer step after regrowth
            if age > self.horizon:
                continue
            weight = self.pruner.layers[layer_i].weight
            # Same incarnation only: alive *and* not dropped-then-regrown since birth.
            still_alive = weight.mask.ravel()[idx] & (self.pruner.grown_at[layer_i].ravel()[idx] == birth)
            if still_alive.any():
                upd = opt.last_update[id(weight)].ravel()[idx[still_alive]]
                ratio = np.abs(upd) / opt.lr
                self.sum_ratio[age - 1] += ratio.sum()
                self.max_ratio[age - 1] = max(self.max_ratio[age - 1], float(ratio.max()))
                self.count[age - 1] += ratio.size
            alive_cohorts.append((layer_i, idx, birth))
        self.cohorts = alive_cohorts
        # Register connections grown in *this* step (they take their first update next step).
        for layer_i, idx in self.pruner.last_grown:
            self.cohorts.append((layer_i, idx, step))

    def mean_ratio(self) -> np.ndarray:
        return np.divide(self.sum_ratio, np.maximum(self.count, 1))

    def report(self) -> dict:
        return {
            "age": list(range(1, self.horizon + 1)),
            "mean_step_over_lr": self.mean_ratio().tolist(),
            "max_step_over_lr": self.max_ratio.tolist(),
            "n_samples": self.count.tolist(),
        }
