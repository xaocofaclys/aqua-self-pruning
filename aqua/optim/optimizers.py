"""Mask-aware optimizers.

The key correctness property (see DESIGN.md, "the revival bug"):

    A connection that is revived at global step T must behave exactly like a
    freshly created parameter that started training at step T.

Resetting Adam's moments on revival is necessary but *not sufficient*.  Adam's
bias correction divides by (1 - beta^t).  With a single global ``t`` (what every
textbook implementation keeps), a revived entry has fresh moments
m = (1-b1) g, v = (1-b2) g^2 but a correction factor of ~1, so its first step
is lr * (1-b1)/sqrt(1-b2) = 3.16 lr, peaking at ~6.5 lr after ~10 steps.
We therefore keep a *per-entry* step counter that only advances while the
entry is alive and is reset when its topology changes.

Both broken behaviours are still available (``bias_correction="global"``,
``mask_aware=False``) so tests and ablations can demonstrate the failure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from aqua.nn.module import Parameter


@dataclass
class _AdamState:
    m: np.ndarray
    v: np.ndarray
    t: np.ndarray | int  # per-entry int32 array, or a global int


class Optimizer:
    def __init__(self, params: Iterable[Parameter], lr: float, *, mask_aware: bool = True, track_updates: bool = False) -> None:
        self.params: list[Parameter] = list(params)
        if not self.params:
            raise ValueError("optimizer got an empty parameter list")
        if len({id(p) for p in self.params}) != len(self.params):
            raise ValueError("duplicate parameters")
        self.lr = float(lr)
        self.mask_aware = mask_aware
        self.track_updates = track_updates
        self.last_update: dict[int, np.ndarray] = {}
        if mask_aware:
            for p in self.params:
                if p.prunable:
                    p.add_mask_listener(self._on_mask_change)

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    def _active(self, p: Parameter) -> np.ndarray | None:
        return p.mask if (self.mask_aware and p.mask is not None) else None

    def _apply(self, p: Parameter, update: np.ndarray, active: np.ndarray | None) -> None:
        if active is not None:
            update = update * active
        p.data -= update
        if active is not None:
            p.data[~active] = 0.0  # keep removed entries a bit-exact zero
        if self.track_updates:
            self.last_update[id(p)] = update

    def _on_mask_change(self, p: Parameter, old: np.ndarray, new: np.ndarray) -> None:  # pragma: no cover
        raise NotImplementedError

    def step(self) -> None:  # pragma: no cover
        raise NotImplementedError


class Adam(Optimizer):
    """Adam / AdamW with per-entry bias correction for dynamic sparsity.

    Parameters
    ----------
    bias_correction: "per_entry" (correct) or "global" (textbook; wrong for
        revived entries even when moments are reset).
    mask_aware: if False, behave like a naive implementation: no state reset,
        masked entries keep being updated from stale momentum ("zombie drift").
    weight_decay: decoupled (AdamW).  Never applied to removed entries.
    """

    def __init__(
        self,
        params: Iterable[Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        *,
        bias_correction: str = "per_entry",
        mask_aware: bool = True,
        track_updates: bool = False,
    ) -> None:
        if bias_correction not in ("per_entry", "global"):
            raise ValueError("bias_correction must be 'per_entry' or 'global'")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError("betas must be in [0, 1)")
        super().__init__(params, lr, mask_aware=mask_aware, track_updates=track_updates)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.bias_correction = bias_correction
        self.global_step = 0
        self.state: dict[int, _AdamState] = {}
        for p in self.params:
            t = np.zeros(p.shape, dtype=np.int32) if bias_correction == "per_entry" else 0
            self.state[id(p)] = _AdamState(np.zeros_like(p.data), np.zeros_like(p.data), t)

    def _on_mask_change(self, p: Parameter, old: np.ndarray, new: np.ndarray) -> None:
        changed = old ^ new  # pruned *and* revived entries start from a clean slate
        st = self.state[id(p)]
        st.m[changed] = 0.0
        st.v[changed] = 0.0
        if isinstance(st.t, np.ndarray):
            st.t[changed] = 0
        # With a global counter there is nothing to reset: that is the bug.

    def step(self) -> None:
        self.global_step += 1
        b1, b2 = self.beta1, self.beta2
        for p in self.params:
            if p.grad is None:
                continue
            st = self.state[id(p)]
            active = self._active(p)
            g = p.grad if active is None else p.grad * active

            st.m *= b1
            st.m += (1.0 - b1) * g
            st.v *= b2
            st.v += (1.0 - b2) * (g * g)

            if isinstance(st.t, np.ndarray):
                if active is None:
                    st.t += 1
                else:
                    st.t += active  # the clock only runs while the entry exists
                t = st.t
                live = t > 0
                bc1 = np.where(live, 1.0 - np.power(b1, t), 1.0)
                bc2 = np.where(live, 1.0 - np.power(b2, t), 1.0)
            else:
                st.t += 1
                bc1 = 1.0 - b1**st.t
                bc2 = 1.0 - b2**st.t

            update = self.lr * (st.m / bc1) / (np.sqrt(st.v / bc2) + self.eps)
            if self.weight_decay:
                update = update + self.lr * self.weight_decay * p.data
            self._apply(p, update, active)


class SGD(Optimizer):
    """SGD with (optionally Nesterov) momentum; velocity reset on topology change."""

    def __init__(
        self,
        params: Iterable[Parameter],
        lr: float = 1e-2,
        momentum: float = 0.9,
        *,
        nesterov: bool = False,
        weight_decay: float = 0.0,
        mask_aware: bool = True,
        track_updates: bool = False,
    ) -> None:
        super().__init__(params, lr, mask_aware=mask_aware, track_updates=track_updates)
        self.momentum, self.nesterov, self.weight_decay = momentum, nesterov, weight_decay
        self.velocity = {id(p): np.zeros_like(p.data) for p in self.params}

    def _on_mask_change(self, p: Parameter, old: np.ndarray, new: np.ndarray) -> None:
        self.velocity[id(p)][old ^ new] = 0.0

    def step(self) -> None:
        for p in self.params:
            if p.grad is None:
                continue
            active = self._active(p)
            g = p.grad if active is None else p.grad * active
            if self.weight_decay:
                g = g + self.weight_decay * p.data
            vel = self.velocity[id(p)]
            vel *= self.momentum
            vel += g
            if active is not None:
                vel *= active
            direction = g + self.momentum * vel if self.nesterov else vel
            self._apply(p, self.lr * direction, active)
