"""Importance criteria: how a connection earns the right to survive.

All criteria return a score per weight where **higher = keep**.  The two
gradient-based ones are estimates of the loss increase caused by removing the
connection, i.e. of

    dL_i = L(theta - w_i e_i) - L(theta)
         ~= -g_i w_i + 1/2 H_ii w_i^2                      (2nd-order Taylor)

(see DESIGN.md for the derivation and the assumptions).

* ``magnitude``: |w|.  Baseline.  Implicitly assumes g = 0 and H = c*I.
* ``saliency``:  E|w g|.  First-order Taylor magnitude (Molchanov et al.
  2017/2019) - the literal |weight x gradient| of the brief, averaged over
  steps so a single noisy mini-batch cannot decide a weight's fate.
* ``taylor2``:   1/2 F w^2 - g_bar w.  Full second-order estimate with the
  diagonal Hessian replaced by the diagonal *per-example* empirical Fisher
  (Optimal Brain Damage with a Fisher curvature).  Signed: a negative value
  predicts that removal *lowers* the loss, so it is pruned first.  Because it
  is in loss units it is comparable *across* layers.
* ``fisher_obd``: 1/2 F w^2.  ``taylor2`` without the first-order term.  The
  default.  Near a minimum E[g] ~ 0, and the noisy signed term -g_bar w makes a
  *local* linear prediction about a *non-local* move (w -> 0); dropping it is
  what the validation set preferred by a wide margin (see DESIGN.md sec. 2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LayerStats:
    """Bias-corrected exponential moving averages, all of shape W."""

    beta: float
    grad: np.ndarray        # E[dL/dW_eff]          (dense: defined at pruned entries too)
    abs_wg: np.ndarray      # E|w * dL/dW_eff|
    fisher: np.ndarray      # E_n[(dL_n/dW_eff)^2]  per-example empirical Fisher diagonal
    count: int = 0

    @classmethod
    def zeros(cls, shape, beta: float) -> "LayerStats":
        return cls(beta, np.zeros(shape), np.zeros(shape), np.zeros(shape), 0)

    def update(self, w: np.ndarray, dense_grad: np.ndarray, fisher_batch: np.ndarray | None) -> None:
        b = self.beta
        self.grad = b * self.grad + (1 - b) * dense_grad
        self.abs_wg = b * self.abs_wg + (1 - b) * np.abs(w * dense_grad)
        if fisher_batch is not None:
            self.fisher = b * self.fisher + (1 - b) * fisher_batch
        self.count += 1

    def _debias(self, x: np.ndarray) -> np.ndarray:
        return x / (1.0 - self.beta**self.count) if self.count else x

    @property
    def mean_grad(self) -> np.ndarray:
        return self._debias(self.grad)

    @property
    def mean_abs_wg(self) -> np.ndarray:
        return self._debias(self.abs_wg)

    @property
    def mean_fisher(self) -> np.ndarray:
        return self._debias(self.fisher)


def per_example_fisher(x: np.ndarray, grad_preact: np.ndarray) -> np.ndarray:
    """Exact per-example empirical Fisher diagonal for a linear layer, in one matmul.

    For y_n = x_n W, the per-example gradient is dL_n/dW_ij = x_ni * d_nj with
    d_n = dL_n/dy_n.  With a mean-reduced loss, the batch gradient at y is
    G_y = d / N, so d = N * G_y and

        F_ij = (1/N) sum_n (x_ni d_nj)^2 = (1/N) (x^2)^T (d^2) = N (x^2)^T (G_y^2).

    The naive alternative, squaring the *mini-batch* gradient, estimates
    (E g)^2 + Var(g)/N instead of E[g^2] and badly under-counts curvature.
    """
    n = x.shape[0]
    return n * ((x * x).T @ (grad_preact * grad_preact))


CRITERIA = ("magnitude", "saliency", "taylor2", "fisher_obd")


def needs_fisher(criterion: str) -> bool:
    return criterion in ("taylor2", "fisher_obd")


def importance(criterion: str, w: np.ndarray, stats: LayerStats | None) -> np.ndarray:
    """Score each weight; higher means more important (keep)."""
    if criterion == "magnitude":
        return np.abs(w)
    if stats is None or stats.count == 0:
        raise RuntimeError(f"criterion {criterion!r} needs gradient statistics; none were collected")
    if criterion == "saliency":
        return stats.mean_abs_wg
    if criterion == "taylor2":
        return 0.5 * stats.mean_fisher * w * w - stats.mean_grad * w
    if criterion == "fisher_obd":
        return 0.5 * stats.mean_fisher * w * w
    raise ValueError(f"unknown criterion {criterion!r}; choose from {CRITERIA}")


def regrowth_score(stats: LayerStats) -> np.ndarray:
    """RigL-style: grow where the dense loss slope is steepest, |E[dL/dW_eff]|."""
    return np.abs(stats.mean_grad)
