"""Finite-difference gradient checking.

For an operation ``f`` with (possibly non-scalar) output we check the scalar
``L = sum(f(x) * R)`` for a fixed random projection ``R``.  A random ``R``
exercises every output element with a distinct weight, so a bug that permutes
or mis-scales gradients cannot hide behind symmetry (as it can with
``R = ones``).  Central differences have O(eps^2) truncation error; in float64
with eps=1e-6 the relative error of a correct op is ~1e-9.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .tensor import Tensor


@dataclass
class GradCheckResult:
    max_rel_error: float
    max_abs_error: float
    per_input: list[float]
    analytic: list[np.ndarray]
    numeric: list[np.ndarray]

    def passed(self, rtol: float = 1e-6) -> bool:
        return self.max_rel_error <= rtol


def _relative_error(a: np.ndarray, n: np.ndarray, floor: float = 1e-8) -> float:
    denom = np.maximum(np.abs(a) + np.abs(n), floor)
    return float(np.max(np.abs(a - n) / denom)) if a.size else 0.0


def gradcheck(
    fn: Callable[..., Tensor],
    inputs: Sequence[Tensor],
    *,
    eps: float = 1e-6,
    seed: int = 0,
) -> GradCheckResult:
    """Compare analytic and central-difference gradients of ``fn(*inputs)``.

    Every input must be a float64 Tensor with ``requires_grad=True``.
    """
    for x in inputs:
        if x.dtype != np.float64:
            raise TypeError("gradcheck requires float64 inputs")
        if not x.requires_grad:
            raise ValueError("every gradcheck input must require grad")
        if not x.data.flags.c_contiguous:
            # reshape(-1) must be a view so that perturbations reach the tensor.
            x.data = np.ascontiguousarray(x.data)

    rng = np.random.default_rng(seed)
    probe = fn(*inputs)
    projection = rng.standard_normal(probe.shape)

    def scalar_loss() -> float:
        return float((fn(*inputs).data * projection).sum())

    for x in inputs:
        x.grad = None
    out = fn(*inputs)
    (out * projection).sum().backward()
    analytic = [np.zeros_like(x.data) if x.grad is None else x.grad.copy() for x in inputs]

    numeric: list[np.ndarray] = []
    for x in inputs:
        num = np.zeros_like(x.data)
        flat, num_flat = x.data.reshape(-1), num.reshape(-1)
        for i in range(flat.size):
            original = flat[i]
            flat[i] = original + eps
            plus = scalar_loss()
            flat[i] = original - eps
            minus = scalar_loss()
            flat[i] = original
            num_flat[i] = (plus - minus) / (2.0 * eps)
        numeric.append(num)

    per_input = [_relative_error(a, n) for a, n in zip(analytic, numeric)]
    max_abs = max((float(np.max(np.abs(a - n))) if a.size else 0.0) for a, n in zip(analytic, numeric))
    return GradCheckResult(
        max_rel_error=max(per_input) if per_input else 0.0,
        max_abs_error=max_abs,
        per_input=per_input,
        analytic=analytic,
        numeric=numeric,
    )
