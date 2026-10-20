"""Small, dependency-free statistics for seed-variance reasoning.

We compare methods with **paired** tests: run ``i`` of method A and run ``i``
of method B share the seed, hence the initialization, the data order and the
held-out split.  Pairing removes the between-seed variance that is common to
both methods, which is usually most of it.
"""

from __future__ import annotations

import itertools

import numpy as np


def paired_permutation_pvalue(diffs: np.ndarray) -> float:
    """Exact two-sided sign-flip permutation test of H0: E[diff] = 0.

    Under H0 (exchangeability within a pair) each paired difference is
    equally likely to have either sign, so we enumerate all 2^n sign patterns.
    Exact for n <= 20; no normality assumption (unlike the paired t-test).
    With n = 10 the smallest attainable p-value is 2/1024 ~= 0.002.
    """
    d = np.asarray(diffs, dtype=float)
    n = d.size
    if n == 0:
        return 1.0
    if n > 20:
        raise ValueError("exact enumeration limited to n <= 20")
    observed = abs(d.mean())
    signs = np.array(list(itertools.product([1.0, -1.0], repeat=n)))
    null = np.abs((signs * d).mean(axis=1))
    return float(np.mean(null >= observed - 1e-12))


def bootstrap_ci(values: np.ndarray, *, level: float = 0.95, n_boot: int = 20_000, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean (fixed seed -> reproducible)."""
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, v.size, size=(n_boot, v.size))].mean(axis=1)
    alpha = (1.0 - level) / 2.0
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))


def describe(values: np.ndarray) -> dict:
    v = np.asarray(values, dtype=float)
    return {
        "n": int(v.size),
        "mean": float(v.mean()),
        "std": float(v.std(ddof=1)) if v.size > 1 else 0.0,
        "sem": float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else 0.0,
        "min": float(v.min()),
        "max": float(v.max()),
    }


def paired_comparison(a: np.ndarray, b: np.ndarray) -> dict:
    """Summarize A - B for paired samples."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired samples must have equal shape")
    d = a - b
    lo, hi = bootstrap_ci(d)
    return {
        "mean_diff": float(d.mean()),
        "std_diff": float(d.std(ddof=1)) if d.size > 1 else 0.0,
        "ci95": [lo, hi],
        "p_value": paired_permutation_pvalue(d),
        "wins": int((d > 0).sum()),
        "ties": int((d == 0).sum()),
        "losses": int((d < 0).sum()),
        "n": int(d.size),
    }
