"""Wall-clock timing with warm-up and robust statistics."""

from __future__ import annotations

import time

import numpy as np


def time_callable(fn, x: np.ndarray, *, repeats: int = 200, warmup: int = 20, inner: int = 1, budget_s: float = 1.5) -> dict:
    """Median and spread of per-call latency in microseconds.

    ``inner`` calls are batched per sample to amortize timer resolution for
    very fast calls.  The median (not the mean) is reported because latency
    distributions are right-skewed by OS scheduling noise.
    """
    t0 = time.perf_counter()
    fn(x)
    single = time.perf_counter() - t0
    # Slow configurations get fewer samples (never fewer than 15) so a sweep stays bounded.
    repeats = int(max(15, min(repeats, budget_s / max(single * inner, 1e-9))))
    warmup = min(warmup, max(2, repeats // 5))
    for _ in range(warmup):
        fn(x)
    samples = np.empty(repeats)
    for i in range(repeats):
        t0 = time.perf_counter_ns()
        for _ in range(inner):
            fn(x)
        samples[i] = (time.perf_counter_ns() - t0) / inner / 1e3
    return {
        "median_us": float(np.median(samples)),
        "p10_us": float(np.percentile(samples, 10)),
        "p90_us": float(np.percentile(samples, 90)),
        "repeats": repeats,
        "inner": inner,
    }
