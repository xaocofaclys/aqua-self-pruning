"""Datasets and mini-batching.

scikit-learn is used **only** to load the bundled UCI digits dataset
(``sklearn.datasets.load_digits``) - never for models, training or gradients.
The digits data ships inside the scikit-learn wheel, so no network access is
needed and every clone sees byte-identical data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass
class Split:
    X: np.ndarray
    y: np.ndarray

    def __len__(self) -> int:
        return len(self.y)


@dataclass
class Dataset:
    train: Split
    val: Split
    test: Split
    n_features: int
    n_classes: int
    name: str


def _stratified_split(y: np.ndarray, fractions: tuple[float, float], rng: np.random.Generator):
    """Return index arrays (train, val, test) with per-class proportions."""
    val_frac, test_frac = fractions
    train_idx, val_idx, test_idx = [], [], []
    for cls in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == cls))
        n_test = int(round(test_frac * len(idx)))
        n_val = int(round(val_frac * len(idx)))
        test_idx.append(idx[:n_test])
        val_idx.append(idx[n_test : n_test + n_val])
        train_idx.append(idx[n_test + n_val :])
    return tuple(np.sort(np.concatenate(parts)) for parts in (train_idx, val_idx, test_idx))


def _standardize(train: np.ndarray, *others: np.ndarray):
    mu = train.mean(axis=0)
    sd = train.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)  # constant (always-blank) pixels
    return [(a - mu) / sd for a in (train, *others)]


def load_digits_dataset(split_seed: int = 0, val_frac: float = 0.2, test_frac: float = 0.2) -> Dataset:
    """UCI handwritten digits: 1797 8x8 images, 10 classes.

    The split depends only on ``split_seed`` (not the model seed), so every
    model seed is evaluated on exactly the same held-out test set.
    """
    from sklearn.datasets import load_digits  # data loading only

    raw = load_digits()
    X = raw.data.astype(np.float64) / 16.0
    y = raw.target.astype(np.int64)
    tr, va, te = _stratified_split(y, (val_frac, test_frac), np.random.default_rng(split_seed))
    Xtr, Xva, Xte = _standardize(X[tr], X[va], X[te])
    return Dataset(Split(Xtr, y[tr]), Split(Xva, y[va]), Split(Xte, y[te]), X.shape[1], 10, "digits")


def make_spirals(n_per_class: int = 300, n_classes: int = 3, noise: float = 0.2, seed: int = 0) -> Dataset:
    """Classic interleaved-spirals toy problem (not linearly separable)."""
    rng = np.random.default_rng(seed)
    X, y = [], []
    for c in range(n_classes):
        r = np.linspace(0.05, 1.0, n_per_class)
        theta = np.linspace(c * 2 * np.pi / n_classes, c * 2 * np.pi / n_classes + 4.0, n_per_class)
        theta = theta + rng.standard_normal(n_per_class) * noise
        X.append(np.stack([r * np.sin(theta), r * np.cos(theta)], axis=1))
        y.append(np.full(n_per_class, c))
    X, y = np.concatenate(X), np.concatenate(y).astype(np.int64)
    tr, va, te = _stratified_split(y, (0.2, 0.2), rng)
    Xtr, Xva, Xte = _standardize(X[tr], X[va], X[te])
    return Dataset(Split(Xtr, y[tr]), Split(Xva, y[va]), Split(Xte, y[te]), 2, n_classes, "spirals")


def load_dataset(name: str, split_seed: int = 0) -> Dataset:
    if name == "digits":
        return load_digits_dataset(split_seed)
    if name == "spirals":
        return make_spirals(seed=split_seed)
    raise ValueError(f"unknown dataset {name!r}")


def iterate_minibatches(split: Split, batch_size: int, rng: np.random.Generator, *, drop_last: bool = True) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Shuffled mini-batches.

    ``drop_last`` keeps every step's batch size identical, which matters for
    the per-example Fisher estimate (it rescales by the batch size).
    """
    order = rng.permutation(len(split))
    stop = len(order) - (len(order) % batch_size) if drop_last else len(order)
    for start in range(0, stop, batch_size):
        idx = order[start : start + batch_size]
        yield split.X[idx], split.y[idx]
