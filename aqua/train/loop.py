"""Mini-batched training loop with hooks (used by the pruner)."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from typing import Protocol

import numpy as np

from aqua.engine import Tensor, no_grad, softmax_cross_entropy
from aqua.nn import MLP
from aqua.optim import SGD, Adam, Optimizer

from .data import Dataset, Split, iterate_minibatches


@dataclass
class TrainConfig:
    epochs: int = 60
    batch_size: int = 64
    lr: float = 2e-3
    lr_schedule: str = "cosine"  # "constant" | "cosine"
    min_lr_frac: float = 0.05
    warmup_steps: int = 50
    weight_decay: float = 0.0
    optimizer: str = "adam"  # "adam" | "sgd"
    momentum: float = 0.9
    betas: tuple[float, float] = (0.9, 0.999)
    bias_correction: str = "per_entry"
    mask_aware: bool = True
    seed: int = 0
    log_every_epoch: bool = False


class Hook(Protocol):
    def on_train_begin(self, trainer: "Trainer") -> None: ...
    def after_backward(self, trainer: "Trainer") -> None: ...
    def after_step(self, trainer: "Trainer") -> None: ...


def evaluate(model: MLP, split: Split, batch_size: int = 1024) -> tuple[float, float]:
    """Mean cross-entropy and accuracy, without building a graph."""
    total_loss, correct = 0.0, 0
    with no_grad():
        for start in range(0, len(split), batch_size):
            X, y = split.X[start : start + batch_size], split.y[start : start + batch_size]
            logits = model(Tensor(X))
            total_loss += softmax_cross_entropy(logits, y, reduction="sum").item()
            correct += int((logits.data.argmax(axis=1) == y).sum())
    return total_loss / len(split), correct / len(split)


def build_optimizer(model: MLP, cfg: TrainConfig, *, track_updates: bool = False) -> Optimizer:
    if cfg.optimizer == "adam":
        return Adam(
            model.parameters(), lr=cfg.lr, betas=cfg.betas, weight_decay=cfg.weight_decay,
            bias_correction=cfg.bias_correction, mask_aware=cfg.mask_aware, track_updates=track_updates,
        )
    if cfg.optimizer == "sgd":
        return SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay,
                   mask_aware=cfg.mask_aware, track_updates=track_updates)
    raise ValueError(f"unknown optimizer {cfg.optimizer!r}")


@dataclass
class Trainer:
    model: MLP
    data: Dataset
    cfg: TrainConfig
    hooks: list = field(default_factory=list)
    optimizer: Optimizer | None = None

    def __post_init__(self) -> None:
        if self.optimizer is None:
            self.optimizer = build_optimizer(self.model, self.cfg)
        self.rng = np.random.default_rng(self.cfg.seed + 10_000)  # data order stream
        self.step = 0
        self.steps_per_epoch = len(self.data.train) // self.cfg.batch_size
        self.total_steps = self.steps_per_epoch * self.cfg.epochs
        self.history: list[dict] = []
        self.batch_size = self.cfg.batch_size
        self.last_loss: float = float("nan")

    def lr_at(self, step: int) -> float:
        cfg = self.cfg
        if step < cfg.warmup_steps:
            return cfg.lr * (step + 1) / cfg.warmup_steps
        if cfg.lr_schedule == "constant":
            return cfg.lr
        progress = min(1.0, (step - cfg.warmup_steps) / max(1, self.total_steps - cfg.warmup_steps))
        floor = cfg.lr * cfg.min_lr_frac
        return floor + 0.5 * (cfg.lr - floor) * (1.0 + math.cos(math.pi * progress))

    def fit(self) -> list[dict]:
        for hook in self.hooks:
            hook.on_train_begin(self)
        self.model.train()
        for epoch in range(1, self.cfg.epochs + 1):
            t0 = time.perf_counter()
            losses, correct, seen = [], 0, 0
            for X, y in iterate_minibatches(self.data.train, self.cfg.batch_size, self.rng):
                self.optimizer.lr = self.lr_at(self.step)
                logits = self.model(Tensor(X))
                loss = softmax_cross_entropy(logits, y)
                if not np.isfinite(loss.item()):
                    raise FloatingPointError(f"non-finite loss {loss.item()} at step {self.step}")
                self.optimizer.zero_grad()
                loss.backward()
                for hook in self.hooks:
                    hook.after_backward(self)
                self.optimizer.step()
                self.step += 1
                for hook in self.hooks:
                    hook.after_step(self)
                self.last_loss = loss.item()
                losses.append(self.last_loss)
                correct += int((logits.data.argmax(axis=1) == y).sum())
                seen += len(y)
            val_loss, val_acc = evaluate(self.model, self.data.val)
            record = {
                "epoch": epoch,
                "step": self.step,
                "train_loss": float(np.mean(losses)),
                "train_acc": correct / seen,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "sparsity": self.model.sparsity(),
                "lr": self.optimizer.lr,
                "epoch_seconds": time.perf_counter() - t0,
            }
            self.history.append(record)
            if self.cfg.log_every_epoch:
                print(
                    f"epoch {epoch:3d} | train loss {record['train_loss']:.4f} acc {record['train_acc']:.4f} | "
                    f"val loss {val_loss:.4f} acc {val_acc:.4f} | sparsity {record['sparsity']:.3f}",
                    flush=True,
                )
        return self.history


def config_dict(cfg: TrainConfig) -> dict:
    d = asdict(cfg)
    d["betas"] = list(cfg.betas)
    return d
