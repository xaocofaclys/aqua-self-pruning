"""One function that runs one fully specified, seeded experiment.

Every script (Part 2 training, Part 3 self-pruning, Part 4 sweep) goes
through :func:`run`, so a row in a results CSV can always be reproduced by
calling ``run`` with the config recorded next to it.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import numpy as np

from aqua.nn import MLP
from aqua.prune import PruneConfig, Pruner, RevivalMonitor
from aqua.train import Dataset, Trainer, TrainConfig, build_optimizer, config_dict, evaluate, load_dataset

DEFAULT_HIDDEN = (256, 256)


@dataclass
class ExperimentConfig:
    dataset: str = "digits"
    hidden: tuple[int, ...] = DEFAULT_HIDDEN
    activation: str = "relu"
    train: TrainConfig = field(default_factory=TrainConfig)
    prune: PruneConfig | None = None
    split_seed: int = 0
    monitor_revival: bool = False

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "hidden": list(self.hidden),
            "activation": self.activation,
            "train": config_dict(self.train),
            "prune": asdict(self.prune) if self.prune else None,
            "split_seed": self.split_seed,
        }


_DATA_CACHE: dict[tuple[str, int], Dataset] = {}


def get_dataset(name: str, split_seed: int) -> Dataset:
    key = (name, split_seed)
    if key not in _DATA_CACHE:
        _DATA_CACHE[key] = load_dataset(name, split_seed)
    return _DATA_CACHE[key]


def build_model(cfg: ExperimentConfig, data: Dataset) -> MLP:
    rng = np.random.default_rng(cfg.train.seed)  # model-init stream
    return MLP([data.n_features, *cfg.hidden, data.n_classes], rng, activation=cfg.activation)


def run(cfg: ExperimentConfig) -> dict:
    data = get_dataset(cfg.dataset, cfg.split_seed)
    model = build_model(cfg, data)
    optimizer = build_optimizer(model, cfg.train, track_updates=cfg.monitor_revival)
    hooks = []
    pruner = monitor = None
    if cfg.prune is not None:
        pruner = Pruner(model, cfg.prune)
        hooks.append(pruner)
        if cfg.monitor_revival:
            monitor = RevivalMonitor(pruner)
            hooks.append(monitor)
    trainer = Trainer(model, data, cfg.train, hooks=hooks, optimizer=optimizer)
    t0 = time.perf_counter()
    history = trainer.fit()
    train_seconds = time.perf_counter() - t0
    test_loss, test_acc = evaluate(model, data.test)
    val_loss, val_acc = evaluate(model, data.val)
    result = {
        "config": cfg.to_dict(),
        "test_acc": test_acc,
        "test_loss": test_loss,
        "val_acc": val_acc,
        "val_loss": val_loss,
        "sparsity": model.sparsity(),
        "params": model.count_parameters(),
        "train_seconds": train_seconds,
        "history": history,
    }
    if pruner is not None:
        result["prune_summary"] = pruner.summary()
        result["prune_events"] = [asdict(e) for e in pruner.events]
    if monitor is not None:
        result["revival"] = monitor.report()
    result["_model"] = model  # stripped before serialization
    return result


def serializable(result: dict) -> dict:
    return {k: v for k, v in result.items() if not k.startswith("_")}
