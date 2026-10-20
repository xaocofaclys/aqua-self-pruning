"""Data loading and the training loop."""

from .data import Dataset, Split, iterate_minibatches, load_dataset, load_digits_dataset, make_spirals
from .loop import Trainer, TrainConfig, build_optimizer, config_dict, evaluate

__all__ = [
    "Dataset", "Split", "Trainer", "TrainConfig", "build_optimizer", "config_dict", "evaluate",
    "iterate_minibatches", "load_dataset", "load_digits_dataset", "make_spirals",
]
