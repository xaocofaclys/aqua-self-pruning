"""Sparse-aware inference: CSR kernels, structural compaction, cost accounting."""

from .benchmark import time_callable
from .csr import CSRMatrix
from .inference import (
    CompactEngine,
    CSREngine,
    DenseEngine,
    LayerSpec,
    compact_layers,
    dense_flops,
    export_layers,
    reference_forward,
)

__all__ = [
    "CSRMatrix", "CSREngine", "CompactEngine", "DenseEngine", "LayerSpec", "compact_layers",
    "dense_flops", "export_layers", "reference_forward", "time_callable",
]
