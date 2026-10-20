"""Graph-free inference engines for a trained (pruned) MLP, and FLOP accounting.

Three engines, all exactly equivalent to the training-time forward pass:

* ``DenseEngine``   - dense BLAS matmul with the masked weights.  This is the
  "dense x zeros" path: it does the *same* work at 0% and 99% sparsity and is
  therefore **not** evidence of any speedup.  Kept as the honest baseline.
* ``CSREngine``     - every layer as a CSR matrix: work proportional to nnz.
* ``CompactEngine`` - structured clean-up first: hidden units whose fan-in or
  fan-out was pruned away entirely are removed exactly (constant units are
  folded into the next bias), unused input features are dropped, then each
  layer picks dense or CSR by its *remaining* density.  This is what one
  would ship.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from aqua.engine import Tensor, no_grad
from aqua.nn import MLP, Activation, MaskedLinear

from .csr import CSRMatrix


def _act_fn(kind: str):
    if kind == "relu":
        return lambda x: np.maximum(x, 0.0)
    if kind == "tanh":
        return np.tanh
    if kind == "sigmoid":
        return lambda x: 0.5 * (1.0 + np.tanh(0.5 * x))
    if kind == "gelu":
        k = np.sqrt(2.0 / np.pi)
        return lambda x: 0.5 * x * (1.0 + np.tanh(k * (x + 0.044715 * x**3)))
    if kind == "linear":
        return lambda x: x
    raise ValueError(kind)


@dataclass
class LayerSpec:
    weight: np.ndarray  # (in, out), masked entries are exact zeros
    bias: np.ndarray    # (out,)
    activation: str     # applied after this layer ("linear" for logits)


def export_layers(model: MLP) -> list[LayerSpec]:
    specs: list[LayerSpec] = []
    layers = model.layers
    for i, layer in enumerate(layers):
        if isinstance(layer, MaskedLinear):
            nxt = layers[i + 1] if i + 1 < len(layers) else None
            act = nxt.kind if isinstance(nxt, Activation) else "linear"
            bias = layer.bias.data.copy() if layer.bias is not None else np.zeros(layer.out_features)
            specs.append(LayerSpec(layer.effective_weight().copy(), bias, act))
    return specs


def dense_flops(specs: list[LayerSpec]) -> int:
    """Multiply-adds counted as 2 FLOPs, per example (bias adds ignored)."""
    return sum(2 * s.weight.shape[0] * s.weight.shape[1] for s in specs)


class DenseEngine:
    name = "dense (masked weights, BLAS)"

    def __init__(self, specs: list[LayerSpec]) -> None:
        self.layers = [(np.ascontiguousarray(s.weight), s.bias, _act_fn(s.activation)) for s in specs]
        self.flops_per_example = dense_flops(specs)
        self.weight_bytes = sum(w.nbytes for w, _, _ in self.layers)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        for w, b, act in self.layers:
            x = act(x @ w + b)
        return x


class CSREngine:
    name = "csr (all layers sparse)"

    def __init__(self, specs: list[LayerSpec]) -> None:
        self.layers = [(CSRMatrix.from_dense(s.weight), s.bias, _act_fn(s.activation)) for s in specs]
        self.flops_per_example = sum(2 * m.nnz for m, _, _ in self.layers)
        self.weight_bytes = sum(m.nbytes() for m, _, _ in self.layers)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        for m, b, act in self.layers:
            x = act(m.matmul(x) + b)
        return x


def compact_layers(specs: list[LayerSpec]) -> tuple[np.ndarray, list[LayerSpec], dict]:
    """Remove dead structure exactly.  Returns (input_index, layers, report).

    A hidden unit j of layer l is dead if
      * fan-out is empty (row j of W_{l+1} is all zero): nothing reads it -> drop; or
      * fan-in is empty (column j of W_l is all zero): it outputs the constant
        a_j = act(b_j) -> add a_j * W_{l+1}[j, :] to b_{l+1}, then drop.
    Removing a unit can make others dead, so we iterate to a fixed point.
    Input features with empty fan-out are never read and are dropped too.
    """
    ws = [s.weight.copy() for s in specs]
    bs = [s.bias.copy() for s in specs]
    acts = [s.activation for s in specs]
    keep_in = np.arange(ws[0].shape[0])
    removed = {"dead_fan_out": 0, "constant_folded": 0, "inputs_dropped": 0}
    changed = True
    while changed:
        changed = False
        for l in range(len(ws) - 1):  # hidden layer = output of layer l
            fan_in_empty = ~ws[l].any(axis=0)
            fan_out_empty = ~ws[l + 1].any(axis=1)
            drop = fan_in_empty | fan_out_empty
            if not drop.any():
                continue
            fold = fan_in_empty & ~fan_out_empty
            if fold.any():
                const = _act_fn(acts[l])(bs[l][fold])
                bs[l + 1] = bs[l + 1] + const @ ws[l + 1][fold]
            removed["constant_folded"] += int(fold.sum())
            removed["dead_fan_out"] += int((drop & ~fold).sum())
            keep = ~drop
            ws[l], bs[l] = ws[l][:, keep], bs[l][keep]
            ws[l + 1] = ws[l + 1][keep, :]
            changed = True
        unused = ~ws[0].any(axis=1)
        if unused.any():
            removed["inputs_dropped"] += int(unused.sum())
            keep_in = keep_in[~unused]
            ws[0] = ws[0][~unused, :]
            changed = True
    layers = [LayerSpec(w, b, a) for w, b, a in zip(ws, bs, acts)]
    removed["shapes"] = [list(w.shape) for w in ws]
    return keep_in, layers, removed


class CompactEngine:
    name = "compacted + per-layer dense/CSR"

    def __init__(self, specs: list[LayerSpec], csr_below_density: float = 0.1) -> None:
        self.input_index, layers, self.report = compact_layers(specs)
        self.layers = []
        flops = 0
        self.weight_bytes = 0
        self.kernels = []
        for s in layers:
            nnz = int(np.count_nonzero(s.weight))
            density = nnz / max(1, s.weight.size)
            if density < csr_below_density:
                op = CSRMatrix.from_dense(s.weight)
                self.layers.append((op.matmul, s.bias, _act_fn(s.activation)))
                flops += 2 * nnz
                self.weight_bytes += op.nbytes()
                self.kernels.append("csr")
            else:
                w = np.ascontiguousarray(s.weight)
                self.layers.append((w.__rmatmul__, s.bias, _act_fn(s.activation)))
                flops += 2 * w.size
                self.weight_bytes += w.nbytes
                self.kernels.append("dense")
        self.flops_per_example = flops

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = x[:, self.input_index]
        for op, b, act in self.layers:
            x = act(op(x) + b)
        return x


def reference_forward(model: MLP, x: np.ndarray) -> np.ndarray:
    with no_grad():
        return model(Tensor(x)).data
