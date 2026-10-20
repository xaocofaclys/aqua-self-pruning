"""Layers: a maskable Linear layer, activations and an MLP."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from aqua.engine import Tensor, as_tensor, gelu, is_grad_enabled, relu, sigmoid, tanh

from .module import Module, Parameter

# Gain for the activation that *follows* the layer (He/Kaiming: Var(W) = gain^2 / fan_in).
_GAINS = {"relu": np.sqrt(2.0), "gelu": np.sqrt(2.0), "tanh": 5.0 / 3.0, "sigmoid": 1.0, "linear": 1.0}


def init_weight(fan_in: int, fan_out: int, nonlinearity: str, rng: np.random.Generator) -> np.ndarray:
    """Variance-preserving init.

    * ReLU/GELU: He-normal, std = sqrt(2 / fan_in).  ReLU zeroes half its input,
      halving the second moment; the factor 2 compensates, so activations keep
      unit scale through depth (He et al., 2015).
    * tanh: gain 5/3 (the PyTorch convention) to offset tanh's contraction.
    * Output (linear) layer: std = 1/sqrt(fan_in) -> logits start O(1), so the
      initial loss is ~log(C) and softmax is not saturated on step 0.
    """
    std = _GAINS[nonlinearity] / np.sqrt(fan_in)
    return rng.standard_normal((fan_in, fan_out)) * std


class MaskedLinear(Module):
    """y = x @ (W * M) + b, with W stored as (in_features, out_features).

    The mask multiplies the weight *inside the graph*.  Consequences:

    * Forward: a masked connection contributes exactly 0, regardless of what
      is stored in W (W is also hard-zeroed by ``Parameter.set_mask``).
    * dL/dW = M * dL/dW_eff: the exact derivative of the function actually
      computed, so finite differences agree with it (a test checks this).
    * dL/dW_eff (kept on ``last_w_eff.grad``) is the *dense* gradient: what the
      loss slope would be if the connection existed.  This is what regrowth
      needs.  Two different quantities, used for two different jobs.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rng: np.random.Generator,
        *,
        nonlinearity: str = "relu",
        bias: bool = True,
        prunable: bool = True,
        name: str = "linear",
    ) -> None:
        self.in_features, self.out_features = in_features, out_features
        self.weight = Parameter(init_weight(in_features, out_features, nonlinearity, rng), name=f"{name}.weight", prunable=prunable)
        self.bias = Parameter(np.zeros(out_features), name=f"{name}.bias") if bias else None
        self._mask_tensor: Tensor | None = None
        self._mask_version = -1
        # Cached for pruning statistics (only populated while grad is enabled).
        self.last_input: np.ndarray | None = None
        self.last_w_eff: Tensor | None = None
        self.last_preact: Tensor | None = None

    def _mask(self) -> Tensor | None:
        if self.weight.mask is None:
            return None
        if self._mask_version != self.weight.mask_version:
            self._mask_tensor = Tensor(self.weight.mask.astype(self.weight.dtype))
            self._mask_version = self.weight.mask_version
        return self._mask_tensor

    def forward(self, x) -> Tensor:
        x = as_tensor(x)
        mask = self._mask()
        w_eff = self.weight * mask if mask is not None else self.weight
        preact = x @ w_eff
        out = preact + self.bias if self.bias is not None else preact
        if is_grad_enabled():
            self.last_input, self.last_w_eff, self.last_preact = x.data, w_eff, preact
        return out

    def effective_weight(self) -> np.ndarray:
        w = self.weight.data
        return w if self.weight.mask is None else w * self.weight.mask

    def __repr__(self) -> str:
        return f"MaskedLinear({self.in_features}, {self.out_features}, density={self.weight.density():.3f})"


class Activation(Module):
    _FNS = {"relu": relu, "tanh": tanh, "sigmoid": sigmoid, "gelu": gelu}

    def __init__(self, kind: str) -> None:
        if kind not in self._FNS:
            raise ValueError(f"unknown activation {kind!r}")
        self.kind = kind

    def forward(self, x: Tensor) -> Tensor:
        return self._FNS[self.kind](x)

    def numpy_forward(self, x: np.ndarray) -> np.ndarray:
        """Graph-free version for the inference path."""
        if self.kind == "relu":
            return np.maximum(x, 0.0)
        with_grad_off = self._FNS[self.kind](Tensor(x))
        return with_grad_off.data

    def __repr__(self) -> str:
        return f"Activation({self.kind})"


class Sequential(Module):
    def __init__(self, *layers: Module) -> None:
        self.layers = list(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def __iter__(self):
        return iter(self.layers)

    def __repr__(self) -> str:
        inner = ",\n  ".join(repr(l) for l in self.layers)
        return f"Sequential(\n  {inner}\n)"


class MLP(Sequential):
    """Linear -> act -> ... -> Linear (logits)."""

    def __init__(
        self,
        sizes: Sequence[int],
        rng: np.random.Generator,
        *,
        activation: str = "relu",
        prune_output_layer: bool = True,
    ) -> None:
        if len(sizes) < 2:
            raise ValueError("need at least input and output sizes")
        layers: list[Module] = []
        n = len(sizes) - 1
        for i, (fan_in, fan_out) in enumerate(zip(sizes[:-1], sizes[1:])):
            last = i == n - 1
            layers.append(
                MaskedLinear(
                    fan_in, fan_out, rng,
                    nonlinearity="linear" if last else activation,
                    prunable=prune_output_layer or not last,
                    name=f"fc{i}",
                )
            )
            if not last:
                layers.append(Activation(activation))
        super().__init__(*layers)
        self.sizes = list(sizes)
        self.activation = activation

    @property
    def linears(self) -> list[MaskedLinear]:
        return [l for l in self.layers if isinstance(l, MaskedLinear)]

    def prunable_weights(self) -> list[Parameter]:
        return [l.weight for l in self.linears if l.weight.prunable]

    def sparsity(self) -> float:
        """Fraction of prunable connections removed (biases excluded)."""
        weights = self.prunable_weights()
        total = sum(w.size for w in weights)
        alive = sum(int(w.mask.sum()) for w in weights)
        return 1.0 - alive / total if total else 0.0

    def count_parameters(self) -> dict[str, int]:
        weights = sum(l.weight.size for l in self.linears)
        active = sum(int(l.weight.mask.sum()) if l.weight.mask is not None else l.weight.size for l in self.linears)
        biases = sum(l.bias.size for l in self.linears if l.bias is not None)
        return {"weights_total": weights, "weights_active": active, "biases": biases}
