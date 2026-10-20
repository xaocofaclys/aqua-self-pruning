"""A minimal reverse-mode automatic differentiation engine on top of NumPy.

Design
------
* A :class:`Tensor` wraps an ``np.ndarray`` and, when it was produced by a
  differentiable operation, remembers its parents and a *pure* backward
  closure ``g_out -> (g_parent_0, g_parent_1, ...)``.
* The engine (``Tensor.backward``) is the only place that touches ``.grad``:
  it walks the graph in reverse topological order, *un-broadcasts* every
  parent gradient back to that parent's shape and accumulates it.  Individual
  operations therefore never have to think about broadcasting or
  accumulation, which keeps every op small enough to verify by eye.
* Every intermediate node that ``requires_grad`` keeps its gradient after
  ``backward``.  The pruning code relies on this: for a masked layer the
  gradient on the *effective* weight ``W_eff = W * M`` is the dense
  "what-if-this-connection-existed" signal used for regrowth (see DESIGN.md).

Masking is *not* special-cased anywhere in this file.  A mask is just a
constant tensor in the graph, so the gradient that reaches a stored weight is
the exact derivative of the function the network actually computes.
"""

from __future__ import annotations

import contextlib
from typing import Callable, Iterator, Sequence

import numpy as np

DEFAULT_DTYPE = np.float64

_grad_enabled: bool = True


@contextlib.contextmanager
def no_grad() -> Iterator[None]:
    """Disable graph construction (inference / evaluation)."""
    global _grad_enabled
    previous = _grad_enabled
    _grad_enabled = False
    try:
        yield
    finally:
        _grad_enabled = previous


def is_grad_enabled() -> bool:
    return _grad_enabled


def unbroadcast(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Reduce ``grad`` to ``shape`` by summing over broadcast dimensions.

    NumPy broadcasting (1) prepends size-1 dimensions and (2) stretches size-1
    dimensions.  The adjoint of "stretch" is "sum", so we sum over the
    prepended axes and over every axis where the original size was 1.
    Example: ``(N, D) + (D,)`` -> the bias gradient is ``grad.sum(axis=0)``.
    """
    if grad.shape == shape:
        return grad
    extra = grad.ndim - len(shape)
    if extra < 0:
        raise ValueError(f"cannot unbroadcast {grad.shape} to {shape}")
    if extra:
        grad = grad.sum(axis=tuple(range(extra)))
    axes = tuple(i for i, s in enumerate(shape) if s == 1 and grad.shape[i] != 1)
    if axes:
        grad = grad.sum(axis=axes, keepdims=True)
    return grad.reshape(shape)


BackwardFn = Callable[[np.ndarray], Sequence["np.ndarray | None"]]


class Tensor:
    """An n-dimensional array that participates in reverse-mode autodiff."""

    # Make ``ndarray (op) Tensor`` dispatch to Tensor's reflected operators.
    __array_priority__ = 1000

    __slots__ = ("data", "grad", "requires_grad", "_parents", "_backward", "op", "name", "__weakref__")

    def __init__(
        self,
        data,
        requires_grad: bool = False,
        *,
        dtype=None,
        name: str | None = None,
        _parents: tuple["Tensor", ...] = (),
        _backward: BackwardFn | None = None,
        _op: str = "",
    ) -> None:
        arr = np.asarray(data)
        if dtype is not None:
            arr = arr.astype(dtype, copy=False)
        elif not np.issubdtype(arr.dtype, np.floating):
            arr = arr.astype(DEFAULT_DTYPE)
        self.data: np.ndarray = arr
        self.grad: np.ndarray | None = None
        self.requires_grad: bool = bool(requires_grad)
        self._parents = _parents
        self._backward = _backward
        self.op = _op
        self.name = name

    # ------------------------------------------------------------------ basics
    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def is_leaf(self) -> bool:
        return self._backward is None

    def numpy(self) -> np.ndarray:
        return self.data

    def item(self) -> float:
        return float(self.data)

    def detach(self) -> "Tensor":
        return Tensor(self.data, requires_grad=False)

    def zero_grad(self) -> None:
        self.grad = None

    def __repr__(self) -> str:
        label = f" name={self.name!r}" if self.name else ""
        op = f" op={self.op}" if self.op else ""
        return f"Tensor(shape={self.shape}, dtype={self.dtype}{op}{label}, requires_grad={self.requires_grad})"

    def __len__(self) -> int:
        return len(self.data)

    # -------------------------------------------------------------- graph glue
    @staticmethod
    def _make(data: np.ndarray, parents: tuple["Tensor", ...], backward: BackwardFn, op: str) -> "Tensor":
        """Create an op output; record the graph only if a parent needs grad."""
        needs = _grad_enabled and any(p.requires_grad for p in parents)
        if needs:
            return Tensor(data, requires_grad=True, _parents=parents, _backward=backward, _op=op)
        return Tensor(data, requires_grad=False, _op=op)

    def _topo_order(self) -> list["Tensor"]:
        """Iterative post-order DFS (no recursion limit on deep graphs)."""
        order: list[Tensor] = []
        visited: set[int] = set()
        stack: list[tuple[Tensor, bool]] = [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                order.append(node)
                continue
            if id(node) in visited:
                continue
            visited.add(id(node))
            stack.append((node, True))
            for parent in node._parents:
                if parent.requires_grad and id(parent) not in visited:
                    stack.append((parent, False))
        return order

    def backward(self, grad: np.ndarray | None = None, *, retain_graph: bool = False) -> None:
        """Back-propagate from this tensor.

        Gradients *accumulate* into ``.grad`` of every node that requires grad
        (call ``zero_grad`` between steps).  With ``retain_graph=False`` the
        closures of interior nodes are released afterwards to free memory, but
        their ``.grad`` values are kept for inspection.
        """
        if not self.requires_grad:
            raise RuntimeError("backward() called on a tensor that does not require grad")
        if grad is None:
            if self.data.size != 1:
                raise RuntimeError("grad must be given for non-scalar outputs")
            grad = np.ones_like(self.data)
        grad = np.asarray(grad, dtype=self.data.dtype)
        if grad.shape != self.shape:
            raise ValueError(f"seed grad shape {grad.shape} != tensor shape {self.shape}")

        order = self._topo_order()
        # Interior nodes get their gradient fresh for this pass; leaves accumulate.
        for node in order:
            if not node.is_leaf:
                node.grad = None
        self.grad = grad.copy() if self.grad is None else self.grad + grad

        for node in reversed(order):
            if node._backward is None or node.grad is None:
                continue
            parent_grads = node._backward(node.grad)
            for parent, g in zip(node._parents, parent_grads):
                if g is None or not parent.requires_grad:
                    continue
                g = unbroadcast(np.asarray(g), parent.shape)
                if parent.grad is None:
                    # Own the buffer: ops may hand back views of node.grad.
                    parent.grad = np.array(g, dtype=parent.data.dtype, copy=True)
                else:
                    parent.grad = parent.grad + g
            if not retain_graph:
                node._backward = None
                node._parents = ()

    # ------------------------------------------------------------- arithmetic
    def __add__(self, other) -> "Tensor":
        return add(self, other)

    def __radd__(self, other) -> "Tensor":
        return add(other, self)

    def __sub__(self, other) -> "Tensor":
        return sub(self, other)

    def __rsub__(self, other) -> "Tensor":
        return sub(other, self)

    def __mul__(self, other) -> "Tensor":
        return mul(self, other)

    def __rmul__(self, other) -> "Tensor":
        return mul(other, self)

    def __truediv__(self, other) -> "Tensor":
        return div(self, other)

    def __rtruediv__(self, other) -> "Tensor":
        return div(other, self)

    def __neg__(self) -> "Tensor":
        return neg(self)

    def __pow__(self, exponent: float) -> "Tensor":
        return power(self, exponent)

    def __matmul__(self, other) -> "Tensor":
        return matmul(self, other)

    def __rmatmul__(self, other) -> "Tensor":
        return matmul(other, self)

    # ---------------------------------------------------------------- methods
    def sum(self, axis=None, keepdims: bool = False) -> "Tensor":
        return tsum(self, axis=axis, keepdims=keepdims)

    def mean(self, axis=None, keepdims: bool = False) -> "Tensor":
        return mean(self, axis=axis, keepdims=keepdims)

    def relu(self) -> "Tensor":
        return relu(self)

    def tanh(self) -> "Tensor":
        return tanh(self)

    def sigmoid(self) -> "Tensor":
        return sigmoid(self)

    def gelu(self) -> "Tensor":
        return gelu(self)

    def exp(self) -> "Tensor":
        return exp(self)

    def log(self) -> "Tensor":
        return log(self)

    def reshape(self, *shape) -> "Tensor":
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return reshape(self, shape)

    def transpose(self, *axes) -> "Tensor":
        return transpose(self, axes or None)

    @property
    def T(self) -> "Tensor":
        return transpose(self, None)


def as_tensor(x, dtype=None) -> Tensor:
    if isinstance(x, Tensor):
        return x
    return Tensor(x, requires_grad=False, dtype=dtype or DEFAULT_DTYPE)


# =============================================================================
# Operations.  Each returns per-parent gradients *before* un-broadcasting.
# =============================================================================


def add(a, b) -> Tensor:
    a, b = as_tensor(a), as_tensor(b)
    return Tensor._make(a.data + b.data, (a, b), lambda g: (g, g), "add")


def sub(a, b) -> Tensor:
    a, b = as_tensor(a), as_tensor(b)
    return Tensor._make(a.data - b.data, (a, b), lambda g: (g, -g), "sub")


def mul(a, b) -> Tensor:
    a, b = as_tensor(a), as_tensor(b)
    ad, bd = a.data, b.data
    return Tensor._make(ad * bd, (a, b), lambda g: (g * bd, g * ad), "mul")


def div(a, b) -> Tensor:
    a, b = as_tensor(a), as_tensor(b)
    ad, bd = a.data, b.data
    out = ad / bd

    def backward(g):
        ga = g / bd
        return ga, -ga * out

    return Tensor._make(out, (a, b), backward, "div")


def neg(a) -> Tensor:
    a = as_tensor(a)
    return Tensor._make(-a.data, (a,), lambda g: (-g,), "neg")


def power(a, exponent: float) -> Tensor:
    a = as_tensor(a)
    if isinstance(exponent, Tensor):
        raise TypeError("only scalar exponents are supported")
    p = float(exponent)
    ad = a.data
    return Tensor._make(ad**p, (a,), lambda g: (g * p * ad ** (p - 1.0),), "pow")


def matmul(a, b) -> Tensor:
    """Matrix product for operands with ndim >= 2 (batched via broadcasting)."""
    a, b = as_tensor(a), as_tensor(b)
    if a.ndim < 2 or b.ndim < 2:
        raise ValueError("matmul requires operands with ndim >= 2; reshape vectors explicitly")
    ad, bd = a.data, b.data

    def backward(g):
        # Batched leading dims (if any) are reduced by unbroadcast in the engine.
        return g @ np.swapaxes(bd, -1, -2), np.swapaxes(ad, -1, -2) @ g

    return Tensor._make(ad @ bd, (a, b), backward, "matmul")


def _normalize_axis(axis, ndim: int):
    if axis is None:
        return None
    axes = (axis,) if isinstance(axis, int) else tuple(axis)
    return tuple(ax % ndim for ax in axes)


def tsum(a, axis=None, keepdims: bool = False) -> Tensor:
    a = as_tensor(a)
    shape = a.shape
    axes = _normalize_axis(axis, a.ndim)

    def backward(g):
        if axes is not None and not keepdims:
            g = np.expand_dims(g, axes)
        return (np.broadcast_to(g, shape),)

    return Tensor._make(a.data.sum(axis=axes, keepdims=keepdims), (a,), backward, "sum")


def mean(a, axis=None, keepdims: bool = False) -> Tensor:
    a = as_tensor(a)
    axes = _normalize_axis(axis, a.ndim)
    count = a.size if axes is None else int(np.prod([a.shape[ax] for ax in axes]))
    return tsum(a, axis=axes, keepdims=keepdims) * (1.0 / count)


def relu(a) -> Tensor:
    a = as_tensor(a)
    positive = a.data > 0
    # Sub-gradient 0 at exactly 0 (the convention every framework uses).
    return Tensor._make(np.where(positive, a.data, 0.0), (a,), lambda g: (g * positive,), "relu")


def tanh(a) -> Tensor:
    a = as_tensor(a)
    t = np.tanh(a.data)
    return Tensor._make(t, (a,), lambda g: (g * (1.0 - t * t),), "tanh")


def _stable_sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def sigmoid(a) -> Tensor:
    a = as_tensor(a)
    s = _stable_sigmoid(a.data)
    return Tensor._make(s, (a,), lambda g: (g * s * (1.0 - s),), "sigmoid")


_GELU_K = np.sqrt(2.0 / np.pi)
_GELU_C = 0.044715


def gelu(a) -> Tensor:
    """GELU, tanh approximation (Hendrycks & Gimpel)."""
    a = as_tensor(a)
    x = a.data
    inner = _GELU_K * (x + _GELU_C * x**3)
    t = np.tanh(inner)
    out = 0.5 * x * (1.0 + t)

    def backward(g):
        d_inner = _GELU_K * (1.0 + 3.0 * _GELU_C * x * x)
        return (g * (0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * d_inner),)

    return Tensor._make(out, (a,), backward, "gelu")


def exp(a) -> Tensor:
    a = as_tensor(a)
    e = np.exp(a.data)
    return Tensor._make(e, (a,), lambda g: (g * e,), "exp")


def log(a) -> Tensor:
    a = as_tensor(a)
    ad = a.data
    return Tensor._make(np.log(ad), (a,), lambda g: (g / ad,), "log")


def reshape(a, shape) -> Tensor:
    a = as_tensor(a)
    original = a.shape
    return Tensor._make(a.data.reshape(shape), (a,), lambda g: (g.reshape(original),), "reshape")


def transpose(a, axes=None) -> Tensor:
    a = as_tensor(a)
    if axes is None:
        inverse = None
    else:
        axes = tuple(axes)
        inverse = tuple(np.argsort(axes))
    return Tensor._make(np.transpose(a.data, axes), (a,), lambda g: (np.transpose(g, inverse),), "transpose")


def log_softmax(logits, axis: int = -1) -> Tensor:
    """Numerically stable log-softmax: x - max - log(sum(exp(x - max)))."""
    logits = as_tensor(logits)
    x = logits.data
    shifted = x - x.max(axis=axis, keepdims=True)
    lse = np.log(np.exp(shifted).sum(axis=axis, keepdims=True))
    out = shifted - lse
    soft = np.exp(out)

    def backward(g):
        return (g - soft * g.sum(axis=axis, keepdims=True),)

    return Tensor._make(out, (logits,), backward, "log_softmax")


def softmax(logits, axis: int = -1) -> Tensor:
    logits = as_tensor(logits)
    x = logits.data
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    s = e / e.sum(axis=axis, keepdims=True)

    def backward(g):
        return (s * (g - (g * s).sum(axis=axis, keepdims=True)),)

    return Tensor._make(s, (logits,), backward, "softmax")


def softmax_cross_entropy(logits, targets, reduction: str = "mean") -> Tensor:
    """Fused, numerically stable softmax + cross-entropy.

    ``logits``: (N, C) tensor.  ``targets``: (N,) integer class indices.
    loss_n = logsumexp(z_n) - z_n[y_n]  and  dloss_n/dz_n = softmax(z_n) - onehot(y_n).
    The max-shift makes it overflow-free for arbitrarily large logits.
    """
    logits = as_tensor(logits)
    if logits.ndim != 2:
        raise ValueError("logits must be (N, C)")
    y = np.asarray(targets)
    if y.ndim != 1 or y.shape[0] != logits.shape[0] or not np.issubdtype(y.dtype, np.integer):
        raise ValueError("targets must be an (N,) integer array matching logits")
    n, c = logits.shape
    if y.min(initial=0) < 0 or y.max(initial=0) >= c:
        raise ValueError("target index out of range")
    z = logits.data
    shifted = z - z.max(axis=1, keepdims=True)
    lse = np.log(np.exp(shifted).sum(axis=1))
    rows = np.arange(n)
    per_example = lse - shifted[rows, y]
    probs = np.exp(shifted - lse[:, None])

    if reduction == "mean":
        out, scale = per_example.mean(), 1.0 / n
    elif reduction == "sum":
        out, scale = per_example.sum(), 1.0
    elif reduction == "none":
        out, scale = per_example, None
    else:
        raise ValueError(f"unknown reduction {reduction!r}")

    def backward(g):
        d = probs.copy()
        d[rows, y] -= 1.0
        if scale is None:
            return (d * g[:, None],)
        return (d * (g * scale),)

    return Tensor._make(np.asarray(out, dtype=z.dtype), (logits,), backward, "softmax_ce")
