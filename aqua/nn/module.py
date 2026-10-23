"""Parameters and composable modules."""

from __future__ import annotations

import weakref
from typing import Callable, Iterator

import numpy as np

from aqua.engine import Tensor

MaskListener = Callable[["Parameter", np.ndarray, np.ndarray], None]


class Parameter(Tensor):
    """A trainable leaf tensor with an optional binary connectivity mask.

    The mask is the single source of truth for which entries exist.  Changing
    it goes through :meth:`set_mask`, which (a) writes a *real* 0.0 into every
    removed entry and (b) notifies subscribers (optimizers) so that their
    per-entry state can never silently drift out of sync with the topology.
    """

    __slots__ = ("mask", "mask_version", "_listeners")

    def __init__(self, data, *, name: str | None = None, prunable: bool = False) -> None:
        super().__init__(np.array(data, dtype=np.float64, copy=True), requires_grad=True, name=name)
        self.mask: np.ndarray | None = self._frozen(np.ones(self.shape, dtype=bool)) if prunable else None
        self.mask_version = 0
        self._listeners: list[weakref.WeakMethod | Callable] = []

    @staticmethod
    def _frozen(mask: np.ndarray) -> np.ndarray:
        # Read-only: an in-place edit (``w.mask[i, j] = False``) would bypass
        # set_mask, so the forward pass would keep using a stale cached mask, the
        # weight would not be zeroed and the optimizer would not reset its state.
        # Make that mistake raise instead of silently corrupting training.
        mask.flags.writeable = False
        return mask

    @property
    def prunable(self) -> bool:
        return self.mask is not None

    def add_mask_listener(self, callback: MaskListener) -> None:
        ref = weakref.WeakMethod(callback) if hasattr(callback, "__self__") else (lambda cb=callback: cb)
        self._listeners.append(ref)

    def set_mask(self, new_mask: np.ndarray) -> None:
        if self.mask is None:
            raise RuntimeError(f"parameter {self.name!r} is not prunable")
        new_mask = np.asarray(new_mask, dtype=bool)
        if new_mask.shape != self.shape:
            raise ValueError(f"mask shape {new_mask.shape} != parameter shape {self.shape}")
        old_mask = self.mask
        self.mask = self._frozen(new_mask.copy())
        self.data[~self.mask] = 0.0  # a genuine zero, not a near-zero weight
        self.mask_version += 1
        alive = []
        for ref in self._listeners:
            cb = ref()
            if cb is not None:
                cb(self, old_mask, self.mask)
                alive.append(ref)
        self._listeners = alive

    def density(self) -> float:
        return 1.0 if self.mask is None else float(self.mask.mean())


class Module:
    """Minimal module base: recursive parameter discovery + train/eval flag."""

    training: bool = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):  # pragma: no cover - abstract
        raise NotImplementedError

    def children(self) -> Iterator["Module"]:
        for value in vars(self).values():
            if isinstance(value, Module):
                yield value
            elif isinstance(value, (list, tuple)):
                yield from (v for v in value if isinstance(v, Module))

    def modules(self) -> Iterator["Module"]:
        yield self
        for child in self.children():
            yield from child.modules()

    def named_parameters(self, prefix: str = "") -> Iterator[tuple[str, Parameter]]:
        seen: set[int] = set()
        for mod_name, module in self._named_modules(prefix):
            for attr, value in vars(module).items():
                if isinstance(value, Parameter) and id(value) not in seen:
                    seen.add(id(value))
                    yield (f"{mod_name}.{attr}" if mod_name else attr), value

    def _named_modules(self, prefix: str) -> Iterator[tuple[str, "Module"]]:
        yield prefix, self
        for attr, value in vars(self).items():
            items = [(attr, value)] if isinstance(value, Module) else (
                [(f"{attr}.{i}", v) for i, v in enumerate(value) if isinstance(v, Module)]
                if isinstance(value, (list, tuple)) else []
            )
            for name, child in items:
                yield from child._named_modules(f"{prefix}.{name}" if prefix else name)

    def parameters(self) -> list[Parameter]:
        return [p for _, p in self.named_parameters()]

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = None

    def train(self, mode: bool = True) -> "Module":
        for m in self.modules():
            m.training = mode
        return self

    def eval(self) -> "Module":
        return self.train(False)

    def state_dict(self) -> dict[str, np.ndarray]:
        state = {}
        for name, p in self.named_parameters():
            state[name] = p.data.copy()
            if p.mask is not None:
                state[f"{name}.__mask__"] = p.mask.copy()
        return state

    def load_state_dict(self, state: dict[str, np.ndarray]) -> None:
        for name, p in self.named_parameters():
            p.data[...] = state[name]
            mask_key = f"{name}.__mask__"
            if p.mask is not None and mask_key in state:
                p.set_mask(state[mask_key])
