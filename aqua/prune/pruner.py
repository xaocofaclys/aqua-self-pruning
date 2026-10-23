"""The self-pruning controller (a training-loop hook).

Per training step:
    forward -> backward -> Pruner.after_backward (collect statistics)
            -> optimizer.step -> Pruner.after_step (maybe change the masks)

Mask changes go through ``Parameter.set_mask`` only, which hard-zeroes removed
weights and makes the optimizer reset state for every entry whose existence
changed.  The pruner itself never touches optimizer internals.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np

from aqua.nn import MLP, MaskedLinear

from .criteria import CRITERIA, LayerStats, importance, needs_fisher, per_example_fisher, regrowth_score
from .schedule import CubicSchedule


@dataclass
class PruneConfig:
    criterion: str = "fisher_obd"
    final_sparsity: float = 0.9
    start_frac: float = 0.1       # ramp start, as a fraction of total training steps
    end_frac: float = 0.65        # ramp end; mask frozen afterwards (fine-tuning)
    update_every: int = 16        # steps between mask updates (~1 epoch here)
    scope: str = "global"         # "global": one ranking in loss units; "layer": same sparsity per layer
    min_layer_density: float = 0.005  # floor per layer for global scope
    stats_beta: float = 0.9       # EMA decay of gradient statistics
    regrow: bool = False
    regrow_frac: float = 0.2      # fraction of a layer's live weights recycled per update, cosine-decayed
    regrow_end_frac: float = 0.8

    def __post_init__(self) -> None:
        if self.criterion not in CRITERIA:
            raise ValueError(f"criterion must be one of {CRITERIA}")
        if self.scope not in ("layer", "global"):
            raise ValueError("scope must be 'layer' or 'global'")
        if not 0.0 <= self.final_sparsity < 1.0:
            raise ValueError("final_sparsity must be in [0, 1)")
        if not 0.0 <= self.start_frac <= self.end_frac <= 1.0:
            raise ValueError("need 0 <= start_frac <= end_frac <= 1")


@dataclass
class PruneEvent:
    step: int
    target: float
    achieved: float
    pruned: int
    grown: int


class Pruner:
    def __init__(self, model: MLP, cfg: PruneConfig) -> None:
        self.model, self.cfg = model, cfg
        self.layers: list[MaskedLinear] = [l for l in model.linears if l.weight.prunable]
        if not self.layers:
            raise ValueError("model has no prunable layers")
        self.stats = [LayerStats.zeros(l.weight.shape, cfg.stats_beta) for l in self.layers]
        self.grown_at = [np.full(l.weight.shape, -(10**9), dtype=np.int64) for l in self.layers]
        self.events: list[PruneEvent] = []
        self.last_grown: list[tuple[int, np.ndarray]] = []  # (layer index, flat indices) of the last regrowth
        self.schedule: CubicSchedule | None = None
        self.total_grown = 0

    # ----------------------------------------------------------- hook protocol
    def on_train_begin(self, trainer) -> None:
        total = trainer.total_steps
        cfg = self.cfg
        self.schedule = CubicSchedule(
            final_sparsity=cfg.final_sparsity,
            start_step=int(round(cfg.start_frac * total)),
            end_step=int(round(cfg.end_frac * total)),
            update_every=cfg.update_every,
        )
        self.regrow_end = int(round(cfg.regrow_end_frac * total)) if cfg.regrow else -1
        self.last_mask_step = max(self.schedule.end_step, self.regrow_end)

    def _collecting(self, step: int) -> bool:
        if self.cfg.final_sparsity == 0.0:
            return False
        # Warm the EMA up for ~one window before the first update; stop after the last.
        return self.schedule.start_step - 2 * self.cfg.update_every <= step <= self.last_mask_step

    def after_backward(self, trainer) -> None:
        if self.cfg.criterion == "magnitude" and not self.cfg.regrow:
            return  # magnitude needs no gradient statistics
        if not self._collecting(trainer.step + 1):
            return
        want_fisher = needs_fisher(self.cfg.criterion)
        for layer, st in zip(self.layers, self.stats):
            dense = layer.last_w_eff.grad
            if dense is None:
                raise RuntimeError("no gradient on W_eff; did backward run?")
            fisher = per_example_fisher(layer.last_input, layer.last_preact.grad) if want_fisher else None
            st.update(layer.weight.data, dense, fisher)

    def after_step(self, trainer) -> None:
        step = trainer.step
        self.last_grown = []
        sched = self.schedule
        pruned = grown = 0
        changed = False
        if sched.is_update_step(step):
            pruned = self._prune_to(sched.sparsity(step), step)
            changed = True
        if self.cfg.regrow and sched.start_step < step <= self.regrow_end and (step - sched.start_step) % self.cfg.update_every == 0:
            grown = self._regrow(step)
            changed = True
        if changed:
            self.events.append(PruneEvent(step, sched.sparsity(step), self.model.sparsity(), pruned, grown))

    # ---------------------------------------------------------------- pruning
    def _scores(self, i: int) -> np.ndarray:
        layer = self.layers[i]
        s = importance(self.cfg.criterion, layer.weight.data, self.stats[i])
        return np.nan_to_num(s, nan=-np.inf)

    def _protected(self, i: int, step: int) -> np.ndarray:
        """Freshly grown weights start at 0 and score ~0: shield them through the
        NEXT mask update, so each gets one full window of training before it is
        judged.  Mask updates are exactly ``update_every`` steps apart, so at the
        next update a weight's age equals ``update_every``: the comparison must be
        ``<=``.  (A strict ``<`` never protects anything; it was the original bug,
        and it let ~79% of regrown weights be deleted after only 16 steps.)"""
        return (step - self.grown_at[i]) <= self.cfg.update_every

    @staticmethod
    def _rank_alive(score: np.ndarray, alive: np.ndarray, protected: np.ndarray) -> np.ndarray:
        """Flat indices of alive entries, least important first (protected last)."""
        idx = np.flatnonzero(alive)
        order = np.lexsort((idx, score.ravel()[idx], protected.ravel()[idx]))
        return idx[order]

    def _prune_to(self, target: float, step: int) -> int:
        if self.cfg.scope == "layer":
            keep_counts = [l.weight.size - int(round(target * l.weight.size)) for l in self.layers]
            ranked = [self._rank_alive(self._scores(i), l.weight.mask, self._protected(i, step)) for i, l in enumerate(self.layers)]
        else:
            keep_counts, ranked = self._global_allocation(target, step)
        removed = 0
        for layer, k_keep, order in zip(self.layers, keep_counts, ranked):
            n_drop = len(order) - k_keep
            if n_drop <= 0:
                continue
            mask = layer.weight.mask.copy()
            mask.ravel()[order[:n_drop]] = False
            layer.weight.set_mask(mask)
            removed += n_drop
        return removed

    def _global_allocation(self, target: float, step: int):
        sizes = [l.weight.size for l in self.layers]
        k_total = sum(sizes) - int(round(target * sum(sizes)))
        ranked = [self._rank_alive(self._scores(i), l.weight.mask, self._protected(i, step)) for i, l in enumerate(self.layers)]
        floors = [min(len(r), math.ceil(self.cfg.min_layer_density * n)) for r, n in zip(ranked, sizes)]
        if sum(floors) > k_total:
            # Floor infeasible under this budget: fall back to proportional floors
            # so the hard budget always wins.
            floors = [min(len(r), (k_total * n) // sum(sizes)) for r, n in zip(ranked, sizes)]
        # Pool everything above each layer's floor and take the globally best.
        pool_scores, pool_layer = [], []
        for i, (order, floor) in enumerate(zip(ranked, floors)):
            rest = order[: len(order) - floor]
            sc = self._scores(i).ravel()[rest]
            prot = self._protected(i, step).ravel()[rest]
            pool_scores.append(np.where(prot, np.inf, sc))
            pool_layer.append(np.full(len(rest), i))
        scores = np.concatenate(pool_scores) if pool_scores else np.zeros(0)
        owner = np.concatenate(pool_layer) if pool_layer else np.zeros(0, dtype=int)
        n_extra = max(0, k_total - sum(floors))
        keep_extra = np.zeros(len(self.layers), dtype=int)
        if n_extra and len(scores):
            best = np.argsort(-scores, kind="stable")[:n_extra]
            keep_extra = np.bincount(owner[best], minlength=len(self.layers))
        keep_counts = [f + int(e) for f, e in zip(floors, keep_extra)]
        return keep_counts, ranked

    # --------------------------------------------------------------- regrowth
    def _regrow_fraction(self, step: int) -> float:
        start = self.schedule.start_step
        progress = (step - start) / max(1, self.regrow_end - start)
        return self.cfg.regrow_frac * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    def _regrow(self, step: int) -> int:
        """Drop-and-grow at constant density (RigL): sparsity budget unchanged."""
        alpha = self._regrow_fraction(step)
        total = 0
        for i, layer in enumerate(self.layers):
            mask = layer.weight.mask.copy()
            n_alive = int(mask.sum())
            k = int(alpha * n_alive)
            dead = np.flatnonzero(~mask.ravel())
            k = min(k, len(dead))
            if k == 0:
                continue
            order = self._rank_alive(self._scores(i), mask, self._protected(i, step))
            drop = order[:k]
            grow_score = regrowth_score(self.stats[i]).ravel()[dead]
            grow = dead[np.argsort(-grow_score, kind="stable")[:k]]
            mask.ravel()[drop] = False
            mask.ravel()[grow] = True
            layer.weight.set_mask(mask)  # grown entries start at exactly 0 with fresh optimizer state
            self.grown_at[i].ravel()[grow] = step
            self.last_grown.append((i, grow))
            total += k
        self.total_grown += total
        return total

    # ---------------------------------------------------------------- report
    def summary(self) -> dict:
        per_layer = {l.weight.name: 1.0 - l.weight.density() for l in self.layers}
        return {
            "config": asdict(self.cfg),
            "target_sparsity": self.cfg.final_sparsity,
            "achieved_sparsity": self.model.sparsity(),
            "per_layer_sparsity": per_layer,
            "n_mask_updates": len(self.events),
            "total_regrown": self.total_grown,
            "schedule": {"start_step": self.schedule.start_step, "end_step": self.schedule.end_step,
                         "update_every": self.schedule.update_every} if self.schedule else None,
        }
