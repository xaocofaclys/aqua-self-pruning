"""Gradual sparsity schedules."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CubicSchedule:
    """Zhu & Gupta (2017) polynomial ramp:

        s(t) = s_f + (s_i - s_f) * (1 - (t - t0) / (t1 - t0))^3,   t0 <= t <= t1

    Why cubic: ds/dt is largest at t0, while the network is still highly
    redundant and recovers quickly, and goes to 0 at t1 so that the last
    (most damaging) removals are small and well separated in time.  The mask
    is frozen after t1, which leaves a fine-tuning phase at a fixed topology.
    """

    final_sparsity: float
    start_step: int
    end_step: int
    update_every: int
    initial_sparsity: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.initial_sparsity <= self.final_sparsity < 1.0:
            raise ValueError("need 0 <= initial <= final < 1")
        if self.end_step < self.start_step or self.update_every < 1:
            raise ValueError("invalid schedule window")

    def sparsity(self, step: int) -> float:
        if step <= self.start_step:
            return self.initial_sparsity
        if step >= self.end_step:
            return self.final_sparsity
        frac = (step - self.start_step) / (self.end_step - self.start_step)
        return self.final_sparsity + (self.initial_sparsity - self.final_sparsity) * (1.0 - frac) ** 3

    def is_update_step(self, step: int) -> bool:
        if self.final_sparsity == 0.0:
            return False
        if step == self.end_step:
            return True  # always land exactly on the budget
        return self.start_step <= step < self.end_step and (step - self.start_step) % self.update_every == 0
