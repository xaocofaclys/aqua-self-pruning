"""Part 3: schedule, criteria statistics, budget enforcement and regrowth."""

import unittest

import numpy as np

from aqua.engine import Tensor, softmax_cross_entropy
from aqua.experiment import ExperimentConfig, run
from aqua.nn import MLP, MaskedLinear
from aqua.prune import CubicSchedule, PruneConfig, Pruner, importance, per_example_fisher
from aqua.prune.criteria import LayerStats
from aqua.train import TrainConfig


class Schedule(unittest.TestCase):
    def test_cubic_ramp_is_monotone_and_hits_target_exactly(self):
        s = CubicSchedule(final_sparsity=0.9, start_step=100, end_step=600, update_every=16)
        values = [s.sparsity(t) for t in range(0, 800)]
        self.assertTrue(all(b >= a for a, b in zip(values, values[1:])))
        self.assertEqual(s.sparsity(100), 0.0)
        self.assertEqual(s.sparsity(600), 0.9)
        self.assertTrue(s.is_update_step(600))
        self.assertFalse(s.is_update_step(601))
        # Fastest removal early, slowest late.
        self.assertGreater(s.sparsity(200) - s.sparsity(100), s.sparsity(600) - s.sparsity(500))


class FisherTrick(unittest.TestCase):
    def test_per_example_fisher_matches_brute_force(self):
        rng = np.random.default_rng(0)
        layer = MaskedLinear(5, 4, rng)
        x = rng.standard_normal((9, 5))
        y = rng.integers(0, 4, size=9)
        softmax_cross_entropy(layer(Tensor(x)), y).backward()
        fast = per_example_fisher(layer.last_input, layer.last_preact.grad)
        brute = np.zeros((5, 4))
        for n in range(9):
            single = MaskedLinear(5, 4, rng)
            single.weight.data[:] = layer.weight.data
            single.bias.data[:] = layer.bias.data
            softmax_cross_entropy(single(Tensor(x[n : n + 1])), y[n : n + 1]).backward()
            brute += single.weight.grad**2
        np.testing.assert_allclose(fast, brute / 9, rtol=1e-10)


class Criteria(unittest.TestCase):
    def test_criteria_rank_known_cases(self):
        st = LayerStats.zeros((3,), beta=0.0)
        w = np.array([1.0, 1.0, 0.1])
        st.update(w, dense_grad=np.array([0.0, 0.0, 0.0]), fisher_batch=np.array([1.0, 100.0, 100.0]))
        obd = importance("fisher_obd", w, st)
        # Same magnitude, higher curvature -> more important; magnitude can't tell them apart.
        self.assertGreater(obd[1], obd[0])
        self.assertEqual(importance("magnitude", w, st)[0], importance("magnitude", w, st)[1])

    def test_gradient_criteria_refuse_to_run_without_statistics(self):
        with self.assertRaises(RuntimeError):
            importance("saliency", np.ones(3), LayerStats.zeros((3,), 0.9))


def _quick(prune_cfg, epochs=8, **train_kw):
    cfg = ExperimentConfig(hidden=(32, 32), train=TrainConfig(epochs=epochs, seed=0, **train_kw), prune=prune_cfg)
    return run(cfg)


class PrunerEndToEnd(unittest.TestCase):
    def test_hard_budget_is_met_exactly_per_layer(self):
        for crit in ("magnitude", "saliency", "fisher_obd", "taylor2"):
            with self.subTest(criterion=crit):
                r = _quick(PruneConfig(criterion=crit, final_sparsity=0.9, scope="layer", update_every=4))
                model = r["_model"]
                for w in model.prunable_weights():
                    self.assertEqual(int((~w.mask).sum()), int(round(0.9 * w.size)))
                    self.assertTrue(np.all(w.data[~w.mask] == 0.0))

    def test_global_scope_meets_budget_and_respects_floor(self):
        r = _quick(PruneConfig(criterion="fisher_obd", final_sparsity=0.95, scope="global", update_every=4, min_layer_density=0.02))
        model = r["_model"]
        total = sum(w.size for w in model.prunable_weights())
        self.assertEqual(sum(int((~w.mask).sum()) for w in model.prunable_weights()), int(round(0.95 * total)))
        for w in model.prunable_weights():
            self.assertGreaterEqual(w.mask.mean(), 0.02 - 1e-9)
        # An infeasible floor (5% per layer under a 5% total budget) must not break the budget.
        r = _quick(PruneConfig(criterion="fisher_obd", final_sparsity=0.95, scope="global", update_every=4, min_layer_density=0.05))
        self.assertEqual(sum(int((~w.mask).sum()) for w in r["_model"].prunable_weights()), int(round(0.95 * total)))

    def test_regrowth_keeps_budget_and_actually_regrows(self):
        r = _quick(PruneConfig(criterion="fisher_obd", final_sparsity=0.9, scope="layer", update_every=4, regrow=True), epochs=10)
        self.assertGreater(r["prune_summary"]["total_regrown"], 0)
        self.assertAlmostEqual(r["sparsity"], 0.9, places=3)
        # After the ramp ends, every regrowth event keeps the sparsity exactly at the budget.
        end = r["prune_summary"]["schedule"]["end_step"]
        for ev in r["prune_events"]:
            if ev["step"] >= end:
                self.assertAlmostEqual(ev["achieved"], 0.9, places=3)

    def test_regrown_weights_survive_their_first_mask_update(self):
        """Regression test for the protection off-by-one: a regrown weight starts
        at 0 and scores ~0, so without protection it is deleted at the very next
        update after only `update_every` steps (79% were, in the 98% sweep runs)."""
        survival = []

        class Watch(Pruner):
            def after_step(self, trainer):
                prev = [(i, g.copy()) for i, g in getattr(self, "_cohort", [])]
                super().after_step(trainer)
                if self.last_grown:  # an update happened: did the previous cohort survive it?
                    if prev:
                        alive = sum(int(self.layers[i].weight.mask.ravel()[g].sum()) for i, g in prev)
                        survival.append(alive / sum(len(g) for _, g in prev))
                    self._cohort = self.last_grown

        import aqua.experiment as E
        orig = E.Pruner
        E.Pruner = Watch
        try:
            _quick(PruneConfig(criterion="fisher_obd", final_sparsity=0.9, scope="layer", update_every=4, regrow=True), epochs=10)
        finally:
            E.Pruner = orig
        self.assertTrue(survival, "no regrowth events observed")
        self.assertGreaterEqual(min(survival), 0.99, f"regrown weights deleted at their first update: {survival}")

    def test_protection_window_boundaries(self):
        model = MLP([4, 5, 3], np.random.default_rng(0))
        p = Pruner(model, PruneConfig(update_every=16, regrow=True))
        p.grown_at[0][0, 0] = 100
        self.assertTrue(p._protected(0, 116)[0, 0])    # the next update: protected
        self.assertFalse(p._protected(0, 132)[0, 0])   # the one after: judged normally

    def test_runs_are_bitwise_reproducible(self):
        a = _quick(PruneConfig(criterion="saliency", final_sparsity=0.8, update_every=4))
        b = _quick(PruneConfig(criterion="saliency", final_sparsity=0.8, update_every=4))
        self.assertEqual(a["test_acc"], b["test_acc"])
        for wa, wb in zip(a["_model"].prunable_weights(), b["_model"].prunable_weights()):
            np.testing.assert_array_equal(wa.data, wb.data)

    def test_training_is_stable(self):
        r = _quick(None, epochs=5)
        self.assertTrue(all(np.isfinite(h["train_loss"]) for h in r["history"]))
        self.assertLess(r["history"][-1]["train_loss"], r["history"][0]["train_loss"])


if __name__ == "__main__":
    unittest.main()
