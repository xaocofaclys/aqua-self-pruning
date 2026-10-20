"""Part 1 correctness requirement: masked weights and optimizer state.

What these tests pin down
-------------------------
1. A masked weight contributes *exactly* zero to the forward pass, whatever
   value is stored in it.
2. dL/dW for a masked entry is exactly 0 - and that is the *true* derivative of
   the masked network (finite differences agree), not a convention.
3. The dense gradient dL/dW_eff is still available for regrowth decisions.
4. Optimizer state of removed entries is exactly zero and they never move
   (no momentum "zombie drift", no decoupled weight decay leakage).
5. THE REVIVAL BUG: a connection revived at step T behaves bit-for-bit like a
   freshly created parameter started at step T.  The textbook Adam (single
   global step counter) fails this even when it resets m and v; we show the
   ~3.16x / ~6.5x over-sized steps it produces.
"""

import unittest

import numpy as np

from aqua.engine import Tensor, gradcheck, softmax_cross_entropy
from aqua.nn import MLP, MaskedLinear, Parameter
from aqua.optim import SGD, Adam


def _layer(seed=0, n_in=5, n_out=4):
    rng = np.random.default_rng(seed)
    layer = MaskedLinear(n_in, n_out, rng)
    layer.bias.data[:] = rng.standard_normal(n_out)
    mask = rng.random((n_in, n_out)) > 0.4
    layer.weight.set_mask(mask)
    x = rng.standard_normal((7, n_in))
    y = rng.integers(0, n_out, size=7)
    return layer, x, y, mask


class MaskedForwardAndGradient(unittest.TestCase):
    def test_masked_entries_are_true_zeros_and_do_not_affect_output(self):
        layer, x, _, mask = _layer()
        self.assertTrue(np.all(layer.weight.data[~mask] == 0.0))
        out_before = layer(Tensor(x)).data.copy()
        # Corrupt the stored values of removed entries: output must not change.
        layer.weight.data[~mask] = 1e6
        np.testing.assert_array_equal(layer(Tensor(x)).data, out_before)

    def test_gradient_of_masked_weight_is_exactly_zero_and_matches_finite_differences(self):
        layer, x, y, mask = _layer()

        # (a) gradcheck the real layer, with the weight Parameter as the input.
        result = gradcheck(lambda w: softmax_cross_entropy(layer(Tensor(x)), y), [layer.weight])
        self.assertLess(result.max_rel_error, 1e-6)
        analytic, numeric = result.analytic[0], result.numeric[0]
        self.assertTrue(np.all(analytic[~mask] == 0.0), "masked grad must be an exact 0")
        self.assertTrue(np.all(numeric[~mask] == 0.0), "perturbing a masked weight cannot change the loss")
        self.assertTrue(np.all(np.abs(analytic[mask]) > 0.0))

        # (b) plain backward on the layer gives the same masked-zero pattern.
        layer.weight.grad = None
        softmax_cross_entropy(layer(Tensor(x)), y).backward()
        self.assertTrue(np.all(layer.weight.grad[~mask] == 0.0))

        # (c) and the full dense function's gradient equals grad_eff * mask.
        np.testing.assert_allclose(layer.weight.grad, layer.last_w_eff.grad * mask, rtol=1e-12)

    def test_dense_gradient_on_effective_weight_is_available_for_regrowth(self):
        layer, x, y, mask = _layer()
        loss = softmax_cross_entropy(layer(Tensor(x)), y)
        loss.backward()
        dense = layer.last_w_eff.grad
        # dL/dW_eff = x^T dL/dpreact: non-zero at masked positions in general.
        expected = x.T @ layer.last_preact.grad
        np.testing.assert_allclose(dense, expected, rtol=1e-12)
        self.assertGreater(np.abs(dense[~mask]).min(), 0.0)
        np.testing.assert_allclose(layer.weight.grad, dense * mask)


class OptimizerStateUnderMasking(unittest.TestCase):
    def _train_steps(self, opt, layer, x, y, n):
        for _ in range(n):
            opt.zero_grad()
            softmax_cross_entropy(layer(Tensor(x)), y).backward()
            opt.step()

    def test_masked_weights_never_move_and_state_stays_zero(self):
        for opt_cls in (Adam, SGD):
            with self.subTest(opt=opt_cls.__name__):
                layer, x, y, _ = _layer()
                kwargs = {"weight_decay": 0.1}
                opt = opt_cls([layer.weight, layer.bias], lr=1e-2, **kwargs)
                self._train_steps(opt, layer, x, y, 20)
                # Prune a few more *after* momentum has built up.
                new_mask = layer.weight.mask.copy()
                alive = np.argwhere(new_mask)
                for r, c in alive[:3]:
                    new_mask[r, c] = False
                layer.weight.set_mask(new_mask)
                self._train_steps(opt, layer, x, y, 30)
                w = layer.weight
                self.assertTrue(np.all(w.data[~w.mask] == 0.0), "pruned entries drifted (zombie weights)")
                if isinstance(opt, Adam):
                    st = opt.state[id(w)]
                    self.assertTrue(np.all(st.m[~w.mask] == 0.0) and np.all(st.v[~w.mask] == 0.0))
                    self.assertTrue(np.all(st.t[~w.mask] == 0))
                else:
                    self.assertTrue(np.all(opt.velocity[id(w)][~w.mask] == 0.0))

    def test_naive_mask_unaware_adam_lets_pruned_weights_drift(self):
        """Documents the first, well-known failure mode (stale momentum)."""
        layer, x, y, _ = _layer()
        opt = Adam([layer.weight, layer.bias], lr=1e-2, mask_aware=False)
        self._train_steps(opt, layer, x, y, 20)
        new_mask = layer.weight.mask.copy()
        r, c = np.argwhere(new_mask)[0]
        new_mask[r, c] = False
        layer.weight.set_mask(new_mask)
        self._train_steps(opt, layer, x, y, 5)
        self.assertNotEqual(layer.weight.data[r, c], 0.0, "naive Adam should drift a pruned weight")


class RevivalBug(unittest.TestCase):
    """The bug: Adam bias correction for revived connections."""

    T_REVIVE = 5000
    LR = 1e-3

    def _revival_trajectory(self, bias_correction, grads):
        p = Parameter(np.zeros(2), prunable=True)
        opt = Adam([p], lr=self.LR, bias_correction=bias_correction)
        # Phase 1: entry 0 alive and training, entry 1 pruned.
        p.set_mask(np.array([True, False]))
        for _ in range(self.T_REVIVE):
            p.grad = np.array([0.3, 0.0])
            opt.step()
        # Phase 2: revive entry 1 (a regrowth event), feed it a gradient stream.
        p.set_mask(np.array([True, True]))
        traj = []
        for g in grads:
            before = p.data[1]
            p.grad = np.array([0.3, g])
            opt.step()
            traj.append(p.data[1] - before)
        return np.array(traj)

    def _fresh_trajectory(self, grads):
        q = Parameter(np.zeros(1))
        opt = Adam([q], lr=self.LR)
        traj = []
        for g in grads:
            before = q.data[0]
            q.grad = np.array([g])
            opt.step()
            traj.append(q.data[0] - before)
        return np.array(traj)

    def test_revived_connection_behaves_exactly_like_a_fresh_parameter(self):
        rng = np.random.default_rng(0)
        grads = rng.standard_normal(200) * 0.05 + 0.02
        revived = self._revival_trajectory("per_entry", grads)
        fresh = self._fresh_trajectory(grads)
        np.testing.assert_allclose(revived, fresh, rtol=1e-12, atol=0)
        self.assertAlmostEqual(abs(revived[0]), self.LR, places=9)  # first Adam step == lr

    def test_textbook_global_step_adam_oversteps_revived_connections(self):
        grads = np.full(50, 0.01)  # constant gradient -> closed form
        buggy = self._revival_trajectory("global", grads)
        b1, b2 = 0.9, 0.999
        k = np.arange(1, 51)
        # Moments were reset, but bias correction uses the global t ~ 5000 (factor ~1):
        bc_global = np.sqrt(1 - b2 ** (self.T_REVIVE + k)) / (1 - b1 ** (self.T_REVIVE + k))
        expected_ratio = (1 - b1**k) / np.sqrt(1 - b2**k) * bc_global
        np.testing.assert_allclose(np.abs(buggy) / self.LR, expected_ratio, rtol=1e-4)  # eps=1e-8 term
        self.assertGreater(abs(buggy[0]) / self.LR, 3.0)   # 3.16x on the very first step
        self.assertGreater(np.max(np.abs(buggy)) / self.LR, 6.0)  # ~6.5x peak around k=10
        fresh = self._fresh_trajectory(grads)
        self.assertTrue(np.allclose(np.abs(fresh) / self.LR, 1.0, atol=1e-6))

    def test_pruned_then_revived_weight_restarts_from_zero_not_a_stale_value(self):
        p = Parameter(np.array([0.5, -0.7]), prunable=True)
        opt = Adam([p], lr=1e-2)
        for _ in range(10):
            p.grad = np.array([0.1, -0.2])
            opt.step()
        p.set_mask(np.array([True, False]))
        for _ in range(10):
            p.grad = np.array([0.1, 0.0])
            opt.step()
        p.set_mask(np.array([True, True]))
        self.assertEqual(p.data[1], 0.0)
        st = opt.state[id(p)]
        self.assertEqual((st.m[1], st.v[1], st.t[1]), (0.0, 0.0, 0))


class ModelLevel(unittest.TestCase):
    def test_mlp_sparsity_accounting_and_state_dict_roundtrip(self):
        rng = np.random.default_rng(0)
        model = MLP([6, 8, 3], rng)
        w = model.prunable_weights()[0]
        mask = np.ones(w.shape, dtype=bool)
        mask[:, :4] = False
        w.set_mask(mask)
        self.assertAlmostEqual(model.sparsity(), 24 / (48 + 24))
        state = model.state_dict()
        clone = MLP([6, 8, 3], np.random.default_rng(1))
        clone.load_state_dict(state)
        x = Tensor(rng.standard_normal((4, 6)))
        np.testing.assert_array_equal(model(x).data, clone(x).data)
        self.assertAlmostEqual(clone.sparsity(), model.sparsity())


if __name__ == "__main__":
    unittest.main()
