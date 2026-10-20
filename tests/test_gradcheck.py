"""Part 1: finite-difference gradient checks for every differentiable op."""

import unittest

import numpy as np

from aqua.engine import Tensor, gradcheck, no_grad, softmax_cross_entropy, unbroadcast
from aqua.engine import tensor as T

RTOL = 1e-6


def leaf(shape, rng, low=None, away_from_zero=False, positive=False):
    x = rng.standard_normal(shape)
    if away_from_zero:  # keep finite differences off ReLU's kink
        x = np.where(np.abs(x) < 0.1, np.sign(x + 1e-12) * 0.1 + x, x)
    if positive:
        x = np.abs(x) + 0.5
    return Tensor(x, requires_grad=True)


class GradCheckOps(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(1234)

    def check(self, fn, *inputs, rtol=RTOL):
        result = gradcheck(fn, list(inputs))
        self.assertLessEqual(result.max_rel_error, rtol, f"rel err {result.max_rel_error:.2e}")
        return result

    # --- element-wise binary ops, including every broadcasting pattern -------
    def test_binary_ops_with_broadcasting(self):
        shapes = [((4, 3), (4, 3)), ((4, 3), (3,)), ((4, 1), (1, 3)), ((2, 4, 3), (4, 1)), ((4, 3), ())]
        for sa, sb in shapes:
            for name, fn in [("add", T.add), ("sub", T.sub), ("mul", T.mul)]:
                with self.subTest(op=name, a=sa, b=sb):
                    self.check(fn, leaf(sa, self.rng), leaf(sb, self.rng))
            with self.subTest(op="div", a=sa, b=sb):
                self.check(T.div, leaf(sa, self.rng), leaf(sb, self.rng, positive=True))

    def test_bias_broadcast_ND_plus_D(self):
        """The case called out in the brief: (N, D) + (D,)."""
        x, b = leaf((5, 3), self.rng), leaf((3,), self.rng)
        result = self.check(lambda x, b: x + b, x, b)
        self.assertEqual(result.analytic[1].shape, (3,))

    def test_operator_overloads_and_reflected(self):
        x = leaf((3, 2), self.rng, positive=True)
        self.check(lambda x: 2.0 - x * 3.0 + 1.0 / x - (-x) ** 2.0 / 4.0, x)

    def test_neg_pow(self):
        self.check(lambda x: -x, leaf((3, 4), self.rng))
        self.check(lambda x: x**3.0, leaf((3, 4), self.rng))
        self.check(lambda x: x**0.5, leaf((3, 4), self.rng, positive=True))

    # --- matmul ----------------------------------------------------------------
    def test_matmul_2d(self):
        self.check(T.matmul, leaf((5, 4), self.rng), leaf((4, 3), self.rng))

    def test_matmul_batched_broadcast(self):
        self.check(T.matmul, leaf((2, 5, 4), self.rng), leaf((4, 3), self.rng))

    def test_matmul_rejects_vectors(self):
        with self.assertRaises(ValueError):
            T.matmul(Tensor(np.ones(3), requires_grad=True), Tensor(np.ones((3, 2))))

    # --- reductions ------------------------------------------------------------
    def test_sum_and_mean_all_axes(self):
        for axis in [None, 0, 1, -1, (0, 1)]:
            for keepdims in [False, True]:
                with self.subTest(axis=axis, keepdims=keepdims):
                    self.check(lambda x: x.sum(axis=axis, keepdims=keepdims), leaf((4, 3), self.rng))
                    self.check(lambda x: x.mean(axis=axis, keepdims=keepdims), leaf((4, 3), self.rng))

    # --- non-linearities -------------------------------------------------------
    def test_relu(self):
        self.check(T.relu, leaf((6, 5), self.rng, away_from_zero=True))

    def test_tanh_sigmoid_gelu_exp_log(self):
        self.check(T.tanh, leaf((6, 5), self.rng))
        self.check(T.sigmoid, Tensor(self.rng.standard_normal((6, 5)) * 3.0, requires_grad=True))
        self.check(T.gelu, leaf((6, 5), self.rng))
        self.check(T.exp, leaf((6, 5), self.rng))
        self.check(T.log, leaf((6, 5), self.rng, positive=True))

    def test_sigmoid_is_stable_at_extremes(self):
        # Finite differences are meaningless where the gradient is ~1e-300, so
        # check the closed form instead: no overflow, s(1-s) gradient.
        x = Tensor(np.array([-800.0, -30.0, 0.0, 30.0, 800.0]), requires_grad=True)
        s = T.sigmoid(x)
        s.sum().backward()
        self.assertTrue(np.all(np.isfinite(s.data)) and np.all(np.isfinite(x.grad)))
        np.testing.assert_allclose(s.data[[0, 2, 4]], [0.0, 0.5, 1.0], atol=1e-300)
        np.testing.assert_allclose(x.grad, s.data * (1 - s.data))

    # --- shape ops -------------------------------------------------------------
    def test_reshape_transpose(self):
        self.check(lambda x: x.reshape(6, 4), leaf((4, 6), self.rng))
        self.check(lambda x: x.T, leaf((4, 6), self.rng))
        self.check(lambda x: x.transpose(2, 0, 1), leaf((2, 3, 4), self.rng))

    # --- softmax family --------------------------------------------------------
    def test_softmax_and_log_softmax(self):
        self.check(T.softmax, leaf((4, 5), self.rng))
        self.check(T.log_softmax, leaf((4, 5), self.rng))

    def test_softmax_cross_entropy_all_reductions(self):
        y = self.rng.integers(0, 5, size=6)
        for reduction in ["mean", "sum", "none"]:
            with self.subTest(reduction=reduction):
                self.check(lambda z: softmax_cross_entropy(z, y, reduction), leaf((6, 5), self.rng))

    def test_cross_entropy_matches_log_softmax_composition(self):
        z = self.rng.standard_normal((7, 4))
        y = self.rng.integers(0, 4, size=7)
        fused = softmax_cross_entropy(Tensor(z), y).item()
        composed = -T.log_softmax(Tensor(z)).data[np.arange(7), y].mean()
        self.assertAlmostEqual(fused, composed, places=12)

    def test_cross_entropy_is_numerically_stable(self):
        z = Tensor(np.array([[1000.0, -1000.0, 0.0], [-5e4, 5e4, 1.0]]), requires_grad=True)
        loss = softmax_cross_entropy(z, np.array([0, 2]))
        loss.backward()
        self.assertTrue(np.isfinite(loss.item()))
        self.assertTrue(np.all(np.isfinite(z.grad)))
        self.assertAlmostEqual(loss.item(), (0.0 + (5e4 - 1.0)) / 2, places=6)

    # --- a whole network -------------------------------------------------------
    def test_composite_mlp(self):
        x = Tensor(self.rng.standard_normal((5, 4)))
        y = self.rng.integers(0, 3, size=5)
        w1, b1 = leaf((4, 6), self.rng), leaf((6,), self.rng)
        w2, b2 = leaf((6, 3), self.rng), leaf((3,), self.rng)
        self.check(lambda w1, b1, w2, b2: softmax_cross_entropy(T.tanh(x @ w1 + b1) @ w2 + b2, y), w1, b1, w2, b2)


class EngineSemantics(unittest.TestCase):
    def test_gradient_accumulates_when_a_node_is_reused(self):
        x = Tensor(np.array([2.0, 3.0]), requires_grad=True)
        (x * x + x).sum().backward()  # d/dx = 2x + 1
        np.testing.assert_allclose(x.grad, [5.0, 7.0])

    def test_diamond_graph_topological_order(self):
        x = Tensor(np.array(1.5), requires_grad=True)
        a = x * 2.0
        b = a.tanh()
        c = a * b  # depends on a twice through two paths
        c.backward()
        t = np.tanh(3.0)
        np.testing.assert_allclose(x.grad, 2.0 * (t + 3.0 * (1 - t * t)))

    def test_leaf_gradients_accumulate_across_backward_calls(self):
        x = Tensor(np.ones(3), requires_grad=True)
        (x * 2.0).sum().backward()
        (x * 3.0).sum().backward()
        np.testing.assert_allclose(x.grad, [5.0, 5.0, 5.0])

    def test_deep_graph_does_not_hit_recursion_limit(self):
        x = Tensor(np.array(1.0), requires_grad=True)
        y = x
        for _ in range(5000):
            y = y * 1.0
        y.backward()
        self.assertEqual(x.grad, 1.0)

    def test_no_grad_builds_no_graph(self):
        x = Tensor(np.ones(3), requires_grad=True)
        with no_grad():
            y = (x * 2.0).sum()
        self.assertFalse(y.requires_grad)
        with self.assertRaises(RuntimeError):
            y.backward()

    def test_grad_buffers_are_not_aliased(self):
        x = Tensor(np.ones(3), requires_grad=True)
        y = x + 0.0
        y.sum().backward()
        x.grad *= 10.0  # in-place edit of a leaf grad must not leak into y.grad
        np.testing.assert_allclose(y.grad, np.ones(3))

    def test_unbroadcast_shapes(self):
        g = np.ones((2, 4, 3))
        self.assertEqual(unbroadcast(g, (3,)).tolist(), [8.0, 8.0, 8.0])
        self.assertEqual(unbroadcast(g, (4, 1)).shape, (4, 1))
        self.assertEqual(float(unbroadcast(g, ())), 24.0)


if __name__ == "__main__":
    unittest.main()
