"""Part 4 cost path: the sparse engines must be exact, and count real work."""

import unittest

import numpy as np

from aqua.nn import MLP
from aqua.sparse import CompactEngine, CSREngine, CSRMatrix, DenseEngine, compact_layers, export_layers, reference_forward


def _random_sparse_mlp(sparsity, seed=0, sizes=(20, 32, 32, 5), activation="relu"):
    rng = np.random.default_rng(seed)
    model = MLP(list(sizes), rng, activation=activation)
    for lin in model.linears:
        lin.bias.data[:] = rng.standard_normal(lin.bias.shape) * 0.1
        lin.weight.set_mask(rng.random(lin.weight.shape) >= sparsity)
    return model, rng.standard_normal((13, sizes[0]))


class CSR(unittest.TestCase):
    def test_roundtrip_and_matmul_including_empty_rows(self):
        rng = np.random.default_rng(0)
        w = rng.standard_normal((7, 6)) * (rng.random((7, 6)) > 0.7)
        w[:, 2] = 0.0  # an output unit with no inputs -> empty CSR row
        m = CSRMatrix.from_dense(w)
        np.testing.assert_array_equal(m.to_dense(), w)
        x = rng.standard_normal((5, 7))
        np.testing.assert_allclose(m.matmul(x), x @ w, rtol=1e-12, atol=1e-14)
        self.assertEqual(m.nnz, int(np.count_nonzero(w)))

    def test_all_zero_matrix(self):
        m = CSRMatrix.from_dense(np.zeros((4, 3)))
        np.testing.assert_array_equal(m.matmul(np.ones((2, 4))), np.zeros((2, 3)))


class Engines(unittest.TestCase):
    def test_all_engines_match_training_forward(self):
        for activation in ("relu", "tanh", "gelu"):
            for sparsity in (0.0, 0.5, 0.9, 0.97):
                with self.subTest(activation=activation, sparsity=sparsity):
                    model, x = _random_sparse_mlp(sparsity, activation=activation)
                    ref = reference_forward(model, x)
                    specs = export_layers(model)
                    for engine in (DenseEngine(specs), CSREngine(specs), CompactEngine(specs)):
                        np.testing.assert_allclose(engine(x), ref, rtol=1e-10, atol=1e-10, err_msg=engine.name)

    def test_csr_flops_scale_with_nnz(self):
        model, _ = _random_sparse_mlp(0.9)
        specs = export_layers(model)
        nnz = sum(int(np.count_nonzero(s.weight)) for s in specs)
        self.assertEqual(CSREngine(specs).flops_per_example, 2 * nnz)
        self.assertEqual(DenseEngine(specs).flops_per_example, 2 * (20 * 32 + 32 * 32 + 32 * 5))

    def test_compaction_removes_dead_units_exactly(self):
        model, x = _random_sparse_mlp(0.0)
        w0, w1 = model.linears[0].weight, model.linears[1].weight
        m0 = w0.mask.copy(); m0[:, 3] = False          # unit 3: no fan-in -> constant, folded
        w0.set_mask(m0)
        m1 = w1.mask.copy(); m1[7, :] = False          # unit 7: no fan-out -> dropped
        w1.set_mask(m1)
        keep_in, layers, report = compact_layers(export_layers(model))
        self.assertEqual(report["constant_folded"], 1)
        self.assertEqual(report["dead_fan_out"], 1)
        self.assertEqual(layers[0].weight.shape[1], 30)
        np.testing.assert_allclose(CompactEngine(export_layers(model))(x), reference_forward(model, x), atol=1e-10)

    def test_compaction_cascades(self):
        model, x = _random_sparse_mlp(0.0)
        # Kill all fan-out of layer-1 hidden units except unit 0 -> layer-0 units feeding only dead units cascade.
        w2 = model.linears[2].weight
        m2 = np.zeros_like(w2.mask); m2[0, :] = True
        w2.set_mask(m2)
        w1 = model.linears[1].weight
        m1 = np.zeros_like(w1.mask); m1[:5, 0] = True
        w1.set_mask(m1)
        engine = CompactEngine(export_layers(model))
        self.assertEqual(engine.report["shapes"][1], [5, 1])
        np.testing.assert_allclose(engine(x), reference_forward(model, x), atol=1e-10)


if __name__ == "__main__":
    unittest.main()
