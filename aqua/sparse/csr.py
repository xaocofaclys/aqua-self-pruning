"""A compressed-sparse-row matrix and a sparse-aware matmul, from scratch.

We store W^T (out_features x in_features) in CSR so that row j lists the
inputs feeding output unit j.  ``matmul`` performs exactly ``nnz`` multiplies
and ``nnz`` adds per example - no multiplication by stored zeros.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CSRMatrix:
    indptr: np.ndarray   # (out + 1,) int64: row j's entries are data[indptr[j]:indptr[j+1]]
    indices: np.ndarray  # (nnz,) int64: input-feature index of each entry
    data: np.ndarray     # (nnz,) float64
    shape: tuple[int, int]  # (in_features, out_features) of the *logical* W

    @classmethod
    def from_dense(cls, w: np.ndarray) -> "CSRMatrix":
        """``w`` is (in, out); zeros are dropped exactly (no tolerance)."""
        wt = np.ascontiguousarray(w.T)
        rows, cols = np.nonzero(wt)  # row-major order -> already sorted by row
        counts = np.bincount(rows, minlength=wt.shape[0])
        indptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        csr = cls(indptr, cols.astype(np.int64), wt[rows, cols].astype(np.float64), (w.shape[0], w.shape[1]))
        csr._prepare()
        return csr

    def _prepare(self) -> None:
        counts = np.diff(self.indptr)
        self._nonempty = np.flatnonzero(counts > 0)
        self._starts = self.indptr[:-1][self._nonempty]

    @property
    def nnz(self) -> int:
        return int(self.data.size)

    @property
    def density(self) -> float:
        return self.nnz / (self.shape[0] * self.shape[1])

    def to_dense(self) -> np.ndarray:
        wt = np.zeros((self.shape[1], self.shape[0]))
        rows = np.repeat(np.arange(self.shape[1]), np.diff(self.indptr))
        wt[rows, self.indices] = self.data
        return wt.T

    # Bound the (rows x nnz) gather temporary to ~32 MB of float64.
    _MAX_TEMP_ELEMS = 1 << 22

    def matmul(self, x: np.ndarray) -> np.ndarray:
        """y = x @ W for x of shape (N, in), processed in memory-bounded row chunks."""
        n = x.shape[0]
        out = np.zeros((n, self.shape[1]), dtype=np.result_type(x, self.data))
        if self.nnz == 0:
            return out
        chunk = max(1, self._MAX_TEMP_ELEMS // self.nnz)
        for start in range(0, n, chunk):
            xs = x[start : start + chunk]
            gathered = xs[:, self.indices] * self.data  # one multiply per stored entry
            out[start : start + chunk, self._nonempty] = np.add.reduceat(gathered, self._starts, axis=1)
        return out

    def nbytes(self) -> int:
        return self.indptr.nbytes + self.indices.nbytes + self.data.nbytes
