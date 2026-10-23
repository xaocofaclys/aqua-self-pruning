"""Re-check the pre-registered claim from the committed raw numbers.

This deliberately recomputes everything from ``results/part4/sweep_raw.csv``
instead of trusting ``claim_result.json``: if someone edits the summary by
hand, or the raw CSV stops supporting the claim, this test fails.
"""

import csv
import json
import unittest
from pathlib import Path

import numpy as np

from aqua.stats import paired_permutation_pvalue

RAW = Path(__file__).resolve().parent.parent / "results" / "part4" / "sweep_raw.csv"
CLAIM = RAW.parent / "claim_result.json"


@unittest.skipUnless(RAW.exists(), "run scripts/sweep.py first")
class PreregisteredClaim(unittest.TestCase):
    def setUp(self):
        with RAW.open() as f:
            self.rows = list(csv.DictReader(f))

    def _acc(self, method, sparsity):
        return {int(r["seed"]): float(r["test_acc"]) for r in self.rows
                if r["method"] == method and abs(float(r["target_sparsity"]) - sparsity) < 1e-9}

    def test_claim_holds_on_raw_numbers(self):
        a, b = self._acc("fisher_obd", 0.98), self._acc("magnitude", 0.98)
        seeds = sorted(set(a) & set(b))
        self.assertEqual(seeds, list(range(100, 110)), "claim is defined on seeds 100-109")
        diffs = np.array([a[s] - b[s] for s in seeds])
        self.assertGreaterEqual(diffs.mean(), 0.005, f"mean paired diff {diffs.mean():+.4f} < +0.5pp")
        self.assertLess(paired_permutation_pvalue(diffs), 0.05)

    def test_summary_file_agrees_with_raw(self):
        claim = json.loads(CLAIM.read_text())
        a, b = self._acc("fisher_obd", 0.98), self._acc("magnitude", 0.98)
        diffs = np.array([a[s] - b[s] for s in sorted(a)])
        self.assertAlmostEqual(claim["mean_diff"], diffs.mean(), places=6)
        self.assertTrue(claim["holds"])

    def test_every_run_met_its_sparsity_budget(self):
        for r in self.rows:
            target, achieved = float(r["target_sparsity"]), float(r["achieved_sparsity"])
            self.assertLess(abs(target - achieved), 2.0 / int(r["total_weights"]), r)


class ClaimNeedsItsSeeds(unittest.TestCase):
    def test_quick_mode_does_not_report_falsified(self):
        import sys
        sys.path.insert(0, str(RAW.parent.parent.parent / "scripts"))
        import sweep
        row = lambda acc: {"test_acc": acc}
        by = {("fisher_obd", 0.98): {100: row(0.97), 101: row(0.96)}, ("magnitude", 0.98): {100: row(0.95), 101: row(0.94)}}
        verdict = sweep.evaluate_claim(by)
        self.assertFalse(verdict["evaluated"])
        self.assertNotIn("holds", verdict)


class StatsSanity(unittest.TestCase):
    def test_permutation_pvalue_known_values(self):
        self.assertEqual(paired_permutation_pvalue(np.ones(10)), 2 / 1024)
        self.assertEqual(paired_permutation_pvalue(np.array([1.0, -1.0])), 1.0)


if __name__ == "__main__":
    unittest.main()
