# Pre-registered falsifiable claim

Committed **before** the confirmatory Part 4 sweep was run (check `git log -- results/PREREGISTERED_CLAIM.md`).

## How the claim was formed

A pilot on model seeds 0–3 (98% sparsity, global ranking) gave test accuracy
Fisher-OBD 95.8% vs magnitude 94.5%.  The criterion (`fisher_obd`) and the scope
(`global`) were selected on the **validation** split of that pilot, never on test.
The confirmatory sweep uses **fresh model seeds 100–109**, which the pilot never touched.

## The claim

> At 98% sparsity (global ranking, identical training budget), Fisher-OBD
> pruning reaches a higher held-out test accuracy than magnitude pruning.
> Over the 10 paired seeds 100–109 the mean paired difference
> (Fisher-OBD − magnitude) is **at least +0.5 percentage points**, and an exact
> two-sided paired sign-flip permutation test gives **p < 0.05**.

It is falsified if either the mean paired difference is below +0.5 pp or p ≥ 0.05.

Evaluated automatically by `scripts/sweep.py` into `results/part4/claim_result.json`
and re-checked from the committed raw CSV by `tests/test_claim.py`.

Note on granularity: the test split has 359 examples, so one example = 0.279 pp.
