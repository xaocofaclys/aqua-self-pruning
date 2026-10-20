"""Part 4: sparsity-accuracy Pareto sweep across seeds, baselines and ablations.

    python scripts/sweep.py                 # full sweep (~450 runs; ~25 min on 2 cores)
    python scripts/sweep.py --quick         # 2 seeds, 3 sparsities: a smoke test

Writes results/part4/:
    sweep_raw.csv        one row per run (the raw numbers; everything else derives from it)
    summary.json         mean/std per (method, sparsity) + paired tests vs magnitude
    claim_result.json    evaluation of results/PREREGISTERED_CLAIM.md
    revival.json         |dw|/lr of regrown weights by age, per optimizer variant
    pareto.png, paired_diff.png, revival.png
"""

from __future__ import annotations

import argparse
import os
import time
from collections import defaultdict
from multiprocessing import Pool

import _common  # noqa: F401
from _common import COLORS, environment, out_dir, setup_matplotlib, write_csv, write_json

import numpy as np

from aqua.experiment import ExperimentConfig, run
from aqua.prune import PruneConfig
from aqua.stats import describe, paired_comparison
from aqua.train import TrainConfig

SPARSITIES = [0.5, 0.75, 0.9, 0.95, 0.98, 0.99]
SEEDS = list(range(100, 110))

# name -> (prune kwargs, train kwargs)
METHODS: dict[str, tuple[dict, dict]] = {
    "magnitude": ({"criterion": "magnitude"}, {}),
    "saliency": ({"criterion": "saliency"}, {}),
    "fisher_obd": ({"criterion": "fisher_obd"}, {}),
    "taylor2": ({"criterion": "taylor2"}, {}),
    "fisher_obd+regrow": ({"criterion": "fisher_obd", "regrow": True}, {}),
}
ABLATIONS: dict[str, tuple[dict, dict, list[float]]] = {
    # scope: same criteria with per-layer uniform sparsity
    "magnitude@layer": ({"criterion": "magnitude", "scope": "layer"}, {}, [0.9, 0.95, 0.98, 0.99]),
    "fisher_obd@layer": ({"criterion": "fisher_obd", "scope": "layer"}, {}, [0.9, 0.95, 0.98, 0.99]),
    # optimizer handling of revived connections
    "regrow_global_t": ({"criterion": "fisher_obd", "regrow": True}, {"bias_correction": "global"}, [0.95, 0.98, 0.99]),
    "regrow_mask_unaware": ({"criterion": "fisher_obd", "regrow": True}, {"bias_correction": "global", "mask_aware": False}, [0.95, 0.98, 0.99]),
}


def _job(spec):
    method, sparsity, seed, prune_kw, train_kw = spec
    train = TrainConfig(seed=seed, **train_kw)
    prune = None if sparsity == 0.0 else PruneConfig(final_sparsity=sparsity, **prune_kw)
    regrow = bool(prune_kw.get("regrow"))
    r = run(ExperimentConfig(train=train, prune=prune, monitor_revival=regrow and prune is not None))
    p = r["params"]
    row = {
        "method": method,
        "criterion": prune_kw.get("criterion", "none") if prune else "none",
        "scope": prune.scope if prune else "none",
        "regrow": int(regrow),
        "bias_correction": train.bias_correction,
        "mask_aware": int(train.mask_aware),
        "target_sparsity": sparsity,
        "achieved_sparsity": round(r["sparsity"], 6),
        "seed": seed,
        "val_acc": round(r["val_acc"], 6),
        "test_acc": round(r["test_acc"], 6),
        "test_loss": round(r["test_loss"], 6),
        "active_weights": p["weights_active"],
        "total_weights": p["weights_total"],
        "train_seconds": round(r["train_seconds"], 2),
    }
    return row, r.get("revival")


def build_jobs(seeds, sparsities):
    jobs = [("dense", 0.0, s, {}, {}) for s in seeds]
    for m, (pk, tk) in METHODS.items():
        jobs += [(m, sp, s, pk, tk) for sp in sparsities for s in seeds]
    for m, (pk, tk, sps) in ABLATIONS.items():
        jobs += [(m, sp, s, pk, tk) for sp in sps if sp in sparsities for s in seeds]
    return jobs


def summarize(rows):
    by = defaultdict(dict)
    for r in rows:
        by[(r["method"], r["target_sparsity"])][r["seed"]] = r
    dense = {seed: r for (m, _), d in by.items() if m == "dense" for seed, r in d.items()}
    summary = {"per_method": {}, "vs_magnitude": {}, "vs_no_regrow": {}, "vs_correct_regrow": {}}
    for (method, sp), d in sorted(by.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        seeds = sorted(d)
        summary["per_method"].setdefault(method, {})[str(sp)] = {
            "test_acc": describe([d[s]["test_acc"] for s in seeds]),
            "val_acc": describe([d[s]["val_acc"] for s in seeds]),
            "achieved_sparsity": describe([d[s]["achieved_sparsity"] for s in seeds]),
            "retained_vs_dense": describe([d[s]["test_acc"] / dense[s]["test_acc"] for s in seeds if s in dense]),
        }

    def compare(a_name, b_name, key):
        for (m, sp), d in by.items():
            if m != a_name or (b_name, sp) not in by:
                continue
            other = by[(b_name, sp)]
            seeds = sorted(set(d) & set(other))
            summary[key].setdefault(a_name, {})[str(sp)] = paired_comparison(
                [d[s]["test_acc"] for s in seeds], [other[s]["test_acc"] for s in seeds])

    for m in METHODS:
        if m != "magnitude":
            compare(m, "magnitude", "vs_magnitude")
    compare("fisher_obd@layer", "magnitude@layer", "vs_magnitude")
    for m in ("fisher_obd+regrow", "regrow_global_t", "regrow_mask_unaware"):
        compare(m, "fisher_obd", "vs_no_regrow")
    compare("regrow_global_t", "fisher_obd+regrow", "vs_correct_regrow")
    compare("regrow_mask_unaware", "fisher_obd+regrow", "vs_correct_regrow")
    return summary, by


def evaluate_claim(by) -> dict:
    a, b = by.get(("fisher_obd", 0.98)), by.get(("magnitude", 0.98))
    if not a or not b:
        return {"evaluated": False, "reason": "98% runs missing (quick mode?)"}
    seeds = sorted(set(a) & set(b))
    cmp = paired_comparison([a[s]["test_acc"] for s in seeds], [b[s]["test_acc"] for s in seeds])
    holds = cmp["mean_diff"] >= 0.005 and cmp["p_value"] < 0.05
    return {
        "evaluated": True,
        "claim": "fisher_obd - magnitude at 98% sparsity: mean paired test-acc diff >= +0.5pp and permutation p < 0.05",
        "seeds": seeds,
        "fisher_obd_mean": float(np.mean([a[s]["test_acc"] for s in seeds])),
        "magnitude_mean": float(np.mean([b[s]["test_acc"] for s in seeds])),
        **cmp,
        "holds": bool(holds),
    }


def _sparsity_axis(s):
    return -np.log10(1.0 - np.asarray(s))  # spreads 90/95/98/99% apart


def plot(by, summary, revival, out):
    plt = setup_matplotlib()
    dense_acc = np.array([r["test_acc"] for r in by[("dense", 0.0)].values()])

    # --- Pareto: zoomed main panel + full-range panel (taylor2 collapses) ------
    ticks = [0.5, 0.75, 0.9, 0.95, 0.98, 0.99]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1.6, 1]})
    for ax, methods, ylim in [(axes[0], [m for m in METHODS if m != "taylor2"], (0.88, 0.985)),
                              (axes[1], list(METHODS), (0.6, 1.0))]:
        ax.axhspan(dense_acc.mean() - dense_acc.std(), dense_acc.mean() + dense_acc.std(), color=COLORS["dense"], alpha=0.12, lw=0)
        ax.axhline(dense_acc.mean(), color=COLORS["dense"], lw=1, ls="--", label=f"dense ({dense_acc.mean():.3f})")
        offsets = np.linspace(-0.025, 0.025, len(methods))
        for off, m in zip(offsets, methods):
            sps = sorted(sp for (mm, sp) in by if mm == m)
            if not sps:
                continue
            xs = [float(np.mean([r["achieved_sparsity"] for r in by[(m, sp)].values()])) for sp in sps]
            accs = [np.array([r["test_acc"] for r in by[(m, sp)].values()]) for sp in sps]
            ax.errorbar(_sparsity_axis(xs) + off, [a.mean() for a in accs], yerr=[a.std(ddof=1) for a in accs],
                        marker="o", ms=4, capsize=3, lw=1.6, color=COLORS[m], label=m)
        ax.set_xticks(_sparsity_axis(ticks), [f"{t:.0%}" for t in ticks])
        ax.set_ylim(*ylim)
        ax.set_xlabel("achieved sparsity (log-odds scale)")
        ax.legend(loc="lower left", fontsize=8)
    axes[0].set(ylabel="test accuracy", title=f"Sparsity vs accuracy (mean ± std over {len(dense_acc)} seeds)")
    axes[1].set(title="Full range, incl. signed 2nd-order Taylor")
    fig.tight_layout()
    fig.savefig(out / "pareto.png")
    plt.close(fig)

    # --- paired difference vs magnitude ---------------------------------------
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.axhline(0, color="#888", lw=1)
    offsets = np.linspace(-0.03, 0.03, len(summary["vs_magnitude"]))
    for off, (m, d) in zip(offsets, summary["vs_magnitude"].items()):
        if "@" in m:
            continue
        sps = sorted(float(k) for k in d)
        mean = np.array([d[str(s)]["mean_diff"] for s in sps]) * 100
        lo = np.array([d[str(s)]["ci95"][0] for s in sps]) * 100
        hi = np.array([d[str(s)]["ci95"][1] for s in sps]) * 100
        x = _sparsity_axis(sps) + off
        ax.errorbar(x, mean, yerr=[mean - lo, hi - mean], marker="o", ms=4, capsize=3, lw=1.4, color=COLORS[m], label=m)
    ax.set_xticks(_sparsity_axis(ticks), [f"{t:.0%}" for t in ticks])
    ax.set(xlabel="target sparsity", ylabel="test acc − magnitude (pp)",
           title="Paired difference vs magnitude pruning (same seeds), 95% bootstrap CI")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "paired_diff.png")
    plt.close(fig)

    # --- revival telemetry ------------------------------------------------------
    if revival:
        fig, ax = plt.subplots(figsize=(11, 3.8))
        labels = {"fisher_obd+regrow": "per-entry bias correction (ours)",
                  "regrow_global_t": "global step counter (textbook Adam, moments reset)",
                  "regrow_mask_unaware": "mask-unaware (no reset, stale moments)"}
        for m, rep in revival.items():
            age = np.array(rep["age"])
            ax.plot(age, rep["mean_step_over_lr"], color=COLORS[m], lw=1.8, label=f"{labels[m]}: mean")
            ax.plot(age, rep["max_step_over_lr"], color=COLORS[m], lw=1, ls=":", label=f"{labels[m]}: max")
        ax.axhline(1.0, color="#888", lw=1, ls="--")
        ax.set(xlabel="optimizer steps since the connection was regrown", ylabel="|Δw| / lr",
               title="How hard do revived connections get kicked? (regrowth runs at 95-99%, 10 seeds)", yscale="log")
        ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0))
        fig.tight_layout()
        fig.savefig(out / "revival.png")
        plt.close(fig)


def _load_rows(out):
    import csv

    ints, floats = {"regrow", "mask_aware", "seed", "active_weights", "total_weights"}, {
        "target_sparsity", "achieved_sparsity", "val_acc", "test_acc", "test_loss", "train_seconds"}
    with (out / "sweep_raw.csv").open() as f:
        return [{k: int(v) if k in ints else float(v) if k in floats else v for k, v in r.items()} for r in csv.DictReader(f)]


def _aggregate_revival(revival_raw):
    """Pool per-run telemetry: sample-weighted mean and overall max, by age."""
    revival = {}
    for m, reps in revival_raw.items():
        counts = np.sum([r["n_samples"] for r in reps], axis=0)
        sums = np.sum([np.array(r["mean_step_over_lr"]) * np.array(r["n_samples"]) for r in reps], axis=0)
        revival[m] = {"age": reps[0]["age"], "mean_step_over_lr": (sums / np.maximum(counts, 1)).tolist(),
                      "max_step_over_lr": np.max([r["max_step_over_lr"] for r in reps], axis=0).tolist(),
                      "n_runs": len(reps), "n_samples": counts.tolist()}
    return revival


def revival_only(out, n_jobs) -> None:
    rows = _load_rows(out)
    committed = {(r["method"], r["target_sparsity"], r["seed"]): r for r in rows}
    jobs = [j for j in build_jobs(SEEDS, SPARSITIES) if j[3].get("regrow") and j[1] >= 0.95]
    print(f"{len(jobs)} regrowth runs", flush=True)
    revival_raw = defaultdict(list)
    with Pool(n_jobs) as pool:
        for row, rev in pool.imap_unordered(_job, jobs, chunksize=1):
            ref = committed[(row["method"], row["target_sparsity"], row["seed"])]
            for k in ("test_acc", "val_acc", "test_loss", "achieved_sparsity", "active_weights"):
                assert row[k] == ref[k], f"not reproduced: {row['method']} {row['target_sparsity']} {row['seed']} {k}"
            revival_raw[row["method"]].append(rev)
    write_json(out / "revival.json", _aggregate_revival(revival_raw))
    print("all regrowth runs reproduced sweep_raw.csv exactly; revival.json rewritten")
    replot(out)


def replot(out) -> None:
    import json

    rows = _load_rows(out)
    summary, by = summarize(rows)
    previous = json.loads((out / "summary.json").read_text())
    write_json(out / "summary.json", {k: previous[k] for k in ("environment", "seeds", "sparsities", "wall_seconds")} | summary)
    write_json(out / "claim_result.json", evaluate_claim(by))
    revival = json.loads((out / "revival.json").read_text())
    plot(by, summary, revival, out)
    print(f"replotted from {len(rows)} rows")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="2 seeds x 3 sparsities smoke run (writes results/part4_quick)")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--revival-only", action="store_true",
                    help="rerun only the regrowth runs (>=95%%) to recompute revival.json; asserts they reproduce sweep_raw.csv exactly")
    ap.add_argument("--replot", action="store_true", help="recompute summaries and figures from the committed sweep_raw.csv (no training)")
    args = ap.parse_args()
    if args.replot:
        return replot(out_dir("part4"))
    if args.revival_only:
        return revival_only(out_dir("part4"), args.jobs)
    seeds = SEEDS[:2] if args.quick else SEEDS
    sparsities = [0.5, 0.9, 0.98] if args.quick else SPARSITIES
    out = out_dir("part4_quick" if args.quick else "part4")

    jobs = build_jobs(seeds, sparsities)
    print(f"{len(jobs)} runs on {args.jobs} processes", flush=True)
    t0 = time.time()
    rows, revival_raw = [], defaultdict(list)
    with Pool(args.jobs) as pool:
        for i, (row, rev) in enumerate(pool.imap_unordered(_job, jobs, chunksize=1), 1):
            rows.append(row)
            if rev and row["target_sparsity"] >= 0.95:
                revival_raw[row["method"]].append(rev)
            if i % 25 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} done ({time.time() - t0:.0f}s)", flush=True)
    rows.sort(key=lambda r: (r["method"], r["target_sparsity"], r["seed"]))
    write_csv(out / "sweep_raw.csv", rows)

    summary, by = summarize(rows)
    claim = evaluate_claim(by)
    revival = _aggregate_revival(revival_raw)
    write_json(out / "summary.json", {"environment": environment(), "seeds": seeds, "sparsities": sparsities,
                                      "wall_seconds": round(time.time() - t0, 1), **summary})
    write_json(out / "claim_result.json", claim)
    write_json(out / "revival.json", revival)
    plot(by, summary, revival, out)

    print("\nmean test accuracy (± std over seeds):")
    methods = ["dense", *METHODS, *ABLATIONS]
    for m in methods:
        cells = []
        for sp in [0.0, *sparsities]:
            if (m, sp) in by:
                a = np.array([r["test_acc"] for r in by[(m, sp)].values()])
                cells.append(f"{sp:.0%}: {a.mean():.4f}±{a.std(ddof=1):.4f}")
        print(f"  {m:22s} " + "  ".join(cells))
    print("\nclaim:", "HOLDS" if claim.get("holds") else "FALSIFIED / not evaluated", claim)


if __name__ == "__main__":
    main()
