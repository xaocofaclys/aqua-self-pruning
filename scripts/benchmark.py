"""Part 4: does pruning actually reduce work?  Measured, not asserted.

    python scripts/benchmark.py        # run on an otherwise idle machine

A. Real pruned models (Fisher-OBD, global, seed 100 - identical to the sweep runs):
   FLOPs, weight bytes and median latency for three inference engines:
     dense   - masked weights through BLAS (dense x zeros: NOT a speedup, the honest baseline)
     csr     - our CSR kernel on every layer
     compact - exact dead-unit removal, then dense/CSR per layer by remaining density
   All engines must agree with the training forward pass (checked here too).
B. Synthetic scaling (random masks on 3 x 2048 layers, sparsity >= 80%): where does our CSR kernel
   overtake BLAS?  Timing only - these weights are not trained.
"""

from __future__ import annotations

import argparse

import _common  # noqa: F401
from _common import COLORS, environment, out_dir, setup_matplotlib, write_csv, write_json

import numpy as np

from aqua.experiment import ExperimentConfig, get_dataset, run
from aqua.prune import PruneConfig
from aqua.sparse import CompactEngine, CSREngine, DenseEngine, LayerSpec, export_layers, reference_forward, time_callable
from aqua.train import TrainConfig

SPARSITIES = [0.0, 0.5, 0.75, 0.9, 0.95, 0.98, 0.99]
BATCHES = [1, 256]


def bench_real(seed: int, repeats: int):
    data = get_dataset("digits", 0)
    rows = []
    for sp in SPARSITIES:
        prune = None if sp == 0 else PruneConfig(criterion="fisher_obd", final_sparsity=sp)
        r = run(ExperimentConfig(train=TrainConfig(seed=seed), prune=prune))
        model = r["_model"]
        specs = export_layers(model)
        ref = reference_forward(model, data.test.X)
        for bs in BATCHES:
            x = np.ascontiguousarray(np.resize(data.test.X, (bs, data.test.X.shape[1])))
            # The compacted engine's kernel plan is autotuned for the serving batch size.
            for eng in (DenseEngine(specs), CSREngine(specs), CompactEngine(specs, tune_batch=bs)):
                out = eng(data.test.X)
                np.testing.assert_allclose(out, ref, rtol=1e-9, atol=1e-9)  # exactness gate
                acc = float((out.argmax(1) == data.test.y).mean())
                t = time_callable(eng, x, repeats=repeats, inner=20 if bs == 1 else 2)
                compact = isinstance(eng, CompactEngine)
                rows.append({
                    "sparsity": sp, "achieved_sparsity": round(r["sparsity"], 6), "engine": eng.name.split(" ")[0],
                    "batch": bs, "flops_per_example": eng.flops_per_example, "weight_bytes": eng.weight_bytes,
                    "median_us": round(t["median_us"], 2), "p10_us": round(t["p10_us"], 2), "p90_us": round(t["p90_us"], 2),
                    "us_per_example": round(t["median_us"] / bs, 3), "test_acc": acc,
                    "compact_shapes": str(eng.report["shapes"]) if compact else "",
                    "compact_kernels": ",".join(eng.kernels) if compact else "",
                })
        print(f"sparsity {sp:.2f}: " + "  ".join(
            f"{row['engine']}@{row['batch']}={row['median_us']:.0f}us" for row in rows if row["sparsity"] == sp), flush=True)
    return rows


def bench_synthetic(repeats: int):
    rng = np.random.default_rng(0)
    width = 2048
    rows = []
    for sp in [0.8, 0.9, 0.95, 0.98, 0.99, 0.995]:
        specs = []
        for _ in range(3):
            w = rng.standard_normal((width, width)) * (rng.random((width, width)) >= sp)
            specs.append(LayerSpec(w, np.zeros(width), "relu"))
        dense, csr = DenseEngine(specs), CSREngine(specs)
        for bs in [1, 64]:
            x = rng.standard_normal((bs, width))
            np.testing.assert_allclose(csr(x), dense(x), rtol=1e-8, atol=1e-8)
            td = time_callable(dense, x, repeats=repeats, warmup=5)
            tc = time_callable(csr, x, repeats=repeats, warmup=5)
            rows.append({"sparsity": sp, "batch": bs, "dense_us": round(td["median_us"], 1), "csr_us": round(tc["median_us"], 1),
                         "speedup": round(td["median_us"] / tc["median_us"], 3),
                         "dense_flops": dense.flops_per_example, "csr_flops": csr.flops_per_example})
        print(f"synthetic {sp:.3f}: " + "  ".join(f"bs{r['batch']} x{r['speedup']:.2f}" for r in rows if r["sparsity"] == sp), flush=True)
    return rows


def plot(real, synth, out):
    plt = setup_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    colors = {"dense": COLORS["dense"], "csr": COLORS["fisher_obd"], "compacted": COLORS["saliency"]}
    for eng in ["dense", "csr", "compacted"]:
        sel = [r for r in real if r["engine"] == eng and r["batch"] == 1]
        axes[0].plot([r["sparsity"] for r in sel], [r["flops_per_example"] / 1e3 for r in sel], marker="o", color=colors[eng], label=eng)
        for bs, ls in [(1, "-"), (256, "--")]:
            sel = [r for r in real if r["engine"] == eng and r["batch"] == bs]
            axes[1].plot([r["sparsity"] for r in sel], [r["us_per_example"] for r in sel], marker="o", ms=3, ls=ls,
                         color=colors[eng], label=f"{eng}, batch {bs}")
    axes[0].set(title="Work per example (counted)", xlabel="sparsity", ylabel="kFLOPs", yscale="log")
    axes[0].legend(fontsize=8)
    axes[1].set(title="Measured latency per example (digits MLP)", xlabel="sparsity", ylabel="µs / example (median)", yscale="log")
    axes[1].legend(fontsize=7)
    for bs, ls in [(1, "-"), (64, "--")]:
        sel = [r for r in synth if r["batch"] == bs]
        axes[2].plot([r["sparsity"] for r in sel], [r["speedup"] for r in sel], marker="o", ls=ls, color=COLORS["fisher_obd"], label=f"batch {bs}")
    axes[2].axhline(1.0, color="#888", lw=1)
    axes[2].set(title="Synthetic 3×2048 layers: CSR speedup over BLAS", xlabel="sparsity", ylabel="dense time / CSR time", yscale="log")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "cost.png")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--repeats", type=int, default=200)
    args = ap.parse_args()
    out = out_dir("part4")
    real = bench_real(args.seed, args.repeats)
    synth = bench_synthetic(max(20, args.repeats // 10))
    write_csv(out / "cost_real_models.csv", real)
    write_csv(out / "cost_synthetic_scaling.csv", synth)
    write_json(out / "cost_environment.json", environment())
    plot(real, synth, out)
    print(f"wrote {out}/cost_*.csv and cost.png")


if __name__ == "__main__":
    main()
