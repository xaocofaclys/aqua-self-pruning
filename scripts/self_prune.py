"""Part 3: one self-pruning run, with the full timeline of the mask.

    python scripts/self_prune.py                          # Fisher-OBD, 90%, regrowth on
    python scripts/self_prune.py --sparsity 0.95 --criterion saliency --no-regrow
"""

from __future__ import annotations

import argparse

import _common  # noqa: F401
from _common import COLORS, environment, out_dir, setup_matplotlib, write_json

import numpy as np

from aqua.experiment import ExperimentConfig, run, serializable
from aqua.prune import CRITERIA, PruneConfig
from aqua.train import TrainConfig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sparsity", type=float, default=0.9)
    ap.add_argument("--criterion", choices=CRITERIA, default="fisher_obd")
    ap.add_argument("--scope", choices=["global", "layer"], default="global")
    ap.add_argument("--no-regrow", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = out_dir("part3")

    train = TrainConfig(seed=args.seed, log_every_epoch=True)
    print("=== dense reference (same seed) ===", flush=True)
    dense = run(ExperimentConfig(train=TrainConfig(seed=args.seed)))
    print(f"dense test acc {dense['test_acc']:.4f}\n=== self-pruning ===", flush=True)
    prune = PruneConfig(criterion=args.criterion, final_sparsity=args.sparsity, scope=args.scope, regrow=not args.no_regrow)
    r = run(ExperimentConfig(train=train, prune=prune, monitor_revival=not args.no_regrow))
    model = r["_model"]

    report = {
        "environment": environment(),
        "target_sparsity": args.sparsity,
        "achieved_sparsity": r["sparsity"],
        "budget_met": abs(r["sparsity"] - args.sparsity) < 1.0 / r["params"]["weights_total"] + 1e-12,
        "per_layer_sparsity": r["prune_summary"]["per_layer_sparsity"],
        "params": r["params"],
        "dense_test_acc": dense["test_acc"],
        "pruned_test_acc": r["test_acc"],
        "accuracy_cost_pp": (dense["test_acc"] - r["test_acc"]) * 100,
        "dead_hidden_units": [int((~lin.weight.mask.any(axis=0)).sum()) for lin in model.linears[:-1]],
        "run": serializable(r),
    }
    write_json(out / "self_prune_run.json", report)

    plt = setup_matplotlib()
    hist = r["history"]
    ep = np.array([h["epoch"] for h in hist])
    fig, ax1 = plt.subplots(figsize=(7.2, 3.8))
    ax1.plot(ep, [h["val_acc"] for h in hist], color=COLORS["fisher_obd"], lw=1.8, label="val accuracy (pruning run)")
    ax1.plot(ep, [h["val_acc"] for h in dense["history"]], color=COLORS["dense"], lw=1.2, ls="--", label="val accuracy (dense, same seed)")
    ax1.set(xlabel="epoch", ylabel="validation accuracy", ylim=(0.85, 1.0))
    ax2 = ax1.twinx()
    ax2.plot(ep, [h["sparsity"] for h in hist], color=COLORS["magnitude"], lw=1.6, label="sparsity")
    ax2.set_ylabel("sparsity", color=COLORS["magnitude"])
    ax2.set_ylim(0, 1)
    ax2.grid(False)
    ax2.spines["right"].set_visible(True)
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="lower center", fontsize=8)
    regrow_txt = "with regrowth" if not args.no_regrow else "no regrowth"
    ax1.set_title(f"Self-pruning to {args.sparsity:.0%} ({args.criterion}, {args.scope}, {regrow_txt}): "
                  f"test {r['test_acc']:.3f} vs dense {dense['test_acc']:.3f}", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "self_prune_timeline.png")

    print(f"\ntarget {args.sparsity:.2%}  achieved {r['sparsity']:.4%}  budget met: {report['budget_met']}")
    print(f"per-layer sparsity: { {k: round(v, 4) for k, v in report['per_layer_sparsity'].items()} }")
    print(f"active weights {r['params']['weights_active']} / {r['params']['weights_total']}")
    print(f"test acc: dense {dense['test_acc']:.4f}  pruned {r['test_acc']:.4f}  cost {report['accuracy_cost_pp']:+.2f} pp")
    print(f"regrown connections over training: {r['prune_summary']['total_regrown']}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
