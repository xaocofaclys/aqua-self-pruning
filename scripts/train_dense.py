"""Part 2: train the dense MLP with our Adam (and SGD-momentum for reference).

    python scripts/train_dense.py            # digits + spirals, writes results/part2/
"""

from __future__ import annotations

import argparse

import _common  # noqa: F401  (thread pinning + sys.path)
from _common import COLORS, environment, out_dir, setup_matplotlib, write_json

from aqua.experiment import ExperimentConfig, run, serializable
from aqua.train import TrainConfig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=60)
    args = ap.parse_args()
    out = out_dir("part2")

    runs = {
        "digits_adam": ExperimentConfig(dataset="digits", train=TrainConfig(epochs=args.epochs, seed=args.seed, log_every_epoch=True)),
        "digits_sgd_momentum": ExperimentConfig(dataset="digits", train=TrainConfig(
            epochs=args.epochs, seed=args.seed, optimizer="sgd", lr=0.05, momentum=0.9)),
        "spirals_adam": ExperimentConfig(dataset="spirals", hidden=(64, 64), train=TrainConfig(
            epochs=150, seed=args.seed, lr=5e-3, batch_size=32)),
    }
    results = {}
    for name, cfg in runs.items():
        print(f"\n=== {name} ===", flush=True)
        r = run(cfg)
        results[name] = serializable(r)
        print(f"{name}: test acc {r['test_acc']:.4f}  test loss {r['test_loss']:.4f}  ({r['train_seconds']:.1f}s)")
    write_json(out / "histories.json", {"environment": environment(), "runs": results})

    plt = setup_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    styles = {"digits_adam": ("Adam (digits)", COLORS["fisher_obd"]),
              "digits_sgd_momentum": ("SGD+momentum (digits)", COLORS["magnitude"]),
              "spirals_adam": ("Adam (spirals)", COLORS["saliency"])}
    for name, r in results.items():
        label, color = styles[name]
        ep = [h["epoch"] for h in r["history"]]
        axes[0].plot(ep, [h["train_loss"] for h in r["history"]], color=color, label=label)
        axes[1].plot(ep, [h["val_loss"] for h in r["history"]], color=color, label=label)
        axes[2].plot(ep, [h["val_acc"] for h in r["history"]], color=color, label=label)
    axes[0].set(title="Training loss", xlabel="epoch", yscale="log")
    axes[1].set(title="Validation loss", xlabel="epoch", yscale="log")
    axes[2].set(title="Validation accuracy", xlabel="epoch", ylim=(0.5, 1.0))
    axes[2].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "learning_curves.png")
    print(f"\nwrote {out/'learning_curves.png'} and {out/'histories.json'}")


if __name__ == "__main__":
    main()
