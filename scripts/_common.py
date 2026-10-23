"""Shared setup for the reproduction scripts.

Imported *first* by every script: pins BLAS to one thread before NumPy is
imported (bitwise-reproducible matmuls, and parallelism comes from running
independent seeds in separate processes instead).
"""

from __future__ import annotations

import csv
import json
import os
import platform
import sys
from pathlib import Path

if os.environ.get("AQUA_ALLOW_THREADS") != "1":
    # Forced, not setdefault: a shell that already exports OMP_NUM_THREADS=8
    # would otherwise silently un-pin BLAS and change the arithmetic.
    for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[_var] = "1"

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "results"

import numpy as np  # noqa: E402  (after the thread pinning)


def out_dir(name: str) -> Path:
    path = RESULTS / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(f"not JSON serializable: {type(o)}")


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, default=_default, sort_keys=False) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("no rows")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def environment() -> dict:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "blas_threads": os.environ.get("OMP_NUM_THREADS"),
    }


def setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 160,
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "legend.frameon": False,
    })
    return plt


# One fixed color per method across every figure.
COLORS = {
    "dense": "#555555",
    "magnitude": "#d1495b",
    "saliency": "#edae49",
    "fisher_obd": "#00798c",
    "taylor2": "#8e7dbe",
    "fisher_obd+regrow": "#30638e",
    "regrow_global_t": "#e07a5f",
    "regrow_mask_unaware": "#9c6644",
}
