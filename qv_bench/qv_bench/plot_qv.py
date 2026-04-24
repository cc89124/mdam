"""Generate publication-ready QV scaling plots from benchmark results.

Reads ``results.csv`` and produces a log-scale comparison of execution
times across simulators as a function of qubit count.

Uses the ``rsmf`` package to match Quantum journal formatting
(font sizes, column widths) derived from ``quantum-template.tex``.

Usage
-----
    python -m qv_bench.plot_qv [--input results.csv] [--output qv_scaling.pdf]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import rsmf  # noqa: E402

_PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

_SIM_STYLE: dict[str, dict[str, object]] = {
    "clifft": {"color": "C0", "marker": "o", "ls": "-", "label": "Clifft"},
    "qiskit": {"color": "C1", "marker": "s", "ls": "--", "label": "Qiskit-Aer"},
    "qulacs": {"color": "C2", "marker": "^", "ls": "-.", "label": "Qulacs"},
    "qsim": {"color": "C3", "marker": "D", "ls": ":", "label": "Qsim"},
    "qrack": {"color": "C4", "marker": "P", "ls": (0, (3, 1, 1, 1)), "label": "Qrack"},
}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot QV benchmark scaling results.")
    p.add_argument("--input", type=str, default=str(_PROJECT_ROOT / "results.csv"), help="Input CSV.")
    p.add_argument(
        "--output",
        type=str,
        default=str(_PROJECT_ROOT / "qv_scaling.pdf"),
        help="Output plot.",
    )
    p.add_argument(
        "--wide",
        action="store_true",
        help="Use two-column (wide) figure width.",
    )
    return p


def plot(
    csv_path: Path,
    output_path: Path,
    *,
    wide: bool = False,
) -> None:
    """Generate the QV scaling plot."""
    fmt = rsmf.setup(r"\documentclass[a4paper,twocolumn,11pt]{quantumarticle}")

    df = pd.read_csv(csv_path)
    success = df[df["status"] == "SUCCESS"].copy()
    success["exec_s"] = success["exec_s"].astype(float)

    stats = success.groupby(["simulator", "N"])["exec_s"].median()

    fig = fmt.figure(wide=wide)
    ax = fig.add_subplot(111)

    for sim in stats.index.get_level_values("simulator").unique():
        style = _SIM_STYLE.get(sim, {"color": "gray", "marker": "x", "ls": "-", "label": sim})
        sim_stats = stats.loc[sim]
        ax.plot(
            sim_stats.index.values,
            sim_stats.values,
            color=style["color"],
            marker=style["marker"],
            ls=style["ls"],
            label=str(style["label"]),
            markersize=2,
            linewidth=1.0,
        )

    ax.set_yscale("log")
    ax.set_xlabel("Number of Qubits ($N$)")
    ax.set_ylabel("Execution Time (s)")
    # Curves grow up and to the right, so upper-left is consistently empty;
    # place the legend there to free the space above the axes.
    ax.legend(
        loc="upper left",
        fontsize=6,
        handlelength=1.5,
        framealpha=0.85,
    )
    ax.grid(True, which="major", ls="-", alpha=0.3)
    ax.grid(True, which="minor", ls="-", alpha=0.1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point."""
    args = _build_parser().parse_args(argv)
    plot(
        Path(args.input),
        Path(args.output),
        wide=args.wide,
    )


if __name__ == "__main__":
    main()
