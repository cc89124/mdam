"""Generate the LaTeX cost-comparison table for magic state cultivation.

Produces the inject+cultivate @ d=5 table (clifft vs SOFT).

Usage:
    uv run python generate_cost_tables.py
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_IC_DIR = _HERE / "results" / "inject_cultivate"


# ---------------------------------------------------------------------------
# External reference data (hard-coded from published papers)
# ---------------------------------------------------------------------------

# SOFT Table III (d=5 T-gate inject+cultivate; Li et al., arXiv:2512.23037).
# See https://arxiv.org/abs/2512.23037 for the source table.
_SOFT_IC = [
    {"p": 0.0005, "total": 134.4e9, "kept": 50.9e9, "errs": 8,  "rate": 1.57e-10},
    {"p": 0.001,  "total": 74.0e9,  "kept": 10.6e9, "errs": 49, "rate": 4.59e-9},
    {"p": 0.002,  "total": 28.9e9,  "kept": 0.60e9, "errs": 22, "rate": 3.41e-8},
]
_SOFT_HARDWARE = r"$16{\times}$ H800"
# Per-shot sampling time for d=5 cultivation on a single H800, from
# Table V of Li et al.  This gives 1/9.37e-5 s ≈ 10,670 shots/s per
# GPU.  Multiplying by the total d=5 shots (sum of Table III) backs
# out the GPU-hours consumed by the d=5 campaign specifically; the
# 20-day × 16-GPU figure in the paper text covers both d=3 and d=5.
_SOFT_TIME_PER_SHOT_S = 9.37e-5


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_ic_totals() -> dict[tuple[str, int], dict]:
    """Aggregate per-stratum IC checkpoints per (circuit, dcolor)."""
    agg: dict[tuple[str, int], dict] = {}
    for path in _IC_DIR.glob("*.json"):
        if path.name == "analysis.json":
            continue
        with open(path) as f:
            d = json.load(f)
        key = (d["circuit"], int(d["dcolor"]))
        a = agg.setdefault(
            key,
            {"total_shots": 0, "passed_shots": 0, "logical_errors": 0, "seconds": 0.0},
        )
        a["total_shots"] += d["total_shots"]
        a["passed_shots"] += d["passed_shots"]
        a["logical_errors"] += d["logical_errors"]
        a["seconds"] += d["seconds"]
    return agg


def _load_ic_rates() -> dict[str, list[dict]]:
    with open(_IC_DIR / "analysis.json") as f:
        return json.load(f)


def _ic_rate_at(rates: dict, circuit: str, dcolor: int, p: float) -> float | None:
    key = f"{circuit}_d{dcolor}"
    for entry in rates.get(key, []):
        if abs(entry["p"] - p) < 1e-12:
            return entry["error_rate"]
    return None


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _fmt_sci(x: float | None, digits: int = 2) -> str:
    """Format as ``$m{\\times}10^{e}$`` (LaTeX math)."""
    if x is None or not math.isfinite(x) or x == 0:
        return "--"
    exp = int(math.floor(math.log10(abs(x))))
    mant = x / 10**exp
    return f"${mant:.{digits}f}{{\\times}}10^{{{exp}}}$"


def _fmt_count(x: float) -> str:
    """Integer/count with SI-like suffix (always math for consistent kerning)."""
    return _fmt_sci(x, digits=1)


def _fmt_rate(x: float) -> str:
    """shots/s with k / M suffix (plain text), 3 sig figs."""
    if x >= 1e6:
        # 2.168M -> 2.17M (matches the round-trip to per-shot prose)
        return f"{x/1e6:.2f}M"
    if x >= 1e4:
        return f"{x/1e3:.0f}k"
    if x >= 1e3:
        return f"{x/1e3:.1f}k"
    return f"{x:.0f}"


def _fmt_hours(seconds: float) -> str:
    """seconds → formatted hours ('k' for thousands)."""
    h = seconds / 3600
    if h >= 10_000:
        return f"{h/1000:.0f}k"
    if h >= 1000:
        return f"{h/1000:.1f}k"
    if h >= 10:
        return f"{h:,.0f}"
    return f"{h:.1f}"


# ---------------------------------------------------------------------------
# Inject+Cultivate cost table @ d=5
# ---------------------------------------------------------------------------


_IC_NOISES = (0.0005, 0.001, 0.002)


def generate_ic_table() -> str:
    ic_totals = _load_ic_totals()
    rates = _load_ic_rates()

    rows: list[dict] = []

    # Our rows (d=5 only): one per circuit (t_gate, s_proxy).
    #
    # The raw `seconds` field in the IC JSON files is per-worker wall-
    # clock summed across all strata (see run_ic_tiered.py:200), and the
    # raw `rate = total_shots / seconds` is per-worker throughput.  We
    # report per-machine quantities by dividing seconds by `parallelism`
    # (= 16 worker processes on a c6i.8xlarge) and multiplying rate by
    # the same factor.  This puts both the Clifft and SOFT rows on a
    # whole-machine grain (one c6i.8xlarge instance vs one 16-H800
    # cluster).
    for circuit, label in (
        ("t_gate",  r"clifft ($T$-gate)"),
        ("s_proxy", r"clifft ($S$-proxy)"),
    ):
        a = ic_totals[(circuit, 5)]
        per_p = [_ic_rate_at(rates, circuit, 5, p) for p in _IC_NOISES]
        rows.append(
            {
                "label": label,
                "hw": "c6i.8xlarge (CPU)",
                "total": a["total_shots"],
                "seconds": a["seconds"],
                "rate": a["total_shots"] / a["seconds"],
                "parallelism": 16,  # 16 worker processes on the c6i.8xlarge
                "per_p": per_p,
            }
        )

    # SOFT row: one aggregated row (T-gate only; no S-proxy reported).
    # Per-cluster machine-hours are backed out from the measured single-
    # H800 per-shot time (Table V of Li et al.) rather than the overall
    # 20-day campaign figure (which also includes d=3 and other work).
    soft_total = sum(r["total"] for r in _SOFT_IC)
    soft_seconds = soft_total * _SOFT_TIME_PER_SHOT_S
    rows.append(
        {
            "label": r"SOFT ($T$-gate)",
            "hw": _SOFT_HARDWARE + " (GPU)",
            "total": soft_total,
            "seconds": soft_seconds,
            "rate": 1.0 / _SOFT_TIME_PER_SHOT_S,
            "parallelism": 16,  # 16 H800s in the published cluster
            "per_p": [r["rate"] for r in _SOFT_IC],  # already sorted 5e-4, 1e-3, 2e-3
        }
    )

    lines = []
    lines.append(r"    \begin{tabular}{l l r r r r r r}")
    lines.append(r"        \toprule")
    lines.append(
        r"        \textbf{Simulator} & \textbf{Hardware}"
        r" & \textbf{Total shots} & \textbf{Shots/s} & \textbf{Machine-h}"
        r" & $\boldsymbol{\epsilon_L}$($p{=}5{\times}10^{-4}$)"
        r" & $\boldsymbol{\epsilon_L}$($p{=}10^{-3}$)"
        r" & $\boldsymbol{\epsilon_L}$($p{=}2{\times}10^{-3}$) \\"
    )
    lines.append(r"        \midrule")
    for r in rows:
        eps = [_fmt_sci(v) for v in r["per_p"]]
        # Convert per-worker / per-GPU values to per-machine values.
        machine_rate = r["rate"] * r["parallelism"]
        machine_seconds = r["seconds"] / r["parallelism"]
        cols = [
            f"        {r['label']}",
            r["hw"],
            _fmt_count(r["total"]),
            _fmt_rate(machine_rate),
            _fmt_hours(machine_seconds),
            eps[0],
            eps[1],
            eps[2],
        ]
        lines.append(" & ".join(cols) + r" \\")
    lines.append(r"        \bottomrule")
    lines.append(r"    \end{tabular}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    caption_note = (
        r"% Note: clifft rows were collected on AWS c6i.8xlarge "
        r"(16 physical cores, Intel Xeon 8375C Ice Lake, 32 vCPU)."
        "\n"
        r"% Shots/s and Machine-h are per-machine: one c6i.8xlarge "
        r"instance for Clifft, one 16-H800 cluster for SOFT."
    )

    print("% Inject+Cultivate cost comparison (d=5)")
    print(generate_ic_table())
    print(caption_note)


if __name__ == "__main__":
    main()
