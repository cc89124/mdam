#!/usr/bin/env python3
"""Generate publication-quality plots from saved simulation data.

Produces:
- Inject+cultivate comparison (one plot per code distance).
- End-to-end combined plot (desaturation on top, T/S infidelity ratio
  on bottom, shared x-axis, one per code distance).

Usage:
    uv run python plot_results.py
    uv run python plot_results.py --e2e-only
    uv run python plot_results.py --ic-only
"""

from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import sinter

from lib.importance_sampling import (
    StratumResult,
    binomial_pmf,
    ratio_estimate,
    survival_rate,
)
from cultiv import split_by_gap_threshold

RESULTS_DIR = pathlib.Path(__file__).parent / "results"
OUTPUT_DIR = pathlib.Path(__file__).parent / "figures"

# Noise levels to report
NOISE_LEVELS_IC_D3 = [0.0005, 0.001, 0.002, 0.003, 0.005, 0.007, 0.01]
NOISE_LEVELS_IC_D5 = [0.0005, 0.0007, 0.001, 0.0015, 0.002]

N_SITES_IC = {3: 518, 5: 3564}

# SOFT Table III (Li et al., arXiv:2512.23037): d=5 T-gate inject+cultivate.
# Each entry: (noise, total_shots, preserved_shots, error_rate).
_SOFT_IC_D5 = [
    (0.0005, 134.4e9, 50.9e9, 1.57e-10),
    (0.001,   74.0e9, 10.6e9, 4.59e-9),
    (0.002,   28.9e9,  0.60e9, 3.41e-8),
]

# Minimum PMF coverage to report a noise level
# 0.995 allows d=5 S-proxy at p=0.001 (99.6% coverage with k=0..9)
# while still catching genuinely truncated points (p=0.002+ at <82%).
MIN_PMF_COVERAGE = 0.995



# =====================================================================
# Data loading
# =====================================================================

def load_strata(data_dir: pathlib.Path) -> dict[str, list[dict]]:
    """Load per-stratum checkpoint files (IC format).

    Sub-chunked strata (e.g. t_gate_d5_k4_chunk0.json, _chunk1.json)
    are aggregated by summing total_shots, passed_shots, and
    logical_errors so that each k appears exactly once per key.
    """
    # First pass: collect all records keyed by (circuit_dcolor, k)
    raw: dict[str, dict[int, list[dict]]] = {}
    for path in sorted(data_dir.glob("*_d*_k*.json")):
        with open(path) as f:
            data = json.load(f)
        key = f"{data['circuit']}_d{data['dcolor']}"
        k = data["k"]
        raw.setdefault(key, {}).setdefault(k, []).append(data)

    # Second pass: aggregate chunks per (key, k)
    results: dict[str, list[dict]] = {}
    for key, k_map in raw.items():
        strata = []
        for k, chunks in sorted(k_map.items()):
            if len(chunks) == 1:
                strata.append(chunks[0])
            else:
                # Aggregate sub-chunks
                agg = {
                    "circuit": chunks[0]["circuit"],
                    "dcolor": chunks[0]["dcolor"],
                    "k": k,
                    "total_shots": sum(c["total_shots"] for c in chunks),
                    "passed_shots": sum(c["passed_shots"] for c in chunks),
                    "logical_errors": sum(c.get("logical_errors", 0) for c in chunks),
                    "seconds": sum(c.get("seconds", 0) for c in chunks),
                }
                strata.append(agg)
        results[key] = strata

    return results


def load_e2e_results(data_dir: pathlib.Path) -> dict[str, dict[float, dict]]:
    """Load E2E result files with gap-binned histograms.

    Result files are named {circuit}_d{d}_p{noise}.json and contain
    shots, errors, discards, and custom_counts with C{gap}/E{gap} bins.

    Returns:
        Dict mapping "{circuit}_d{dcolor}" to {noise: {shots, errors,
        discards, custom_counts}}.
    """
    results: dict[str, dict[float, dict]] = {}

    for path in sorted(data_dir.glob("*_d*_p*.json")):
        try:
            with open(path) as f:
                data = json.load(f)
            # Must have custom_counts to be an E2E histogram file
            if "custom_counts" not in data:
                continue
            circuit = data["circuit"]
            dcolor = data["dcolor"]
            noise = data["noise"]
            key = f"{circuit}_d{dcolor}"
            results.setdefault(key, {})[noise] = data
        except (json.JSONDecodeError, KeyError):
            continue

    return results


# =====================================================================
# Convert importance-sampled data to sinter TaskStats
# =====================================================================

def _ic_strata_to_taskstats(
    strata_data: list[dict],
    N_sites: int,
    noise_levels: list[float],
    circuit_label: str,
    dcolor: int,
) -> list[sinter.TaskStats]:
    """Convert inject+cultivate strata to sinter TaskStats per noise level.

    Uses the ratio estimator and Delta Method std error from
    lib/importance_sampling to derive effective shots and errors for
    sinter's confidence regions. This correctly represents the
    importance-sampling variance at each reweighted noise level.

    PMF coverage is checked against actually sampled k values (not
    the contiguous range 0..max_k), so partial/resumed runs with
    missing strata are correctly detected.

    Zero-error points are included so that early runs show up as
    upper bounds rather than silently disappearing.
    """
    strata = []
    for d in strata_data:
        strata.append(StratumResult(
            k=d["k"], total_shots=d["total_shots"],
            passed_shots=d["passed_shots"],
            n_errors=d.get("logical_errors", 0),
        ))

    if not strata:
        return []

    max_k = max(sr.k for sr in strata)
    sampled_ks = {sr.k for sr in strata}
    gate = "T" if circuit_label == "t_gate" else "S"
    result = []

    for p in noise_levels:
        P_K = binomial_pmf(N_sites, p, max_k)

        # Check coverage against actually sampled strata
        coverage = sum(P_K[k] for k in sampled_ks if k <= max_k)
        if coverage < MIN_PMF_COVERAGE:
            continue

        est, std_err = ratio_estimate(P_K, strata)
        surv = survival_rate(P_K, strata)

        if surv <= 0:
            continue

        # Derive effective shots/errors for sinter from the ratio
        # estimator. If std_err > 0, effective_n = est*(1-est)/std_err^2
        # gives the binomial sample size with matching variance.
        # After rounding errors to an integer, recalculate effective_kept
        # from errors/est so sinter's MLE dot (errors/effective_kept)
        # matches our ratio estimate exactly.
        if est > 0 and std_err > 0:
            effective_n = max(est * (1 - est) / (std_err ** 2), 1.0)
            errors = max(1, int(round(est * effective_n)))
            effective_kept = int(round(errors / est))
        elif est > 0:
            # std_err == 0 (single stratum or degenerate): use raw counts
            total_errors = sum(int(sr.sum_U) for sr in strata)
            errors = max(1, total_errors)
            effective_kept = int(round(errors / est))
        else:
            # Zero error rate: show as upper bound (0 errors)
            total_passed = sum(sr.passed_shots for sr in strata)
            errors = 0
            effective_kept = max(1, total_passed)

        effective_shots = int(round(effective_kept / surv)) if surv > 0 else effective_kept
        discards = effective_shots - effective_kept

        result.append(sinter.TaskStats(
            strong_id=f"{circuit_label}_d{dcolor}_p{p}",
            decoder="clifft",
            json_metadata={
                "d1": dcolor, "p": p, "gate": gate,
                "r1": dcolor,
            },
            shots=effective_shots,
            errors=errors,
            discards=discards,
        ))

    return result


# =====================================================================
# Plotting
# =====================================================================

def _compute_ic_ratio(
    t_stats: list[sinter.TaskStats],
    s_stats: list[sinter.TaskStats],
    rng: np.random.Generator,
) -> dict | None:
    """Bootstrap the T/S error-rate ratio across shared noise levels.

    Uses the effective (errors, kept) counts already encoded in the IC
    TaskStats by ``_ic_strata_to_taskstats``.  Those counts are
    moment-matched to reproduce the importance-sampling variance under
    a Binomial model, which makes it valid to feed them into the same
    Beta(errors + 0.5, kept - errors + 0.5) posterior we use for the
    raw-Monte-Carlo end-to-end ratio.  Returns arrays sorted by
    ``xs_apk`` (== survival-driven attempts per kept shot), plus the
    per-point ``err_mins`` needed for tiered rendering.
    """
    t_by_p = {s.json_metadata["p"]: s for s in t_stats}
    s_by_p = {s.json_metadata["p"]: s for s in s_stats}
    shared_ps = sorted(set(t_by_p) & set(s_by_p))

    xs_apk, medians, lows, highs, err_mins, ps_out = [], [], [], [], [], []
    for p in shared_ps:
        t = t_by_p[p]
        s = s_by_p[p]
        t_kept = t.shots - t.discards
        s_kept = s.shots - s.discards
        if t_kept < 10 or s_kept < 10:
            continue
        samples_t = rng.beta(
            t.errors + 0.5, t_kept - t.errors + 0.5, size=N_BOOTSTRAP,
        )
        samples_s = rng.beta(
            s.errors + 0.5, s_kept - s.errors + 0.5, size=N_BOOTSTRAP,
        )
        ratio = samples_t / samples_s
        xs_apk.append((t.shots + 1) / (t_kept + 2))
        medians.append(float(np.median(ratio)))
        lo, hi = np.percentile(ratio, [2.5, 97.5])
        lows.append(float(lo))
        highs.append(float(hi))
        err_mins.append(min(t.errors, s.errors))
        ps_out.append(p)

    if not xs_apk:
        return None

    order = np.argsort(xs_apk)
    return {
        "xs_apk": np.array(xs_apk)[order],
        "ps": [ps_out[i] for i in order],
        "medians": np.array(medians)[order],
        "lows": np.array(lows)[order],
        "highs": np.array(highs)[order],
        "err_mins": np.array(err_mins)[order],
    }


def _make_figure(figsize: tuple[float, float], *, rsmf_fmt=None, wide: bool = False,
                 nrows: int = 1, gridspec_kw=None, sharex: bool = False):
    """Create a figure + axes pair, honoring an optional rsmf formatter.

    When ``rsmf_fmt`` is provided, the figure width is governed by the
    Quantum template and the height is derived from the caller's
    ``figsize`` aspect ratio (so multi-panel layouts keep the vertical
    space their gridspec was designed for).  Without a formatter,
    ``figsize`` is used directly.
    """
    if rsmf_fmt is not None:
        aspect_ratio = figsize[1] / figsize[0]
        fig = rsmf_fmt.figure(wide=wide, aspect_ratio=aspect_ratio)
        if nrows == 1:
            ax = fig.add_subplot(111)
            return fig, ax
        axes = fig.subplots(nrows, 1, gridspec_kw=gridspec_kw, sharex=sharex)
        return fig, axes
    if nrows == 1:
        fig, ax = plt.subplots(figsize=figsize)
        return fig, ax
    fig, axes = plt.subplots(
        nrows, 1, figsize=figsize, gridspec_kw=gridspec_kw, sharex=sharex,
    )
    return fig, axes


def _plot_ic_single(
    all_results: dict[str, list[dict]],
    dcolor: int,
    output_path: pathlib.Path,
    *,
    rsmf_fmt=None,
):
    """Plot IC comparison (desaturation + T/S ratio) for a single distance."""
    noise_levels = NOISE_LEVELS_IC_D3 if dcolor == 3 else NOISE_LEVELS_IC_D5
    n_sites = N_SITES_IC.get(dcolor, 518)

    t_stats: list[sinter.TaskStats] = []
    s_stats: list[sinter.TaskStats] = []
    for key, strata_data in sorted(all_results.items()):
        parts = key.split("_d")
        circuit_label = parts[0]
        if int(parts[1]) != dcolor:
            continue
        stats = _ic_strata_to_taskstats(
            strata_data, n_sites, noise_levels, circuit_label, dcolor,
        )
        if circuit_label == "t_gate":
            t_stats.extend(stats)
        elif circuit_label == "s_proxy":
            s_stats.extend(stats)

    all_stats = t_stats + s_stats
    if not all_stats:
        print(f"No inject+cultivate data for d={dcolor}")
        return

    ratio_data = None
    if t_stats and s_stats:
        ratio_data = _compute_ic_ratio(
            t_stats, s_stats, rng=np.random.default_rng(42),
        )

    if ratio_data is not None:
        fig, (ax_top, ax_bot) = _make_figure(
            (10, 8),
            rsmf_fmt=rsmf_fmt, wide=True, nrows=2,
            gridspec_kw={"height_ratios": [3, 1.5]}, sharex=True,
        )
    else:
        fig, ax_top = _make_figure((10, 7), rsmf_fmt=rsmf_fmt, wide=True)
        ax_bot = None

    # --- Top panel: desaturation ---
    # Color encodes gate; same marker "o" and solid line everywhere.
    _GATE_COLOR = {"T": "tab:blue", "S": "tab:orange"}
    _GATE_LABEL = {"T": "T-Gate", "S": "S-Proxy"}

    def _x_func(stat):
        return (stat.shots + 1) / (stat.shots - stat.discards + 2)

    sinter.plot_error_rate(
        ax=ax_top,
        stats=all_stats,
        x_func=_x_func,
        group_func=lambda stat: f"gate={stat.json_metadata['gate']}",
        plot_args_func=lambda idx, group_key, group_stats: {
            "color": _GATE_COLOR.get(
                group_stats[0].json_metadata["gate"], "gray"
            ),
            "linestyle": "-",
            "marker": "o",
            "label": _GATE_LABEL.get(
                group_stats[0].json_metadata["gate"],
                group_stats[0].json_metadata["gate"],
            ),
        },
        highlight_max_likelihood_factor=1000,
    )

    # Overlay SOFT reference points on the d=5 T-gate curve.
    if dcolor == 5:
        soft_x = [total / kept for _, total, kept, _ in _SOFT_IC_D5]
        soft_y = [rate for _, _, _, rate in _SOFT_IC_D5]
        ax_top.scatter(
            soft_x, soft_y,
            marker="*", s=140, facecolors="none", edgecolors="black",
            linewidths=1.2, zorder=5, label="SOFT (Li et al.)",
        )

    ax_top.set_xscale("log")
    ax_top.set_yscale("log")
    ax_top.set_ylabel("Logical Error Rate (per Kept Shot)")
    ax_top.grid(True, which="major", axis="y", linewidth=0.8, alpha=0.5, color="gray")
    ax_top.grid(True, which="minor", axis="y", linewidth=0.5, alpha=0.4, color="gray")
    ax_top.grid(True, which="major", axis="x", linewidth=0.5, alpha=0.3, color="gray")

    # Inline p-labels: one per noise level, anchored at the T dot if
    # available (else S), placed just to the right of the point.
    t_by_p = {s.json_metadata["p"]: s for s in t_stats}
    s_by_p = {s.json_metadata["p"]: s for s in s_stats}
    for p in sorted(set(t_by_p) | set(s_by_p)):
        anchor_stat = t_by_p.get(p, s_by_p.get(p))
        if anchor_stat is None:
            continue
        kept = max(1, anchor_stat.shots - anchor_stat.discards)
        x_a = _x_func(anchor_stat)
        y_a = (anchor_stat.errors + 0.5) / (kept + 1)
        ax_top.annotate(
            f"p={p}",
            xy=(x_a, y_a),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
        )

    # Dedupe legend to 2 entries.
    handles, labels = ax_top.get_legend_handles_labels()
    seen: set[str] = set()
    dedup: list[tuple] = []
    for h, l in zip(handles, labels):
        if l in seen:
            continue
        seen.add(l)
        dedup.append((h, l))
    ax_top.legend(
        [h for h, _ in dedup],
        [l for _, l in dedup],
        loc="upper left",
        fontsize=10,
    )

    # --- Bottom panel: ratio ---
    # Single gray shade; the 95% credible-interval band width carries
    # the confidence signal on its own, so no tier gating is applied.
    if ax_bot is not None and ratio_data is not None:
        xs = ratio_data["xs_apk"]
        m = ratio_data["medians"]
        lo = ratio_data["lows"]
        hi = ratio_data["highs"]

        SHADE = "0.3"
        ax_bot.plot(xs, m, "o-", color=SHADE, markersize=4)
        ax_bot.fill_between(xs, lo, hi, color=SHADE, alpha=0.2, linewidth=0)

        ax_bot.set_xscale("log")
        ax_bot.set_xlabel("Expected Attempts per Kept Shot")
        ax_bot.set_ylabel("Error Ratio (T / S)")
        ax_bot.grid(True, which="major", linewidth=0.8, alpha=0.5, color="black")
        ax_bot.grid(True, which="minor", linewidth=0.4, alpha=0.2, color="black")
    else:
        ax_top.set_xlabel("Expected Attempts per Kept Shot")

    # Stretch the x-axis on the right so the rightmost inline p-label
    # has room to render without clipping.
    _, xmax = ax_top.get_xlim()
    ax_top.set_xlim(right=xmax * 1.6)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    print(f"Saved: {output_path}")
    plt.close(fig)


def plot_ic_comparison(
    all_results: dict[str, list[dict]],
    output_dir: pathlib.Path,
    *,
    fmt: str = "png",
    rsmf_fmt=None,
):
    """Inject+cultivate: separate combined plots per distance."""
    dcolors_present = set()
    for key in all_results:
        parts = key.split("_d")
        dcolors_present.add(int(parts[1]))

    for dcolor in sorted(dcolors_present):
        _plot_ic_single(
            all_results, dcolor,
            output_dir / f"ic_comparison_d{dcolor}.{fmt}",
            rsmf_fmt=rsmf_fmt,
        )


def _e2e_results_to_taskstats(
    result_data: dict,
    circuit_label: str,
    dcolor: int,
    gap_rounding: int = 2,
) -> list[sinter.TaskStats]:
    """Convert E2E histogram result to gap-threshold TaskStats.

    Uses Gidney's split_by_gap_threshold to expand custom_counts into
    cumulative threshold points, matching his plotting style exactly.
    """
    counts = result_data.get("custom_counts", {})
    if not counts:
        return []

    import collections
    gate = "T" if circuit_label == "t_gate" else "S"
    noise = result_data["noise"]

    # Build a single TaskStats with the histogram in custom_counts
    stat = sinter.TaskStats(
        strong_id=f"{circuit_label}_d{dcolor}_p{noise}",
        decoder="clifft",
        json_metadata={
            "d1": dcolor, "p": noise, "gate": gate, "r1": dcolor,
        },
        shots=result_data["shots"],
        errors=result_data["errors"],
        discards=result_data["discards"],
        custom_counts=collections.Counter(counts),
    )

    # Use Gidney's split_by_gap_threshold to expand into per-threshold points
    return split_by_gap_threshold([stat], gap_rounding=gap_rounding)


def _load_gidney_reference(
    ref_csv: pathlib.Path,
    gap_rounding: int = 2,
    dcolor_filter: int | None = None,
    noise_filter: set[float] | None = None,
) -> list[sinter.TaskStats]:
    """Load Gidney et al. E2E reference data and expand via gap threshold."""
    if not ref_csv.exists():
        return []
    raw_stats = sinter.read_stats_from_csv_files(str(ref_csv))
    # Tag each stat with gate="S" (source="gidney" preserves distinction
    # from internal s_proxy when needed; the filter logic in the plot
    # functions keeps Gidney data as the single S source where available).
    tagged = []
    for s in raw_stats:
        m = dict(s.json_metadata)
        if dcolor_filter is not None and m.get("d1") != dcolor_filter:
            continue
        if noise_filter is not None and m.get("p") not in noise_filter:
            continue
        m["gate"] = "S"
        m["source"] = "gidney"
        tagged.append(sinter.TaskStats(
            strong_id=s.strong_id,
            decoder=s.decoder,
            json_metadata=m,
            shots=s.shots,
            errors=s.errors,
            discards=s.discards,
            custom_counts=s.custom_counts,
        ))
    return split_by_gap_threshold(tagged, gap_rounding=gap_rounding)


def _compute_e2e_error_counts(
    result_data: dict,
    circuit_label: str,
) -> list[tuple[float, int, int, int]]:
    """Compute (gap, total_shots, kept_shots, errors) from gap histogram.

    Uses split_by_gap_threshold to get cumulative threshold points.
    """
    stats = _e2e_results_to_taskstats(
        result_data, circuit_label,
        int(result_data["dcolor"]),
    )
    total_shots = result_data["shots"]
    result = []
    for stat in stats:
        gap = stat.json_metadata.get("gap", 0)
        n_kept = stat.shots - stat.discards
        n_errors = stat.errors
        if n_kept > 0:
            result.append((gap, total_shots, n_kept, n_errors))
    return result


N_BOOTSTRAP = 100_000


def _gidney_gap_counts(
    ref_csv: pathlib.Path,
    dcolor: int,
    noise: float,
    gap_rounding: int = 2,
) -> list[tuple[float, int, int, int]]:
    """Extract (gap, total_shots, kept_shots, errors) from Gidney reference.

    Returns the same format as _compute_e2e_error_counts so the ratio
    code can use either source interchangeably.
    """
    stats = _load_gidney_reference(
        ref_csv, gap_rounding=gap_rounding,
        dcolor_filter=dcolor, noise_filter={noise},
    )
    result = []
    for stat in stats:
        gap = stat.json_metadata.get("gap", 0)
        n_kept = stat.shots - stat.discards
        n_errors = stat.errors
        total_shots = stat.shots  # Gidney stats already have raw shot count
        if n_kept > 0:
            result.append((gap, total_shots, n_kept, n_errors))
    return result


def _compute_ratio_series(
    bf_results: dict[str, dict[float, dict]],
    dcolor: int,
    ref_csv: pathlib.Path | None = None,
    min_err: int = 0,
    rng: np.random.Generator | None = None,
) -> list[tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]]:
    """Compute ratio series for a single distance.

    Returns list of (noise, xs_gap, xs_apk, medians, lows, highs, s_label)
    tuples — one per noise level with data.  xs_gap and xs_apk are parallel
    arrays so the caller can choose which x-axis to use.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    t_gate_data: dict[float, dict] = {}
    for key, noise_data in bf_results.items():
        parts = key.split("_d")
        circuit_label = parts[0]
        key_dc = int(parts[1])
        if circuit_label != "t_gate" or key_dc != dcolor:
            continue
        for noise, chunk_data in noise_data.items():
            t_gate_data[noise] = chunk_data

    series = []
    for noise in sorted(t_gate_data):
        t_data = t_gate_data[noise]
        t_counts = _compute_e2e_error_counts(t_data, "t_gate")
        t_by_gap = {gap: (total, kept, errs) for gap, total, kept, errs in t_counts}

        s_counts = []
        s_label = "S"
        if ref_csv is not None and ref_csv.exists():
            s_counts = _gidney_gap_counts(ref_csv, dcolor, noise)
        if not s_counts:
            s_key = f"s_proxy_d{dcolor}"
            if s_key in bf_results and noise in bf_results[s_key]:
                s_counts = _compute_e2e_error_counts(
                    bf_results[s_key][noise], "s_proxy",
                )
        if not s_counts:
            continue

        s_by_gap = {gap: (total, kept, errs) for gap, total, kept, errs in s_counts}

        xs_gap = []
        xs_apk = []
        medians = []
        lows = []
        highs = []
        err_mins = []  # per-point min(t_errs, s_errs) for tiered plotting

        shared_gaps = sorted(set(t_by_gap) & set(s_by_gap))
        for gap_thr in shared_gaps:
            t_total, t_kept, t_errs = t_by_gap[gap_thr]
            s_total, s_kept, s_errs = s_by_gap[gap_thr]

            if t_kept < 10 or s_kept < 10:
                continue
            if t_errs < min_err or s_errs < min_err:
                continue

            samples_t = rng.beta(
                t_errs + 0.5, t_kept - t_errs + 0.5, size=N_BOOTSTRAP,
            )
            samples_s = rng.beta(
                s_errs + 0.5, s_kept - s_errs + 0.5, size=N_BOOTSTRAP,
            )
            ratio_samples = samples_t / samples_s

            xs_gap.append(gap_thr)
            xs_apk.append(t_total / t_kept if t_kept > 0 else 1.0)
            medians.append(float(np.median(ratio_samples)))
            lo, hi = np.percentile(ratio_samples, [2.5, 97.5])
            lows.append(float(lo))
            highs.append(float(hi))
            err_mins.append(min(t_errs, s_errs))

        if not xs_gap:
            continue

        xs_gap = np.array(xs_gap)
        xs_apk = np.array(xs_apk)
        medians = np.array(medians)
        lows = np.array(lows)
        highs = np.array(highs)
        err_mins = np.array(err_mins)

        order = np.argsort(xs_apk)
        xs_gap = xs_gap[order]
        xs_apk = xs_apk[order]
        medians = medians[order]
        lows = lows[order]
        highs = highs[order]
        err_mins = err_mins[order]

        series.append((noise, xs_gap, xs_apk, medians, lows, highs, s_label, err_mins))

    return series


def _ic_ratios_for_dcolor(
    dcolor: int, ic_dir: pathlib.Path,
) -> dict[float, tuple[float, float, float]]:
    """Return ``{p: (median, low, high)}`` for the IC T/S ratio at ``dcolor``.

    Used by ``plot_e2e_combined`` to overlay the IC asymptote as a dashed
    reference line on the E2E ratio panel.  Empty dict if IC data is
    missing or computation failed.
    """
    if not ic_dir.exists():
        return {}
    ic_data = load_strata(ic_dir)
    noise_levels = NOISE_LEVELS_IC_D3 if dcolor == 3 else NOISE_LEVELS_IC_D5
    n_sites = N_SITES_IC.get(dcolor, 518)

    t_stats, s_stats = [], []
    for key, strata in sorted(ic_data.items()):
        parts = key.split("_d")
        if int(parts[1]) != dcolor:
            continue
        stats = _ic_strata_to_taskstats(
            strata, n_sites, noise_levels, parts[0], dcolor,
        )
        if parts[0] == "t_gate":
            t_stats.extend(stats)
        elif parts[0] == "s_proxy":
            s_stats.extend(stats)

    rd = _compute_ic_ratio(t_stats, s_stats, rng=np.random.default_rng(42))
    if rd is None:
        return {}
    return {
        p: (float(m), float(lo), float(hi))
        for p, m, lo, hi in zip(
            rd["ps"], rd["medians"], rd["lows"], rd["highs"],
        )
    }


def plot_e2e_combined(
    bf_results: dict[str, dict[float, dict]],
    output_dir: pathlib.Path,
    ref_csv: pathlib.Path | None = None,
    noise_filter: set[float] | None = None,
    min_errors: dict[int, int] | int = 0,
    *,
    fmt: str = "png",
    rsmf_fmt=None,
):
    """Stacked 2-panel figure: desaturation on top, ratio below, shared x-axis."""
    dcolors = set()
    for key in bf_results:
        parts = key.split("_d")
        dcolors.add(int(parts[1]))

    for dcolor in sorted(dcolors):
        min_err = min_errors.get(dcolor, 0) if isinstance(min_errors, dict) else min_errors
        # Compute the full ratio series (no error-count threshold).
        # Per-point error counts decide how each point gets rendered
        # (confident tier / nonzero tail / zero-error tail).
        ratio_full = _compute_ratio_series(
            bf_results, dcolor, ref_csv=ref_csv, min_err=0,
            rng=np.random.default_rng(42),
        )

        # Build desaturation stats (same logic as _plot_e2e_desaturation_single)
        all_stats = []
        for key, noise_data in sorted(bf_results.items()):
            parts = key.split("_d")
            circuit_label = parts[0]
            key_dcolor = int(parts[1])
            if key_dcolor != dcolor:
                continue
            for noise, result_data in sorted(noise_data.items()):
                stats = _e2e_results_to_taskstats(
                    result_data, circuit_label, key_dcolor,
                )
                all_stats.extend(stats)

        if ref_csv is not None:
            ref_stats = _load_gidney_reference(ref_csv, dcolor_filter=dcolor,
                                               noise_filter=noise_filter)
            if ref_stats:
                gidney_noises = {s.json_metadata["p"] for s in ref_stats}
                all_stats = [
                    s for s in all_stats
                    if not (s.json_metadata.get("gate") == "S"
                            and s.json_metadata["p"] in gidney_noises)
                ]
                all_stats.extend(ref_stats)

        # Combined plot only shows noise levels where we have both T and S
        # curves, so every top-panel pair has a corresponding bottom-panel
        # ratio curve.
        t_ps = {s.json_metadata["p"] for s in all_stats
                if s.json_metadata["gate"] == "T"}
        s_ps = {s.json_metadata["p"] for s in all_stats
                if s.json_metadata["gate"] == "S"}
        both_ps = t_ps & s_ps
        all_stats = [s for s in all_stats if s.json_metadata["p"] in both_ps]

        if not all_stats or not ratio_full:
            continue

        fig, (ax_top, ax_bot) = _make_figure(
            (12, 10),
            rsmf_fmt=rsmf_fmt, wide=True, nrows=2,
            gridspec_kw={"height_ratios": [3, 1.5]}, sharex=True,
        )

        # --- Top panel: desaturation ---
        # Color encodes gate only (T=blue, S=orange); same marker "o" for
        # every curve.  Per-noise identification is via inline p-labels
        # placed at the rightmost point of each curve pair.
        _GATE_COLOR = {"T": "tab:blue", "S": "tab:orange"}
        _GATE_LABEL = {"T": "T-Gate", "S": "S-Proxy"}

        def _x_func(stat):
            return (stat.shots + 1) / (stat.shots - stat.discards + 2)

        sinter.plot_error_rate(
            ax=ax_top,
            stats=all_stats,
            x_func=_x_func,
            group_func=lambda stat: (
                f"gate={stat.json_metadata['gate']}, "
                f"p={stat.json_metadata['p']}"
            ),
            plot_args_func=lambda idx, group_key, group_stats: {
                "color": _GATE_COLOR.get(group_stats[0].json_metadata["gate"], "gray"),
                "linestyle": "-",
                "marker": "o",
                "label": _GATE_LABEL.get(
                    group_stats[0].json_metadata["gate"],
                    group_stats[0].json_metadata["gate"],
                ),
            },
            highlight_max_likelihood_factor=1000,
        )

        ax_top.set_xscale("log")
        ax_top.set_yscale("log")
        ax_top.set_ylabel("Infidelity (per Kept Shot)")
        ax_top.grid(True, which="major", axis="y", linewidth=0.8, alpha=0.5, color="gray")
        ax_top.grid(True, which="minor", axis="y", linewidth=0.5, alpha=0.4, color="gray")
        ax_top.grid(True, which="major", axis="x", linewidth=0.5, alpha=0.3, color="gray")

        # Inline p-labels: one per noise level, placed at the rightmost
        # point of the T curve if available (else the S curve), positioned
        # halfway between the T and S points when both are present.
        ps = sorted({s.json_metadata["p"] for s in all_stats})
        for p in ps:
            anchor = None
            for preferred_gate in ("T", "S"):
                candidates = [
                    s for s in all_stats
                    if s.json_metadata["gate"] == preferred_gate
                    and s.json_metadata["p"] == p
                ]
                if candidates:
                    s_right = max(candidates, key=_x_func)
                    kept = max(1, s_right.shots - s_right.discards)
                    x_a = _x_func(s_right)
                    y_a = (s_right.errors + 0.5) / (kept + 1)
                    anchor = (x_a, y_a)
                    break
            if anchor is None:
                continue
            ax_top.annotate(
                f"p={p}",
                xy=anchor,
                xytext=(8, 0),
                textcoords="offset points",
                va="center",
                fontsize=10,
            )

        # Deduplicate legend so it only shows one entry per gate color.
        handles, labels = ax_top.get_legend_handles_labels()
        seen: set[str] = set()
        dedup_h, dedup_l = [], []
        for h, l in zip(handles, labels):
            if l in seen:
                continue
            seen.add(l)
            dedup_h.append(h)
            dedup_l.append(l)
        ax_top.legend(dedup_h, dedup_l, loc="upper right", fontsize=10)

        # --- Bottom panel: ratio ---
        # Grayscale so the bottom doesn't imply a shared color-language
        # with the top panel (which uses hues for gate).  Single shade
        # for all curves; the 95% credible-interval band width carries
        # the confidence signal.  Each curve is clipped to the rightmost
        # x actually plotted in the desaturation panel above so the
        # ratio doesn't extend past where the reader can see data.
        def _x_func_stat(s):
            return (s.shots + 1) / (s.shots - s.discards + 2)

        max_x_top_by_p: dict[float, float] = {}
        for s in all_stats:
            p = s.json_metadata["p"]
            x = _x_func_stat(s)
            if x > max_x_top_by_p.get(p, 0.0):
                max_x_top_by_p[p] = x

        # Single gray shade for all curves; band width alone carries
        # the confidence signal.  Zero-error points are dropped entirely
        # since their Beta(0.5, kept+0.5) posteriors are prior-dominated
        # and the ratio is ill-defined.
        SHADE = "0.3"
        # For y-axis scaling we only include CI bounds from "confident"
        # points (err_min >= CI_CONF_THRESHOLD); points in the very
        # noisy tail still plot but don't drive the y-limits, so the
        # figure stays focused on the scaling dynamics.
        CI_CONF_THRESHOLD = 100
        y_vals: list[float] = []

        # IC asymptote per noise level (dashed overlay).  Shows that
        # E2E(T/S) → IC(T/S) as the gap threshold tightens.
        ic_ref = _ic_ratios_for_dcolor(dcolor, RESULTS_DIR / "inject_cultivate")
        ic_line_drawn = False

        for noise, xs_gap, xs_apk, medians, lows, highs, sl, err_mins in ratio_full:
            cap = max_x_top_by_p.get(noise)
            if cap is None:
                continue
            mask = (xs_apk <= cap) & (err_mins > 0)
            if not mask.any():
                continue
            xs = xs_apk[mask]
            m = medians[mask]
            lo = lows[mask]
            hi = highs[mask]
            e = err_mins[mask]

            ax_bot.plot(xs, m, "o-", color=SHADE, markersize=4)
            ax_bot.fill_between(xs, lo, hi, color=SHADE, alpha=0.2, linewidth=0)

            # y-axis tracking: always include medians; include CI only
            # for points past the "confident" threshold so the noisy
            # tail doesn't blow out the y-range.
            y_vals.extend(m.tolist())
            conf = e >= CI_CONF_THRESHOLD
            if conf.any():
                y_vals.extend(lo[conf].tolist())
                y_vals.extend(hi[conf].tolist())

            # IC reference: dashed line at the IC median, spanning the
            # x-range of this E2E curve.  The IC CI is not drawn here
            # to keep the panel clean (it's shown in the IC plots).
            if noise in ic_ref:
                ic_med, ic_lo, ic_hi = ic_ref[noise]
                label = "I+C limit" if not ic_line_drawn else None
                ax_bot.plot(
                    [xs[0], xs[-1]], [ic_med, ic_med],
                    linestyle="--", color=SHADE, linewidth=1.2,
                    alpha=0.9, zorder=1.5, label=label,
                )
                ic_line_drawn = True
                y_vals.extend([ic_med, ic_lo, ic_hi])

            # p-label anchored at the rightmost plotted dot.  White
            # background keeps the label readable near neighboring curves.
            ax_bot.annotate(
                f"p={noise}",
                xy=(xs[-1], m[-1]),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                fontsize=9,
                color="black",
                bbox=dict(
                    facecolor="white", edgecolor="none",
                    pad=1.5, alpha=0.85,
                ),
            )

        if ic_line_drawn:
            ax_bot.legend(loc="upper left", fontsize=9)

        # Axis labels / decorations.
        ax_bot.set_xscale("log")
        ax_bot.set_xlabel("Expected Attempts per Kept Shot")
        ax_bot.set_ylabel("Infidelity Ratio (T / S)")
        ax_bot.grid(True, which="major", linewidth=0.8, alpha=0.5, color="black")
        ax_bot.grid(True, which="minor", linewidth=0.4, alpha=0.2, color="black")

        # Y-range covers all plotted medians + CI bands.  Zero-error
        # points are already excluded by the mask above.
        if y_vals:
            y_lo = min(y_vals)
            y_hi = max(y_vals)
            pad = 0.1 * (y_hi - y_lo) if y_hi > y_lo else 0.1
            ax_bot.set_ylim(max(0.0, y_lo - pad), y_hi + pad)

        # Stretch the x-axis slightly on the right so the rightmost
        # inline p-label has room to render without clipping.
        _, xmax = ax_top.get_xlim()
        ax_top.set_xlim(left=0.9, right=xmax * 1.6)

        fig.tight_layout()
        path = output_dir / f"e2e_combined_d{dcolor}.{fmt}"
        fig.savefig(path, dpi=200)
        print(f"Saved: {path}")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate publication plots")
    parser.add_argument("--e2e-only", action="store_true")
    parser.add_argument("--ic-only", action="store_true")
    parser.add_argument("--dcolor", type=int, help="Filter to a single code distance")
    parser.add_argument("--noise", type=float, action="append",
                        help="Filter to specific noise level(s); repeatable")
    parser.add_argument(
        "--format", choices=("png", "pdf"), default="png",
        help=(
            "Output format.  'pdf' uses the rsmf package to match Quantum "
            "journal formatting (requires the 'plot' extra)."
        ),
    )
    args = parser.parse_args()

    rsmf_fmt = None
    if args.format == "pdf":
        try:
            import rsmf
        except ImportError as exc:  # pragma: no cover - surfaces to user
            raise SystemExit(
                "--format pdf requires the 'rsmf' package; install with "
                "`uv sync --extra plot` from the magic_state_cultivation/ "
                "directory."
            ) from exc
        rsmf_fmt = rsmf.setup(
            r"\documentclass[a4paper,twocolumn,11pt]{quantumarticle}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not args.ic_only:
        e2e_dir = RESULTS_DIR / "end2end"
        if e2e_dir.exists():
            print("Loading end-to-end data...")
            e2e_hist = load_e2e_results(e2e_dir)

            # Apply --dcolor filter
            if args.dcolor is not None:
                e2e_hist = {
                    k: v for k, v in e2e_hist.items()
                    if k.endswith(f"_d{args.dcolor}")
                }

            # Apply --noise filter
            if args.noise:
                noise_set = set(args.noise)
                e2e_hist = {
                    k: {n: d for n, d in v.items() if n in noise_set}
                    for k, v in e2e_hist.items()
                }
                e2e_hist = {k: v for k, v in e2e_hist.items() if v}

            if e2e_hist:
                for k, noise_data in e2e_hist.items():
                    total = sum(d["shots"] for d in noise_data.values())
                    print(f"  {k} ({total:,} shots across {len(noise_data)} noise levels)")
                ref_csv = RESULTS_DIR / "reference" / "gidney_e2e.csv"
                noise_set = set(args.noise) if args.noise else None
                plot_e2e_combined(
                    e2e_hist, OUTPUT_DIR,
                    ref_csv=ref_csv if ref_csv.exists() else None,
                    noise_filter=noise_set,
                    min_errors={5: 100},
                    fmt=args.format,
                    rsmf_fmt=rsmf_fmt,
                )

    if not args.e2e_only:
        ic_dir = RESULTS_DIR / "inject_cultivate"
        if ic_dir.exists():
            print("Loading inject+cultivate data...")
            ic_data = load_strata(ic_dir)
            if ic_data:
                print(f"  {', '.join(f'{k} ({len(v)} strata)' for k, v in ic_data.items())}")
                plot_ic_comparison(
                    ic_data, OUTPUT_DIR,
                    fmt=args.format, rsmf_fmt=rsmf_fmt,
                )


if __name__ == "__main__":
    main()
