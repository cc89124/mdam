#!/usr/bin/env python3
"""Inject+cultivate driver with per-circuit tiered shot budgets and sub-chunking.

Uses Clifft importance sampling (sample_k_survivors) with stratified ratio
estimation and Delta Method error bars. Reweights across noise levels
from a single simulation run.

Sub-chunking
~~~~~~~~~~~~
Large strata (e.g. 4B shots) are split into sub-chunks of SUB_CHUNK_SIZE
(default 500M) and submitted as separate work items. Each sub-chunk produces
its own checkpoint file named ``{key}_k{k}_chunk{chunk_id}.json``. During
analysis, all chunks for the same (key, k) are aggregated by summing
total_shots, passed_shots, and logical_errors.

Per-circuit tiers
~~~~~~~~~~~~~~~~~
Each circuit type (t_gate, s_proxy) can have different shot budgets at
different noise-index (k) ranges. For example, d=5 T-gate strata at
k=4..7 use 1B shots while S-proxy at k=5..9 uses 4B shots.

PMF coverage check
~~~~~~~~~~~~~~~~~~
Before computing error rates at each noise level, the analysis checks that
the sampled strata cover at least MIN_PMF_COVERAGE (99.9%) of the binomial
PMF. Noise levels with insufficient coverage are skipped with a warning.

Usage:
    uv run python run_ic_tiered.py                    # both d=3 and d=5
    uv run python run_ic_tiered.py --smoke             # quick validation (d=3, k=0..6)
    uv run python run_ic_tiered.py --d3-only           # d=3 only
    uv run python run_ic_tiered.py --d5-only           # d=5 only
    uv run python run_ic_tiered.py --workers 24        # set parallelism
"""

from __future__ import annotations

import argparse
import functools
import json
import multiprocessing
import os
import pathlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

multiprocessing.set_start_method("spawn", force=True)
print = functools.partial(print, flush=True)  # unbuffered output

import numpy as np

import clifft

from convert_s_to_t import (
    make_d3_inject_cultivate,
    make_d5_inject_cultivate,
)
from lib.importance_sampling import (
    StratumResult,
    binomial_pmf,
    ratio_estimate,
    survival_rate,
)
from cultiv import make_inject_and_cultivate_circuit
from gen import NoiseModel


# =====================================================================
# Constants
# =====================================================================

RESULTS_DIR = pathlib.Path(__file__).parent / "results" / "inject_cultivate"

# Noise level used for the single simulation run.
# Results are reweighted to other noise levels without re-simulating.
SIM_NOISE = 0.001

# Noise levels to report in the final sweep (superset of Tuloup's points).
NOISE_LEVELS = [0.0001, 0.0002, 0.0003, 0.0005, 0.0007, 0.001, 0.0015,
                0.002, 0.003, 0.005, 0.007, 0.01]

SUB_CHUNK_SIZE = 20_000_000  # 20M shots per sub-chunk (~200s at k=4 on c6i)

MIN_PMF_COVERAGE = 0.999

# =====================================================================
# Per-circuit tier definitions: {circuit_label: [(min_k, max_k, shots_per_k), ...]}
# =====================================================================

D3_TIERS = {
    "t_gate": [(0, 12, 10_000_000)],
    "s_proxy": [(0, 12, 10_000_000)],
}

D5_TIERS = {
    "t_gate": [
        (0, 2, 10_000),
        (3, 3, 2_000_000_000),
        (4, 4, 18_000_000_000),
        (5, 5, 32_000_000_000),
        (6, 6, 13_000_000_000),
        (7, 7, 11_000_000_000),
        (8, 8, 12_000_000_000),
        (9, 9, 2_000_000_000),
        (10, 14, 500_000_000),
        (15, 20, 100_000_000),
    ],
    "s_proxy": [
        (0, 4, 10_000),
        (5, 5, 4_000_000_000),
        (6, 6, 92_000_000_000),
        (7, 7, 124_000_000_000),
        (8, 8, 26_000_000_000),
        (9, 9, 9_000_000_000),
        (10, 10, 5_000_000_000),
        (11, 12, 4_000_000_000),
        (13, 15, 1_000_000_000),
        (16, 17, 500_000_000),
    ],
}

CIRCUITS = ["t_gate", "s_proxy"]


# =====================================================================
# Circuit compilation
# =====================================================================


def compile_t_gate(dcolor: int, noise: float) -> clifft.Program:
    """Compile a T-gate inject+cultivate circuit with all-detector postselection."""
    if dcolor == 3:
        text = make_d3_inject_cultivate(noise_strength=noise)
    elif dcolor == 5:
        text = make_d5_inject_cultivate(noise_strength=noise)
    else:
        raise ValueError(f"Unsupported dcolor={dcolor}")

    prog = clifft.compile(
        text,
        normalize_syndromes=True,
        hir_passes=clifft.default_hir_pass_manager(),
        bytecode_passes=clifft.default_bytecode_pass_manager(),
    )
    # All-detector postselection mask
    mask = [1] * prog.num_detectors
    return clifft.compile(
        text,
        normalize_syndromes=True,
        postselection_mask=mask,
        hir_passes=clifft.default_hir_pass_manager(),
        bytecode_passes=clifft.default_bytecode_pass_manager(),
    )


def compile_s_proxy(dcolor: int, noise: float) -> clifft.Program:
    """Compile an S-proxy inject+cultivate circuit for Clifft (Clifford)."""
    circuit = make_inject_and_cultivate_circuit(
        dcolor=dcolor, inject_style="unitary", basis="Y"
    )
    noisy = NoiseModel.uniform_depolarizing(noise).noisy_circuit_skipping_mpp_boundaries(
        circuit
    )
    text = str(noisy)

    prog = clifft.compile(
        text,
        normalize_syndromes=True,
        hir_passes=clifft.default_hir_pass_manager(),
        bytecode_passes=clifft.default_bytecode_pass_manager(),
    )
    mask = [1] * prog.num_detectors
    return clifft.compile(
        text,
        normalize_syndromes=True,
        postselection_mask=mask,
        hir_passes=clifft.default_hir_pass_manager(),
        bytecode_passes=clifft.default_bytecode_pass_manager(),
    )


# =====================================================================
# Per-stratum sampling (runs in worker processes)
# =====================================================================


def _sample_stratum(
    circuit_label: str,
    dcolor: int,
    noise: float,
    k: int,
    shots: int,
) -> dict:
    """Sample a single stratum and return serializable results.

    This function runs in a worker process. It compiles the circuit
    fresh (Clifft programs are not picklable).
    """
    t0 = time.monotonic()

    if circuit_label == "t_gate":
        prog = compile_t_gate(dcolor, noise)
    else:
        prog = compile_s_proxy(dcolor, noise)

    result = clifft.sample_k_survivors(prog, shots=shots, k=k)

    return {
        "circuit": circuit_label,
        "dcolor": dcolor,
        "k": k,
        "total_shots": result.total_shots,
        "passed_shots": result.passed_shots,
        "logical_errors": result.logical_errors,
        "seconds": time.monotonic() - t0,
    }


# =====================================================================
# Work item construction
# =====================================================================


def _existing_shots_and_max_chunk(key: str, k: int) -> tuple[int, int]:
    """Sum total shots from all existing checkpoint files for a (key, k).

    Returns (total_shots, max_chunk_id). max_chunk_id is -1 if only an
    unchunked file exists, or -2 if no files exist.
    """
    import glob as globmod

    total = 0
    max_cid = -2

    # Check unchunked file
    unchunked = RESULTS_DIR / f"{key}_k{k}.json"
    if unchunked.exists():
        with open(unchunked) as f:
            total += json.load(f)["total_shots"]
        max_cid = -1

    # Check all chunked files
    for path in sorted(RESULTS_DIR.glob(f"{key}_k{k}_chunk*.json")):
        with open(path) as f:
            total += json.load(f)["total_shots"]
        # Extract chunk ID from filename
        stem = path.stem  # e.g. "t_gate_d5_k4_chunk12"
        cid = int(stem.rsplit("chunk", 1)[1])
        max_cid = max(max_cid, cid)

    return total, max_cid


def _build_work_items(
    dcolors: list[int],
    smoke: bool = False,
) -> list[tuple[str, int, int, int, int]]:
    """Build work items to reach tier shot targets, accounting for existing data.

    For each (circuit, k), sums total_shots from all existing checkpoint
    files regardless of chunk size. Only creates new chunks for the
    remaining budget. New chunk IDs start after the highest existing ID.

    Returns list of (circuit_label, dcolor, k, shots, chunk_id) tuples.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    items: list[tuple[str, int, int, int, int]] = []
    skipped_shots = 0

    for dcolor in dcolors:
        tier_map = D3_TIERS if dcolor == 3 else D5_TIERS

        for circuit_label in CIRCUITS:
            tiers = tier_map.get(circuit_label, [])
            for min_k, max_k, shots in tiers:
                if smoke:
                    if dcolor != 3:
                        continue
                    min_k = max(min_k, 0)
                    max_k = min(max_k, 6)
                    shots = 100_000
                    if min_k > 6:
                        continue

                for k in range(min_k, max_k + 1):
                    key = f"{circuit_label}_d{dcolor}"
                    existing_shots, max_cid = _existing_shots_and_max_chunk(key, k)

                    if existing_shots >= shots:
                        skipped_shots += existing_shots
                        continue

                    remaining = shots - existing_shots
                    start_cid = max(max_cid + 1, 0)
                    cid = start_cid

                    while remaining > 0:
                        chunk_shots = min(SUB_CHUNK_SIZE, remaining)
                        items.append((circuit_label, dcolor, k, chunk_shots, cid))
                        remaining -= chunk_shots
                        cid += 1

    if skipped_shots:
        print(f"Skipping strata with {skipped_shots:,} existing shots (at or above target)")

    return items


def _chunk_count_for_shots(shots: int) -> int:
    """Return the number of sub-chunks for a given shot count."""
    if shots <= SUB_CHUNK_SIZE:
        return 1
    return (shots + SUB_CHUNK_SIZE - 1) // SUB_CHUNK_SIZE


# =====================================================================
# Checkpoint aggregation
# =====================================================================


def _load_and_aggregate(
    dcolors: list[int],
) -> dict[str, list[dict]]:
    """Load all checkpoint files and aggregate per (key, k).

    Discovers all checkpoint files by glob pattern rather than
    enumerating chunk IDs, so it works regardless of chunk size
    changes between runs. Both unchunked ({key}_k{k}.json) and
    chunked ({key}_k{k}_chunk{N}.json) files are aggregated.
    """
    all_results: dict[str, list[dict]] = {}

    for dcolor in dcolors:
        tier_map = D3_TIERS if dcolor == 3 else D5_TIERS

        for circuit_label in CIRCUITS:
            key = f"{circuit_label}_d{dcolor}"
            tiers = tier_map.get(circuit_label, [])

            # Collect all k values for this key
            k_values: set[int] = set()
            for min_k, max_k, shots in tiers:
                for k in range(min_k, max_k + 1):
                    k_values.add(k)

            strata: list[dict] = []
            for k in sorted(k_values):
                chunks_data: list[dict] = []

                # Load unchunked file
                unchunked = RESULTS_DIR / f"{key}_k{k}.json"
                if unchunked.exists():
                    with open(unchunked) as f:
                        chunks_data.append(json.load(f))

                # Load all chunked files
                for path in sorted(RESULTS_DIR.glob(f"{key}_k{k}_chunk*.json")):
                    with open(path) as f:
                        chunks_data.append(json.load(f))

                if not chunks_data:
                    continue

                agg = {
                    "circuit": circuit_label,
                    "dcolor": dcolor,
                    "k": k,
                    "total_shots": sum(c["total_shots"] for c in chunks_data),
                    "passed_shots": sum(c["passed_shots"] for c in chunks_data),
                    "logical_errors": sum(c["logical_errors"] for c in chunks_data),
                    "seconds": sum(c.get("seconds", 0) for c in chunks_data),
                }
                strata.append(agg)

            all_results[key] = strata

    return all_results


# =====================================================================
# Analysis with PMF coverage check
# =====================================================================


def analyze_results(
    all_results: dict[str, list[dict]],
    dcolors: list[int],
) -> dict:
    """Analyze results: compute error rates at each noise level via reweighting.

    Checks that sampled strata cover >= MIN_PMF_COVERAGE of the binomial
    PMF at each noise level. Skips noise levels with insufficient coverage.
    """
    analysis: dict[str, list[dict]] = {}

    for dcolor in dcolors:
        prog_probe = compile_t_gate(dcolor, SIM_NOISE)
        N_sites = len(prog_probe.noise_site_probabilities)

        for circuit_label in CIRCUITS:
            key = f"{circuit_label}_d{dcolor}"
            stratum_dicts = all_results.get(key, [])
            if not stratum_dicts:
                continue

            # Build StratumResult objects
            strata = []
            for d in stratum_dicts:
                strata.append(
                    StratumResult(
                        k=d["k"],
                        total_shots=d["total_shots"],
                        passed_shots=d["passed_shots"],
                        n_errors=d["logical_errors"],
                    )
                )

            max_k = max(sr.k for sr in strata)
            sampled_ks = set(sr.k for sr in strata)

            sweep: list[dict] = []
            for p in NOISE_LEVELS:
                P_K = binomial_pmf(N_sites, p, max_k)

                # PMF coverage check: sum P_K for sampled strata
                coverage = sum(P_K[k] for k in sampled_ks if k <= max_k)
                if coverage < MIN_PMF_COVERAGE:
                    print(
                        f"  WARNING [{key}] p={p:.4f}: PMF coverage "
                        f"{coverage:.4f} < {MIN_PMF_COVERAGE} -- skipping"
                    )
                    continue

                est, err = ratio_estimate(P_K, strata)
                surv = survival_rate(P_K, strata)
                sweep.append({
                    "p": p,
                    "error_rate": est,
                    "std_error": err,
                    "survival_rate": surv,
                    "attempts_per_kept": 1.0 / surv if surv > 0 else float("inf"),
                    "pmf_coverage": coverage,
                })

            analysis[key] = sweep

    return analysis


# =====================================================================
# Main
# =====================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Inject+cultivate driver with sub-chunking and per-circuit tiers",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Quick validation: 100k shots, d=3 only, k=0..6",
    )
    parser.add_argument("--d3-only", action="store_true", help="Run d=3 only")
    parser.add_argument("--d5-only", action="store_true", help="Run d=5 only")
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Number of parallel workers (default: physical cores / 2)",
    )
    args = parser.parse_args()

    num_workers = args.workers or max(1, (os.cpu_count() or 2) // 2)

    if args.d3_only and args.d5_only:
        print("ERROR: --d3-only and --d5-only cannot both be set")
        sys.exit(1)

    dcolors: list[int] = []
    if not args.d5_only:
        dcolors.append(3)
    if not args.d3_only:
        dcolors.append(5)

    if args.smoke:
        dcolors = [3]

    # ---- Print plan ----
    print("=" * 60)
    print("Inject+cultivate simulation")
    print("=" * 60)
    if args.smoke:
        print("  MODE: smoke test (d=3, k=0..6, 100k shots)")
    for dcolor in dcolors:
        tier_map = D3_TIERS if dcolor == 3 else D5_TIERS
        print(f"\n  d={dcolor}:")
        for circuit_label in CIRCUITS:
            tiers = tier_map.get(circuit_label, [])
            print(f"    {circuit_label}:")
            for min_k, max_k, shots in tiers:
                if args.smoke:
                    if dcolor != 3:
                        continue
                    min_k_eff = max(min_k, 0)
                    max_k_eff = min(max_k, 6)
                    shots_eff = 100_000
                    if min_k_eff > 6:
                        continue
                    n_strata = max_k_eff - min_k_eff + 1
                    n_chunks = 1
                    print(
                        f"      k={min_k_eff:2d}..{max_k_eff:2d}  "
                        f"({n_strata:2d} strata)  "
                        f"{shots_eff:>15,} shots/k"
                    )
                else:
                    n_strata = max_k - min_k + 1
                    n_chunks = _chunk_count_for_shots(shots)
                    chunk_info = f" ({n_chunks} chunks/k)" if n_chunks > 1 else ""
                    print(
                        f"      k={min_k:2d}..{max_k:2d}  "
                        f"({n_strata:2d} strata)  "
                        f"{shots:>15,} shots/k{chunk_info}"
                    )
    print(f"\n  Sub-chunk size: {SUB_CHUNK_SIZE:,}")
    print(f"  Workers: {num_workers}")
    print("=" * 60)
    print()

    # ---- Build work items ----
    items = _build_work_items(dcolors, smoke=args.smoke)
    if not items:
        print("All work items already completed. Proceeding to analysis.")
    else:
        print(f"Submitting {len(items)} work items to {num_workers} workers\n")

        # ---- Submit to pool ----
        t0 = time.monotonic()
        done_count = 0
        total_count = len(items)

        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            futures = {}
            for circuit_label, dcolor, k, shots, chunk_id in items:
                fut = pool.submit(
                    _sample_stratum, circuit_label, dcolor, SIM_NOISE, k, shots,
                )
                futures[fut] = (circuit_label, dcolor, k, shots, chunk_id)

            for fut in as_completed(futures):
                circuit_label, dcolor, k, shots, chunk_id = futures[fut]
                data = fut.result()
                done_count += 1

                # Save checkpoint — always use chunked naming so that
                # future runs with different chunk sizes can aggregate.
                key = f"{circuit_label}_d{dcolor}"
                checkpoint = RESULTS_DIR / f"{key}_k{k}_chunk{chunk_id}.json"

                with open(checkpoint, "w") as f:
                    json.dump(data, f)

                elapsed = time.monotonic() - t0
                avg = elapsed / done_count
                remaining = avg * (total_count - done_count)
                eta_min = remaining / 60

                print(
                    f"[{key}] k={k:2d} chunk{chunk_id}: "
                    f"passed={data['passed_shots']:,}/{data['total_shots']:,} "
                    f"errors={data['logical_errors']} "
                    f"({done_count}/{total_count}, "
                    f"{elapsed:.0f}s elapsed, "
                    f"ETA ~{remaining:.0f}s / {eta_min:.1f}m)"
                )

        total_elapsed = time.monotonic() - t0
        print(f"\nAll {total_count} work items completed in "
              f"{total_elapsed:.0f}s ({total_elapsed / 60:.1f}m)")

    # ---- Analysis ----
    print("\n=== ANALYSIS ===")
    all_results = _load_and_aggregate(dcolors)
    analysis = analyze_results(all_results, dcolors)

    for key, sweep in analysis.items():
        print(f"\n{key}:")
        for s in sweep:
            print(
                f"  p={s['p']:.4f}: "
                f"error_rate={s['error_rate']:.3e} +/- {s['std_error']:.3e} "
                f"(survival={s['survival_rate']:.3f}, "
                f"pmf_cov={s['pmf_coverage']:.4f})"
            )

    analysis_path = RESULTS_DIR / "analysis.json"
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\nAnalysis saved to {analysis_path}")


if __name__ == "__main__":
    main()
