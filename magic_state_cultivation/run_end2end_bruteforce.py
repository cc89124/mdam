#!/usr/bin/env python3
"""Phase 2: End-to-end infidelity via brute-force Monte Carlo.

Generates shots at a fixed noise level, decodes with the paper's
two-pass desaturation decoder, and bins results into gap histograms
(matching Gidney's stats.csv format).

Storage is extremely compact: ~200 integers per (circuit, noise)
instead of millions of per-shot floats. Results accumulate across
runs — re-running with higher --total-shots merges new counts into
existing histograms.

Usage:
    uv run python run_end2end_bruteforce.py --noise 0.001 --total-shots 1e6
    uv run python run_end2end_bruteforce.py --noise 0.0005 0.001 0.002 --total-shots 1e8
    uv run python run_end2end_bruteforce.py --noise 0.001 --total-shots 1e9 --workers 24
    uv run python run_end2end_bruteforce.py --smoke --noise 0.001
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import multiprocessing
import os
import pathlib
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

multiprocessing.set_start_method("spawn", force=True)
print = functools.partial(print, flush=True)  # unbuffered output

import numpy as np

import clifft

from convert_s_to_t import (
    make_d3_end2end_expval,
    make_d5_end2end_expval,
    make_d3_end2end_expval_sproxy,
    make_d5_end2end_expval_sproxy,
)
from lib.dual_decoder import (
    build_decoder,
    calibrate_static_sign,
)

RESULTS_DIR = pathlib.Path(__file__).parent / "results" / "end2end"

# Default chunk size: each chunk is one independent batch of shots.
# Smaller chunks = more frequent saves + finer progress updates.
# Larger chunks = less overhead from compilation + decoder init.
DEFAULT_CHUNK_SIZE = 250_000


# =====================================================================
# Circuit compilation
# =====================================================================


def _compile_expval(circuit_text: str) -> clifft.Program:
    """Compile an EXP_VAL circuit with cultivation-only postselection."""
    import stim as _stim
    import re as _re

    stim_text = _re.sub(r"^(T_DAG|T)\b", "I", circuit_text, flags=_re.MULTILINE)
    stim_text = _re.sub(r"^EXP_VAL\b.*$", "", stim_text, flags=_re.MULTILINE)
    stim_circuit = _stim.Circuit(stim_text)
    det_coords = stim_circuit.get_detector_coordinates()
    num_dets = stim_circuit.num_detectors

    # Post-select on cultivation-stage detectors only. The paper's
    # circuit convention uses coord[4] == -9 to mark injection/cultivation
    # detectors (see Gidney et al.'s step1_make_circuits). Detectors
    # with <= 4 coordinates are also cultivation-stage (pre-escape).
    # Escape-stage detectors are decoded, not post-selected.
    mask = [0] * num_dets
    for d in range(num_dets):
        coords = det_coords.get(d, [])
        if len(coords) <= 4 or (len(coords) > 4 and coords[4] == -9):
            mask[d] = 1

    return clifft.compile(
        circuit_text,
        normalize_syndromes=True,
        postselection_mask=mask,
        hir_passes=clifft.default_hir_pass_manager(),
        bytecode_passes=clifft.default_bytecode_pass_manager(),
    )


def _make_expval_text(circuit_label: str, dcolor: int, noise: float) -> str:
    """Generate EXP_VAL circuit text for a given circuit and distance."""
    if circuit_label == "t_gate":
        if dcolor == 3:
            return make_d3_end2end_expval(noise_strength=noise)
        else:
            return make_d5_end2end_expval(noise_strength=noise)
    else:
        if dcolor == 3:
            return make_d3_end2end_expval_sproxy(noise_strength=noise)
        else:
            return make_d5_end2end_expval_sproxy(noise_strength=noise)


# =====================================================================
# Per-worker caches (populated on first use, reused across chunks)
# =====================================================================

_PROG_CACHE: dict[str, clifft.Program] = {}
_DECODER_CACHE: dict[str, object] = {}
_SIGN_CACHE: dict[str, float] = {}


def _get_compiled(circuit_label: str, dcolor: int, noise: float) -> clifft.Program:
    """Get or compile an EXP_VAL program (cached per worker process)."""
    key = f"{circuit_label}_d{dcolor}_p{noise}"
    if key not in _PROG_CACHE:
        text = _make_expval_text(circuit_label, dcolor, noise)
        _PROG_CACHE[key] = _compile_expval(text)
    return _PROG_CACHE[key]


def _get_decoder_and_sign(circuit_label: str, dcolor: int, noise: float):
    """Get or build decoder + static sign (cached per worker process)."""
    dkey = f"d{dcolor}_p{noise}"
    skey = f"{circuit_label}_d{dcolor}_p{noise}"
    if dkey not in _DECODER_CACHE:
        _DECODER_CACHE[dkey] = build_decoder(dcolor=dcolor, noise_strength=noise)
    decoder = _DECODER_CACHE[dkey]
    if skey not in _SIGN_CACHE:
        text = _make_expval_text(circuit_label, dcolor, noise)
        ss_Y, _ = calibrate_static_sign(text, decoder.track_YL)
        _SIGN_CACHE[skey] = ss_Y
    return decoder, _SIGN_CACHE[skey]


# =====================================================================
# Per-chunk sampling (runs in worker processes)
# =====================================================================


def _sample_chunk(
    circuit_label: str,
    dcolor: int,
    noise: float,
    shots: int,
    chunk_id: int,
) -> dict:
    """Sample one chunk, decode, and return gap-binned histogram counts.

    Bins survivors by round(gap) into C{gap_bin} (correct) and E{gap_bin}
    (error) counts, matching Gidney's stats.csv format. This is extremely
    compact: ~200 integers instead of millions of per-shot floats.

    Runs in a worker process. Circuit compilation and decoder
    construction are cached per worker — only the first chunk pays
    the ~20s init cost.
    """
    t0 = time.monotonic()
    import collections

    prog = _get_compiled(circuit_label, dcolor, noise)
    decoder, ss_Y = _get_decoder_and_sign(circuit_label, dcolor, noise)

    # Sample with postselection — sample_survivors discards shots where
    # cultivation detectors fire (early exit optimization) and returns
    # only survivors. keep_records=True populates the result arrays.
    result = clifft.sample_survivors(prog, shots=shots, keep_records=True)
    total_shots = result.total_shots
    passed_shots = result.passed_shots

    if passed_shots == 0:
        return {
            "total_shots": total_shots,
            "passed_shots": 0,
            "errors": 0,
            "discards": total_shots,
            "custom_counts": {},
            "seconds": time.monotonic() - t0,
        }

    dets = result.detectors
    meas = result.measurements
    raw_Y = result.exp_vals[:, 1]

    # Postselect (ablatable + single-basis obs detectors)
    keep = decoder.postselect(dets)
    dets = dets[keep]
    meas = meas[keep]
    raw_Y = raw_Y[keep]

    if len(raw_Y) == 0:
        return {
            "total_shots": total_shots,
            "passed_shots": 0,
            "errors": 0,
            "discards": total_shots,
            "custom_counts": {},
            "seconds": time.monotonic() - t0,
        }

    # Decode + frame correct
    Y_corr, gaps = decoder.decode_and_correct(dets, meas, raw_Y, ss_Y)

    # Compute per-shot error indicator
    if circuit_label == "s_proxy":
        F = (1 + Y_corr) / 2
    else:
        F = 0.5 + Y_corr / np.sqrt(2)
    is_error = F < 0.5

    # Bin by round(gap), matching Gidney's stats.csv format:
    # C{n} = survivors in gap bin n, E{n} = errors in gap bin n
    counter: dict[str, int] = collections.Counter()
    for gap, err in zip(gaps, is_error):
        gap_bin = round(float(gap))
        counter[f"E{gap_bin}" if err else f"C{gap_bin}"] += 1

    n_errors = int(np.sum(is_error))
    n_survivors = len(Y_corr)

    return {
        "total_shots": total_shots,
        "passed_shots": n_survivors,
        "errors": n_errors,
        "discards": total_shots - n_survivors,
        "custom_counts": dict(counter),
        "seconds": time.monotonic() - t0,
    }


# =====================================================================
# Result file helpers (one JSON per circuit/noise, accumulates)
# =====================================================================


def _result_path(
    circuit_label: str, dcolor: int, noise: float,
) -> pathlib.Path:
    """Path for the accumulated result file."""
    return RESULTS_DIR / f"{circuit_label}_d{dcolor}_p{noise}.json"


def _load_result(
    circuit_label: str, dcolor: int, noise: float,
) -> dict:
    """Load existing result or return empty template."""
    path = _result_path(circuit_label, dcolor, noise)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {
        "circuit": circuit_label,
        "dcolor": dcolor,
        "noise": noise,
        "shots": 0,
        "errors": 0,
        "discards": 0,
        "custom_counts": {},
    }


def _merge_counts(existing: dict, new: dict) -> dict:
    """Merge new chunk histogram into existing accumulated result."""
    existing["shots"] += new["total_shots"]
    existing["errors"] += new["errors"]
    existing["discards"] += new["discards"]
    for k, v in new["custom_counts"].items():
        existing["custom_counts"][k] = existing["custom_counts"].get(k, 0) + v
    if new.get("seconds", 0) > 0:
        chunk_rate = new["total_shots"] / new["seconds"]
        old_rate = existing.get("shots_per_second")
        if old_rate is None:
            existing["shots_per_second"] = chunk_rate
        else:
            existing["shots_per_second"] = (old_rate + chunk_rate) / 2
    return existing


def _save_result(result: dict) -> None:
    """Save accumulated result to disk (atomic via temp + rename)."""
    path = _result_path(result["circuit"], result["dcolor"], result["noise"])
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(result, f, indent=2)
    os.replace(tmp, path)


# =====================================================================
# Main driver
# =====================================================================


def run_simulation(
    dcolors: list[int],
    circuits: list[str],
    noise_levels: list[float],
    total_shots: int,
    chunk_size: int,
    num_workers: int,
):
    """Run brute-force Monte Carlo for all circuits and noise levels.

    Results are accumulated into per-(circuit, noise) JSON files with
    gap-binned histograms. Each chunk's counts are merged into the
    existing file after completion. Re-running with higher --total-shots
    just adds more counts.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Build work items, checking existing accumulated shots
    work_items: list[tuple[str, int, float, int, int]] = []
    # Track accumulated results per key for merging
    accum: dict[str, dict] = {}

    for noise in noise_levels:
        for dcolor in dcolors:
            for circuit_label in circuits:
                key = f"{circuit_label}_d{dcolor}_p{noise}"
                existing = _load_result(circuit_label, dcolor, noise)
                accum[key] = existing
                existing_shots = existing["shots"]
                remaining_shots = max(0, total_shots - existing_shots)

                if remaining_shots <= 0:
                    print(
                        f"[{key}] Already have {existing_shots:,} "
                        f"shots (>= {total_shots:,} target). Skipping."
                    )
                    continue

                num_new_chunks = math.ceil(remaining_shots / chunk_size)
                print(
                    f"[{key}] {existing_shots:,} done, "
                    f"{remaining_shots:,} remaining "
                    f"({num_new_chunks} chunks of {chunk_size:,})"
                )

                for i in range(num_new_chunks):
                    shots_this_chunk = min(
                        chunk_size,
                        remaining_shots - i * chunk_size,
                    )
                    work_items.append(
                        (circuit_label, dcolor, noise, i, shots_this_chunk)
                    )

    if not work_items:
        print("\nAll targets already met. Nothing to do.")
        return

    print(f"\nSubmitting {len(work_items)} chunks to {num_workers} workers\n")

    t0 = time.monotonic()
    done_count = 0
    total_count = len(work_items)

    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        futures = {}
        for circuit_label, dcolor, noise, chunk_id, shots in work_items:
            fut = pool.submit(
                _sample_chunk, circuit_label, dcolor, noise, shots, chunk_id,
            )
            futures[fut] = (circuit_label, dcolor, noise)

        for fut in as_completed(futures):
            circuit_label, dcolor, noise = futures[fut]
            data = fut.result()
            done_count += 1
            key = f"{circuit_label}_d{dcolor}_p{noise}"

            # Merge into accumulated result and save
            _merge_counts(accum[key], data)
            _save_result(accum[key])

            chunk_errors = data["errors"]

            # Progress + ETA
            elapsed = time.monotonic() - t0
            avg = elapsed / done_count
            remaining_time = avg * (total_count - done_count)
            total_key_shots = accum[key]["shots"]
            total_key_errors = accum[key]["errors"]
            print(
                f"[{key}] "
                f"chunk: {data['passed_shots']:,}/{data['total_shots']:,} "
                f"errors={chunk_errors} | "
                f"total: {total_key_shots:,} shots, {total_key_errors} errors | "
                f"({done_count}/{total_count}, "
                f"{elapsed:.0f}s elapsed, ~{remaining_time:.0f}s remaining)"
            )

    total_elapsed = time.monotonic() - t0
    print(f"\nAll {total_count} chunks completed in "
          f"{total_elapsed:.0f}s ({total_elapsed / 60:.1f}m)")
    for key in sorted(accum):
        r = accum[key]
        if r["shots"] > 0:
            print(f"  {key}: {r['shots']:,} shots, {r['errors']} errors")


# =====================================================================
# CLI
# =====================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: End-to-end brute-force Monte Carlo",
    )
    parser.add_argument(
        "--noise", type=float, nargs="+", required=True,
        help="Physical noise level(s) (e.g. 0.001 or 0.0005 0.001 0.002)",
    )
    parser.add_argument(
        "--total-shots", type=float, default=None,
        help="Target total shots per (circuit, dcolor, noise). Supports "
             "scientific notation (e.g. 1e8). Required unless --smoke.",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
        help=f"Shots per chunk (default: {DEFAULT_CHUNK_SIZE:,})",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Number of parallel workers (default: physical cores / 2)",
    )
    parser.add_argument(
        "--d3-only", action="store_true",
        help="Only run d=3 circuits",
    )
    parser.add_argument(
        "--d5-only", action="store_true",
        help="Only run d=5 circuits",
    )
    parser.add_argument(
        "--t-gate-only", action="store_true",
        help="Only run T-gate circuits",
    )
    parser.add_argument(
        "--s-proxy-only", action="store_true",
        help="Only run S-proxy circuits",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Quick validation: 1000 shots, d=3 only",
    )
    args = parser.parse_args()

    num_workers = args.workers or max(1, (os.cpu_count() or 2) // 2)

    if args.d3_only:
        dcolors = [3]
    elif args.d5_only:
        dcolors = [5]
    else:
        dcolors = [3, 5]

    if args.t_gate_only:
        circuits = ["t_gate"]
    elif args.s_proxy_only:
        circuits = ["s_proxy"]
    else:
        circuits = ["t_gate", "s_proxy"]

    if args.smoke:
        noise_levels = args.noise
        total_shots = 1000
        chunk_size = 500
        dcolors = [3]
        print("=== SMOKE TEST MODE ===")
    else:
        if args.total_shots is None:
            parser.error("--total-shots is required (unless using --smoke)")
        noise_levels = args.noise
        total_shots = int(args.total_shots)
        chunk_size = args.chunk_size

    print(f"Noise levels: {noise_levels}")
    print(f"Total shots target (per circuit/distance/noise): {total_shots:,}")
    print(f"Chunk size: {chunk_size:,}")
    print(f"Workers: {num_workers}")
    print(f"Distances: {dcolors}")
    print(f"Circuits: {circuits}")
    print()

    run_simulation(dcolors, circuits, noise_levels, total_shots, chunk_size, num_workers)

    print("Done. Run 'uv run python plot_results.py --e2e-only' to generate figures.")


if __name__ == "__main__":
    main()
