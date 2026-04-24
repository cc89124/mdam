"""Probe tsim compilation and sampling on all paper matrix circuits.

For each circuit that tsim should run, tries both compilation strategies
(default and cutting), runs a short probe sample to measure timing, and
saves the best configuration to ``configs/tsim_modes.json``.

Requires: bloqade-tsim[cuda13] >= 0.1.2, stim

Usage:
    uv run python -m tsim_compile_check
    uv run python -m tsim_compile_check --timeout 120
    uv run python -m tsim_compile_check --probe-shots 10000
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import textwrap
from pathlib import Path

from run_all import _build_matrix

_HERE = Path(__file__).resolve().parent
_CONFIGS_DIR = _HERE / "configs"
_DEFAULT_CONFIG_PATH = _CONFIGS_DIR / "tsim_modes.json"

# Subprocess script: compile circuit, run probe sample, print JSON result.
_PROBE_SCRIPT = textwrap.dedent("""\
    import json, sys, time
    import tsim
    circuit_text = sys.stdin.read()
    strategy = sys.argv[1]
    probe_shots = int(sys.argv[2])
    tc = tsim.Circuit(circuit_text)
    t0 = time.time()
    sampler = tc.compile_detector_sampler(strategy=strategy)
    compile_s = time.time() - t0
    total_graphs = sum(
        csg.num_graphs
        for comp in sampler._program.components
        for csg in comp.compiled_scalar_graphs
    )
    t1 = time.time()
    sampler.sample(probe_shots, separate_observables=True)
    sample_s = time.time() - t1
    json.dump({
        "compile_s": round(compile_s, 3),
        "sample_s": round(sample_s, 3),
        "num_graphs": total_graphs,
        "tsim_version": tsim.__version__,
    }, sys.stdout)
""")


def _probe(
    label: str,
    circuit_text: str,
    strategy: str,
    probe_shots: int,
    timeout: int,
) -> dict | None:
    """Run a compile+sample probe in a subprocess. Returns result or None."""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _PROBE_SCRIPT, strategy, str(probe_shots)],
            input=circuit_text,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if proc.returncode == 0:
            result = json.loads(proc.stdout)
            print(
                f"  {label:45s} [{strategy:7s}]  "
                f"OK  compile={result['compile_s']:.1f}s  "
                f"sample={result['sample_s']:.3f}s  "
                f"graphs={result['num_graphs']}"
            )
            return result
        else:
            err = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "unknown"
            print(f"  {label:45s} [{strategy:7s}]  ERROR: {err}")
            return None

    except subprocess.TimeoutExpired:
        print(f"  {label:45s} [{strategy:7s}]  TIMEOUT ({timeout}s)")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe tsim on all paper matrix circuits and save config.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Per-circuit timeout in seconds (default: 120).",
    )
    parser.add_argument(
        "--probe-shots",
        type=int,
        default=1000,
        help="Shots for probe sampling (default: 1000).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(_DEFAULT_CONFIG_PATH),
        help=f"Output config path (default: {_DEFAULT_CONFIG_PATH.relative_to(_HERE)}).",
    )
    args = parser.parse_args()

    strategies = ["default", "cutting"]

    # Filter to circuits that include tsim as a backend.
    matrix = _build_matrix()
    tsim_entries = [(b, lbl, meta, gen) for b, lbl, meta, gen, backs in matrix
                    if "tsim" in backs]

    print(
        f"Probing {len(tsim_entries)} circuits x {len(strategies)} strategies "
        f"(timeout={args.timeout}s, probe_shots={args.probe_shots})\n"
    )

    config: dict[str, dict] = {}

    for benchmark, label, meta, gen in tsim_entries:
        circuit_text = gen()

        best_strategy = None
        best_sample_s = float("inf")
        best_result = None

        for strategy in strategies:
            result = _probe(
                label,
                circuit_text,
                strategy,
                args.probe_shots,
                args.timeout,
            )
            if result is not None and result["sample_s"] < best_sample_s:
                best_strategy = strategy
                best_sample_s = result["sample_s"]
                best_result = result

        entry: dict[str, object] = {
            "benchmark": benchmark,
            "supported": best_strategy is not None,
        }
        if best_result is not None:
            entry["chosen_strategy"] = best_strategy
            entry["probe_compile_s"] = best_result["compile_s"]
            entry["probe_sample_s"] = best_result["sample_s"]
            entry["num_graphs"] = best_result["num_graphs"]
            entry["tsim_version"] = best_result["tsim_version"]
            entry["probe_shots"] = args.probe_shots

        config[label] = entry
        print()

    # Write config.
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(config, indent=2) + "\n")

    # Summary.
    supported = sum(1 for v in config.values() if v["supported"])
    unsupported = len(config) - supported
    print(f"{'=' * 70}")
    print(f"  {supported} supported, {unsupported} unsupported")
    print(f"  Config written to {output_path}")


if __name__ == "__main__":
    main()
