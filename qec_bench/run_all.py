"""Run all paper benchmarks for a given backend.

Primary workflow:
    # Step 1 (GPU): probe tsim and save config
    uv run python -m tsim_compile_check

    # Step 2 (GPU): run all benchmarks for tsim
    uv run python -m run_all tsim

    # Step 3 (CPU): run all benchmarks for clifft
    uv run python -m run_all clifft

    # Step 4 (CPU): run clifford_bench for stim
    uv run python -m run_all stim

Smoke test:
    uv run python -m run_all clifft --shots 1000 --repeats 1
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Sequence

import stim

from bench_common import RESULTS_DIR, run_benchmark_loop

_HERE = Path(__file__).resolve().parent
_DEFAULT_TSIM_CONFIG = _HERE / "configs" / "tsim_modes.json"

REPEATS = 3


# ---------------------------------------------------------------------------
# Circuit generation helpers
# ---------------------------------------------------------------------------

_CIRCUITS_DIR = _HERE / "circuits"


def _clifford_circuit(distance: int, rounds: int, phys_error_rate: float) -> str:
    return str(
        stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            rounds=rounds,
            distance=distance,
            after_clifford_depolarization=phys_error_rate,
            after_reset_flip_probability=phys_error_rate,
            before_measure_flip_probability=phys_error_rate,
            before_round_data_depolarization=phys_error_rate,
        )
    )


_CULTIVATION_TEMPLATE_RATE = 0.001
_CULTIVATION_NOISE_RE = re.compile(
    r"(?P<prefix>(?:X_ERROR|Y_ERROR|Z_ERROR|DEPOLARIZE1|DEPOLARIZE2|MX?|MY|MZ)\()"
    + re.escape(str(_CULTIVATION_TEMPLATE_RATE))
    + r"(?P<suffix>\))"
)


def _cultivation_circuit(distance: int, phys_error_rate: float) -> str:
    text = (_CIRCUITS_DIR / f"cultivation_d{distance}.stim").read_text()
    if phys_error_rate != _CULTIVATION_TEMPLATE_RATE:
        text = _CULTIVATION_NOISE_RE.sub(
            rf"\g<prefix>{phys_error_rate}\g<suffix>", text
        )
    return text


_DISTILLATION_TEMPLATE_PREP = 0.05
_DISTILLATION_TEMPLATE_CIRCUIT = 0.01  # prep / 5
_DISTILLATION_NOISE_RATIO = 5

_DISTILLATION_PREP_RE = re.compile(
    r"(?P<prefix>DEPOLARIZE1\()"
    + re.escape(str(_DISTILLATION_TEMPLATE_PREP))
    + r"(?P<suffix>\))"
)
_DISTILLATION_CIRCUIT_RE = re.compile(
    r"(?P<prefix>(?:DEPOLARIZE1|DEPOLARIZE2)\()"
    + re.escape(str(_DISTILLATION_TEMPLATE_CIRCUIT))
    + r"(?P<suffix>\))"
)


def _distillation_circuit(prep_noise: float) -> str:
    text = (_CIRCUITS_DIR / "distillation.stim").read_text()
    circuit_noise = prep_noise / _DISTILLATION_NOISE_RATIO
    if circuit_noise != _DISTILLATION_TEMPLATE_CIRCUIT:
        text = _DISTILLATION_CIRCUIT_RE.sub(
            rf"\g<prefix>{circuit_noise}\g<suffix>", text
        )
    if prep_noise != _DISTILLATION_TEMPLATE_PREP:
        text = _DISTILLATION_PREP_RE.sub(
            rf"\g<prefix>{prep_noise}\g<suffix>", text
        )
    return text


def _coherent_noise_circuit(
    distance: int, rounds: int, phys_error_rate: float, rz_angle: float
) -> str:
    c = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        rounds=rounds,
        distance=distance,
        after_clifford_depolarization=phys_error_rate,
        after_reset_flip_probability=phys_error_rate,
        before_measure_flip_probability=phys_error_rate,
        before_round_data_depolarization=phys_error_rate,
    )
    lines = []
    for line in str(c).split("\n"):
        s = line.strip()
        if s.startswith("DEPOLARIZE1(") or s.startswith("DEPOLARIZE2("):
            targets = s.split(")")[1].strip()
            lines.append(f"R_Z({rz_angle}) {targets}")
        else:
            lines.append(line)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Paper benchmark matrix
# ---------------------------------------------------------------------------

# Each entry: (benchmark_name, label, metadata_dict, circuit_generator, backends)
# circuit_generator is a callable returning circuit text (str).

BenchmarkEntry = tuple[str, str, dict[str, object], callable, tuple[str, ...]]


def _build_matrix() -> list[BenchmarkEntry]:
    """Define every circuit configuration for the paper."""
    matrix: list[BenchmarkEntry] = []

    # Clifford bench: surface code memory-Z (k=0)
    matrix.append((
        "clifford_bench",
        "d=7 r=7 p=1e-3",
        {"distance": 7, "rounds": 7, "phys_error_rate": 1e-3, "shots": 1_000_000},
        lambda: _clifford_circuit(7, 7, 1e-3),
        ("clifft", "stim", "tsim"),
    ))

    # Cultivation bench: magic state cultivation
    for d in [3, 5]:
        matrix.append((
            "cultivation_bench",
            f"d={d} p=1e-3",
            {"distance": d, "phys_error_rate": 1e-3, "shots": 1_000_000},
            lambda d=d: _cultivation_circuit(d, 1e-3),
            ("clifft", "tsim"),
        ))

    # Distillation bench: 85-qubit logical magic-state distillation (k=5)
    matrix.append((
        "distillation_bench",
        "prep=0.05",
        {"prep_noise": 0.05, "shots": 1_000_000},
        lambda: _distillation_circuit(0.05),
        ("clifft", "tsim"),
    ))

    # Coherent noise bench: surface code with R_Z over-rotation
    # Shot counts calibrated to peak active rank:
    #   d=3 r=1 (k=5), d=3 r=3 (k=8): fast  -> 1M shots
    #   d=5 r=1 (k=13): ~10k shots/s          -> 100k shots
    #   d=5 r=5 (k=24): ~0.1 shots/s           -> 20 shots
    _coherent_shots = {(3, 1): 1_000_000, (3, 3): 1_000_000,
                       (5, 1): 100_000, (5, 5): 20}
    for d in [3, 5]:
        for r in sorted({1, d}):
            matrix.append((
                "coherent_noise_bench",
                f"d={d} r={r} p=1e-3 rz=0.02",
                {
                    "distance": d,
                    "rounds": r,
                    "phys_error_rate": 1e-3,
                    "rz_angle": 0.02,
                    "shots": _coherent_shots[(d, r)],
                },
                lambda d=d, r=r: _coherent_noise_circuit(d, r, 1e-3, 0.02),
                ("clifft", "tsim"),
            ))

    return matrix


# ---------------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run all paper benchmarks for a given backend.",
    )
    p.add_argument(
        "backend",
        choices=["clifft", "stim", "tsim"],
        help="Backend to run: clifft, stim, or tsim.",
    )
    p.add_argument(
        "--shots",
        type=int,
        default=None,
        help="Override per-circuit shot counts (useful for smoke testing).",
    )
    p.add_argument(
        "--repeats",
        type=int,
        default=REPEATS,
        help=f"Repetitions per (circuit, backend) combo (default: {REPEATS}).",
    )
    p.add_argument(
        "--results-dir",
        type=str,
        default=str(RESULTS_DIR),
        help="Directory for benchmark CSV outputs (default: results).",
    )
    def _positive_int(value: str) -> int:
        n = int(value)
        if n < 1:
            raise argparse.ArgumentTypeError(f"must be >= 1, got {n}")
        return n

    p.add_argument(
        "--threads",
        type=_positive_int,
        default=1,
        help="Number of parallel worker processes (default: 1).",
    )
    p.add_argument(
        "--tsim-config",
        type=str,
        default=str(_DEFAULT_TSIM_CONFIG),
        help="Path to tsim_modes.json (default: configs/tsim_modes.json).",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help=(
            "Overwrite an existing non-empty results CSV.  Without this "
            "flag run_all refuses to overwrite committed reference data."
        ),
    )
    return p


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    backend = args.backend
    results_dir = Path(args.results_dir)

    matrix = _build_matrix()

    # Filter to entries that include this backend.
    entries = [(b, lbl, meta, gen, backs) for b, lbl, meta, gen, backs in matrix
               if backend in backs]

    # For tsim, load the probe config and filter to supported circuits.
    tsim_config: dict[str, dict] = {}
    if backend == "tsim":
        config_path = Path(args.tsim_config)
        if not config_path.exists():
            raise SystemExit(
                f"tsim config not found: {config_path}\n"
                "Run 'python -m tsim_compile_check' first to generate it."
            )
        tsim_config = json.loads(config_path.read_text())

        supported = []
        for entry in entries:
            label = entry[1]
            cfg = tsim_config.get(label)
            if cfg is None:
                print(f"  WARNING: {label} not in tsim config, skipping")
            elif not cfg.get("supported"):
                print(f"  Skipping {label} (unsupported by tsim)")
            else:
                supported.append(entry)
        entries = supported
        print()

    # Build (metadata, circuit_text) pairs for all entries.
    circuits: list[tuple[dict[str, object], str]] = []
    for benchmark, label, meta, gen, _ in entries:
        row_meta = {"benchmark": benchmark, "circuit": label, **meta}
        if args.shots is not None:
            row_meta["shots"] = args.shots
        circuits.append((row_meta, gen()))

    # Build per-circuit tsim strategy map from the probe config.
    strategy: str | dict[str, str] = "default"
    if backend == "tsim":
        strategy = {
            label: tsim_config[label]["chosen_strategy"]
            for _, label, _, _, _ in entries
        }

    output_path = results_dir / f"{backend}.csv"
    if output_path.exists() and output_path.stat().st_size > 0 and not args.force:
        raise SystemExit(
            f"Refusing to overwrite existing results at {output_path}.\n"
            f"  Pass --force to overwrite, or set --results-dir <other-path>\n"
            f"  (e.g. --results-dir /tmp/smoke) to keep the committed data."
        )
    print(f"=== {backend} -> {output_path} ===")
    run_benchmark_loop(
        circuits=circuits,
        simulators=[backend],
        repeats=args.repeats,
        output_csv=output_path,
        tsim_strategy=strategy,
        threads=args.threads,
    )
    print(f"\nResults written to {output_path}")


if __name__ == "__main__":
    main()
