"""Generate the LaTeX throughput table from QEC benchmark results.

Usage:
    uv run python generate_table.py
    uv run python generate_table.py --results-dir results
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import clifft
import pandas as pd

from run_all import (
    _clifford_circuit,
    _coherent_noise_circuit,
    _cultivation_circuit,
    _distillation_circuit,
)

_HERE = Path(__file__).resolve().parent
_DEFAULT_RESULTS = _HERE / "results"


# ---------------------------------------------------------------------------
# Circuit metadata helpers
# ---------------------------------------------------------------------------


# Gate types that are counted as "non-Clifford" for the circuit metadata
# column.  Parametric rotations are always listed here even though R_Z(0)
# happens to be Clifford — the circuits in this table use them at
# continuously tuned angles.
_NON_CLIFFORD_GATES = frozenset({
    clifft.GateType.T,
    clifft.GateType.T_DAG,
    clifft.GateType.R_X,
    clifft.GateType.R_Y,
    clifft.GateType.R_Z,
    clifft.GateType.R_XX,
    clifft.GateType.R_YY,
    clifft.GateType.R_ZZ,
    clifft.GateType.R_PAULI,
})


def _circuit_metadata(circ_str: str) -> tuple[int, int, int, int]:
    """Return (num_qubits, num_gate_ops, num_non_clifford, peak_k) for a circuit.

    Uses clifft to parse and compile the circuit:
    - num_qubits: from the parsed circuit
    - num_gate_ops: total gate operations (one per qubit target, REPEAT
      blocks unrolled) from the parsed AST nodes
    - num_non_clifford: subset of num_gate_ops whose gate is T/T_DAG or a
      parametric rotation (R_X/R_Y/R_Z, their two-qubit siblings, or
      R_PAULI)
    - peak_k: peak active non-stabilizer rank from the compiled program
    """
    parsed = clifft.parse(circ_str)
    prog = clifft.compile(
        circ_str,
        hir_passes=clifft.default_hir_pass_manager(),
        bytecode_passes=clifft.default_bytecode_pass_manager(),
    )
    k_hist = list(prog.active_k_history)
    _ANNOTATIONS = {
        clifft.GateType.TICK,
        clifft.GateType.DETECTOR,
        clifft.GateType.OBSERVABLE_INCLUDE,
    }
    num_ops = sum(1 for n in parsed.nodes if n.gate not in _ANNOTATIONS)
    num_nc = sum(1 for n in parsed.nodes if n.gate in _NON_CLIFFORD_GATES)
    return parsed.num_qubits, num_ops, num_nc, max(k_hist) if k_hist else 0


# ---------------------------------------------------------------------------
# Table definition
# ---------------------------------------------------------------------------

# Each row: (display_name, benchmark_key, circuit_key, circuit_generator,
#             has_stim, has_tsim, section, source_key)
_ROWS: list[tuple[str, str, str, callable, bool, bool, str, str]] = [
    (
        r"Surface code $d=7, r=7$",
        "clifford_bench",
        "d=7 r=7 p=1e-3",
        lambda: _clifford_circuit(7, 7, 1e-3),
        True,
        True,
        "clifford",
        "stim",
    ),
    (
        r"Cultivation $d=3$",
        "cultivation_bench",
        "d=3 p=1e-3",
        lambda: _cultivation_circuit(3, 1e-3),
        False,
        True,
        "magic",
        "cultivation",
    ),
    (
        r"Cultivation $d=5$",
        "cultivation_bench",
        "d=5 p=1e-3",
        lambda: _cultivation_circuit(5, 1e-3),
        False,
        False,  # DNC
        "magic",
        "cultivation",
    ),
    (
        r"Distillation",
        "distillation_bench",
        "prep=0.05",
        lambda: _distillation_circuit(0.05),
        False,
        True,
        "magic",
        "distillation",
    ),
    (
        r"Surface code $d=3, r=1$",
        "coherent_noise_bench",
        "d=3 r=1 p=1e-3 rz=0.02",
        lambda: _coherent_noise_circuit(3, 1, 1e-3, 0.02),
        False,
        True,
        "coherent",
        "coherent",
    ),
    (
        r"Surface code $d=3, r=3$",
        "coherent_noise_bench",
        "d=3 r=3 p=1e-3 rz=0.02",
        lambda: _coherent_noise_circuit(3, 3, 1e-3, 0.02),
        False,
        False,  # DNC
        "coherent",
        "coherent",
    ),
    (
        r"Surface code $d=5, r=1$",
        "coherent_noise_bench",
        "d=5 r=1 p=1e-3 rz=0.02",
        lambda: _coherent_noise_circuit(5, 1, 1e-3, 0.02),
        False,
        True,
        "coherent",
        "coherent",
    ),
    (
        r"Surface code $d=5, r=5$",
        "coherent_noise_bench",
        "d=5 r=5 p=1e-3 rz=0.02",
        lambda: _coherent_noise_circuit(5, 5, 1e-3, 0.02),
        False,
        False,  # DNC
        "coherent",
        "coherent",
    ),
]

_SECTION_HEADERS = {
    "clifford": r"\textit{Pure Clifford}",
    "magic": r"\textit{Near-Clifford: Magic State}",
    "coherent": r"\textit{Near-Clifford: Coherent Noise}",
}


# Playground deep links.  Each (benchmark_key, circuit_key) maps to a
# STIM source file in ``qec_bench/circuits/`` (populated by
# ``save_table_circuits.py``).  The rendered link opens the clifft
# playground with that file loaded via CORS.
_CIRCUIT_FILES: dict[tuple[str, str], str] = {
    ("clifford_bench",      "d=7 r=7 p=1e-3"):        "surface_d7_r7.stim",
    ("cultivation_bench",   "d=3 p=1e-3"):            "cultivation_d3.stim",
    ("cultivation_bench",   "d=5 p=1e-3"):            "cultivation_d5.stim",
    ("distillation_bench",  "prep=0.05"):             "distillation.stim",
    ("coherent_noise_bench", "d=3 r=1 p=1e-3 rz=0.02"): "coherent_d3_r1.stim",
    ("coherent_noise_bench", "d=3 r=3 p=1e-3 rz=0.02"): "coherent_d3_r3.stim",
    ("coherent_noise_bench", "d=5 r=1 p=1e-3 rz=0.02"): "coherent_d5_r1.stim",
    ("coherent_noise_bench", "d=5 r=5 p=1e-3 rz=0.02"): "coherent_d5_r5.stim",
}

# Paper uses a \clifftplay{filename.stim} macro that expands to the
# full hyperref to the clifft playground.  Paper defines it once as:
#
#   \newcommand{\clifftplay}[1]{\href{https://unitaryfoundation.github.io/
#       clifft/playground/?url=https://raw.githubusercontent.com/
#       unitaryfoundation/clifft-paper/main/qec_bench/circuits/#1}{\faPlay}}
#
# Emitting \clifftplay{...} keeps rows compact and regen-friendly.


# Circuit-source attributions.  Each row tags a ``source_key`` that maps
# to a footnote letter (b, c, ...) and the corresponding citation block
# below the table.  Marker "a" is reserved for the pre-existing DNC note.
_SOURCES: list[tuple[str, str]] = [
    ("stim",         r"Generated via Stim~\cite{gidneyStimFastStabilizer2021}."),
    ("cultivation",  r"Adapted from~\cite{gidneyMagicStateCultivation2024}."),
    ("distillation", r"Adapted from~\cite{salesrodriguezExperimentalDemonstrationLogical2025,haenelTsimFastUniversal2026}."),
    ("coherent",     r"Adapted from~\cite{tuloupComputingLogicalError2026a}."),
]
_SOURCE_LETTERS: dict[str, str] = {
    key: chr(ord("b") + i) for i, (key, _) in enumerate(_SOURCES)
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_results(results_dir: Path) -> dict[tuple[str, str, str], list[float]]:
    """Load CSVs and return {(benchmark, circuit, simulator): [shots/s values]}.

    The table only shows the 16-core runs for Clifft/Stim and the GPU
    tsim run, so only those CSVs are loaded.
    """
    data: dict[tuple[str, str, str], list[float]] = {}

    csv_files = [
        results_dir / "clifft.csv",
        results_dir / "stim.csv",
        results_dir / "tsim.csv",
    ]

    for csv_path in csv_files:
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            if row.get("status") != "SUCCESS":
                continue
            key = (row["benchmark"], row["circuit"], row["simulator"])
            data.setdefault(key, []).append(float(row["effective_shots_per_s"]))

    return data


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def _fmt_number(median: float) -> str:
    """Format a raw median throughput as a compact human-readable string."""
    if median >= 1_000_000:
        return f"{median / 1_000_000:.1f}M"
    if median >= 1_000:
        return f"{median / 1_000:.1f}k"
    return f"{median:.1f}"


# ---------------------------------------------------------------------------
# Table generation
# ---------------------------------------------------------------------------


def generate_table(results_dir: Path) -> str:
    data = _load_results(results_dir)

    # Compute circuit metadata via clifft
    # (qubits, gate_ops, non_clifford, k)
    circuit_meta: list[tuple[int, int, int, int]] = []
    for _, _, _, gen, _, _, _, _ in _ROWS:
        circuit_meta.append(_circuit_metadata(gen()))

    lines = []
    lines.append(r"    \begin{tabular}{l r r r r | r r r}")
    lines.append(r"        \toprule")
    lines.append(
        r"        \textbf{Circuit} & $\mathbf{N}$ & \textbf{Ops}"
        r" & \textbf{Non-Cliff.} & $\mathbf{k_{\max}}$"
        r" & \textbf{Clifft} & \textbf{Stim} & \textbf{Tsim (GPU)} \\"
    )
    lines.append(r"        \midrule")

    # If a section's rows all share one source, attach the footnote
    # marker to the section header and skip per-row markers there.
    # Sections with mixed sources (e.g. Magic State: cultivation + distillation)
    # keep per-row markers.
    section_sources: dict[str, set[str]] = {}
    for row in _ROWS:
        section_sources.setdefault(row[6], set()).add(row[7])

    def _section_header(section_key: str) -> str:
        header = _SECTION_HEADERS[section_key]
        sources = section_sources[section_key]
        if len(sources) == 1:
            (only_source,) = sources
            header += rf"\textsuperscript{{{_SOURCE_LETTERS[only_source]}}}"
        return header

    prev_section = None
    for i, (name, bench, circuit, _, has_stim, has_tsim, section, source) in enumerate(_ROWS):
        if section != prev_section:
            if prev_section is not None:
                lines.append(r"        \midrule")
            lines.append(
                f"        \\multicolumn{{8}}{{l}}{{{_section_header(section)}}} \\\\"
            )
            prev_section = section

        qubits, instr, nc, k = circuit_meta[i]

        # Raw medians (None if no data, which renders as "--" / "DNC").
        def _median(samples: list[float] | None) -> float | None:
            return statistics.median(samples) if samples else None

        clifft_m = _median(data.get((bench, circuit, "clifft")))
        stim_m = _median(data.get((bench, circuit, "stim"))) if has_stim else None
        tsim_m = _median(data.get((bench, circuit, "tsim"))) if has_tsim else None

        # Bold the winning simulator(s) on each row.  The "winner" is
        # the column whose formatted (displayed) value matches that of
        # the highest raw median — so two columns that round to the
        # same displayed value (e.g. both "1.5M") are treated as a tie
        # and both get bolded.
        concrete = {
            sim: m for sim, m in
            (("clifft", clifft_m), ("stim", stim_m), ("tsim", tsim_m))
            if m is not None
        }
        if concrete:
            top_sim = max(concrete, key=concrete.get)
            top_text = _fmt_number(concrete[top_sim])
            winners = {
                sim for sim, m in concrete.items() if _fmt_number(m) == top_text
            }
        else:
            winners = set()

        def _cell(median: float | None, sim: str, *, placeholder: str) -> str:
            if median is None:
                return placeholder
            text = _fmt_number(median)
            return rf"\textbf{{{text}}}" if sim in winners else text

        clifft_cell = _cell(clifft_m, "clifft", placeholder="--")
        stim_cell = _cell(stim_m, "stim", placeholder="--") if has_stim else "--"
        tsim_cell = _cell(
            tsim_m, "tsim",
            placeholder=r"DNC\textsuperscript{a}" if not has_tsim else "--",
        )

        filename = _CIRCUIT_FILES[(bench, circuit)]
        # Drop the per-row marker when the section header already carries it.
        if len(section_sources[section]) > 1:
            marker = rf"\textsuperscript{{{_SOURCE_LETTERS[source]}}}"
        else:
            marker = ""
        name_with_marker = rf"{name}{marker}~\clifftplay{{{filename}}}"
        cols = [
            f"        {name_with_marker}",
            str(qubits),
            str(instr),
            str(nc),
            str(k),
            clifft_cell,
            stim_cell,
            tsim_cell,
        ]
        lines.append(" & ".join(cols) + r" \\")

    lines.append(r"        \bottomrule")
    lines.append(r"    \end{tabular}")
    lines.append("")
    lines.append(
        r"    \textsuperscript{a} DNC: did not compile within a 2 minute time budget."
    )
    for source_key, footnote in _SOURCES:
        letter = _SOURCE_LETTERS[source_key]
        lines.append(rf"    \textsuperscript{{{letter}}} {footnote}")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LaTeX throughput table.")
    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(_DEFAULT_RESULTS),
        help="Directory containing benchmark CSV results.",
    )
    args = parser.parse_args()
    print(generate_table(Path(args.results_dir)))


if __name__ == "__main__":
    main()
