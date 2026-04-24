"""Regenerate the STIM circuit files referenced by the throughput table.

The LaTeX table emitted by ``generate_table.py`` embeds deep links to
the clifft playground for each benchmark circuit.  Those links resolve
raw.githubusercontent.com URLs that point at the files written by this
script, so the reader sees the exact circuit that was timed.

Running this script is idempotent: identical inputs produce identical
output files.  Re-run whenever the generators in ``run_all.py`` change.

Usage:
    uv run python save_table_circuits.py
"""

from __future__ import annotations

from pathlib import Path

from run_all import (
    _CIRCUITS_DIR,
    _clifford_circuit,
    _coherent_noise_circuit,
    _cultivation_circuit,
    _distillation_circuit,
)

# Each entry: (filename, circuit-producing thunk).  The filenames match
# those used by ``generate_table.py`` to build playground deep links.
_TABLE_CIRCUITS: list[tuple[str, callable]] = [
    ("surface_d7_r7.stim",     lambda: _clifford_circuit(7, 7, 1e-3)),
    ("cultivation_d3.stim",    lambda: _cultivation_circuit(3, 1e-3)),
    ("cultivation_d5.stim",    lambda: _cultivation_circuit(5, 1e-3)),
    ("distillation.stim",      lambda: _distillation_circuit(0.05)),
    ("coherent_d3_r1.stim",    lambda: _coherent_noise_circuit(3, 1, 1e-3, 0.02)),
    ("coherent_d3_r3.stim",    lambda: _coherent_noise_circuit(3, 3, 1e-3, 0.02)),
    ("coherent_d5_r1.stim",    lambda: _coherent_noise_circuit(5, 1, 1e-3, 0.02)),
    ("coherent_d5_r5.stim",    lambda: _coherent_noise_circuit(5, 5, 1e-3, 0.02)),
]


def main() -> None:
    _CIRCUITS_DIR.mkdir(exist_ok=True)
    for name, make in _TABLE_CIRCUITS:
        path = _CIRCUITS_DIR / name
        text = make()
        if not text.endswith("\n"):
            text += "\n"
        path.write_text(text)
        print(f"Wrote {path.relative_to(_CIRCUITS_DIR.parent)} ({len(text):,} bytes)")


if __name__ == "__main__":
    main()
