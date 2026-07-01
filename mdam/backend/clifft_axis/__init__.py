"""Clifft-axis compatibility mode for the near-Clifford backend.

A canonical active-axis dense engine that holds the magic state as ONE dense
register `phi` over the active magic qubits (the clifft active rank k), evolves it
with STRICTLY IN-PLACE pairwise kernels (no full temporary vector, no chi0/chi1
branch vectors, no Pauli-sum / MPS / projected-TN dispatch), and enforces a HARD
memory budget: the peak live complex-word count is capped at 2^k_clifft and the
engine raises if a kernel would exceed it.

The math (pullback through the Clifford frame, promote-on-demand, Born collapse,
parity reduction to the active rank) is inherited verbatim from the clifft-validated
``VirtualAxisNearClifford``; this mode REPLACES only the dense kernels (rotation
apply, expectation, Born collapse) with in-place pairwise versions and adds the
budget guard + a per-measurement certificate / log.

See ``engine.CliftAxisNearClifford`` and ``budget.DenseMemoryBudget``.
"""
from mdam.backend.clifft_axis.budget import (
    DenseMemoryBudget, MemoryBudgetExceeded)
from mdam.backend.clifft_axis.engine import CliftAxisNearClifford
from mdam.backend.clifft_axis.bounded import (
    CliftAxisBoundedNearClifford, compile_bounded)

__all__ = ["CliftAxisNearClifford", "CliftAxisBoundedNearClifford",
           "compile_bounded", "DenseMemoryBudget", "MemoryBudgetExceeded"]
