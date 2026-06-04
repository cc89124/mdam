"""Bounded RASL candidate builders.

v1 implements executable-safe analysis candidates only. Builder B is present as
a conservative placeholder and rejects dormant/mixed cases instead of emitting
untracked phase-affecting bytecode.
"""

from __future__ import annotations

from .candidate import CliffordOp, LocalizationCandidate, verify_candidate
from .symplectic import PauliVec


def default_candidate(step_id: int, mapped_pauli: PauliVec, ops: list[CliffordOp],
                      target_axis: int | None) -> LocalizationCandidate:
    c = LocalizationCandidate(step_id=step_id, kind="default",
                              target_axis=target_axis, ops=list(ops))
    verify_candidate(mapped_pauli, c)
    return c


def _normalization_ops(mapped_pauli: PauliVec, support: list[int]) -> list[CliffordOp]:
    ops = []
    for q in support:
        t = mapped_pauli.pauli_type(q)
        if t == "X":
            ops.append(CliffordOp("H", q))
        elif t == "Y":
            ops.append(CliffordOp("S", q))
            ops.append(CliffordOp("H", q))
    return ops


def active_z_route_candidates(step_id: int, mapped_pauli: PauliVec, active_axes: set[int],
                              layout_cost=None, max_support: int = 10) -> list[LocalizationCandidate]:
    support = sorted(mapped_pauli.support())
    if not support or len(support) > max_support:
        return []
    if not set(support) <= set(active_axes):
        return []
    candidates = []
    norm = _normalization_ops(mapped_pauli, support)
    for target in support:
        ops = list(norm)
        for q in support:
            if q != target:
                ops.append(CliffordOp("CNOT", q, target))
        c = LocalizationCandidate(step_id=step_id, kind="active_z_route_star",
                                  target_axis=target, ops=ops)
        verify_candidate(mapped_pauli, c)
        candidates.append(c)

    # Greedy candidate: repeatedly fold the currently cheapest pair into the
    # chosen target. This stays bounded and deterministic.
    if layout_cost is not None and len(support) >= 3:
        for target in support:
            remaining = set(support)
            ops = list(norm)
            while len(remaining) > 1:
                best = None
                for q in sorted(remaining):
                    if q == target:
                        continue
                    cost = 0.0
                    try:
                        cost = layout_cost.path_cost_axes(q, target)
                    except AttributeError:
                        cost = 0.0
                    key = (cost, q)
                    if best is None or key < best[0]:
                        best = (key, q)
                q = best[1]
                ops.append(CliffordOp("CNOT", q, target))
                remaining.remove(q)
            c = LocalizationCandidate(step_id=step_id, kind="active_z_route_greedy",
                                      target_axis=target, ops=ops)
            verify_candidate(mapped_pauli, c)
            candidates.append(c)
    return candidates


def symplectic_greedy_candidates(step_id: int, mapped_pauli: PauliVec, active_axes: set[int],
                                 max_support: int = 10) -> list[LocalizationCandidate]:
    support = mapped_pauli.support()
    if not support <= set(active_axes):
        c = LocalizationCandidate(step_id=step_id, kind="symp_greedy",
                                  target_axis=None, ops=[])
        c.valid = False
        c.reject_reason = "dormant_or_mixed_support_not_emitted_in_v1"
        return [c]
    # For active-only v1, Builder B delegates to the safe active Z-normalization
    # family instead of emitting a separate unphased mixed-symplectic sequence.
    return active_z_route_candidates(step_id, mapped_pauli, active_axes, max_support=max_support)

