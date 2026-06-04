"""RASL candidate data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import inf
from typing import Optional

from .symplectic import apply_ops


@dataclass(frozen=True)
class CliffordOp:
    name: str
    a: int
    b: Optional[int] = None

    def is_2q(self) -> bool:
        return self.name in ("CNOT", "CZ") and self.b is not None


@dataclass
class LocalizationCandidate:
    step_id: int
    kind: str
    target_axis: int | None
    ops: list[CliffordOp] = field(default_factory=list)
    final_pauli_type: str | None = None
    proxy_path_cost: float = inf
    proxy_workspace: float = inf
    proxy_resident_bound: float = inf
    exact_local_resident: float | None = None
    exact_global_resident: float | None = None
    refactor_cost: float = inf
    valid: bool = False
    reject_reason: str | None = None

    def num_2q_ops(self) -> int:
        return sum(1 for op in self.ops if op.is_2q())


def verify_candidate(mapped_pauli, candidate: LocalizationCandidate) -> bool:
    out = apply_ops(mapped_pauli, candidate.ops)
    candidate.valid = out.weight() == 1
    if candidate.valid:
        support = sorted(out.support())
        candidate.final_pauli_type = out.pauli_type(support[0]) if support else None
        candidate.reject_reason = None
    else:
        candidate.final_pauli_type = None
        candidate.reject_reason = f"final_weight={out.weight()}"
    return candidate.valid

