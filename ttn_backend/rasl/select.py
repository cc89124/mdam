"""Selection rule for RASL candidates."""

from __future__ import annotations


def choose_candidate(default, candidates):
    valid = [c for c in candidates if c.valid]
    all_candidates = [default] + valid
    feasible = [
        c for c in all_candidates
        if c.proxy_resident_bound <= default.proxy_resident_bound
    ]
    if not feasible:
        return default
    return min(
        feasible,
        key=lambda c: (
            c.proxy_resident_bound,
            c.proxy_workspace,
            c.refactor_cost,
            c.proxy_path_cost,
            c.num_2q_ops(),
            c.kind,
        ),
    )

