"""Regression tests for the offline selector and targeted peel.

Premise: under fixed Pauli noise the structural schedule (clifft active rank k, NC
block b) is SHOT-INVARIANT, so one offline pass fixes the backend for all shots.
Targeted peel is record-bit-identical to the full factor scan.

Runs under pytest if available; also runnable directly with the clifft env:
    python tests/test_selector_invariance.py            # fast circuits
    python tests/test_selector_invariance.py --slow     # + coherent_d5_r5
"""
import os
import sys

sys.setrecursionlimit(50000)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import clifft
from nearclifford_backend import selector
from nearclifford_backend.backend import NearCliffordBackend

SEEDS = (1, 7, 42, 123, 999)

# (circuit, expected backend, k_max, b_max) -- pinned from the measured schedule.
FAST = [
    ("distillation",   "nc",     5, 4),
    ("cultivation_d3", "clifft", 4, 5),
    ("cultivation_d5", "clifft", 10, 14),
]
SLOW = [("coherent_d5_r5", "nc", 24, 19)]


def check_invariance(circ, exp_be, exp_k, exp_b):
    info = selector.analyze(circ, seeds=SEEDS)
    assert info["invariant"], (
        f"{circ}: k/b schedule varied across seeds: "
        f"{[(r['seed'], r['k_max'], r['b_max']) for r in info['runs']]}")
    assert info["k_max"] == exp_k, f"{circ}: k_max {info['k_max']} != {exp_k}"
    assert info["b_max"] == exp_b, f"{circ}: b_max {info['b_max']} != {exp_b}"
    be, sinfo = selector.select(circ, margin=1, seeds=(42,))
    assert be == exp_be, f"{circ}: picked {be}, expected {exp_be}"
    assert sinfo["peak_qubits"] <= sinfo["k_max"]        # never worse than clifft


def check_targeted_peel_exact(circ):
    """Targeted peel must be record-bit-identical to the full factor scan, and the
    debug re-scan must find zero missed peels (the recorded peel set is complete)."""
    prog = clifft.compile(open(f"qec_bench/circuits/{circ}.stim").read())
    for seed in (42, 7):
        base = NearCliffordBackend(block=True, targeted_peel=False)
        rb = dict(base.run_shot(prog, seed))
        tgt = NearCliffordBackend(block=True, targeted_peel=True,
                                  structure_once_debug=True)
        rt = dict(tgt.run_shot(prog, seed))
        assert rb == rt, f"{circ} seed {seed}: targeted peel changed the record"
        assert tgt.last_peel_mismatch == 0, (
            f"{circ} seed {seed}: {tgt.last_peel_mismatch} missed peels")


# ----- pytest entry points (parametrized) -----
try:
    import pytest

    @pytest.mark.parametrize("circ,be,k,b", FAST)
    def test_invariance(circ, be, k, b):
        check_invariance(circ, be, k, b)

    @pytest.mark.parametrize("circ", [c[0] for c in FAST])
    def test_targeted_peel_exact(circ):
        check_targeted_peel_exact(circ)
except ImportError:
    pass


# ----- standalone runner (no pytest in the clifft env) -----
if __name__ == "__main__":
    cases = FAST + (SLOW if "--slow" in sys.argv else [])
    for circ, be, k, b in cases:
        check_invariance(circ, be, k, b)
        check_targeted_peel_exact(circ)
        print(f"[OK] {circ:16} selector={be:7} k={k} b={b} "
              f"invariant + targeted-peel bit-identical")
    print("ALL PASS")
