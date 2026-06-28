"""Verify the structure-once measurement fast path (NearCliffordBackend(structure_once=True)).

The anticommuting core flushed at each measurement is outcome-independent for these circuits
(measurement records steer only the deferred Pauli FRAME, never the active tableau/rotations),
so the same cached structure pass that finds the dead uids ALSO records a
``{meas_idx -> core uids}`` table. At runtime each measurement looks the core up and gathers
it from the ``pending`` uid map -- instead of re-scanning all pending with
``_core_indices`` / ``_commute_xz``. Single-shot fast path (no shot batching yet).

This checks:
  1. RECORD-BIT-IDENTICAL: structure_once vs the live-scan path produce the same record
     byte-for-byte (same seed), AND structure_once_debug cross-checks every measurement's
     precomputed core against a live scan with **0 mismatches**.
  2. Counters: the commute_xz / dynamic-scan work is eliminated from the runtime
     (replaced by O(1) core lookups).

Reported per circuit (the requested counters):
  structure_once_enabled, prepass_time_ms, runtime_time_ms,
  dynamic_core_scan_calls (before/after), fastpath_core_lookups,
  commute_xz_calls_before / commute_xz_calls_after.

Usage:  python -m nearclifford_backend.scripts.verify_structure_once [circ ...]
"""
from __future__ import annotations
import sys
import time
import numpy as np
import clifft

from nearclifford_backend.backend import NearCliffordBackend

DEFAULT = ["coherent_d3_r1", "coherent_d3_r3", "coherent_d5_r1",
           "distillation", "cultivation_d3", "cultivation_d5"]
SEEDS = [1, 2, 3, 7, 42, 100, 777, 2024, 99991, 1234567]
WALL_K = 80


def load(circ):
    return clifft.compile(open(f"qec_bench/circuits/{circ}.stim").read())


def _wall(make, prog, k, warm=True):
    be = make()
    if warm:
        be._structure_for(prog)              # pay the (cached) pre-pass before timing
    m = np.random.default_rng(5)
    t0 = time.perf_counter()
    for _ in range(k):
        be.run_shot(prog, int(m.integers(0, 2 ** 63 - 1)))
    return be, (time.perf_counter() - t0) / k * 1e3


def main():
    circs = [a for a in sys.argv[1:] if not a.startswith("-")] or DEFAULT
    allok = True
    print(f"{'circuit':15s} {'bit-id':6s} {'SOen':5s} {'dbgMism':7s} "
          f"{'cx before->after':17s} {'dynScan b->a':12s} {'fastLk':6s} "
          f"{'wall b->a (ms)':16s} {'x':5s} {'prepass_ms':10s}", flush=True)
    for c in circs:
        prog = load(c)

        # (1) bit-identity + debug cross-check (0 mismatches required)
        be_dyn = NearCliffordBackend(block=True, structure_once=False)
        be_dbg = NearCliffordBackend(block=True, structure_once=True,
                                     structure_once_debug=True)
        bit = True
        mism = 0
        for sd in SEEDS:
            r0 = dict(be_dyn.run_shot(prog, sd))
            r1 = dict(be_dbg.run_shot(prog, sd))
            mism += be_dbg.last_fast_mismatch
            if r0 != r1:
                bit = False
        cx_before = be_dyn.last_commute_xz
        dyn_before = be_dyn.last_dynamic_core_scan

        # (2) production fast path (debug OFF): counters + wall-clock
        bd, wd = _wall(lambda: NearCliffordBackend(block=True, structure_once=False),
                       prog, WALL_K)
        bf, wf = _wall(lambda: NearCliffordBackend(block=True, structure_once=True),
                       prog, WALL_K)
        cx_after = bf.last_commute_xz
        dyn_after = bf.last_dynamic_core_scan
        fastlk = bf.last_fastpath_lookup
        so_en = bf.last_structure_once_enabled

        ok = bit and mism == 0
        allok &= ok
        print(f"{c:15s} {str(bit):6s} {str(so_en):5s} {mism:7d} "
              f"{cx_before:7d}->{cx_after:<8d} {dyn_before:4d}->{dyn_after:<6d} "
              f"{fastlk:6d} {wd:6.2f}->{wf:<8.2f} {wd/wf:4.2f} {bf.last_prepass_ms:9.1f}",
              flush=True)
    print("\nALL", "PASS" if allok else "FAIL")
    print("note: commute_xz/dynScan -> 0 means the per-measurement commute-judgment FLOPs are "
          "removed from the runtime; wall-clock gain is small where the dense magic flush "
          "(not the core scan) dominates -- the big win is multi-shot batching (not yet done).")
    return allok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
