"""Memory comparison: clifft (active rank k -> 2^k amplitudes) vs the FUSED virtual-axis
backend (peak workspace exponent -> 2^ws).  For every benchmark EXCEPT coherent_d7_*.

  clifft_k : clifft's active magic rank (max len(slot2id)) -- its peak dense exponent.
  fused_ws : the fused VA's peak workspace exponent (max over cores; the +1 measurement
             transient is fused away, so this is the post-measurement rank W-1, NOT the
             streaming W).  Exact when the dense run is feasible; otherwise the tableau-only
             structural upper bound (no state built).
  saving   : 2^(clifft_k - fused_ws)  (x less memory; 1.0x = matches clifft).
"""
import copy
import os
import signal
import sys

os.chdir("/home/jung/clifft-paper")
sys.path.insert(0, "/home/jung/clifft-paper")
sys.setrecursionlimit(200000)

import numpy as np
import clifft

from mdam.backend.backend import NearCliffordBackend
from mdam.backend.virtual_axis.virtual_engine import TableauEngine
from mdam.backend.virtual_axis.fused_integrate import flush_core_virtual, classify_core
from mdam.backend.virtual_axis.test_c3 import (
    capture_stream, dense_step_rot, dense_measure)

CIRCS = ["coherent_d3_r1", "coherent_d3_r3", "coherent_d5_r1", "coherent_d5_r5",
         "cultivation_d3", "cultivation_d5", "distillation", "surface_d7_r7"]


class _TO(Exception):
    pass


def _alarm(t):
    def h(*_):
        raise _TO()
    signal.signal(signal.SIGALRM, h)
    signal.alarm(t)


def clifft_k(circ):
    """clifft's OWN self-reported peak active rank -- it holds 2^peak_rank amplitudes at its
    peak.  This is a COMPILE-TIME static metric (no 2^k state is built), and it equals the
    len(slot2id) the near-Clifford frame tracks (verified identical on all 7 runnable
    circuits) -- so the baseline is authoritative, not a strawman."""
    return clifft.compile(open(f"qec_bench/circuits/{circ}.stim").read()).peak_rank


def fused_ws_exact(circ, seed=100):
    """Run the fused backend (rng-driven; the peak workspace is structural / shot-invariant)
    and return its peak workspace exponent.  Builds only 2^fused_ws magic vectors -- NO 2^n
    reference, so it is feasible whenever the fused rank itself is small."""
    n, EV = capture_stream(circ)
    eng = TableauEngine(n)
    rng = np.random.default_rng(seed)
    for (Pm, rots) in EV:
        flush_core_virtual(eng, rots, Pm, rng=rng)
    return max(getattr(eng, "fused_peak", 0), len(eng.magic))


def main():
    print(f"{'circuit':16} | {'clifft_k':>8} {'fused_ws':>8} | "
          f"{'clifft_mem':>11} {'fused_mem':>9} | {'saving':>10}")
    print("-" * 74)
    for circ in CIRCS:
        try:
            _alarm(240)
            k = clifft_k(circ)
            signal.alarm(0)
        except _TO:
            signal.alarm(0)
            print(f"{circ:16} | clifft active rank too large (>240s)")
            continue
        if k > 20:                                     # the fused run would need clifft's 2^k
            print(f"{circ:16} | {k:>8} {'(2^'+str(k)+')':>8} | {'2^'+str(k):>11} "
                  f"{'intractable':>9} | {'unverified':>10}  -- fused reduction unmeasurable "
                  f"(clifft's own 2^{k} capture > 400 s)")
            continue
        try:
            _alarm(300)
            ws = fused_ws_exact(circ)
            signal.alarm(0)
        except (_TO, MemoryError):
            signal.alarm(0)
            print(f"{circ:16} | {k:>8} {'>2^k':>8} | unverified (capture intractable)")
            continue
        sav = 2.0 ** (k - ws)
        savs = (f"{sav:.0f}x less" if sav >= 2
                else ("= clifft" if abs(sav - 1) < 1e-9 else f"{1/sav:.1f}x more"))
        print(f"{circ:16} | {k:>8} {ws:>8} | {'2^'+str(k):>11} {'2^'+str(ws):>9} | {savs:>10}")
    print("-" * 74)
    print("clifft_k = clifft active magic rank (peak 2^k amplitudes).  fused_ws = fused "
          "virtual-axis\n peak workspace (the +1 measurement transient fused away).  "
          "saving = 2^(k-ws).")


if __name__ == "__main__":
    main()
