"""Instrumentation: for every measurement core in a circuit, report the structure the
fused integration must handle. Drives the STREAMING engine (ground truth) and records,
per core:
  r_in   = magic axes at core start
  W      = magic axes after the rotations, before the measurement (tableau-only work basis)
  a_kind = measured-axis kind: 'fresh' (a >= r_in, a newly-opened |0> ancilla) or
           'preexist' (a < r_in, the measured axis is an existing magic axis)
  Psupp  = support count of P_meas's mask over the W axes (1 = single-axis; >1 = multi-axis
           Pauli needing a synthesised Clifford to a single Z)
  hasX   = P_meas mask has X-part on some axis (needs H/Y rotation, not just Z)
  r_drop = magic axes after measure_drop (includes its internal compress)
  extra  = (W - 1) - r_drop  = axes dropped BEYOND the single measured one (compress tail)
Counts how often each generalisation actually fires, so the fused path only implements what
the circuits exercise -- and flags any core where the simple single-ephemeral-axis contraction
would leave a transient above the post-core resident rank.
"""
import copy
import os
import sys

os.chdir("/home/jung/clifft-paper")
sys.path.insert(0, "/home/jung/clifft-paper")
sys.setrecursionlimit(50000)

import numpy as np

from nearclifford_backend.virtual_axis.virtual_engine import TableauEngine
from nearclifford_backend.virtual_axis.test_c3 import (
    capture_stream, dense_step_rot, dense_measure)


def probe(circ, seed=100):
    n, EV = capture_stream(circ)
    rng = np.random.default_rng(seed)
    psi = np.zeros(1 << n, dtype=complex); psi[0] = 1.0
    outs = []
    for (Pm, rots) in EV:
        for (P, th) in rots:
            psi = dense_step_rot(psi, n, P, th)
        psi, out, p0 = dense_measure(psi, n, Pm, rng); outs.append(out)

    eng = TableauEngine(n)
    rows = []
    peak_stream = 0
    for mi, (Pm, rots) in enumerate(EV):
        r_in = len(eng.magic)
        # tableau-only probe of the work basis (no phi growth)
        probe_eng = copy.deepcopy(eng)
        saved = probe_eng.phi
        probe_eng.phi = None
        for (P, th) in rots:
            probe_eng._mask_for(P)
        W = len(probe_eng.magic)
        mmx, mmz, _ = probe_eng._mask_for(Pm)
        supp = [s for s in range(len(probe_eng.magic))
                if ((mmx >> s) & 1) or ((mmz >> s) & 1)]
        hasX = any((mmx >> s) & 1 for s in supp)
        # which axis would the measurement collapse?  (the X-axis if any, else the Z-axis)
        if supp:
            xax = [s for s in supp if (mmx >> s) & 1]
            a = xax[0] if xax else supp[0]
            a_kind = 'fresh' if a >= r_in else 'preexist'
        else:
            a = None; a_kind = 'trivial'
        probe_eng.phi = saved

        # now the REAL streaming step (with phi + compress) for ground-truth r_drop
        for (P, th) in rots:
            eng.apply_rotation(P, th)
        peak_stream = max(peak_stream, eng.max_k)
        eng.measure_drop(Pm, forced=outs[mi])
        r_drop = len(eng.magic)
        extra = (W - 1) - r_drop
        rows.append((mi, r_in, W, len(supp), int(hasX), a_kind, r_drop, extra))

    # summary
    multiaxis = sum(1 for r in rows if r[3] > 1)
    hasx = sum(1 for r in rows if r[4])
    preexist = sum(1 for r in rows if r[5] == 'preexist')
    extrapos = [r for r in rows if r[7] > 0]
    maxW = max(r[2] for r in rows)
    print(f"\n=== {circ}  (n={n}, cores={len(rows)}, streaming peak max_k={peak_stream}) ===")
    print(f"  maxW(work basis)={maxW}   multi-axis P_meas={multiaxis}   "
          f"P_meas hasX={hasx}   measured-axis preexisting={preexist}")
    print(f"  cores with compress-tail (extra>0, simple fused leaves transient W-1>r_drop): "
          f"{len(extrapos)}")
    for r in extrapos[:20]:
        print(f"    mi={r[0]:3d} r_in={r[1]} W={r[2]} Psupp={r[3]} hasX={r[4]} "
              f"a={r[5]} r_drop={r[6]} extra={r[7]}")
    # the cores that drive the peak
    drivers = sorted(rows, key=lambda r: -r[2])[:6]
    print("  top-W cores:")
    for r in drivers:
        print(f"    mi={r[0]:3d} r_in={r[1]} W={r[2]} Psupp={r[3]} hasX={r[4]} "
              f"a={r[5]} r_drop={r[6]} extra={r[7]}")
    return rows


if __name__ == "__main__":
    for c in (sys.argv[1:] or ["cultivation_d5", "coherent_d3_r3",
                               "cultivation_d3", "distillation"]):
        probe(c)
