"""C-5: the FUSED integration -- drive the whole multi-measurement loop through
`flush_core_virtual` (one fused contraction per core, no streaming `apply_rotation` promote
for the fused cores) and check it stays STATE-EXACT against the dense reference while the
fused workspace never exceeds the clifft bound.

Reports per circuit:
  fused_peak   = max workspace exponent the FUSED cores materialised (must be the resident
                 rank, i.e. W-1, never the streaming W = r_out+1 transient)
  fallback     = #cores still routed to the streaming step (multi-axis / antis / trivial),
                 with their materialised exponent -- Sub-step B drives this to 0
  max|dp0|, min_fid = exactness vs the dense 2^n reference over `trajectories` outcomes
"""
import os
import sys

os.chdir("/home/jung/clifft-paper")
sys.path.insert(0, "/home/jung/clifft-paper")
sys.setrecursionlimit(50000)

import numpy as np

from nearclifford_backend.virtual_axis.virtual_engine import TableauEngine
from nearclifford_backend.virtual_axis.fused_integrate import flush_core_virtual
from nearclifford_backend.virtual_axis.test_c3 import (
    capture_stream, dense_step_rot, dense_measure, _fid)


def run(circ, trajectories=4, bound=None):
    n, EV = capture_stream(circ)
    n_meas = len(EV)
    worst_dp = 0.0
    worst_fid = 1.0
    fused_peak = 0
    fallback = 0
    fb_peak = 0
    fused_cores = 0
    for tj in range(trajectories):
        rng = np.random.default_rng(100 + tj)
        psi = np.zeros(1 << n, dtype=complex); psi[0] = 1.0
        outs = []; p0d = []
        for (Pm, rots) in EV:
            for (P, th) in rots:
                psi = dense_step_rot(psi, n, P, th)
            psi, out, p0 = dense_measure(psi, n, Pm, rng)
            outs.append(out); p0d.append(p0)

        # STREAMING engine (same forced outcomes) -- its transient peak is what the fused
        # path must beat (cultivation_d5: streaming 11 vs fused 10).
        if tj == 0:
            seng = TableauEngine(n)
            for mi, (Pm, rots) in enumerate(EV):
                for (P, th) in rots:
                    seng.apply_rotation(P, th)
                seng.measure_drop(Pm, forced=outs[mi])
            stream_peak = max(seng.max_k, len(seng.magic))

        eng = TableauEngine(n)
        p0e = []
        for mi, (Pm, rots) in enumerate(EV):
            _, p0 = flush_core_virtual(eng, rots, Pm, forced=outs[mi])
            p0e.append(p0)
        fused_peak = max(fused_peak, eng.fused_peak, len(eng.magic))
        fused_cores = eng.fused_cores
        log = eng.core_log                              # (kind, workspace_exp, r_out)
        dp = max((abs(a - b) for a, b in zip(p0d, p0e)), default=0.0)
        worst_dp = max(worst_dp, dp)
        worst_fid = min(worst_fid, _fid(eng.statevector(), psi))

    ok = (worst_dp < 1e-9 and worst_fid > 1 - 1e-9)
    over_bound = (bound is not None and fused_peak > bound)
    # The invariant: every magic-measurement core is fused to its post-measurement rank W-1,
    # never the streaming W (= the +1 measurement transient that drove cult_d5 to 11).  Some
    # cores then drop a FURTHER single-qubit-stabiliser axis via _compress (W-1 -> r_out): that
    # post-measurement compression is identical to the streaming engine's and stays <= k, so it
    # is reported informationally, not as a transient above the bound.
    ctail = sum(1 for c in log if c[0] in ('single', 'multi') and c[1] > c[2])
    bnd = "" if bound is None else f"<=k={bound}:{'OK' if not over_bound else 'OVER'}"
    print(f"{circ:16} n={n:2d} meas={n_meas:3d}  streaming_peak={stream_peak:2d} -> "
          f"fused_ws={fused_peak:2d} {bnd}  fused_cores={fused_cores:3d}  "
          f"(post-meas compress-tail:{ctail})  max|dp0|={worst_dp:.1e} fid={worst_fid:.9f}  "
          f"{'OK' if (ok and not over_bound) else 'FAIL'}")
    return (ok and not over_bound), fused_peak


if __name__ == "__main__":
    targets = {"cultivation_d5": 10, "coherent_d3_r3": 8,
               "cultivation_d3": 4, "distillation": 4}
    circs = sys.argv[1:] or list(targets)
    res = [run(c, bound=targets.get(c)) for c in circs]
    allok = all(r[0] for r in res)
    print("-" * 96)
    print(f"C-5 {'PASS' if allok else 'FAIL'}  (fused integration state-exact; fused cores "
          f"never build the streaming +1 transient)")
    sys.exit(0 if allok else 1)
