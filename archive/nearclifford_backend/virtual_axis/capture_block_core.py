"""Measure the fused virtual-axis peak workspace (fused_ws) on circuits whose clifft active
rank is too large for the dense capture (coherent_d5_r5: clifft 2^24), WITHOUT building any
2^k state and WITHOUT clifft.

Key: the BLOCK backend runs these in seconds at 2^B with VALID Born outcomes and a VALID
Clifford frame (it reuses LazyNearClifford's tableau + pending-rotation ledger; it only
stores the magic register block-factored instead of as a flat 2^k vector).  So the set of
pending rotations forming each measurement core -- and their pullback through the frame --
is EXACTLY the lazy/clifft-streaming core; only the magic *storage* differs.

We therefore capture `_dynamic_core(0,1<<q)` (the full anticommuting lazy core) at each
measurement, pulled back through the block backend's valid frame, then replay those cores on
the fused TableauEngine.  (The earlier capture_block.py grabbed `_flush_one` calls = the
block-FACTORED groups, which gave the wrong, block-structured workspace.)

The fused peak is structural over a valid trajectory; the validation below confirms the
block-captured cores reproduce the dense-capture fused_ws bit-for-bit on every small circuit,
so the coherent_d5_r5 number it then reports is trustworthy.
"""
import os
import sys
import time

os.chdir("/home/jung/clifft-paper")
sys.path.insert(0, "/home/jung/clifft-paper")
sys.setrecursionlimit(400000)

import numpy as np
import clifft

from nearclifford_backend.backend import NearCliffordBackend, count_idents
from nearclifford_backend.block_magic import BlockLazyNearClifford
from nearclifford_backend.virtual_axis.virtual_engine import TableauEngine
from nearclifford_backend.virtual_axis.fused_integrate import flush_core_virtual


def capture_block_core(circ, seed=1):
    """(n, EV, OUTS): EV = [(Pm,[(P,theta)...])...] the LAZY cores pulled back through the
    block backend's valid frame; OUTS[i] = the realised outcome of measurement i.  Magic is
    held block-factored (2^B), never 2^k."""
    EV = []
    OUTS = []
    o_mz = BlockLazyNearClifford.measure_z

    def mz(self, q):
        # capture the full lazy core (pending rotations anticommuting with Z_q), pulled back
        # through the CURRENT (valid) frame -- BEFORE the measurement collapses it
        core = self._dynamic_core(0, 1 << q)               # increasing-uid order (= dense)
        Pm = self._pullback(0, 1 << q)
        rots = [(self._pullback(x, z), theta) for (x, z, p, theta, uid) in core]
        EV.append((Pm, rots))
        out = o_mz(self, q)
        OUTS.append(int(out))
        return out

    BlockLazyNearClifford.measure_z = mz
    try:
        prog = clifft.compile(open(f"qec_bench/circuits/{circ}.stim").read())
        n = count_idents(prog)
        be = NearCliffordBackend(block=True)
        be.run_shot(prog, seed)
    finally:
        BlockLazyNearClifford.measure_z = o_mz
    return n, EV, OUTS


def fused_ws_block(circ, seed=1, return_detail=False):
    n, EV, OUTS = capture_block_core(circ, seed)
    eng = TableauEngine(n)
    for i, (Pm, rots) in enumerate(EV):
        forced = OUTS[i] if i < len(OUTS) else None
        flush_core_virtual(eng, rots, Pm, forced=forced)
    ws = max(getattr(eng, "fused_peak", 0), len(eng.magic))
    if return_detail:
        return ws, n, len(EV)
    return ws


if __name__ == "__main__":
    from nearclifford_backend.virtual_axis.bench_memory import fused_ws_exact, clifft_k
    circs = sys.argv[1:] or ["cultivation_d3", "cultivation_d5", "coherent_d3_r3",
                             "distillation", "coherent_d3_r1", "coherent_d5_r5"]
    for circ in circs:
        t = time.time()
        ws, n, nc = fused_ws_block(circ, return_detail=True)
        dt = time.time() - t
        try:
            k = clifft_k(circ)
        except Exception:
            k = "?"
        ref = ""
        if circ != "coherent_d5_r5":
            try:
                wd = fused_ws_exact(circ)
                ref = f"  dense_fused_ws={wd} MATCH={ws == wd}"
            except Exception as e:
                ref = f"  (dense infeasible: {type(e).__name__})"
        sv = (f"  saving=2^{k - ws}" if isinstance(k, int) else "")
        print(f"{circ:16} clifft_k={k}  fused_ws={ws}{sv}  (n={n},cores={nc}){ref}  ({dt:.1f}s)")
