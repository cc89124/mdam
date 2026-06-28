"""Measure the fused virtual-axis peak workspace on circuits whose clifft active rank is too
large for the dense capture (coherent_d5_r5: clifft 2^24).

The BLOCK backend runs them in seconds at 2^B (B=13 for coherent_d5_r5), with VALID Born
outcomes.  We capture its pulled-back measurement cores AND the realised outcomes, then
replay them on the fused TableauEngine with those outcomes FORCED -- a genuine trajectory,
never building clifft's 2^24 state.  (The cores are outcome-dependent, so the outcomes must
be forced, not re-sampled.)
"""
import os
import sys
import time

os.chdir("/home/jung/clifft-paper")
sys.path.insert(0, "/home/jung/clifft-paper")
sys.setrecursionlimit(200000)

import numpy as np
import clifft

from nearclifford_backend.backend import NearCliffordBackend, count_idents
from nearclifford_backend.block_magic import BlockLazyNearClifford
from nearclifford_backend.virtual_axis.virtual_engine import TableauEngine
from nearclifford_backend.virtual_axis.fused_integrate import flush_core_virtual


def capture_block(circ, seed=1):
    """(n, EV, OUTS): EV = [(Pm,[(P,theta)...])...] pulled-back cores, OUTS[i] = the block
    backend's realised outcome for core i.  Magic held block-factored (2^B), not 2^k."""
    EV = []
    OUTS = []
    o_fc = BlockLazyNearClifford._flush_core
    o_f1 = BlockLazyNearClifford._flush_one
    o_mz = BlockLazyNearClifford.measure_z

    def fc(self, qx, qz):
        EV.append((self._pullback(qx, qz), []))
        return o_fc(self, qx, qz)

    def f1(self, x, z, theta):
        if EV:
            EV[-1][1].append((self._pullback(x, z), theta))
        return o_f1(self, x, z, theta)

    def mz(self, q):
        before = len(EV)
        out = o_mz(self, q)
        if len(EV) > len(OUTS):                # one core flushed -> record its outcome
            OUTS.append(int(out))
        return out

    BlockLazyNearClifford._flush_core = fc
    BlockLazyNearClifford._flush_one = f1
    BlockLazyNearClifford.measure_z = mz
    try:
        prog = clifft.compile(open(f"qec_bench/circuits/{circ}.stim").read())
        n = count_idents(prog)
        be = NearCliffordBackend(block=True)
        be.run_shot(prog, seed)
    finally:
        BlockLazyNearClifford._flush_core = o_fc
        BlockLazyNearClifford._flush_one = o_f1
        BlockLazyNearClifford.measure_z = o_mz
    return n, EV, OUTS


def fused_ws(circ, seed=1):
    n, EV, OUTS = capture_block(circ, seed)
    eng = TableauEngine(n)
    for i, (Pm, rots) in enumerate(EV):
        forced = OUTS[i] if i < len(OUTS) else None
        flush_core_virtual(eng, rots, Pm, forced=forced)
    return max(getattr(eng, "fused_peak", 0), len(eng.magic))


if __name__ == "__main__":
    # validate against the dense-capture fused_ws on small circuits, then run the big one
    from nearclifford_backend.virtual_axis.test_c3 import capture_stream
    from nearclifford_backend.virtual_axis.bench_memory import fused_ws_exact
    circs = sys.argv[1:] or ["cultivation_d3", "cultivation_d5", "coherent_d3_r3",
                             "distillation", "coherent_d5_r5"]
    for circ in circs:
        t = time.time()
        wb = fused_ws(circ)
        dt = time.time() - t
        ref = ""
        try:
            wd = fused_ws_exact(circ)
            ref = f"  (dense-capture fused_ws={wd}, match={wb == wd})"
        except Exception:
            ref = "  (dense-capture infeasible)"
        print(f"{circ:16} block-capture fused_ws={wb}{ref}  ({dt:.1f}s)")
