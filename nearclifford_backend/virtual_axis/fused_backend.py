"""Run the fused virtual-axis engine AS a backend (co-evolved with clifft's Clifford frame),
so the fused peak workspace is measured WITHOUT ever building clifft's 2^k dense active state
-- the same way the block backend runs coherent_d5_r5 at 2^B, not 2^24.

The lazy frame (Clifford tableau + pending rotation ledger) is kept; phi is NEVER built
(`_promote` skips the kron).  At each measurement the whole anticommuting core is pulled
back and handed to a fused TableauEngine, which (a) decides the outcome -- Born for a magic
measurement, uniform for a stabiliser one -- and (b) tracks its own peak workspace.  The
lazy frame is updated with that SAME outcome (forced), so the trajectory is self-consistent.
"""
from __future__ import annotations

import os
import sys

os.chdir("/home/jung/clifft-paper")
sys.path.insert(0, "/home/jung/clifft-paper")
sys.setrecursionlimit(400000)

import numpy as np

from nearclifford_backend.simulator import pauli_commute, pauli_mul
from nearclifford_backend.lazy import LazyNearClifford
from nearclifford_backend.virtual_axis.virtual_engine import TableauEngine
from nearclifford_backend.virtual_axis.fused_integrate import flush_core_virtual


class FusedLazyNearClifford(LazyNearClifford):
    def __init__(self, n):
        super().__init__(n)
        self.feng = TableauEngine(n)
        self.max_fused_ws = 0

    # phi is never materialised -- promotion is a FRAME (M) update only
    def _promote(self, q):
        if q not in self.M:
            self.M.append(q)

    def _ag_measure_out(self, Pm, anti_s, out):
        """Gottesman-Knill frame update for stabiliser Pauli Pm, FORCED to outcome `out`."""
        p = anti_s[0]
        Sp = self.Zc[p]
        for i in range(self.n):
            if i != p and not pauli_commute(self.Zc[i], Pm):
                self.Zc[i] = pauli_mul(self.Zc[i], Sp)
            if not pauli_commute(self.Xc[i], Pm):
                self.Xc[i] = pauli_mul(self.Xc[i], Sp)
        self.Xc[p] = Sp
        self.Zc[p] = (Pm[0], Pm[1], (Pm[2] + 2 * out) & 3)
        self._frame_ver += 1

    def measure_z(self, q):
        Pm = (0, 1 << q, 0)
        # extract the anticommuting core from the lazy pending ledger, pull each rotation back
        core = self._dynamic_core(0, 1 << q)
        for r in core:
            self.pending.pop(r[4], None)
        rots = []
        for (x, z, p, theta, uid) in core:
            rots.append((self._pullback(x, z), theta))
        Pmp = self._pullback(0, 1 << q)

        # the fused engine is the SINGLE magic authority: it classifies the measurement
        # (antis / single / multi) on its OWN tableau, projects (magic Born) or collapses
        # (stabiliser) its register, and returns the realised outcome.  We do NOT pre-classify
        # on the lazy frame -- doing so double-evolved the two frames whenever they disagreed
        # (lazy 'antis' vs engine 'single'), which corrupted every later pulled-back core.
        out, _ = flush_core_virtual(self.feng, rots, Pmp, rng=self.rng)

        # mirror that SAME outcome into the lazy Clifford frame.  M is the engine's compressed
        # magic membership (it _compress'd out the axes that became stabilisers), so the lazy
        # antis test matches the engine's classification: if a non-magic stabiliser still
        # anticommutes with Z_q this was a stabiliser measurement -> Gottesman-Knill collapse
        # with `out`; otherwise it was a magic measurement and Zc/Xc are unchanged.
        self.M = list(self.feng.magic)
        magset = set(self.M)
        anti_s = [i for i in range(self.n)
                  if i not in magset and not pauli_commute(self.Zc[i], Pm)]
        if anti_s:
            self._ag_measure_out(Pm, anti_s, out)
        self.max_fused_ws = max(self.max_fused_ws,
                                getattr(self.feng, "fused_peak", 0), len(self.feng.magic))
        return out

    def statevector(self):                                # not supported (no phi)
        raise NotImplementedError("fused backend tracks rank only, not the dense state")


def fused_ws_backend(circ, seed=1):
    """Peak fused workspace exponent for `circ`, run as a backend (no clifft 2^k state)."""
    import clifft
    import nearclifford_backend.backend as bk
    prog = clifft.compile(open(f"qec_bench/circuits/{circ}.stim").read())
    orig = bk.LazyNearClifford
    bk.LazyNearClifford = FusedLazyNearClifford           # _reset picks this for lazy=True
    try:
        be = bk.NearCliffordBackend(lazy=True, drop_dead=False, structure_once=False)
        be.run_shot(prog, seed)
        return be.nc.max_fused_ws
    finally:
        bk.LazyNearClifford = orig


if __name__ == "__main__":
    import time
    from nearclifford_backend.virtual_axis.bench_memory import fused_ws_exact
    circs = sys.argv[1:] or ["cultivation_d3", "cultivation_d5", "coherent_d3_r3",
                             "distillation", "coherent_d3_r1"]
    for circ in circs:
        t = time.time()
        wb = fused_ws_backend(circ)
        dt = time.time() - t
        try:
            wd = fused_ws_exact(circ)
            tag = f"dense={wd} match={wb == wd}"
        except Exception as e:
            tag = f"dense infeasible ({type(e).__name__})"
        print(f"{circ:16} backend fused_ws={wb}  {tag}  ({dt:.1f}s)")
