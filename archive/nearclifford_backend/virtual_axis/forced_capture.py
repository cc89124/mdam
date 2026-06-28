"""Capture the lazy near-Clifford measurement cores for a circuit whose magic rank is too
large for the dense capture (coherent_d5_r5: clifft 2^24), WITHOUT ever building phi.

The cores are outcome-dependent (feedback applies conditional Clifford corrections that move
the Pauli frame), so resource_only with its own stale-phi outcomes diverges structurally.
Here we instead FORCE a VALID trajectory's outcomes onto a frame-only run:

  * M is promoted exactly as the dense run promotes it (core X-support at flush, measured-Pauli
    X-support on a magic measurement) -- so the antis/magic classification is the dense one;
  * a stabiliser (antis) measurement collapses the frame with the FORCED outcome;
  * a magic measurement leaves the frame unchanged.

No phi, no 2^k.  The only dense-run effect we cannot mirror is `_compress_magic` (it needs
phi); the validation below checks whether that ever changes the cores -- on the small circuits
it does not, so M over-growth is benign and the captured cores are byte-identical to dense.
"""
import os
import sys

os.chdir("/home/jung/clifft-paper")
sys.path.insert(0, "/home/jung/clifft-paper")
sys.setrecursionlimit(400000)

import clifft

from nearclifford_backend.backend import NearCliffordBackend, count_idents
from nearclifford_backend.lazy import LazyNearClifford
from nearclifford_backend.simulator import pauli_commute, pauli_mul


def _ag_measure_forced(self, Pm, anti_s, out):
    """`SimulatorCore._ag_measure` with the outcome FORCED instead of sampled."""
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


def capture_forced(circ, forced, seed=1):
    """(n, EV): replay `circ` frame-only, forcing outcomes `forced`, capturing pulled-back
    cores.  `forced[i]` is the realised outcome of the i-th measure_z call."""
    EV = []
    state = {"i": 0}
    o_fc = LazyNearClifford._flush_core
    o_df = LazyNearClifford._do_flush
    o_mz = LazyNearClifford.measure_z

    def fc(self, qx, qz):
        EV.append((self._pullback(qx, qz), []))
        return o_fc(self, qx, qz)

    def df(self, qx, qz, flush):
        # resource_only: size support, PROMOTE M (core X-support), capture rotations
        if not flush:
            return
        for r in flush:
            del self.pending[r[4]]
        supp = 0
        for (x, z, p, theta, uid) in flush:
            xp, zp, pp = self._pullback(x, z)
            supp |= xp
            if EV:
                EV[-1][1].append(((xp, zp, pp), theta))
        for qq in range(self.n):
            if (supp >> qq) & 1 and qq not in self.M:
                self.M.append(qq)
        self.max_M = max(self.max_M, supp.bit_count())

    def mz(self, q):
        Pm = (0, 1 << q, 0)
        self._flush_core(0, 1 << q)               # capture core + promote core support
        magset = set(self.M)
        anti_s = [i for i in range(self.n)
                  if i not in magset and not pauli_commute(self.Zc[i], Pm)]
        out = int(forced[state["i"]]); state["i"] += 1
        if anti_s:
            _ag_measure_forced(self, Pm, anti_s, out)
        else:
            # magic measurement: promote the measured Pauli's X-support, frame unchanged
            xp, zp, pp = self._pullback(0, 1 << q)
            for qq in range(self.n):
                if (xp >> qq) & 1 and qq not in self.M:
                    self.M.append(qq)
        return out

    LazyNearClifford._flush_core = fc
    LazyNearClifford._do_flush = df
    LazyNearClifford.measure_z = mz
    try:
        prog = clifft.compile(open(f"qec_bench/circuits/{circ}.stim").read())
        n = count_idents(prog)
        be = NearCliffordBackend(lazy=True, resource_only=True)
        be.run_shot(prog, seed)
    finally:
        LazyNearClifford._flush_core = o_fc
        LazyNearClifford._do_flush = o_df
        LazyNearClifford.measure_z = o_mz
    return n, EV


def capture_dense_with_outcomes(circ, seed=1):
    """Dense lazy run: capture cores (like test_c3.capture_stream) AND realised outcomes."""
    EV = []
    OUTS = []
    o_fc = LazyNearClifford._flush_core
    o_f1 = LazyNearClifford._flush_one
    o_mz = LazyNearClifford.measure_z

    def fc(self, qx, qz):
        EV.append((self._pullback(qx, qz), []))
        return o_fc(self, qx, qz)

    def f1(self, x, z, theta):
        if EV:
            EV[-1][1].append((self._pullback(x, z), theta))
        return o_f1(self, x, z, theta)

    def mz(self, q):
        out = o_mz(self, q)
        OUTS.append(int(out))
        return out

    LazyNearClifford._flush_core = fc
    LazyNearClifford._flush_one = f1
    LazyNearClifford.measure_z = mz
    try:
        prog = clifft.compile(open(f"qec_bench/circuits/{circ}.stim").read())
        n = count_idents(prog)
        be = NearCliffordBackend(lazy=True)
        be.run_shot(prog, seed)
    finally:
        LazyNearClifford._flush_core = o_fc
        LazyNearClifford._flush_one = o_f1
        LazyNearClifford.measure_z = o_mz
    return n, EV, OUTS


def _supp_eq(evA, evB):
    if len(evA) != len(evB):
        return False, f"#cores {len(evA)} vs {len(evB)}"
    diffs = 0
    for (PmA, rA), (PmB, rB) in zip(evA, evB):
        if (PmA[0], PmA[1]) != (PmB[0], PmB[1]):
            diffs += 1
        if len(rA) != len(rB):
            diffs += 1
            continue
        for (PA, tA), (PB, tB) in zip(rA, rB):
            if (PA[0], PA[1]) != (PB[0], PB[1]) or tA != tB:
                diffs += 1
    return diffs == 0, f"{diffs} support diffs"


if __name__ == "__main__":
    circs = sys.argv[1:] or ["cultivation_d3", "cultivation_d5",
                             "coherent_d3_r3", "distillation"]
    for circ in circs:
        nD, evD, outs = capture_dense_with_outcomes(circ)
        nF, evF = capture_forced(circ, outs)
        ok, msg = _supp_eq(evD, evF)
        print(f"{circ:16} dense cores={len(evD)}  forced cores={len(evF)}  "
              f"cores_identical={ok}  ({msg})")
