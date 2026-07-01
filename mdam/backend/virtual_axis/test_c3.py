"""C-3 verification: the streaming virtual-axis engine reproduces the magic register's
full evolution over MANY measurements.

The magic register state over physical qubits is persistent (Clifford gates / stabiliser
measurements touch only the outer frame, never phi), so replaying the pulled-back operator
stream -- per measurement: the flushed core rotations then the measured Pauli -- on a
persistent register is exactly what the monolithic backend does to phi. We capture that
stream, then:
  * DENSE reference: a 2^n register; apply each rotation exp(-i th P/2), measure each Pauli
    (a seeded rng picks every outcome, recording p0).
  * ENGINE: TableauEngine(n); same stream, FORCED to the dense reference's outcomes.
Compare p0 at every measurement and the final magic-register statevector fidelity. A
stabiliser-type measurement (Pauli on a |0> direction) is handled uniformly by both
(engine's antis branch / dense projection), so no magic/stabiliser split is needed.
"""
import os
import sys

os.chdir("/home/jung/clifft-paper")
sys.path.insert(0, "/home/jung/clifft-paper")
sys.setrecursionlimit(50000)

import numpy as np
import clifft

from mdam.backend.backend import NearCliffordBackend, count_idents
from mdam.backend.lazy import LazyNearClifford
from mdam.backend.block_magic import _apply_pauli_local
from mdam.backend.virtual_axis.virtual_engine import TableauEngine


def capture_stream(circ, seed=1):
    """Return (n, events) where events = [(Pm, [(P,theta)...]) ...] per measurement, in
    order, all pulled back through the frame at flush time."""
    EV = []
    o_fc = LazyNearClifford._flush_core
    o_f1 = LazyNearClifford._flush_one

    def fc(self, qx, qz):
        EV.append((self._pullback(qx, qz), []))
        return o_fc(self, qx, qz)

    def f1(self, x, z, theta):
        if EV:
            EV[-1][1].append((self._pullback(x, z), theta))
        return o_f1(self, x, z, theta)

    LazyNearClifford._flush_core = fc
    LazyNearClifford._flush_one = f1
    try:
        prog = clifft.compile(open(f"qec_bench/circuits/{circ}.stim").read())
        n = count_idents(prog)
        be = NearCliffordBackend(lazy=True)
        be.run_shot(prog, seed)
    finally:
        LazyNearClifford._flush_core = o_fc
        LazyNearClifford._flush_one = o_f1
    return n, EV


def dense_step_rot(psi, n, P, theta):
    Pv = _apply_pauli_local(list(range(n)), psi, P[0], P[1], P[2])
    return np.cos(theta / 2.0) * psi - 1j * np.sin(theta / 2.0) * Pv


def dense_measure(psi, n, P, rng):
    Pv = _apply_pauli_local(list(range(n)), psi, P[0], P[1], P[2])
    exp = float(np.real(np.vdot(psi, Pv)))
    p0 = min(1.0, max(0.0, 0.5 * (1.0 + exp)))
    out = 0 if float(rng.random()) < p0 else 1
    sign = 1.0 if out == 0 else -1.0
    proj = 0.5 * (psi + sign * Pv)
    nrm = np.linalg.norm(proj)
    if nrm > 1e-12:
        psi = proj / nrm
    return psi, out, p0


def _fid(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 1.0 if (na < 1e-12 and nb < 1e-12) else 0.0
    return abs(complex(np.vdot(a, b))) / (na * nb)


def run(circ, trajectories=4):
    n, EV = capture_stream(circ)
    n_meas = len(EV)
    worst_dp = 0.0
    worst_fid = 1.0
    max_k = 0
    promote_total = 0
    for tj in range(trajectories):
        rng = np.random.default_rng(100 + tj)
        # ---- DENSE reference generates the outcome stream + p0 ----
        psi = np.zeros(1 << n, dtype=complex); psi[0] = 1.0
        outs = []
        p0d = []
        for (Pm, rots) in EV:
            for (P, th) in rots:
                psi = dense_step_rot(psi, n, P, th)
            psi, out, p0 = dense_measure(psi, n, Pm, rng)
            outs.append(out); p0d.append(p0)
        # ---- ENGINE forced to the same outcomes (measure_drop = C-3b minimal-rank) ----
        eng = TableauEngine(n)
        p0e = []
        for mi, (Pm, rots) in enumerate(EV):
            for (P, th) in rots:
                eng.apply_rotation(P, th)
            _, p0 = eng.measure_drop(Pm, forced=outs[mi])
            p0e.append(p0)
        max_k = max(max_k, eng.max_k, len(eng.magic))
        promote_total = eng.promote_calls
        dp = max((abs(a - b) for a, b in zip(p0d, p0e)), default=0.0)
        worst_dp = max(worst_dp, dp)
        worst_fid = min(worst_fid, _fid(eng.statevector(), psi))

    ok = (worst_dp < 1e-9 and worst_fid > 1 - 1e-9)
    print(f"{circ:16}  n={n:2d} meas={n_meas:3d}  max_k={max_k:2d}  promotes={promote_total:3d}  "
          f"max|dp0|={worst_dp:.1e}  min_fid={worst_fid:.9f}  {'OK' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    circs = sys.argv[1:] or ["distillation", "cultivation_d3", "coherent_d3_r3",
                             "cultivation_d5"]
    res = [run(c) for c in circs]
    allok = all(res)
    print("-" * 78)
    print(f"C-3 {'PASS' if allok else 'FAIL'}  (streaming engine state-exact over the full "
          f"multi-measurement loop, with measurement DROP -> active rank)")
    sys.exit(0 if allok else 1)
