"""C-2 verification: persistent virtual state carried across TWO measurements by a
basis change. Isolates the basis-change mechanism:

  Path D (dense)   : 2^B union block -- |0>, rots1, measure A, rots2, measure B.
  Path A (1 basis) : basis2 confines {rots1,A,rots2,B}; evolve everything in basis2.
  Path B (change)  : basis1 confines {rots1,A}; evolve rots1, measure A -> phi_1a;
                     change_basis(phi_1a, basis1, basis2); evolve rots2, measure B.

Path A and Path D agree => C-1 machinery over the union is exact. Path B == Path A then
proves the BASIS CHANGE (the only differing step) is exact: the persistent state carried
from basis1 to basis2 reproduces the from-scratch basis2 evolution. Checked for every
forced outcome (a,b) in {0,1}^2: Born p0_A, p0_B|a, and final statevector fidelity.

Cores are captured from the MONOLITHIC near-Clifford backend (block=False): its magic
register is a clean dense phi with no purge/frame-folding, so the captured pulled-back
(rotation, measurement) stream is a faithful magic-block reference.
"""
import os
import sys

os.chdir("/home/jung/clifft-paper")
sys.path.insert(0, "/home/jung/clifft-paper")
sys.setrecursionlimit(50000)

import numpy as np
import clifft

from nearclifford_backend.backend import NearCliffordBackend
from nearclifford_backend.lazy import LazyNearClifford
from nearclifford_backend.block_magic import _apply_pauli_local
from nearclifford_backend.virtual_axis.virtual_axis import _bits
from nearclifford_backend.virtual_axis.virtual_runtime import (
    build_basis, express, change_basis, reconstruct_physical, junk_x_bits)


def capture_first_two_magic(circ, seed=1, want=2):
    """Capture the first `want` measurements that flush a non-empty magic core from the
    monolithic backend: each as (rots=[(xp,zp,pp,theta)...], meas_pb=(x,z,ph))."""
    REC = []
    o_fc = LazyNearClifford._flush_core
    o_f1 = LazyNearClifford._flush_one

    def fc(self, qx, qz):
        REC.append({"meas": self._pullback(qx, qz), "rots": []})
        return o_fc(self, qx, qz)

    def f1(self, x, z, theta):
        xp, zp, pp = self._pullback(x, z)
        if REC:
            REC[-1]["rots"].append((xp, zp, pp, theta))
        return o_f1(self, x, z, theta)

    LazyNearClifford._flush_core = fc
    LazyNearClifford._flush_one = f1
    try:
        prog = clifft.compile(open(f"qec_bench/circuits/{circ}.stim").read())
        be = NearCliffordBackend(lazy=True)
        be.run_shot(prog, seed)
    finally:
        LazyNearClifford._flush_core = o_fc
        LazyNearClifford._flush_one = o_f1
    magic = [e for e in REC if e["rots"]]
    return magic[:want]


# ----------------------------------------------------- dense + virtual kernels
def _vapply_rot(phi, mask, r, theta):
    Pv = _apply_pauli_local(list(range(r)), phi, mask.x, mask.z, mask.phase)
    return np.cos(theta / 2.0) * phi - 1j * np.sin(theta / 2.0) * Pv


def _vmeasure(phi, mask, r, out):
    Pv = _apply_pauli_local(list(range(r)), phi, mask.x, mask.z, mask.phase)
    exp = float(np.real(np.vdot(phi, Pv)))
    p0 = min(1.0, max(0.0, 0.5 * (1.0 + exp)))
    sign = 1.0 if out == 0 else -1.0
    proj = 0.5 * (phi + sign * Pv)
    nrm = np.linalg.norm(proj)
    return p0, (proj / nrm if nrm > 1e-12 else proj)


def _fid(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 1.0 if (na < 1e-12 and nb < 1e-12) else 0.0
    return abs(complex(np.vdot(a, b))) / (na * nb)


def verify_two(label, rots1, A, rots2, B):
    sup = set()
    for r in (rots1, rots2):
        for (x, z, p, th) in r:
            sup |= set(_bits(x)) | set(_bits(z))
    for P in (A, B):
        sup |= set(_bits(P[0])) | set(_bits(P[1]))
    support = sorted(sup)
    Bn = len(support)
    posn = {q: i for i, q in enumerate(support)}

    def loc(P):
        lx = lz = 0
        for q in _bits(P[0]):
            lx |= 1 << posn[q]
        for q in _bits(P[1]):
            lz |= 1 << posn[q]
        return lx, lz, (P[2] if len(P) > 2 else 0)

    # bases
    rotP1 = [(x, z, p) for (x, z, p, th) in rots1]
    rotP2 = [(x, z, p) for (x, z, p, th) in rots2]
    basis1 = build_basis(rotP1 + [A], support)
    basis2 = build_basis(rotP1 + rotP2 + [A, B], support)

    # soundness: every operator expressed must have NO X on its basis's junk axes
    sound = True
    for P in rotP1 + [A]:
        _, Q = express(P, basis1)
        if junk_x_bits(Q, basis1["pivots"]):
            sound = False
    for P in rotP1 + rotP2 + [A, B]:
        _, Q = express(P, basis2)
        if junk_x_bits(Q, basis2["pivots"]):
            sound = False

    r1, r2 = basis1["r"], basis2["r"]
    # masks
    m1_in_b1 = [express(P, basis1)[0] for P in rotP1]
    A_in_b1 = express(A, basis1)[0]
    m1_in_b2 = [express(P, basis2)[0] for P in rotP1]
    A_in_b2 = express(A, basis2)[0]
    m2_in_b2 = [express(P, basis2)[0] for P in rotP2]
    B_in_b2 = express(B, basis2)[0]

    worst = 0.0
    worst_fid = 1.0
    max_lost = 0.0
    n_change_gates = len(basis1["cnots"]) + len(basis2["cnots"])
    for a in (0, 1):
        for b in (0, 1):
            # ---- Path D: dense 2^B union ----
            blk = np.zeros(1 << Bn, dtype=complex)
            blk[0] = 1.0
            for (x, z, p, th) in rots1:
                lx, lz, _ = loc((x, z, p))
                blk = np.cos(th / 2) * blk - 1j * np.sin(th / 2) * _apply_pauli_local(
                    list(range(Bn)), blk, lx, lz, p)
            alx, alz, ap = loc(A)
            PvA = _apply_pauli_local(list(range(Bn)), blk, alx, alz, ap)
            p0A_d = min(1.0, max(0.0, 0.5 * (1.0 + float(np.real(np.vdot(blk, PvA))))))
            sgn = 1.0 if a == 0 else -1.0
            blk = 0.5 * (blk + sgn * PvA)
            nb = np.linalg.norm(blk)
            if nb < 1e-12:
                continue                         # impossible branch; skip
            blk /= nb
            for (x, z, p, th) in rots2:
                lx, lz, _ = loc((x, z, p))
                blk = np.cos(th / 2) * blk - 1j * np.sin(th / 2) * _apply_pauli_local(
                    list(range(Bn)), blk, lx, lz, p)
            blx, blz, bp = loc(B)
            PvB = _apply_pauli_local(list(range(Bn)), blk, blx, blz, bp)
            p0B_d = min(1.0, max(0.0, 0.5 * (1.0 + float(np.real(np.vdot(blk, PvB))))))
            sgn = 1.0 if b == 0 else -1.0
            blkD = 0.5 * (blk + sgn * PvB)
            nb = np.linalg.norm(blkD)
            blkD = blkD / nb if nb > 1e-12 else blkD

            # ---- Path A: single basis2 ----
            phi = np.zeros(1 << r2, dtype=complex); phi[0] = 1.0
            for mk, (x, z, p, th) in zip(m1_in_b2, rots1):
                phi = _vapply_rot(phi, mk, r2, th)
            p0A_A, phi = _vmeasure(phi, A_in_b2, r2, a)
            for mk, (x, z, p, th) in zip(m2_in_b2, rots2):
                phi = _vapply_rot(phi, mk, r2, th)
            p0B_A, phiA = _vmeasure(phi, B_in_b2, r2, b)

            # ---- Path B: basis1 -> change_basis -> basis2 ----
            phi = np.zeros(1 << r1, dtype=complex); phi[0] = 1.0
            for mk, (x, z, p, th) in zip(m1_in_b1, rots1):
                phi = _vapply_rot(phi, mk, r1, th)
            p0A_B, phi = _vmeasure(phi, A_in_b1, r1, a)
            phi, lost = change_basis(phi, basis1, basis2)
            max_lost = max(max_lost, lost)
            for mk, (x, z, p, th) in zip(m2_in_b2, rots2):
                phi = _vapply_rot(phi, mk, r2, th)
            p0B_B, phiB = _vmeasure(phi, B_in_b2, r2, b)

            # ---- compare (Born + final fidelity) all three paths ----
            recA = reconstruct_physical(phiA, basis2)
            recB = reconstruct_physical(phiB, basis2)
            errs = [abs(p0A_d - p0A_A), abs(p0A_d - p0A_B),
                    abs(p0B_d - p0B_A), abs(p0B_d - p0B_B)]
            worst = max([worst] + errs)
            worst_fid = min(worst_fid, _fid(recA, blkD), _fid(recB, blkD))

    ok = (sound and worst < 1e-9 and worst_fid > 1 - 1e-9 and max_lost < 1e-12)
    print(f"{label:22}  B={Bn:2d} r1={r1:2d} r2={r2:2d}  change_gates={n_change_gates:3d}  "
          f"max|dp0|={worst:.1e}  min_fid={worst_fid:.9f}  lost={max_lost:.1e}  "
          f"sound={sound}  {'OK' if ok else 'FAIL'}")
    return ok, n_change_gates


def run_circuit(circ):
    mm = capture_first_two_magic(circ)
    if len(mm) < 2:
        print(f"{circ:22}  (<2 magic-core measurements; skip)")
        return None
    m1, m2 = mm
    return verify_two(circ, m1["rots"], m1["meas"], m2["rots"], m2["meas"])


def synthetic_two():
    """Scenarios with X-redundancy so the CNOT basis change is NON-TRIVIAL (change_gates>0):
    a persistent entangled magic state must be re-confined into the second basis."""
    t1, t2 = 0.7, 0.9
    out = []
    # m1: entangle X0X1, measure Z0Z1 ; m2: entangle X1X2, measure Z1  (3 qubits)
    out.append(("synth:X0X1|X1X2",
                [(0b011, 0, 0, t1)], (0, 0b011, 0),
                [(0b110, 0, 0, t2)], (0, 0b010, 0)))
    # m1: X0X1 twice (dup, X-rank1) measure Z0 ; m2: X0X1X2 measure X2
    out.append(("synth:dupX0X1|X012",
                [(0b011, 0, 0, t1), (0b011, 0, 0, 0.3)], (0, 0b001, 0),
                [(0b111, 0, 0, t2)], (0b100, 0, 0)))
    # m1: Y0Y1 measure Z0Z1 ; m2: X0 measure Z0  (Y entanglement carried)
    out.append(("synth:Y0Y1|X0",
                [(0b011, 0b011, 0, t1)], (0, 0b011, 0),
                [(0b001, 0, 0, t2)], (0, 0b001, 0)))
    return out


if __name__ == "__main__":
    print("== real circuits (first two magic-core measurements) ==")
    circs = sys.argv[1:] or ["distillation", "cultivation_d3", "cultivation_d5",
                             "coherent_d3_r3"]
    res = [run_circuit(c) for c in circs]
    print("== synthetic (non-trivial CNOT basis change) ==")
    res += [verify_two(lbl, r1, A, r2, B) for (lbl, r1, A, r2, B) in synthetic_two()]
    res = [x for x in res if x is not None]
    allok = all(x[0] for x in res)
    anychange = any(x[1] > 0 for x in res)
    print("-" * 78)
    print(f"C-2 {'PASS' if (allok and anychange) else 'FAIL'}  (persistent state carried "
          f"across a basis change, exact; non-trivial change exercised: {anychange})")
    sys.exit(0 if (allok and anychange) else 1)
