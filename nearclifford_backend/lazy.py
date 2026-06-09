"""Lazy near-Clifford simulator: defer non-Clifford rotations as PENDING Pauli
rotations and materialise only the anticommutation-connected core at each
measurement. Realises "push all Cliffords into the frame, store each RZ as a Pauli
rotation, group commuting rotations, materialise only the anticommuting core".

State form (all Cliffords pushed to the right of the rotations):

    |psi> = ( prod_j R_{L_j}(theta_j) )  U_C  ( |0>_{notM} (x) |phi>_M )

* L_j -- the rotation generators in the PHYSICAL (lab) frame: an RZ on qubit q is
  R_{Z_q}, so L_j = Z_q at application time. Subsequent Cliffords G conjugate every
  pending generator (L_j <- G L_j G^dag) -- O(1) per pending Pauli per 1-2q gate.
  Physical-frame storage is the crux: it is INVARIANT under a measurement's tableau
  relabelling (if U_C -> U_C V then U_C P U_C^dag is unchanged), so a pending
  rotation survives across measurements correctly. (Pre-frame storage is NOT
  invariant -- the projection relabels the |0> frame.)
* U_C, |phi>_M -- inherited Clifford tableau + dense magic register (NearClifford).

A measurement of physical Z_q: the generators L_j that commute with Z_q (directly,
and transitively through the anticommutation graph) commute through the projector
and stay pending; only the anticommutation-connected core of Z_q is flushed
(applied to the dense register, pulling each generator back through U_C). A qubit
born |+> with a coherent R_Z read later in Z keeps L = Z_q commuting with the Z_q
measurement -> never flushed (free). An ancilla whose R_Z is turned to X_anc by a
syndrome-extraction H anticommutes with its Z measurement -> flushed, but it is
re-measured each round so the live core stays small.

Verified against dense (scripts/verify_lazy.py) and clifft (scripts/verify_backend.py --lazy).
"""
from __future__ import annotations

import numpy as np

from nearclifford_backend.simulator import NearClifford


# --- single-Pauli Clifford conjugations P -> G P G^dag, P=(x,z,phase) ---------
def _conj_h(P, q):
    x, z, p = P; bit = 1 << q
    xq = (x >> q) & 1; zq = (z >> q) & 1
    x2 = (x & ~bit) | (zq << q)
    z2 = (z & ~bit) | (xq << q)
    return (x2, z2, (p + 2 * (xq & zq)) & 3)


def _conj_s(P, q, dag):
    x, z, p = P
    xq = (x >> q) & 1
    z2 = z ^ (xq << q)
    return (x, z2, (p + (xq * (3 if dag else 1))) & 3)


def _conj_cx(P, c, t):
    x, z, p = P; bt = 1 << t; bc = 1 << c
    xc = (x >> c) & 1; xt = (x >> t) & 1
    zc = (z >> c) & 1; zt = (z >> t) & 1
    x2 = x | (xc << t) if xc else x
    x2 = (x2 & ~bt) | (((xt ^ xc) & 1) << t)
    z2 = (z & ~bc) | (((zc ^ zt) & 1) << c)
    return (x2, z2, p)


def _commute_xz(ax, az, bx, bz):
    return (((ax & bz).bit_count() + (az & bx).bit_count()) & 1) == 0


class LazyNearClifford(NearClifford):
    def __init__(self, n):
        super().__init__(n)
        self.pending = []          # list of [x, z, phase, theta] PHYSICAL generators
        self.max_M = 0
        self.cap = None            # if set, raise MagicCapExceeded when |M| exceeds
        # resource_only: do NOT materialise the dense register; just record the core
        # SUPPORT size (the |M| a flush would need) and evolve the tableau as a
        # Gottesman-Knill stabilizer sim. Gives exact core sizes with no 2^|M| cost.
        self.resource_only = False

    # ---- Clifford gates: update tableau (super) AND conjugate pending ----
    def h(self, q):
        super().h(q)
        self.pending = [[*_conj_h((r[0], r[1], r[2]), q), r[3]] for r in self.pending]

    def s(self, q, dag=False):
        super().s(q, dag)
        self.pending = [[*_conj_s((r[0], r[1], r[2]), q, dag), r[3]] for r in self.pending]

    def cx(self, c, t):
        super().cx(c, t)
        self.pending = [[*_conj_cx((r[0], r[1], r[2]), c, t), r[3]] for r in self.pending]

    def cz(self, a, b):
        # cz = H_b CX(a,b) H_b ; reuse the composed conjugations on pending via super
        super().cz(a, b)
        new = []
        for r in self.pending:
            P = (r[0], r[1], r[2])
            P = _conj_h(P, b); P = _conj_cx(P, a, b); P = _conj_h(P, b)
            new.append([P[0], P[1], P[2], r[3]])
        self.pending = new

    # ---- defer a rotation (physical generator (x,z)) instead of applying ----
    def apply_rotation(self, x, z, theta):
        self.pending.append([x, z, 0, theta])

    # ---- apply an already-PHYSICAL generator to the dense register (flush) ----
    def _flush_one(self, x, z, theta):
        xp, zp, pp = self._pullback(x, z)        # physical -> pre-frame
        for qq in range(self.n):
            if (xp >> qq) & 1 and qq not in self.M:
                self._promote(qq)
        if len(self.M) > self.max_M:
            self.max_M = len(self.M)
        if self.cap is not None and len(self.M) > self.cap:
            from nearclifford_backend.backend import MagicCapExceeded
            raise MagicCapExceeded(-1, len(self.M))
        Pphi = self._apply_magic_pauli(xp, zp, pp)
        c = np.cos(theta / 2.0); s = np.sin(theta / 2.0)
        self.phi = c * self.phi - 1j * s * Pphi

    # ---- anticommutation-connected core of measured physical Pauli (qx,qz) ----
    def _core_indices(self, qx, qz):
        n = len(self.pending)
        in_core = [False] * n
        stack = []
        for j, r in enumerate(self.pending):
            if not _commute_xz(qx, qz, r[0], r[1]):
                in_core[j] = True; stack.append(j)
        while stack:
            j = stack.pop(); rj = self.pending[j]
            for k in range(n):
                if not in_core[k]:
                    rk = self.pending[k]
                    if not _commute_xz(rj[0], rj[1], rk[0], rk[1]):
                        in_core[k] = True; stack.append(k)
        return in_core

    def _flush_core(self, qx, qz):
        in_core = self._core_indices(qx, qz)
        if not any(in_core):
            return
        keep = []; flush = []
        for j, r in enumerate(self.pending):
            (flush if in_core[j] else keep).append(r)
        self.pending = keep
        if self.resource_only:
            # |M| this flush would need = #qubits in the union X-support of the
            # pulled-back core generators + the measured qubit's pulled-back X-support.
            supp = 0
            qxp, qzp, _ = self._pullback(qx, qz)
            supp |= qxp
            for (x, z, p, theta) in flush:
                xp, zp, pp = self._pullback(x, z)
                supp |= xp
            self.max_M = max(self.max_M, supp.bit_count())
            return
        for (x, z, p, theta) in flush:       # append order preserved
            self._flush_one(x, z, theta)

    # ---- measurement: flush core for the measured physical Pauli, then measure ----
    def measure_z(self, q):
        self._flush_core(0, 1 << q)            # measured physical Pauli = Z_q
        return super().measure_z(q)

    # ---- verification helper: realise the full state ----
    def statevector(self):
        for (x, z, p, theta) in self.pending:
            self._flush_one(x, z, theta)
        self.pending = []
        return super().statevector()

    def live_magic(self):
        return len(self.M)
