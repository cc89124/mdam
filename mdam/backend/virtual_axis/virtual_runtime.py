"""Step C-1: persistent VirtualRuntimeState + single-measurement engine.

|phi> is a dense vector over r VIRTUAL axes (NOT physical qubits). At a measurement
the runtime applies ONLY precompiled masks (rotation_masks, measurement_mask) to the
2^r vector -- no physical promote, no rank elimination, no symplectic work at runtime.

The masks come from an OFFLINE basis change W: a |0>-fixing CNOT network that confines
all rotation/measurement X-support to r = X-rank `pivot` axes. The remaining B-r `junk`
axes then carry only Z (Z acts as +1 on |0>), so they factor out. Because CNOT|0>=|0>,
W|0_B> = |0_B>, so the virtual block starts at |0_r> EXACTLY -- this is the initial-
state handling the pure-symplectic localization lacked (its virtual-Z axes were images
of core Paulis, not of the |0_B> stabilisers, so |0_B> did not map to |0_r>).

Correctness (proved + numerically verified in test_c1.py):
  |block_phys> = (prod_j exp(-i th_j P'_j/2)) |0_B>            (the physical 2^B block)
  W|block_phys> = (prod_j exp(-i th_j Q_j/2)) |0_B>,  Q_j = W P'_j W^dag
  Q_j has X confined to the r pivots and only-Z on junk -> junk stays |0> (Z|0>=+|0>),
  so the r-pivot reduced evolution = |phi_r>, and |block_phys> = W^dag(|phi_r> (x) |0_junk>).
W is exact (conj_cx is phase-exact, tested), so Born prob + projected state are exact.
"""
from __future__ import annotations

import numpy as np

from mdam.backend.block_magic import _apply_pauli_local, _vec_cx
from mdam.backend.virtual_axis.clifford_synth import conj_cx
from mdam.backend.virtual_axis.virtual_axis import VirtualPauliMask, _bits


# --------------------------------------------------------------------------- offline
def _confine_x(paulis_local, B):
    """`paulis_local`: list of (x,z,ph) over B local qubits. Return (pivots, cnots):
    the CNOT network (list of (ctrl,tgt)) confining every Pauli's X-support to the
    `pivots` columns (r = X-rank = GF(2) rank of the x-masks); junk columns keep only
    Z. Every CNOT has a PIVOT control and a DEPENDENT target, so it fixes |0_B> and
    never disturbs an already-confined column."""
    # column c -> bit-vector over the Paulis (which Paulis have X on qubit c)
    col = [0] * B
    for c in range(B):
        m = 0
        for i, (x, z, ph) in enumerate(paulis_local):
            if (x >> c) & 1:
                m |= 1 << i
        col[c] = m
    # GF(2) column reduction with COEFFICIENT tracking: `cm` is a bitmask over the
    # ORIGINAL column ids such that XOR_{k in cm} col[k] == the reduced vector. When a
    # column reduces to 0, cm tells which ORIGINAL PIVOT columns XOR to it (every member
    # != c is a pivot), so CNOT(pivot, c) for each zeroes col[c] EXACTLY. (Tracking only
    # the reduced-basis ids -- the earlier bug -- misses the pivots folded in during
    # reduction, e.g. it dropped CNOT(0,2) when col2 = col0 XOR col1.)
    basis = []                                  # (pivot_bit, reduced_vec, coeff_mask)
    pivots = []
    cnots = []
    for c in range(B):
        v = col[c]
        cm = 1 << c
        for (pb, bv, bcm) in basis:
            if (v >> pb) & 1:
                v ^= bv
                cm ^= bcm
        if v:                                   # independent column -> a pivot (active axis)
            pb = (v & -v).bit_length() - 1
            basis.append((pb, v, cm))
            pivots.append(c)
        else:                                   # col[c] = XOR of the pivot columns in cm
            for p in _bits(cm):                 # CNOT(ctrl=pivot p, tgt=c): col_c ^= col_p
                if p != c:
                    cnots.append((p, c))
    return pivots, cnots


def _conj_cnots(P, cnots):
    Q = P
    for (c, t) in cnots:
        Q = conj_cx(Q, c, t)
    return Q


def _to_mask(Q, pivots):
    """Restrict conjugated Pauli Q=(x,z,ph) to the pivot axes (junk x must be 0; junk z
    -- acting as +1 on |0_junk> -- is dropped). Returns a VirtualPauliMask on r axes."""
    r = len(pivots)
    x = z = 0
    for i, p in enumerate(pivots):
        if (Q[0] >> p) & 1:
            x |= 1 << i
        if (Q[1] >> p) & 1:
            z |= 1 << i
    return VirtualPauliMask(x, z, Q[2] & 3, r)


def _loc(P, posn):
    """Global Pauli (x,z,ph) -> local (over `support`) Pauli."""
    x, z, ph = P
    lx = lz = 0
    for q in _bits(x):
        lx |= 1 << posn[q]
    for q in _bits(z):
        lz |= 1 << posn[q]
    return (lx, lz, ph)


def build_basis(confine_paulis, support):
    """Offline: build a |0>-fixing CNOT basis over `support` that confines every Pauli
    in `confine_paulis` (GLOBAL (x,z,ph)) to r = X-rank pivot axes. Returns the basis
    descriptor (pivots, cnots, posn) -- any Pauli in the confined span is then expressed
    on the r axes by `express`."""
    posn = {q: i for i, q in enumerate(support)}
    B = len(support)
    locs = [_loc(P, posn) for P in confine_paulis]
    pivots, cnots = _confine_x(locs, B)
    return {"r": len(pivots), "B": B, "pivots": pivots, "cnots": cnots,
            "support": support, "posn": posn}


def express(P_global, basis):
    """Express a GLOBAL Pauli on `basis`'s r axes. Returns (VirtualPauliMask, Q) where Q
    is the conjugated local Pauli (its junk-axis X bits MUST be 0 -- i.e. P is in the
    confined span; the caller asserts this)."""
    Q = _conj_cnots(_loc(P_global, basis["posn"]), basis["cnots"])
    return _to_mask(Q, basis["pivots"]), Q


def junk_x_bits(Q, pivots):
    """X-support of conjugated Pauli Q that fell OUTSIDE the pivot axes (must be 0 for a
    sound drop). Returns the offending bitmask (0 == sound)."""
    px = 0
    for p in pivots:
        px |= 1 << p
    return Q[0] & ~px


def build_single_meas_plan(rot_paulis, theta_list, meas_pauli, support):
    """Offline plan for ONE measurement (C-1). rot_paulis / meas_pauli: GLOBAL (x,z,ph)
    Paulis (already pulled back through the frame). Returns r, the |0>-fixing basis change,
    per-rotation masks + thetas, and the measurement mask -- all the runtime needs."""
    basis = build_basis(list(rot_paulis) + [meas_pauli], support)
    rot_masks = [express(P, basis)[0] for P in rot_paulis]
    meas_mask = express(meas_pauli, basis)[0]
    return {**basis, "rot_masks": rot_masks, "thetas": list(theta_list),
            "meas_mask": meas_mask}


def change_basis(phi1, basis1, basis2):
    """Carry a persistent virtual state from basis1 to basis2 (both over the SAME support).
    Physical state = W1^dag(phi1 on pivots1, junk1=|0>); re-confine with W2, restrict to
    pivots2. Exact iff every basis1-live direction is confined by basis2 (junk2 = |0>);
    the caller guarantees this by building basis2 over a superset of basis1's generators.

    NOTE: transiently materialises the 2^B union vector -- fine for the C-2 correctness
    check (B small across two measurements). The streaming engine (C-3) avoids it by
    adding/dropping ONE axis at a time so the live vector never exceeds the active rank."""
    B = basis1["B"]
    assert basis2["B"] == B and basis2["support"] == basis1["support"]
    vfull = np.zeros(1 << B, dtype=complex)
    for a in range(1 << basis1["r"]):              # embed phi1 on basis1 pivots, junk=|0>
        idx = 0
        for i, p in enumerate(basis1["pivots"]):
            if (a >> i) & 1:
                idx |= 1 << p
        vfull[idx] = phi1[a]
    for (c, t) in reversed(basis1["cnots"]):       # W1^dag
        vfull = _vec_cx(vfull, c, t)
    for (c, t) in basis2["cnots"]:                 # W2
        vfull = _vec_cx(vfull, c, t)
    r2 = basis2["r"]
    phi2 = np.zeros(1 << r2, dtype=complex)
    junk2 = [q for q in range(B) if q not in basis2["pivots"]]
    lost = 0.0
    for full in range(1 << B):
        amp = vfull[full]
        if amp == 0:
            continue
        if any((full >> q) & 1 for q in junk2):    # amplitude on a junk axis -> would be lost
            lost += abs(amp) ** 2
            continue
        a = 0
        for i, p in enumerate(basis2["pivots"]):
            if (full >> p) & 1:
                a |= 1 << i
        phi2[a] = amp
    return phi2, lost


# --------------------------------------------------------------------------- runtime
class VirtualRuntimeState:
    """Dense |phi> over r VIRTUAL axes. The runtime touches ONLY this 2^r vector via
    precompiled masks; it never sees a physical qubit, a rank, or a symplectic step."""

    def __init__(self, r):
        self.r = r
        self.phi = np.zeros(1 << r, dtype=complex)
        self.phi[0] = 1.0                       # |0_r>  (W|0_B> = |0_B> guarantees this)
        self._axes = list(range(r))             # cached qubit list for the mask kernel

    def _apply(self, mask):
        """i^phase X^x Z^z (mask over axes 0..r-1) applied to phi -> new vector."""
        return _apply_pauli_local(self._axes, self.phi, mask.x, mask.z, mask.phase)

    def apply_rotation(self, mask, theta):
        Pv = self._apply(mask)
        self.phi = np.cos(theta / 2.0) * self.phi - 1j * np.sin(theta / 2.0) * Pv

    def measure(self, mask, u):
        """Born-sample +-1 Pauli `mask` with uniform draw u in [0,1); collapse phi.
        Returns (outcome 0/1, p0). p0 is exact -> same u gives the same record bit as
        the exact backend's p0 (verified to ~1e-12 in C-1)."""
        Pv = self._apply(mask)
        exp = float(np.real(np.vdot(self.phi, Pv)))
        p0 = min(1.0, max(0.0, 0.5 * (1.0 + exp)))
        out = 0 if u < p0 else 1
        sign = 1.0 if out == 0 else -1.0
        proj = 0.5 * (self.phi + sign * Pv)
        nrm = np.linalg.norm(proj)
        if nrm > 1e-12:
            self.phi = proj / nrm
        return out, p0


# ----------------------------------------------- verification helper (NOT runtime path)
def reconstruct_physical(phi_r, plan):
    """Map a virtual 2^r vector back to the physical 2^B block: |block> = W^dag(|phi_r>
    (x) |0_junk>). Build |phi_r> on the pivot axes with junk=|0>, then apply W^dag (the
    CNOTs in reverse). Used ONLY by the test for statevector-fidelity; the runtime never
    calls this (it would defeat the 2^r memory goal)."""
    B, pivots, r = plan["B"], plan["pivots"], plan["r"]
    vfull = np.zeros(1 << B, dtype=complex)
    for a in range(1 << r):
        idx = 0
        for i in range(r):
            if (a >> i) & 1:
                idx |= 1 << pivots[i]
        vfull[idx] = phi_r[a]
    for (c, t) in reversed(plan["cnots"]):      # W^dag = product of CNOTs in reverse
        vfull = _vec_cx(vfull, c, t)
    return vfull
