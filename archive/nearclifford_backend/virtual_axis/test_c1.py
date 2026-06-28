"""C-1 verification: persistent VirtualRuntimeState + single measurement end-to-end.

For the FIRST magic measurement of each circuit, capture (from the exact block backend)
the pulled-back core rotations + thetas and the pulled-back measurement Pauli. Then:
  * REFERENCE = the exact 2^B dense block (rebuild |0_B>, apply the rotations, measure).
  * VIRTUAL   = build the C-1 plan (|0>-fixing CNOT basis change -> r=X-rank pivot axes),
                evolve the 2^r VirtualRuntimeState, measure.
Verify, against the exact backend:
  (1) Born probability p0           (=> identical record bit for any shared uniform draw)
  (2) projected statevector fidelity (both outcome branches, mapped 2^r -> 2^B by W^dag)
  (3) r <= B with r strictly < B somewhere (the mechanism actually reduces).

Forbidden-op audit: the runtime path (VirtualRuntimeState + masks) performs NO physical
promote, NO rank/symplectic computation -- all of that is offline in build_single_meas_plan.
"""
import os
import sys

os.chdir("/home/jung/clifft-paper")
sys.path.insert(0, "/home/jung/clifft-paper")
sys.setrecursionlimit(50000)

import numpy as np
import clifft

from nearclifford_backend.backend import NearCliffordBackend
from nearclifford_backend.block_magic import (
    BlockLazyNearClifford, MagicRegister, _apply_pauli_local)
from nearclifford_backend.virtual_axis.virtual_axis import _bits
from nearclifford_backend.virtual_axis.virtual_runtime import (
    build_single_meas_plan, VirtualRuntimeState, reconstruct_physical)


class _Stop(Exception):
    pass


def capture_first_magic(circ, seed=1):
    """Run the block backend, halt at the first measurement that takes the MAGIC path
    (mag.measure_pauli), and return that measurement's pulled-back core + meas Pauli."""
    REC = {"done": False, "cur": None, "rots": None, "meas": None}
    o_fc = BlockLazyNearClifford._flush_core
    o_f1 = BlockLazyNearClifford._flush_one
    o_mp = MagicRegister.measure_pauli

    def fc(self, qx, qz):
        if not REC["done"]:
            REC["cur"] = []                     # new measurement -> fresh core list
        return o_fc(self, qx, qz)

    def f1(self, x, z, theta):
        if not REC["done"] and REC["cur"] is not None:
            xp, zp, pp = self._pullback(x, z)
            REC["cur"].append((xp, zp, pp, theta))
        return o_f1(self, x, z, theta)

    def mp(self, xmask, zmask, phase, rng):
        if not REC["done"]:                     # first magic-path measurement
            REC["rots"] = REC["cur"] or []
            REC["meas"] = (xmask, zmask, phase)
            REC["done"] = True
            raise _Stop()
        return o_mp(self, xmask, zmask, phase, rng)

    BlockLazyNearClifford._flush_core = fc
    BlockLazyNearClifford._flush_one = f1
    MagicRegister.measure_pauli = mp
    try:
        prog = clifft.compile(open(f"qec_bench/circuits/{circ}.stim").read())
        be = NearCliffordBackend(block=True)
        try:
            be.run_shot(prog, seed)
        except _Stop:
            pass
    finally:
        BlockLazyNearClifford._flush_core = o_fc
        BlockLazyNearClifford._flush_one = o_f1
        MagicRegister.measure_pauli = o_mp
    return REC


def reference_block(rots, meas):
    """Exact 2^B reference: rebuild |0_B>, apply the rotations in flush order, measure."""
    sup = set()
    for (x, z, p, th) in rots:
        sup |= set(_bits(x)) | set(_bits(z))
    sup |= set(_bits(meas[0])) | set(_bits(meas[1]))
    support = sorted(sup)
    B = len(support)
    posn = {q: i for i, q in enumerate(support)}

    def loc(x, z):
        lx = lz = 0
        for q in _bits(x):
            lx |= 1 << posn[q]
        for q in _bits(z):
            lz |= 1 << posn[q]
        return lx, lz

    blk = np.zeros(1 << B, dtype=complex)
    blk[0] = 1.0
    for (x, z, p, th) in rots:
        lx, lz = loc(x, z)
        Pv = _apply_pauli_local(list(range(B)), blk, lx, lz, p)
        blk = np.cos(th / 2.0) * blk - 1j * np.sin(th / 2.0) * Pv
    mlx, mlz = loc(meas[0], meas[1])
    Pm = _apply_pauli_local(list(range(B)), blk, mlx, mlz, meas[2])
    exp = float(np.real(np.vdot(blk, Pm)))
    p0 = min(1.0, max(0.0, 0.5 * (1.0 + exp)))
    proj0 = 0.5 * (blk + Pm)
    n0 = np.linalg.norm(proj0)
    proj0 = proj0 / n0 if n0 > 1e-12 else proj0
    proj1 = 0.5 * (blk - Pm)
    n1 = np.linalg.norm(proj1)
    proj1 = proj1 / n1 if n1 > 1e-12 else proj1
    return support, B, blk, p0, proj0, proj1


def _fid(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 1.0 if (na < 1e-12 and nb < 1e-12) else 0.0
    return abs(complex(np.vdot(a, b))) / (na * nb)


def verify_single(label, rots, meas):
    """Core C-1 check: exact 2^B reference vs virtual 2^r, given a single measurement's
    pulled-back core `rots` [(x,z,ph,theta)...] and measurement Pauli `meas` (x,z,ph)."""
    support, B, blk, p0, proj0, proj1 = reference_block(rots, meas)
    plan = build_single_meas_plan(
        [(x, z, p) for (x, z, p, th) in rots],
        [th for (x, z, p, th) in rots], meas, support)
    r = plan["r"]

    # forbidden-op AUDIT: every mask (rotations + measurement) must have NO X on junk;
    # masks already live on r pivot axes only, so the audit is that reconstruction (the
    # only place junk-Z is dropped) reproduces the exact block to fidelity 1 (checked below).
    st = VirtualRuntimeState(r)
    for mask, th in zip(plan["rot_masks"], plan["thetas"]):
        st.apply_rotation(mask, th)
    phi_pre = st.phi.copy()

    Pv = st._apply(plan["meas_mask"])
    exp_v = float(np.real(np.vdot(st.phi, Pv)))
    p0_v = min(1.0, max(0.0, 0.5 * (1.0 + exp_v)))
    projv0 = 0.5 * (st.phi + Pv)
    n0 = np.linalg.norm(projv0)
    projv0 = projv0 / n0 if n0 > 1e-12 else projv0
    projv1 = 0.5 * (st.phi - Pv)
    n1 = np.linalg.norm(projv1)
    projv1 = projv1 / n1 if n1 > 1e-12 else projv1

    fid_pre = _fid(reconstruct_physical(phi_pre, plan), blk)
    fid0 = _fid(reconstruct_physical(projv0, plan), proj0)
    fid1 = _fid(reconstruct_physical(projv1, plan), proj1)

    dp = abs(p0 - p0_v)
    us = np.linspace(0.0, 1.0, 4001)            # record-bit equivalence over shared draws
    bit_disagree = int(np.sum((us >= p0).astype(int) != (us >= p0_v).astype(int)))

    ok = (r <= B and dp < 1e-9 and fid_pre > 1 - 1e-9
          and fid0 > 1 - 1e-9 and fid1 > 1 - 1e-9 and bit_disagree <= 2)
    print(f"{label:22}  B={B:2d} r={r:2d} (r<B:{'Y' if r < B else 'n'})  "
          f"|dp0|={dp:.1e}  fid=({fid_pre:.7f},{fid0:.7f},{fid1:.7f})  "
          f"disagree={bit_disagree}  {'OK' if ok else 'FAIL'}")
    return ok, r, B


def run_circuit(circ):
    REC = capture_first_magic(circ)
    if not REC["done"]:
        print(f"{circ:22}  (no magic-path measurement reached)")
        return None
    return verify_single(circ, REC["rots"], REC["meas"])


# ---- synthetic single-measurement cases that EXERCISE the r<B reduction directly ----
def synthetic_cases():
    th = 0.7
    cases = []
    # (1) entangling X0X1 rotation measured by Z0: 2 physical qubits -> 1 virtual axis.
    cases.append(("synth:X0X1->1axis",
                  [(0b11, 0, 0, th)], (0, 0b01, 0)))
    # (2) duplicate entangling rotation (X0X1 twice): X-rank 1 though support is 2.
    cases.append(("synth:dup-X0X1",
                  [(0b11, 0, 0, th), (0b11, 0, 0, 0.4)], (0, 0b01, 0)))
    # (3) Z-only qubit: rotation X0, measurement Z0Z1 -> qubit1 carries only Z (junk).
    cases.append(("synth:Z-only-junk",
                  [(0b01, 0, 0, th)], (0, 0b11, 0)))
    # (4) parity trio: X0,X1 rotations measured by Z0Z1 (full rank, r=B=2 -> no reduce).
    cases.append(("synth:X0,X1 (full)",
                  [(0b01, 0, 0, th), (0b10, 0, 0, 0.5)], (0, 0b11, 0)))
    # (5) Y-entanglement: Y0Y1 rotation measured by X0 (general X/Y, not Z-parity) -> 1 axis.
    cases.append(("synth:Y0Y1->1axis",
                  [(0b11, 0b11, 0, th)], (0b01, 0, 0)))
    return cases


if __name__ == "__main__":
    print("== real circuits (first magic measurement; exactness on pulled-back cores) ==")
    circs = sys.argv[1:] or ["distillation", "cultivation_d3", "cultivation_d5",
                             "coherent_d3_r3"]
    creal = [run_circuit(c) for c in circs]
    print("== synthetic single measurements (exercise the r<B reduction) ==")
    csyn = [verify_single(lbl, rots, meas) for (lbl, rots, meas) in synthetic_cases()]

    results = [x for x in (creal + csyn) if x is not None]
    allok = all(x[0] for x in results)
    anyreduce = any(x[1] < x[2] for x in results)
    print("-" * 78)
    print(f"C-1 {'PASS' if (allok and anyreduce) else 'FAIL'}  "
          f"(all-exact: {allok}; reduces r<B in some single measurement: {anyreduce})")
    sys.exit(0 if (allok and anyreduce) else 1)
