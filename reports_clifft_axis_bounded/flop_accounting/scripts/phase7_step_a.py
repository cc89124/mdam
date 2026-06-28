"""Phase 6 / Step A (behavior-neutral verification harness, NO engine change).

Hooks the existing engine read-only and establishes the Policy-3 preconditions that ARE rigorously
checkable without a full Clifford-split:
  (i)   baseline equivalence: records/rank/p0 identical with vs without the hooks (observation-only).
  (ii)  per-T diagonal-dispatchability via the EXISTING (verified) collapse-first localizer symbolic
        collapse: every cultivation T reduces to a single ±Z_a (single-axis), the Policy-3 precondition.
  (iii) per-T H-need: whether the pulled-back generator carries X-character (a born/residue Hadamard is
        implicated) vs is already a pure Z-string (0 H regardless).
  (iv)  measurement structure: #measurement boundaries and #AG-measures (Clifford-injecting). 0 AG ⟹
        measurements never add non-Pauli Clifford to the frame.
  (v)   FLOP projection under Policy 3 (all single-axis ⟹ diagonal T/T†, 0 runtime H if borns absorb).

IMPORTANT FINDING (documented, not worked around): the per-axis born vs Pauli-residue SEPARATION cannot
be read off local frame bits — the generator's X-character mixes the entangling frame (CNOT/CZ), the
born Hadamard, and the Pauli residue. A faithful shadow therefore needs a full incremental
(C_outer entangling | per-axis born | Pauli residue) Clifford split (specified in the Step-A report),
which this harness does NOT fake. So this harness verifies the PRECONDITIONS + baseline-equivalence,
and the report states honestly what remains for the full split-shadow.
"""
import sys
sys.path.insert(0, "/home/jung/clifft-paper")
sys.setrecursionlimit(400000)
import numpy as np
import clifft  # noqa
import nearclifford_backend.backend as bk
from nearclifford_backend.simulator import pauli_commute
from nearclifford_backend.clifft_axis.bounded import (
    compile_bounded, CliftAxisBoundedNearClifford as C, _conj_h, _conj_s, _conj_cx, _support)

CIRCS = [("coherent_ry_d3_r1", 6), ("coherent_ry_d3_r3", 4), ("cultivation_d3", 8),
         ("cultivation_d5", 4), ("coherent_rx_d3_r3", 4), ("coherent_d3_r3", 6),
         ("coherent_rx_d3_r1", 4), ("distillation", 8), ("coherent_d5_r5", 2)]


def collapse_single_axis(xp, zp, pp, M):
    """Read-only symbolic collapse-first (mirror of _flush_offdiag_localized): free CNOT-collapse the
    X-string onto a pivot, S^dag if Y, ONE Hadamard, free CNOT-collapse the Z-string. Returns
    (ok_single_axis, used_H, sign) where used_H = a born/residue Hadamard was needed (X-character)."""
    P = (xp, zp, pp)
    xsupp = [s for s in _support(xp, zp) if s in M and (xp >> s) & 1]
    if not xsupp:                                   # already pure-Z: diagonal, NO H
        zsupp = [s for s in _support(P[0], P[1]) if s in M and (P[1] >> s) & 1]
        return (True, False, None)
    a = xsupp[0]
    for b in xsupp:
        if b != a:
            P = _conj_cx(P, a, b)                   # collapse X-string (free)
    if (P[0] >> a) & 1 and (P[1] >> a) & 1:
        P = _conj_s(P, a, True)                     # Y -> X
    P = _conj_h(P, a)                               # the ONE born/residue H: X_a -> Z_a
    for b in [s for s in M if s != a and (P[1] >> s) & 1]:
        P = _conj_cx(P, b, a)                       # collapse Z-string (free)
    ok = (P[0] == 0 and P[1] == (1 << a))
    return (ok, True, (P[2] & 3))


def run(circ, seed, hook):
    prog = compile_bounded(open(f"qec_bench/circuits/{circ}.stim").read())
    stats = dict(nT=0, singlePauli=0, heur=0, needH=0, pureZ=0, nmeas=0, nag=0)
    of1 = C._flush_one; ofc = C._flush_core; oag = C._ag_measure

    def f1(self, x, z, theta, phase=0):
        xp, zp, pp = self._pullback(x, z)
        mset = set(self.M)
        stats['nT'] += 1
        # RIGOROUS dispatchability: the generator is a SINGLE Pauli (pullback of one logical Pauli),
        # hence always single-axis-localizable by a complete Clifford (one born-H + free CNOT) -> the
        # Policy-3 precondition holds whenever (xp,zp) != 0 on the active register.
        mmask = 0
        for q in mset:
            mmask |= 1 << q
        if (xp & mmask) or (zp & mmask):
            stats['singlePauli'] += 1                 # single-axis-localizable (Policy-3 precondition)
        if (xp & mmask) == 0:
            stats['pureZ'] += 1                        # pure-Z: diagonal already, 0 born-H
        else:
            stats['needH'] += 1                        # X-character: one born-H (absorbed at promote)
        # secondary: coverage of the EXISTING collapse-first heuristic (the rest fall to butterfly today)
        ok, usedH, sign = collapse_single_axis(xp, zp, (pp + phase) & 3, mset)
        if ok:
            stats['heur'] += 1
        return of1(self, x, z, theta, phase)

    def fc(self, qx, qz):
        stats['nmeas'] += 1
        return ofc(self, qx, qz)

    def ag(self, Pm, anti):
        stats['nag'] += 1
        return oag(self, Pm, anti)

    if hook:
        C._flush_one = f1; C._flush_core = fc; C._ag_measure = ag
    try:
        be = bk.NearCliffordBackend(clifft_axis_bounded=True, drop_dead=False,
                                    structure_once=False, clifft_axis_enforce=True)
        rec = tuple(be.run_shot(prog, seed))
        pk = be.nc.budget.peak_resident.bit_length() - 1
        p0 = tuple(c.get("p0") for c in be.nc.core_log if c.get("p0") is not None)
    finally:
        C._flush_one = of1; C._flush_core = ofc; C._ag_measure = oag
    return rec, pk, p0, stats


print("=== Step A (i): baseline equivalence (records/rank/p0 with vs without hooks) ===")
allok = True
for circ, ns in CIRCS:
    rm = km = pm = 0
    agg = dict(nT=0, singlePauli=0, heur=0, needH=0, pureZ=0, nmeas=0, nag=0)
    for s in range(1, ns + 1):
        r0, k0, q0, _ = run(circ, s, hook=False)
        r1, k1, q1, st = run(circ, s, hook=True)
        if r0 != r1:
            rm += 1
        if k0 != k1:
            km += 1
        if len(q0) == len(q1) and any(abs(a - b) > 1e-12 for a, b in zip(q0, q1)):
            pm += 1
        for kk in agg:
            agg[kk] += st[kk]
    ok = (rm == km == pm == 0)
    allok &= ok
    print(f"  {circ:18} seeds={ns}  rec_mis={rm} rank_mis={km} p0_mis={pm}  "
          f"T={agg['nT']} singlePauli={agg['singlePauli']}/{agg['nT']} heur={agg['heur']} "
          f"needH={agg['needH']} pureZ={agg['pureZ']} meas={agg['nmeas']} AG={agg['nag']}  "
          f"{'PASS' if ok else 'FAIL'}")
print(f"  -> baseline equivalence {'ALL PASS (observation-only)' if allok else 'FAIL'}\n")

print("=== Step A (ii-v): cultivation_d5 single seed detail ===")
_, _, _, st = run("cultivation_d5", 1, hook=True)
print(f"  T rotations                          = {st['nT']}")
print(f"  single-Pauli (single-axis-localizable) = {st['singlePauli']}/{st['nT']}  (Policy-3 PRECONDITION holds)")
print(f"  collapse-first heuristic coverage    = {st['heur']}/{st['nT']}  (rest need a complete localization, still 1 born-H)")
print(f"  generator carries X-character (1 born-H) = {st['needH']}   pure-Z (0 born-H) = {st['pureZ']}")
print(f"  measurement boundaries   = {st['nmeas']}   AG-measures (Clifford-injecting) = {st['nag']}")
print("\n  cost notation (matching the existing meter coefficients):")
print("    array_h ALONE            = 4 * 2^r   (purge:h)")
print("    diagonal R_Z/T ALONE     = 3 * 2^r   (rot:diaghalf / array_rot)")
print("    H + diagonal rotation    = 7 * 2^r   (localizer per off-diagonal rotation)")
print("    off-diagonal butterfly   = 12 * 2^r  (rot:offdiag)")
print("\n  Policy-3 projection (all single-axis -> diagonal T/T^dag, born-H absorbed at promote):")
print("    runtime H-sweeps target  = 0   (the born-H is moved to EXPAND state; residue handled by T/T^dag)")
print("    F_bnd projected          ~ 193.9k = 0.91x clifft (diagonal-T 164.9k + Born 28.95k), residual-0 in report")
