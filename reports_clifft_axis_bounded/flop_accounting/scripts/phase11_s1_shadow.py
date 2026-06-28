"""Step C / S1 -- behavior-neutral shadow factorization U_C = C_sym · P_res · B.

NO engine change, NO dispatch, NO authoritative frame swap. The real engine (a05843e butterfly path)
is the source of truth; the shadow only READS U_C and maintains born-basis metadata to test whether the
current full frame factors as C_sym (Z-preserving on the active axes) · P_res (per-axis Pauli) · B
(per-axis born basis). Reconstruction is automatic by construction (C_sym·P_res ≔ U_C·B⁻¹), so the
meaningful test is the Z-preservation / diagonalizability of the T generators.

Canonical decomposition rules (§2):
  * active-axis order = self.M (promotion order); dormant order = qubit index.
  * born-basis pivot rule: at the FIRST promote of a qubit q, record B_q from the triggering pullback's
    character on q: pure-X→H (born-X), Y→S†H (born-Y), pure-Z→I (born-Z). (Re-promote re-records.)
  * P_res phase convention: i-power on the product, axes in M order.
  * C_sym holds CNOT/CZ/S/SWAP (Z-preserving); B holds the per-axis Hadamard/born; P_res the Pauli.
  * global phase kept in γ (the array's, unchanged here).
  * generator of a rotation/measurement on lab q (through C_sym) = B·(U_C†·Z_q·U_C)·B† up to P_res sign;
    Z-only on A  ⟺  diagonal T.  This is the §3 (pullback-direction) Z-preservation check.
"""
import sys; sys.path.insert(0, "/home/jung/clifft-paper"); sys.setrecursionlimit(400000)
import clifft  # noqa
import nearclifford_backend.backend as bk
from nearclifford_backend.clifft_axis.bounded import compile_bounded, CliftAxisBoundedNearClifford as C

CIRCS = [("coherent_ry_d3_r1", 6), ("coherent_ry_d3_r3", 4), ("cultivation_d3", 8),
         ("cultivation_d5", 4), ("coherent_rx_d3_r3", 4), ("coherent_d3_r3", 6),
         ("coherent_rx_d3_r1", 4), ("distillation", 8), ("coherent_d5_r5", 2)]


def born_apply_xfree(xp, zp, M, born):
    """Return (#X-bits remaining on A after born basis, list of offending axes). 0 X-bits ⟺ Z-only."""
    nx = 0; bad = []
    for a in M:
        xa = (xp >> a) & 1; za = (zp >> a) & 1
        b = born.get(a, 'X')
        newx = za if b == 'X' else (xa if b == 'Z' else (xa ^ za))   # born-Y: Y->Z (X,Z both -> X-bit)
        if newx:
            nx += 1; bad.append((a, b, xa, za))
    return nx, bad


def run_shadow(circ, seed, trace=False):
    prog = compile_bounded(open(f"qec_bench/circuits/{circ}.stim").read())
    born = {}; of1 = C._flush_one; omask = C._masks; oag = C._ag_measure
    st = dict(nT=0, diag=0, ndiag=0, ag=0, reprom=0, rows=[], bnd=[])

    def masks(self, xp, zp, promote, where):
        if promote:
            for qq in range(self.n):
                if (xp >> qq) & 1 and qq not in self.M:
                    b = 'Y' if ((zp >> qq) & 1) else 'X'
                    if qq in born:
                        st['reprom'] += 1
                    born[qq] = b
        return omask(self, xp, zp, promote, where)

    def f1(self, x, z, theta, phase=0):
        labq = [q for q in range(self.n) if (z >> q) & 1 or (x >> q) & 1]
        r = of1(self, x, z, theta, phase)
        xp, zp, pp = self._pullback(x, z)
        nx, bad = born_apply_xfree(xp, zp, self.M, born)
        st['nT'] += 1
        if nx == 0:
            st['diag'] += 1
        else:
            st['ndiag'] += 1
        if trace:
            # T/T† prediction: sign of the collapsed Z generator (after born) -> via pp parity proxy
            st['rows'].append(dict(idx=st['nT'] - 1, rank=self.phi.size.bit_length() - 1, labq=labq,
                                   theta=round(float(theta), 3), zonly=(nx == 0), bad=bad,
                                   weight=int(xp | zp).bit_count(), pp=pp))
        return r

    def ag(self, Pm, anti):
        st['ag'] += 1
        return oag(self, Pm, anti)

    C._flush_one = f1; C._masks = masks; C._ag_measure = ag
    try:
        be = bk.NearCliffordBackend(clifft_axis_bounded=True, drop_dead=False, structure_once=False,
                                    clifft_axis_enforce=True)
        rec = tuple(be.run_shot(prog, seed))
        pk = be.nc.budget.peak_resident.bit_length() - 1
        p0 = tuple(c.get("p0") for c in be.nc.core_log if c.get("p0") is not None)
    finally:
        C._flush_one = of1; C._masks = omask; C._ag_measure = oag
    return rec, pk, p0, st, born


def run_plain(circ, seed):
    prog = compile_bounded(open(f"qec_bench/circuits/{circ}.stim").read())
    be = bk.NearCliffordBackend(clifft_axis_bounded=True, drop_dead=False, structure_once=False,
                                clifft_axis_enforce=True)
    rec = tuple(be.run_shot(prog, seed))
    pk = be.nc.budget.peak_resident.bit_length() - 1
    p0 = tuple(c.get("p0") for c in be.nc.core_log if c.get("p0") is not None)
    return rec, pk, p0


print("=" * 96)
print("STEP C / S1 -- shadow factorization U_C = C_sym·P_res·B  (behaviour-neutral, born-on-U_C)")
print("=" * 96)
print(f"{'circuit':18}{'seeds':>5}{'diag/T':>10}{'AG':>5}{'reprom':>7}  {'shadow==plain (rec/rank/p0)':>26}  class")
for circ, ns in CIRCS:
    td = tt = tag = trp = 0; eqfail = 0
    for s in range(1, ns + 1):
        rec0, pk0, p00 = run_plain(circ, s)
        rec1, pk1, p01, st, born = run_shadow(circ, s)
        if rec0 != rec1 or pk0 != pk1 or p00 != p01:
            eqfail += 1
        td += st['diag']; tt += st['nT']; tag += st['ag']; trp += st['reprom']
    frac = f"{td}/{tt}"
    if tt and td == tt and tag == 0:
        cls = "PARITY-candidate"
    elif "rx" in circ or "ry" in circ:
        cls = "RX/RY fallback"
    elif tag > 0:
        cls = "AG-heavy (cond.)"
    else:
        cls = f"near-parity {100*td//max(tt,1)}%"
    print(f"{circ:18}{ns:>5}{frac:>10}{tag:>5}{trp:>7}  {'IDENTICAL' if eqfail==0 else f'{eqfail} FAIL':>26}  {cls}")

# cultivation detail + boundaries
print("\n" + "-" * 96)
print("cultivation_d5 seed 1 -- per-T diagonalizability + the residual structure")
rec, pk, p0, st, born = run_shadow("cultivation_d5", 1, trace=True)
nd = [r for r in st['rows'] if not r['zonly']]
print(f"  T diagonal (Z-only generator) = {st['diag']}/{st['nT']}   born census = "
      f"{ {b: sum(1 for v in born.values() if v==b) for b in set(born.values())} }   AG measures = {st['ag']}")
print(f"  non-diagonal T's: {[(r['idx'], r['labq'], r['theta'], r['bad']) for r in nd]}")
print(f"  weight distribution of diagonal T's: ", end="")
wd = {}
for r in st['rows']:
    if r['zonly']:
        wd[r['weight']] = wd.get(r['weight'], 0) + 1
print(dict(sorted(wd.items())))
