"""Phase 2A / section 1 -- post-Phase-1 FLOP attribution.

Three backends on the SAME circuit/seed/shot, ONE FLOP convention (cmul=6, rcmul=2, cadd=2,
sqmag=4, vdot=8; compile-time matrix algebra and memcpy excluded):
  1. Clifft UNFUSED   (clifft.compile(..., bytecode_passes=None), C++ CostMeter)
  2. bounded ORIGINAL (pre-Phase-1 measure_z, reconstructed inline)
  3. bounded PHASE 1  (current measure_z)
Both bounded versions and clifft are bucketed into the SAME categories from their REAL kernel
events, then the post-Phase-1 gap  dF = F(bounded P1) - F(clifft unfused)  is decomposed per
category and checked to sum EXACTLY to the measured total gap (no hidden residual).

Categories: diag_rot, offdiag_rot, array_clifford, born_sqnorm, purge, other.
"""
import sys, csv
sys.path.insert(0, "/home/jung/clifft-paper")
sys.setrecursionlimit(400000)
from collections import defaultdict
import numpy as np
import clifft
from clifft import _clifft_core as cc
import nearclifford_backend.backend as bk
from nearclifford_backend.simulator import pauli_commute
from nearclifford_backend.clifft_axis import budget as _bud
from nearclifford_backend.clifft_axis.bounded import compile_bounded, CliftAxisBoundedNearClifford as C

CONV = dict(cmul=6, rcmul=2, cadd=2, sqmag=4, vdot=8)
CATS = ["diag_rot", "offdiag_rot", "array_clifford", "born_sqnorm", "purge", "other"]
NEW_measure_z = C.measure_z

# ---- clifft kernel -> category ----
CLF_CAT = {
    'array_rot': 'diag_rot',
    'array_h': 'array_clifford', 'array_s': 'array_clifford', 'array_s_dag': 'array_clifford',
    'array_t': 'array_clifford', 'array_t_dag': 'array_clifford', 'array_cz': 'array_clifford',
    'array_multi_cz': 'array_clifford', 'array_u2': 'array_clifford', 'array_u4': 'array_clifford',
    'array_cnot': 'array_clifford', 'array_swap': 'array_clifford', 'array_multi_cnot': 'array_clifford',
    'expand': 'other', 'expand_t': 'other', 'expand_t_dag': 'other', 'expand_rot': 'other',
    'meas_diagonal': 'born_sqnorm', 'meas_interfere': 'born_sqnorm',
    'swap_meas_interfere': 'born_sqnorm', 'exp_val': 'born_sqnorm',
}


def clifft_flop(circ, seed=1):
    src = open(f"qec_bench/circuits/{circ}.stim").read()
    prog = clifft.compile(src, bytecode_passes=None)
    cc.cost_meter_reset(); cc.cost_meter_enable(True)
    clifft.sample(prog, 1, seed)
    cc.cost_meter_enable(False)
    snap = cc.cost_meter_snapshot()
    cat = defaultdict(float)
    for name, s in snap.items():
        f = sum(CONV[k] * s[k] for k in CONV)
        cat[CLF_CAT.get(name, 'other')] += f
    return cat, prog.peak_rank


# ---- bounded charge `where` -> (FLOP/word, category) ----
BCOEF = {'rot:offdiag': (12, 'offdiag_rot'), 'rot:offdiag-scalar': (12, 'offdiag_rot'),
         'collapse:offdiag': (12, 'born_sqnorm'),
         'rot:diaghalf': (3, 'diag_rot'), 'rot:diag': (6, 'diag_rot'), 'rot:diag0': (6, 'diag_rot'), 'rot:diag-scalar': (6, 'diag_rot'),
         'collapse:diag': (6, 'born_sqnorm'), 'collapse:diag0': (6, 'born_sqnorm'),
         'meas': (10, 'born_sqnorm'), 'exp': (10, 'born_sqnorm'), 'reduce:verify': (10, 'purge'),
         'sqnorm': (2, 'born_sqnorm'), 'normalize': (2, 'born_sqnorm'),
         'purge:h': (5, 'array_clifford'), 'purge:s': (3, 'array_clifford'),
         'purge:cnot': (0, 'purge'), 'reduce:cnot': (0, 'purge'), 'reduce:gf2scan': (0, 'purge'),
         'drop': (0, 'other'), 'promote': (0, 'other'), 'init': (0, 'other'), 'post-reduce': (0, 'other')}


def OLD_measure_z(self, q):
    self._flush_core(0, 1 << q)
    Pm = (0, 1 << q, 0)
    magset = set(self.M)
    anti_s = [i for i in range(self.n)
              if i not in magset and not pauli_commute(self.Zc[i], Pm)]
    p0 = None
    if anti_s:
        out = self._ag_measure(Pm, anti_s)
    else:
        xp, zp, pp = self._pullback(0, 1 << q)
        r, sign = self._localize_to_Z(xp, zp, pp, prefer=q)
        if r is None:
            p0 = max(0.0, min(1.0, (1.0 + sign) / 2.0))
            out = 0 if float(self.rng.random()) < p0 else 1
        else:
            jr = self.M.index(r)
            p0r = self._branch_sqnorm(jr, 0)
            p0 = p0r if sign > 0 else (1.0 - p0r)
            out = 0 if float(self.rng.random()) < p0 else 1
            plus_bit = 0 if sign > 0 else 1
            keepbit = plus_bit if out == 0 else (1 - plus_bit)
            v = self.phi.reshape(-1, 2, 1 << jr)
            v[:, 1 - keepbit, :] = 0.0
            nrm2 = self._sqnorm_1d(self.phi)
            if nrm2 > 1e-24:
                self.budget.charge(self.phi.size, 0, "normalize")     # complete OLD accounting too
                self.phi /= nrm2 ** 0.5
            self._compress_magic()
    self._reduce_full()
    if len(self.M) > self.max_M:
        self.max_M = len(self.M)
    self.budget.note_resident(self.phi.size, "post-reduce")
    self._meas_log_ctr += 1
    return out


# OLD's _sqnorm_1d is an UNCHARGED full sweep -> to make OLD a COMPLETE total, wrap it to charge.
_orig_sq1d = C._sqnorm_1d.__func__ if hasattr(C._sqnorm_1d, "__func__") else C._sqnorm_1d


def bounded_flop(circ, old, seed=1):
    prog = compile_bounded(open(f"qec_bench/circuits/{circ}.stim").read())
    cat = defaultdict(float)
    orig = _bud.DenseMemoryBudget.charge

    def charge(self, resident, transient=0, where=""):
        N = int(resident)
        coeff, c = BCOEF.get(where, (0, 'other'))
        cat[c] += coeff * N
        if where.startswith('collapse'):
            cat['born_sqnorm'] += 6 * N            # deferred-norm the collapse pays (clifft O(1))
        return orig(self, resident, transient, where)

    _bud.DenseMemoryBudget.charge = charge
    C.measure_z = OLD_measure_z if old else NEW_measure_z
    # for OLD: charge its _sqnorm_1d full sweep (sqmag over 2^r = 4*2^(r-1)=2*2^r) as born_sqnorm
    if old:
        def sq1d(arr):
            cat['born_sqnorm'] += 2 * int(arr.size)
            return _orig_sq1d(arr)
        C._sqnorm_1d = staticmethod(sq1d)
    try:
        be = bk.NearCliffordBackend(clifft_axis_bounded=True, drop_dead=False,
                                    structure_once=False, clifft_axis_enforce=True)
        be.run_shot(prog, seed)
        pk = be.nc.budget.peak_resident.bit_length() - 1
    finally:
        _bud.DenseMemoryBudget.charge = orig
        C.measure_z = NEW_measure_z
        C._sqnorm_1d = staticmethod(_orig_sq1d)
    return cat, pk


def H(x):
    a = abs(x)
    for u, s in ((1e9, 'G'), (1e6, 'M'), (1e3, 'k')):
        if a >= u:
            return f"{x/u:.2f}{s}"
    return f"{x:.0f}"


CIRCS = ["coherent_ry_d3_r1", "coherent_ry_d3_r3", "cultivation_d3", "cultivation_d5",
         "coherent_rx_d3_r1", "coherent_rx_d3_r3", "coherent_d3_r3", "coherent_d5_r5",
         "distillation"]

rows = []
for circ in CIRCS:
    clf, kc = clifft_flop(circ)
    bo, _ = bounded_flop(circ, old=True)
    bp, kp = bounded_flop(circ, old=False)
    tot_clf = sum(clf.values()); tot_bo = sum(bo.values()); tot_bp = sum(bp.values())
    print(f"\n{'='*100}\n{circ}   clifft k={kc} | bounded peak={kp}   "
          f"[clifft-unf {H(tot_clf)} | bnd orig {H(tot_bo)} | bnd P1 {H(tot_bp)}]")
    print(f"  {'category':16}{'clifft-unf':>13}{'bnd orig':>13}{'bnd P1':>13}"
          f"{'dF (P1-clf)':>14}")
    dF_sum = 0.0
    for c in CATS:
        a, b, d = clf.get(c, 0.0), bo.get(c, 0.0), bp.get(c, 0.0)
        dF = d - a
        dF_sum += dF
        if abs(a) + abs(b) + abs(d) < 1e-9:
            continue
        print(f"  {c:16}{H(a):>13}{H(b):>13}{H(d):>13}{H(dF):>14}")
    gap = tot_bp - tot_clf
    print(f"  {'TOTAL':16}{H(tot_clf):>13}{H(tot_bo):>13}{H(tot_bp):>13}{H(gap):>14}")
    print(f"  dF decomposition sums to {H(dF_sum)}  (measured gap {H(gap)})  "
          f"-> residual {H(dF_sum - gap):>8} {'OK' if abs(dF_sum-gap)<1 else 'MISMATCH!'}")
    # which category dominates the gap?
    by = sorted(((bp.get(c, 0) - clf.get(c, 0), c) for c in CATS), key=lambda t: -abs(t[0]))
    top = ", ".join(f"{c} {H(v)}" for v, c in by[:3] if abs(v) > 1)
    print(f"  gap driven by: {top}")
    rows.append([circ, kc, kp, tot_clf, tot_bo, tot_bp,
                 *[clf.get(c, 0) for c in CATS], *[bp.get(c, 0) for c in CATS]])

with open("reports_clifft_axis_bounded/flop_accounting/data/phase2a_attribution.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["circuit", "clifft_k", "bounded_peak", "clifft_unf_total", "bnd_orig_total",
                "bnd_p1_total"] + [f"clf_{c}" for c in CATS] + [f"bndP1_{c}" for c in CATS])
    w.writerows(rows)
print("\n-> data/phase2a_attribution.csv")
