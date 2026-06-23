"""Phase-1 bounded FLOP before/after, all attribution circuits -- BOUNDED ONLY.

clifft's FLOP is NOT remeasured (it is unchanged): the stored clifft-unfused (B) numbers
from data/flop_attribution.csv are reused for context.  Only the bounded backend changed,
so only it is re-measured -- OLD (reconstructed pre-Phase-1 magic branch) vs NEW (current),
both via the budget.charge FLOP hook, same convention as flop_attribution.py.
"""
import sys, csv
sys.path.insert(0, "/home/jung/clifft-paper")
sys.setrecursionlimit(400000)
from collections import defaultdict
import numpy as np
import nearclifford_backend.backend as bk
from nearclifford_backend.simulator import pauli_commute
from nearclifford_backend.clifft_axis import budget as _bud
from nearclifford_backend.clifft_axis.bounded import compile_bounded, CliftAxisBoundedNearClifford as C

COEF = {'rot:offdiag': 12, 'rot:offdiag-scalar': 12, 'collapse:offdiag': 12,
        'rot:diag': 6, 'rot:diaghalf': 3, 'rot:diag0': 6, 'rot:diag-scalar': 6, 'collapse:diag': 6, 'collapse:diag0': 6,
        'meas': 10, 'exp': 10, 'reduce:verify': 10, 'sqnorm': 2, 'normalize': 2, 'purge:h': 5, 'purge:s': 3,
        'purge:cnot': 0, 'reduce:cnot': 0, 'drop': 0, 'promote': 0, 'reduce:gf2scan': 0,
        'init': 0, 'post-reduce': 0}
NEW_measure_z = C.measure_z


def OLD_measure_z(self, q):
    self._flush_core(0, 1 << q)
    Pm = (0, 1 << q, 0)
    magset = set(self.M)
    anti_s = [i for i in range(self.n)
              if i not in magset and not pauli_commute(self.Zc[i], Pm)]
    M_before = len(self.M)
    p0 = None
    if anti_s:
        out = self._ag_measure(Pm, anti_s); branch = "stabilizer"
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
                self.phi /= nrm2 ** 0.5
            self._compress_magic()
        branch = "magic"
    self._reduce_full()
    if len(self.M) > self.max_M:
        self.max_M = len(self.M)
    self.budget.note_resident(self.phi.size, "post-reduce")
    self._meas_log_ctr += 1
    return out


def bounded_flop(circ, old, seed=1):
    prog = compile_bounded(open(f"qec_bench/circuits/{circ}.stim").read())
    cat = defaultdict(float)
    orig = _bud.DenseMemoryBudget.charge

    def charge(self, resident, transient=0, where=""):
        N = int(resident)
        cat[where] += COEF.get(where, 0) * N
        if where.startswith('collapse'):
            cat['normalization'] += 6 * N
        return orig(self, resident, transient, where)

    _bud.DenseMemoryBudget.charge = charge
    C.measure_z = OLD_measure_z if old else NEW_measure_z
    try:
        be = bk.NearCliffordBackend(clifft_axis_bounded=True, drop_dead=False,
                                    structure_once=False, clifft_axis_enforce=True)
        be.run_shot(prog, seed)
        peak = be.nc.budget.peak_resident.bit_length() - 1
    finally:
        _bud.DenseMemoryBudget.charge = orig
        C.measure_z = NEW_measure_z
    sq = cat.get('sqnorm', 0.0)
    return sum(cat.values()), sq, peak


def H(x):
    a = abs(x)
    for u, s in ((1e9, 'G'), (1e6, 'M'), (1e3, 'k')):
        if a >= u:
            return f"{x/u:.2f}{s}"
    return f"{x:.0f}"


# stored clifft-unfused (B) FLOP -- NOT remeasured (clifft unchanged)
CLF_UNF = {"coherent_ry_d3_r1": 12.29e6, "coherent_ry_d3_r3": None, "cultivation_d3": 1.80e3,
           "cultivation_d5": 212.82e3, "coherent_rx_d3_r3": 2.59e6, "coherent_d3_r3": 52.56e3,
           "coherent_rx_d3_r1": 870.38e3, "coherent_d5_r5": 17.99e9, "distillation": 1.90e3}
CIRCS = list(CLF_UNF)

print(f"{'circuit':20}{'peak':>5}{'clifft-unf':>12}{'bnd BEFORE':>12}{'bnd AFTER':>12}"
      f"{'delta':>8}{'sqBEFORE':>10}{'sqAFTER':>10}")
print("-" * 99)
rows = []
for c in CIRCS:
    b_tot, b_sq, pk = bounded_flop(c, old=True)
    n_tot, n_sq, pk2 = bounded_flop(c, old=False)
    clf = CLF_UNF[c]
    d = (n_tot - b_tot) / b_tot * 100
    print(f"{c:20}{pk:>5}{(H(clf) if clf else '-'):>12}{H(b_tot):>12}{H(n_tot):>12}"
          f"{d:>7.0f}%{H(b_sq):>10}{H(n_sq):>10}", flush=True)
    rows.append([c, pk, clf, b_tot, n_tot, b_sq, n_sq])

with open("reports_clifft_axis_bounded/flop_accounting/data/phase1_bounded_flop.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["circuit", "peak_rank", "clifft_unfused_flop", "bounded_before_flop",
                "bounded_after_flop", "sqnorm_before_flop", "sqnorm_after_flop"])
    w.writerows(rows)
print("\n-> data/phase1_bounded_flop.csv")
