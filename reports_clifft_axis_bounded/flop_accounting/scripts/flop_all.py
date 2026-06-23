"""Full-benchmark FLOP / traffic table via the VALIDATED budget.charge hook.
bounded = ALGORITHMIC FLOPs (stated convention).  clifft = MODELED (compiled .so, charged at 2^k).
Off-axis d5 (R_X/R_Y, k=38/47 > 2^26) are reported INFEASIBLE (not run)."""
import sys, time, math, csv, os
sys.path.insert(0, "/home/jung/clifft-paper")
sys.setrecursionlimit(400000)
import nearclifford_backend.backend as bk
from nearclifford_backend.clifft_axis import budget as _bud
from nearclifford_backend.clifft_axis.bounded import compile_bounded

VALID = {  # (flop, R, W, clifft-shared) per CHARGED element
    'rot:offdiag': (12, 1, 1, 1), 'rot:offdiag-scalar': (12, 1, 1, 1), 'collapse:offdiag': (12, 1, 1, 1),
    'rot:diag': (6, 1, 1, 1), 'rot:diag0': (6, 1, 1, 1), 'rot:diag-scalar': (6, 1, 1, 1),
    'collapse:diag': (6, 1, 1, 1), 'collapse:diag0': (6, 1, 1, 1),
    'meas': (10, 1.5, 0, 1), 'exp': (10, 1.5, 0, 1), 'reduce:verify': (10, 1.5, 0, 0),
    'sqnorm': (2, 0.5, 0, 1), 'purge:h': (5, 1, 1, 0), 'purge:s': (3, 0.5, 0.5, 0),
    'purge:cnot': (0, 0.5, 0.5, 0), 'reduce:cnot': (0, 0.5, 0.5, 0),
    'drop': (0, 0.5, 0.5, 0), 'promote': (0, 0, 1, 0),
    'reduce:gf2scan': (0, 1, 0, 0), 'init': (0, 0, 0, 0), 'post-reduce': (0, 0, 0, 0),
}


class Prod:
    def __init__(self):
        self.fb = self.fc = self.R = self.W = 0.0; self.kern = {}; self.inv = 0; self._o = None
    def enable(self):
        self._o = _bud.DenseMemoryBudget.charge; P = self; orig = self._o
        def charge(self, resident, transient=0, where=""):
            f, rc, wc, sh = VALID.get(where, (0, 0, 0, 0)); N = int(resident)
            P.fb += f * N; P.R += rc * N; P.W += wc * N; P.inv += 1
            d = P.kern.setdefault(where, [0.0, 0]); d[0] += f * N; d[1] += 1
            if sh: P.fc += f * self.cap
            if where.startswith('collapse'):
                P.fb += 6 * N; P.R += N; P.W += N; P.fc += 6 * self.cap
            return orig(self, resident, transient, where)
        _bud.DenseMemoryBudget.charge = charge
    def disable(self): _bud.DenseMemoryBudget.charge = self._o


def axis_of(c):
    if 'rx' in c: return 'R_X'
    if 'ry' in c: return 'R_Y'
    if c.startswith('cultivation') or c.startswith('distill'): return 'T'
    return 'R_Z'


def state_volume(circ):
    p = f"reports_clifft_axis_bounded/bounded_{circ}_per_step.csv"
    if not os.path.exists(p): return None, None
    Sb = Sc = 0
    for r in csv.DictReader(open(p)):
        Sb += 1 << int(r['bounded_resident_qubits']); Sc += 1 << int(r['n_active'])
    return Sb, Sc


def run(circ, seed=1, kmax=26):
    prog = compile_bounded(open(f"qec_bench/circuits/{circ}.stim").read())
    if prog.peak_rank > kmax:
        return dict(circ=circ, k=prog.peak_rank, infeasible=True)
    P = Prod(); P.enable(); t0 = time.time()
    try:
        be = bk.NearCliffordBackend(clifft_axis_bounded=True, drop_dead=False,
                                    structure_once=False, clifft_axis_enforce=True)
        be.run_shot(prog, seed)
    finally:
        P.disable()
    dt = time.time() - t0; Sb, Sc = state_volume(circ)
    return dict(circ=circ, k=prog.peak_rank, maxM=be.last_max_M, fb=P.fb, fc=P.fc,
                R=P.R, W=P.W, bytes=16 * (P.R + P.W), inv=P.inv, kern=P.kern, dt=dt,
                Sb=Sb, Sc=Sc, nm=prog.num_measurements, infeasible=False)


def H(x):
    if x is None: return "-"
    for u, s in ((1e12, 'T'), (1e9, 'G'), (1e6, 'M'), (1e3, 'k')):
        if abs(x) >= u: return f"{x/u:.2f}{s}"
    return f"{x:.0f}"


CIRCS = ["coherent_d3_r1", "coherent_d3_r3", "coherent_d5_r1", "coherent_d5_r5",
         "coherent_d7_r1", "coherent_d7_r7", "surface_d7_r7",
         "coherent_rx_d3_r1", "coherent_rx_d3_r3", "coherent_rx_d5_r1", "coherent_rx_d5_r5",
         "coherent_ry_d3_r1", "coherent_ry_d3_r3", "coherent_ry_d5_r1", "coherent_ry_d5_r5",
         "cultivation_d3", "cultivation_d5", "distillation"]

rows = []
print(f"{'circuit':18}{'ax':4}{'k':>3}{'maxM':>5}{'bnd FLOP':>11}{'clifft FLOP':>12}"
      f"{'F_cl/F_bn':>10}{'R words':>9}{'W words':>9}{'bytes':>9}{'S_cl/S_bn':>10}{'inv':>6}{'ms':>7}")
for c in CIRCS:
    try:
        r = run(c)
    except Exception as e:
        print(f"{c:18}{axis_of(c):4}  ERROR {type(e).__name__}: {str(e)[:40]}"); continue
    ax = axis_of(c)
    if r.get('infeasible'):
        why = "off-axis magic unbounded" if ax in ("R_X", "R_Y") else "magic accumulates over rounds"
        print(f"{c:18}{ax:4}{r['k']:>3}     INFEASIBLE (2^{r['k']} > 2^26): {why}")
        rows.append([c, ax, r['k'], 'INFEASIBLE', '', '', '', '', '', '', '', '', '']); continue
    fr = (r['fc'] / r['fb']) if r['fb'] else float('nan')
    sv = (r['Sc'] / r['Sb']) if r['Sb'] else float('nan')
    print(f"{c:18}{ax:4}{r['k']:>3}{r['maxM']:>5}{H(r['fb']):>11}{H(r['fc']):>12}"
          f"{fr:>10.1f}{H(r['R']):>9}{H(r['W']):>9}{H(r['bytes']):>9}{sv:>10.1f}{r['inv']:>6}{r['dt']*1e3:>7.0f}")
    rows.append([c, ax, r['k'], r['maxM'], int(r['fb']), int(r['fc']),
                 f"{fr:.2f}" if r['fb'] else '', int(r['R']), int(r['W']), int(r['bytes']),
                 f"{sv:.2f}" if r['Sb'] else '', r['inv'], f"{r['dt']*1e3:.0f}"])

with open("reports_clifft_axis_bounded/flop_accounting/data/flop_all.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["circuit", "axis", "k_clifft", "max_M", "bounded_FLOP", "clifft_FLOP_modeled",
                "F_cl_over_F_bn", "R_words", "W_words", "bytes", "S_cl_over_S_bn", "invocations", "ms"])
    w.writerows(rows)
print("\n-> reports_clifft_axis_bounded/flop_accounting/data/flop_all.csv")
