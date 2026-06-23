"""PRODUCTION FLOP / memory-traffic accounting for clifft_axis_bounded, via the VALIDATED
budget.charge() hook (coeffs cross-checked 1:1 against direct kernel-event instrumentation,
exact at r=1..6, on real circuits, and unit-called for unexercised kernels).

bounded numbers are ALGORITHMIC FLOPs under the stated convention.
clifft numbers are MODELED ('estimated'): clifft's core is a compiled extension
(_clifft_core.abi3.so) so its kernels CANNOT be instrumented; we charge each clifft-SHARED
event (rotation flush / Born / norm) at the active 2^k array (cap = 2^k_clifft) instead of
bounded's localised resident -- i.e. clifft holding the full active state, no localize-and-drop.
The assumption-light comparison is the state-volume proxy  S = sum_t 2^{r_t}.

FLOP convention: complex mult=6, add/sub=2, real*complex=2, |z|^2=4.  word = complex128 = 16 B.
Per-charged-N coeffs (validated): see VALID below.  Permutations (cnot/drop/promote) = 0 FLOP.
"""
import sys, time
sys.path.insert(0, "/home/jung/clifft-paper")
sys.setrecursionlimit(400000)
import numpy as np
import nearclifford_backend.backend as bk
from nearclifford_backend.clifft_axis import budget as _bud
from nearclifford_backend.clifft_axis.bounded import compile_bounded

# (flop, R, W) per element of the CHARGED N ; clifft-shared flag
VALID = {
    'rot:offdiag': (12, 1, 1, 1), 'rot:offdiag-scalar': (12, 1, 1, 1),
    'collapse:offdiag': (12, 1, 1, 1),
    'rot:diag': (6, 1, 1, 1), 'rot:diag0': (6, 1, 1, 1), 'rot:diag-scalar': (6, 1, 1, 1),
    'collapse:diag': (6, 1, 1, 1), 'collapse:diag0': (6, 1, 1, 1),
    'meas': (10, 1.5, 0, 1), 'exp': (10, 1.5, 0, 1), 'reduce:verify': (10, 1.5, 0, 0),
    'sqnorm': (2, 0.5, 0, 1),                       # 4*(N/2)/N flop ; reads N/2
    'purge:h': (5, 1, 1, 0), 'purge:s': (3, 0.5, 0.5, 0),
    'purge:cnot': (0, 0.5, 0.5, 0), 'reduce:cnot': (0, 0.5, 0.5, 0),
    'drop': (0, 0.5, 0.5, 0), 'promote': (0, 0, 1, 0),
    'reduce:gf2scan': (0, 1, 0, 0), 'init': (0, 0, 0, 0), 'post-reduce': (0, 0, 0, 0),
}


class Prod:
    def __init__(self):
        self.fb = 0.0; self.fc = 0.0; self.R = 0.0; self.W = 0.0
        self.kern = {}            # label -> [flop, count]
        self.inv = 0
        self._orig = None
    def enable(self):
        self._orig = _bud.DenseMemoryBudget.charge
        P = self; orig = self._orig
        def charge(self, resident, transient=0, where=""):
            f, rc, wc, shared = VALID.get(where, (0, 0, 0, 0))
            N = int(resident)
            fb = f * N
            P.fb += fb; P.R += rc * N; P.W += wc * N; P.inv += 1
            d = P.kern.setdefault(where, [0.0, 0]); d[0] += fb; d[1] += 1
            if shared:
                P.fc += f * self.cap                       # clifft at full active 2^k
            if where.startswith('collapse'):               # modeled norm+renorm (6N), uncharged
                P.fb += 6 * N; P.R += N; P.W += N
                P.fc += 6 * self.cap
                d2 = P.kern.setdefault('norm+renorm(model)', [0.0, 0]); d2[0] += 6 * N; d2[1] += 1
            return orig(self, resident, transient, where)
        _bud.DenseMemoryBudget.charge = charge
    def disable(self):
        _bud.DenseMemoryBudget.charge = self._orig


def state_volume(circ):
    """assumption-light proxy from the per-step rank trace: S=sum 2^resident (bounded) vs
    sum 2^k (clifft holding the full active array every step)."""
    import csv, os
    p = f"reports_clifft_axis_bounded/bounded_{circ}_per_step.csv"
    if not os.path.exists(p):
        return None, None
    Sb = Sc = 0
    for r in csv.DictReader(open(p)):
        Sb += 1 << int(r['bounded_resident_qubits'])
        Sc += 1 << int(r['n_active'])
    return Sb, Sc


def run(circ, seed=1):
    prog = compile_bounded(open(f"qec_bench/circuits/{circ}.stim").read())
    P = Prod(); P.enable()
    t0 = time.time()
    try:
        be = bk.NearCliffordBackend(clifft_axis_bounded=True, drop_dead=False,
                                    structure_once=False, clifft_axis_enforce=True)
        be.run_shot(prog, seed)
    finally:
        P.disable()
    dt = time.time() - t0
    Sb, Sc = state_volume(circ)
    return dict(k=prog.peak_rank, maxM=be.last_max_M, fb=P.fb, fc=P.fc, R=P.R, W=P.W,
                bytes=16 * (P.R + P.W), inv=P.inv, kern=P.kern, dt=dt, Sb=Sb, Sc=Sc,
                nm=prog.num_measurements)


def H(x):
    for u, s in ((1e12, 'T'), (1e9, 'G'), (1e6, 'M'), (1e3, 'k')):
        if abs(x) >= u: return f"{x/u:.2f}{s}"
    return f"{x:.0f}"


CIRCS = [("coherent_ry_d3_r1", "R_Y"), ("coherent_ry_d3_r3", "R_Y"),     # corrected R_Y FIRST
         ("coherent_rx_d3_r1", "R_X"), ("coherent_rx_d3_r3", "R_X"),
         ("coherent_d3_r1", "R_Z"), ("coherent_d3_r3", "R_Z"),
         ("cultivation_d3", "T"), ("distillation", "T")]
print("=== PRODUCTION FLOP / traffic : bounded (ALGORITHMIC) vs clifft 2^k (MODELED) ===")
print(f"{'circuit':18}{'ax':4}{'k':>3}{'maxM':>5}{'bnd FLOP':>11}{'clifft FLOP':>12}"
      f"{'F_cl/F_bn':>10}{'bytes':>10}{'S_bnd':>9}{'S_clifft':>10}{'S_cl/S_bn':>10}{'  ms':>7}")
for c, ax in CIRCS:
    r = run(c)
    fr = (r['fc'] / r['fb']) if r['fb'] else float('nan')
    sv = (r['Sc'] / r['Sb']) if r['Sb'] else float('nan')
    print(f"{c:18}{ax:4}{r['k']:>3}{r['maxM']:>5}{H(r['fb']):>11}{H(r['fc']):>12}"
          f"{fr:>10.1f}{H(r['bytes']):>10}{H(r['Sb']):>9}{H(r['Sc']):>10}{sv:>10.1f}{r['dt']*1e3:>7.0f}")

# per-kernel breakdown for the two corrected R_Y circuits
for c in ("coherent_ry_d3_r1", "coherent_ry_d3_r3"):
    r = run(c)
    print(f"\n=== {c}: per-kernel (bounded algorithmic FLOP) ===")
    for k in sorted(r['kern'], key=lambda x: -r['kern'][x][0]):
        f, n = r['kern'][k]
        if f or n:
            print(f"  {k:22} calls={n:>5}  FLOP={H(f):>10} ({100*f/r['fb']:4.1f}%)")
    print(f"  TOTAL bnd FLOP={H(r['fb'])}  words R={H(r['R'])} W={H(r['W'])}  bytes={H(r['bytes'])}  "
          f"invocations={r['inv']}")
