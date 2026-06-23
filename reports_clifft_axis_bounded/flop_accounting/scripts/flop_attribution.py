"""Quantitative FLOP attribution: WHY does bounded use less peak memory than Clifft yet MORE FLOP?

Compares THREE runs on the same circuit/seed/shot under ONE FLOP convention (cmul=6, rcmul=2,
cadd=2, sqmag=4, vdot=8 -- compile-time matrix algebra and memcpy excluded):
  A. Clifft fused   (native)
  B. Clifft UNFUSED (bytecode_passes=None)  <- the architecture-fair baseline (fusion ruled out)
  C. bounded        (compile_bounded, unfused)
  D. bounded counterfactual: off-diagonal rotations charged at the diagonal rate (isolates the
     off-diagonal penalty).  Accounting-only, not a real implementation.

Both backends are bucketed into the SAME categories and each category is decomposed as
  FLOP = calls x (mean amplitudes/call) x (FLOP/amplitude).
Outputs: per-circuit category table, counterfactual D, W1=sum 2^r, event counts, rank histograms.
"""
import sys, csv, math
sys.path.insert(0, "/home/jung/clifft-paper")
sys.setrecursionlimit(400000)
from collections import defaultdict
import numpy as np
import clifft
from clifft import _clifft_core as cc
import nearclifford_backend.backend as bk
from nearclifford_backend.clifft_axis import budget as _bud
from nearclifford_backend.clifft_axis.bounded import compile_bounded

CONV = dict(cmul=6, rcmul=2, cadd=2, sqmag=4, vdot=8)

# unified category order
CATS = ["diag_rot", "offdiag_rot", "array_gate", "promote", "born", "projection",
        "sqnorm", "normalization", "purge", "swap_copy", "other"]

# ---------------- Clifft side (CostMeter primitives) ----------------
def clifft_run(circ, fused, seed=1):
    src = open(f"qec_bench/circuits/{circ}.stim").read()
    prog = clifft.compile(src) if fused else clifft.compile(src, bytecode_passes=None)
    cc.cost_meter_reset(); cc.cost_meter_enable(True)
    clifft.sample(prog, 1, seed)
    cc.cost_meter_enable(False)
    snap = {k: dict(v) for k, v in cc.cost_meter_snapshot().items()}
    hist = {int(k): dict(v) for k, v in cc.cost_meter_rank_histogram().items()}
    cat = {c: [0, 0, 0.0] for c in CATS}          # calls, sum_pow2k, flop
    def add(c, calls, S, f):
        cat[c][0] += calls; cat[c][1] += S; cat[c][2] += f
    for name, s in snap.items():
        S = s['sum_pow2k']; calls = s['calls']
        tot = sum(CONV[k] * s[k] for k in CONV)
        born = 4 * s['sqmag']; proj = 2 * s['cadd'] + 2 * s['rcmul']
        if name == 'array_rot':
            add('diag_rot', calls, S, tot)
        elif name in ('array_h', 'array_s', 'array_s_dag', 'array_t', 'array_t_dag',
                      'array_cz', 'array_multi_cz', 'array_u2', 'array_u4'):
            add('array_gate', calls, S, tot)
        elif name in ('expand', 'expand_t', 'expand_t_dag', 'expand_rot'):
            add('promote', calls, S, tot)
        elif name in ('meas_diagonal', 'meas_interfere', 'swap_meas_interfere'):
            add('born', calls, S, born); add('projection', 0, 0, proj)
        elif name == 'exp_val':
            add('born', calls, S, tot)
        elif name in ('array_cnot', 'array_swap', 'array_multi_cnot'):
            add('swap_copy', calls, S, tot)
        else:
            add('other', calls, S, tot)
    total = sum(c[2] for c in cat.values())
    W1 = sum(c[1] for c in cat.values())
    nev = sum(c[0] for c in cat.values())
    return dict(cat=cat, total=total, W1=W1, nev=nev, k=prog.peak_rank, hist=hist)

# ---------------- bounded side (budget.charge hook) ----------------
COEF = {'rot:offdiag': 12, 'rot:offdiag-scalar': 12, 'collapse:offdiag': 12,
        'rot:diag': 6, 'rot:diag0': 6, 'rot:diag-scalar': 6, 'collapse:diag': 6, 'collapse:diag0': 6,
        'meas': 10, 'exp': 10, 'reduce:verify': 10, 'sqnorm': 2, 'normalize': 2, 'purge:h': 5, 'purge:s': 3,
        'purge:cnot': 0, 'reduce:cnot': 0, 'drop': 0, 'promote': 0, 'reduce:gf2scan': 0,
        'init': 0, 'post-reduce': 0}
CAT_B = {'rot:offdiag': 'offdiag_rot', 'rot:offdiag-scalar': 'offdiag_rot',
         'rot:diag': 'diag_rot', 'rot:diag0': 'diag_rot', 'rot:diag-scalar': 'diag_rot',
         'collapse:offdiag': 'projection', 'collapse:diag': 'projection', 'collapse:diag0': 'projection',
         'meas': 'born', 'exp': 'born', 'reduce:verify': 'purge', 'sqnorm': 'sqnorm',
         'purge:h': 'purge', 'purge:s': 'purge', 'purge:cnot': 'purge',
         'reduce:cnot': 'purge', 'reduce:gf2scan': 'purge', 'drop': 'swap_copy', 'promote': 'promote',
         'init': 'other', 'post-reduce': 'other'}

def bounded_run(circ, seed=1):
    prog = compile_bounded(open(f"qec_bench/circuits/{circ}.stim").read())
    cat = {c: [0, 0, 0.0] for c in CATS}
    hist = defaultdict(lambda: [0, 0, 0.0])       # rank -> [events, sum_pow2k, flop]
    orig = _bud.DenseMemoryBudget.charge
    def charge(self, resident, transient=0, where=""):
        N = int(resident); coeff = COEF.get(where, 0); ct = CAT_B.get(where, 'other')
        f = coeff * N
        cat[ct][0] += 1; cat[ct][1] += N; cat[ct][2] += f
        if where.startswith('collapse'):          # modeled deferred-norm bounded pays, Clifft O(1)
            cat['normalization'][0] += 1; cat['normalization'][1] += N; cat['normalization'][2] += 6 * N
            f += 6 * N
        r = int(round(math.log2(N))) if N >= 1 else 0
        h = hist[r]; h[0] += 1; h[1] += N; h[2] += f
        return orig(self, resident, transient, where)
    _bud.DenseMemoryBudget.charge = charge
    try:
        be = bk.NearCliffordBackend(clifft_axis_bounded=True, drop_dead=False,
                                    structure_once=False, clifft_axis_enforce=True)
        be.run_shot(prog, seed)
    finally:
        _bud.DenseMemoryBudget.charge = orig
    total = sum(c[2] for c in cat.values())
    W1 = sum(c[1] for c in cat.values())
    nev = sum(c[0] for c in cat.values())
    return dict(cat=cat, total=total, W1=W1, nev=nev, k=prog.peak_rank,
                hist={r: list(v) for r, v in sorted(hist.items())})

def H(x):
    if x is None: return "-"
    a = abs(x)
    for u, s in ((1e9, 'G'), (1e6, 'M'), (1e3, 'k')):
        if a >= u: return f"{x/u:.2f}{s}"
    return f"{x:.0f}"

CIRCS = [("coherent_ry_d3_r1", 16), ("coherent_ry_d3_r3", 16), ("cultivation_d3", 4),
         ("cultivation_d5", 10), ("coherent_rx_d3_r3", 14), ("coherent_d3_r3", 8),
         ("coherent_rx_d3_r1", 14), ("coherent_d5_r5", 24), ("distillation", 5)]

allrows = []
for circ, kc in CIRCS:
    A = clifft_run(circ, fused=True)
    B = clifft_run(circ, fused=False)     # architecture-fair baseline
    C = bounded_run(circ)
    offdiag_flop = C['cat']['offdiag_rot'][2]
    D_total = C['total'] - 0.5 * offdiag_flop     # counterfactual: off-diag -> diag rate
    print(f"\n{'='*92}\n{circ}   Clifft k={A['k']}, bounded peak={C['k']}   "
          f"[A fused {H(A['total'])} | B unfused {H(B['total'])} | C bounded {H(C['total'])} | "
          f"D bnd-diag {H(D_total)}]")
    print(f"  W1=sum2^r: Clifft-unfused {H(B['W1'])} ({B['nev']} ev) | bounded {H(C['W1'])} ({C['nev']} ev)")
    print(f"  {'category':14}{'clifft-unf calls':>17}{'clifft FLOP':>13}{'bnd calls':>11}{'bnd FLOP':>11}{'bnd-clf diff':>13}")
    for c in CATS:
        bc, bs, bf = B['cat'][c]; cc_, cs, cf = C['cat'][c]
        if bf == 0 and cf == 0:
            continue
        print(f"  {c:14}{bc:>17}{H(bf):>13}{cc_:>11}{H(cf):>11}{H(cf - bf):>13}")
    print(f"  {'TOTAL':14}{B['nev']:>17}{H(B['total']):>13}{C['nev']:>11}{H(C['total']):>11}{H(C['total'] - B['total']):>13}")
    print(f"  counterfactual D (bounded off-diag charged as diagonal): {H(D_total)}  "
          f"(off-diag penalty = {H(0.5*offdiag_flop)} = {100*0.5*offdiag_flop/max(C['total'],1):.0f}% of bounded total)")
    # store rows for csv
    for c in CATS:
        allrows.append([circ, c, B['cat'][c][0], int(B['cat'][c][2]), C['cat'][c][0], int(C['cat'][c][2])])

with open("reports_clifft_axis_bounded/flop_accounting/data/flop_attribution.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["circuit", "category", "clifft_unfused_calls", "clifft_unfused_flop",
                "bounded_calls", "bounded_flop"])
    w.writerows(allrows)
print("\n-> reports_clifft_axis_bounded/flop_accounting/data/flop_attribution.csv")
