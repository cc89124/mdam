"""Phase 3 (ANALYSIS ONLY -- no optimization) -- per-event dense-state trace for the common
cost model  F = sum_e c_e 2^{r_e}.

For each circuit dumps:
  clifft (unfused, bytecode_passes=None): the FULL CostMeter snapshot, per kernel --
     calls, sum_pow2k (= sum_e 2^{r_e}), processed (touched words), FLOP, eff coeff = FLOP/sum_pow2k.
     The decisive question: does clifft apply localization H/CNOT to the ARRAY (array_h/array_cnot
     nonzero) or only to the symbolic frame?  array_rot eff-coeff tells whether rotations land
     diagonal (3) only, or carry extra sweeps.
  bounded (P2 production): per `where` -- count, sum_resident (= sum_e 2^{r_e}), FLOP, eff coeff.

Separates: resident peak / state-size sum (sum 2^r) / touched-word traffic / FLOP / (wall omitted).
NO kernels are modified; this only READS the cost meter and the budget charges.
"""
import sys
sys.path.insert(0, "/home/jung/clifft-paper")
sys.setrecursionlimit(400000)
from collections import defaultdict
import clifft
from clifft import _clifft_core as cc
import nearclifford_backend.backend as bk
from nearclifford_backend.clifft_axis import budget as _bud
from nearclifford_backend.clifft_axis.bounded import compile_bounded

CONV = dict(cmul=6, rcmul=2, cadd=2, sqmag=4, vdot=8)
# bounded where -> FLOP/amplitude coefficient (the c_e), matching the kernels in engine.py/bounded.py
BCOEF = {'rot:offdiag': 12, 'rot:offdiag-scalar': 12, 'rot:diaghalf': 3, 'rot:diag': 6,
         'rot:diag0': 6, 'rot:diag-scalar': 6, 'collapse:offdiag': 12, 'collapse:diag': 6,
         'collapse:diag0': 6, 'meas': 10, 'exp': 10, 'reduce:verify': 10, 'sqnorm': 2,
         'normalize': 2, 'purge:h': 4, 'purge:s': 2, 'purge:cnot': 0, 'reduce:cnot': 0,
         'reduce:gf2scan': 0, 'drop': 0, 'promote': 0, 'init': 0, 'post-reduce': 0}


def clifft_trace(circ, seed=1):
    prog = clifft.compile(open(f"qec_bench/circuits/{circ}.stim").read(), bytecode_passes=None)
    cc.cost_meter_reset(); cc.cost_meter_enable(True)
    clifft.sample(prog, 1, seed)
    cc.cost_meter_enable(False)
    snap = cc.cost_meter_snapshot()
    rows = []
    for name, s in snap.items():
        flop = sum(CONV[k] * s[k] for k in CONV)
        if s['calls'] == 0 and flop == 0:
            continue
        s2 = s.get('sum_pow2k', 0)
        rows.append((name, s['calls'], s2, s.get('processed', 0), flop,
                     (flop / s2 if s2 else 0.0)))
    rows.sort(key=lambda r: -r[4])
    return rows, prog.peak_rank


def bounded_trace(circ, seed=1):
    prog = compile_bounded(open(f"qec_bench/circuits/{circ}.stim").read())
    agg = defaultdict(lambda: [0, 0, 0])     # where -> [count, sum_resident, touched_words]
    orig = _bud.DenseMemoryBudget.charge

    def charge(self, resident, transient=0, where=""):
        a = agg[where]; a[0] += 1; a[1] += int(resident); a[2] += int(resident)
        return orig(self, resident, transient, where)
    _bud.DenseMemoryBudget.charge = charge
    try:
        be = bk.NearCliffordBackend(clifft_axis_bounded=True, drop_dead=False,
                                    structure_once=False, clifft_axis_enforce=True)
        be.run_shot(prog, seed)
        pk = be.nc.budget.peak_resident.bit_length() - 1
    finally:
        _bud.DenseMemoryBudget.charge = orig
    rows = []
    for where, (cnt, sres, tw) in agg.items():
        c = BCOEF.get(where, 0)
        rows.append((where, cnt, sres, tw, c * sres, float(c)))
    rows.sort(key=lambda r: -r[4])
    return rows, pk


def H(x):
    a = abs(x)
    for u, s in ((1e9, 'G'), (1e6, 'M'), (1e3, 'k')):
        if a >= u:
            return f"{x/u:.2f}{s}"
    return f"{x:.0f}"


for circ in ["cultivation_d5", "coherent_ry_d3_r1", "coherent_d5_r5"]:
    print("\n" + "=" * 92 + f"\n  {circ}\n" + "=" * 92)
    cr, kc = clifft_trace(circ)
    br, kb = bounded_trace(circ)
    ftc = sum(r[4] for r in cr); ftb = sum(r[4] for r in br)
    s2c = sum(r[2] for r in cr); s2b = sum(r[2] for r in br)
    twc = sum(r[3] for r in cr); twb = sum(r[3] for r in br)
    print(f"\n  CLIFFT-unfused (k={kc}):  FLOP={H(ftc)}  sum2^r={H(s2c)}  traffic={H(twc)}")
    print(f"  {'kernel':22}{'calls':>7}{'sum2^r':>10}{'touched':>10}{'FLOP':>10}{'c=F/s2r':>9}")
    for nm, cnt, s2, tw, fl, c in cr:
        print(f"  {nm:22}{cnt:>7}{H(s2):>10}{H(tw):>10}{H(fl):>10}{c:>9.2f}")
    print(f"\n  BOUNDED-P2 (k={kb}):  FLOP={H(ftb)}  sum2^r={H(s2b)}  traffic={H(twb)}")
    print(f"  {'where':22}{'calls':>7}{'sum2^r':>10}{'touched':>10}{'FLOP':>10}{'c':>9}")
    for nm, cnt, s2, tw, fl, c in br:
        print(f"  {nm:22}{cnt:>7}{H(s2):>10}{H(tw):>10}{H(fl):>10}{c:>9.2f}")
    print(f"\n  --> FLOP ratio bnd/clf = {ftb/ftc:.2f}   sum2^r ratio = {s2b/s2c:.2f}   "
          f"traffic ratio = {twb/twc:.2f}")
