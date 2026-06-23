"""Phase 2 / Step 2 bit-exact verification: single-axis X/Y rotation localized to a diagonal
Z_a (via the verified _localize_to_Z) vs the off-diagonal butterfly.  Same process, toggling
CliftAxisBoundedNearClifford._step2_localize.  The localization is an EXACT state transform and
the diagonal landing drops only a global phase, so records, peak rank, and Born p0 must be
IDENTICAL.  Also reports FLOP, off/diag rotation counts, and wall-clock delta.
"""
import sys, time, statistics
sys.path.insert(0, "/home/jung/clifft-paper")
sys.setrecursionlimit(400000)
import numpy as np
import clifft
import nearclifford_backend.backend as bk
from nearclifford_backend.clifft_axis import budget as _bud
from nearclifford_backend.clifft_axis.bounded import compile_bounded, CliftAxisBoundedNearClifford as C

COEF = {'rot:offdiag': 12, 'rot:offdiag-scalar': 12, 'rot:diag': 6, 'rot:diag0': 6,
        'rot:diag-scalar': 6, 'rot:diaghalf': 3, 'collapse:offdiag': 12, 'collapse:diag': 6,
        'collapse:diag0': 6, 'meas': 10, 'exp': 10, 'reduce:verify': 10, 'sqnorm': 2,
        'normalize': 2, 'purge:h': 4, 'purge:s': 2, 'purge:cnot': 0, 'reduce:cnot': 0,
        'drop': 0, 'promote': 0, 'reduce:gf2scan': 0, 'init': 0, 'post-reduce': 0}
OFFD = {'rot:offdiag', 'rot:offdiag-scalar'}
DIAGH = {'rot:diaghalf'}


def run(circ, seed, step2, want_acct=False):
    prog = compile_bounded(open(f"qec_bench/circuits/{circ}.stim").read())
    C._step2_localize = step2
    flop = [0.0]; noff = [0]; ndh = [0]; nh = [0]
    orig = _bud.DenseMemoryBudget.charge
    if want_acct:
        def charge(self, resident, transient=0, where=""):
            flop[0] += COEF.get(where, 0) * int(resident)
            if where in OFFD:
                noff[0] += 1
            if where in DIAGH:
                ndh[0] += 1
            if where == 'purge:h':
                nh[0] += 1
            return orig(self, resident, transient, where)
        _bud.DenseMemoryBudget.charge = charge
    try:
        be = bk.NearCliffordBackend(clifft_axis_bounded=True, drop_dead=False,
                                    structure_once=False, clifft_axis_enforce=True)
        rec = dict(be.run_shot(prog, seed))
        pk = be.nc.budget.peak_resident.bit_length() - 1
        p0 = [c.get("p0") for c in be.nc.core_log if c.get("p0") is not None]
    finally:
        _bud.DenseMemoryBudget.charge = orig
        C._step2_localize = True
    return rec, pk, p0, flop[0], noff[0], ndh[0], nh[0]


def wall(circ, seed, step2, iters=5):
    prog = compile_bounded(open(f"qec_bench/circuits/{circ}.stim").read())
    C._step2_localize = step2
    try:
        for _ in range(2):
            bk.NearCliffordBackend(clifft_axis_bounded=True, drop_dead=False,
                                   structure_once=False, clifft_axis_enforce=True).run_shot(prog, seed)
        ts = []
        for _ in range(iters):
            b = bk.NearCliffordBackend(clifft_axis_bounded=True, drop_dead=False,
                                       structure_once=False, clifft_axis_enforce=True)
            t0 = time.perf_counter(); b.run_shot(prog, seed); ts.append(time.perf_counter() - t0)
    finally:
        C._step2_localize = True
    return statistics.median(ts)


CIRCS = [("coherent_ry_d3_r1", 10), ("coherent_ry_d3_r3", 6), ("cultivation_d3", 16),
         ("cultivation_d5", 8), ("coherent_rx_d3_r3", 8), ("coherent_d3_r3", 10),
         ("coherent_rx_d3_r1", 8), ("distillation", 16), ("coherent_d5_r5", 2)]

print("=== Phase 2 Step 2: single-axis X/Y localization, bit-exact A/B (on vs off) ===\n")
allok = True
for circ, ns in CIRCS:
    rm = km = 0; pm = 0.0
    for s in range(1, ns + 1):
        r1, k1, q1, *_ = run(circ, s, True)
        r0, k0, q0, *_ = run(circ, s, False)
        if r1 != r0:
            rm += 1
        if k1 != k0:
            km += 1
        pm = max([pm] + [abs(a - b) for a, b in zip(q1, q0)]) if len(q1) == len(q0) else 1.0
    ok = rm == 0 and km == 0 and pm < 1e-9
    allok &= ok
    print(f"  {circ:20} seeds={ns}  rec_mismatch={rm}  rank_mismatch={km}  "
          f"max|dp0|={pm:.1e}  {'PASS' if ok else 'FAIL'}", flush=True)
print(f"\n{'ALL EXACT' if allok else 'SOME FAIL'}\n")

print("=== Step 2 FLOP + rotation-class counts + wall-clock (off -> on) ===")
print(f"{'circuit':18}{'FLOP off':>11}{'FLOP on':>11}{'d%':>6}{'offd o/n':>11}{'diagH o/n':>11}"
      f"{'+H':>5}{'wall o/n (ms)':>16}")
for circ, _ in CIRCS:
    _, _, _, fo, oo, dho, ho = run(circ, 1, False, want_acct=True)
    _, _, _, fn, on, dhn, hn = run(circ, 1, True, want_acct=True)
    wo = wall(circ, 1, False); wn = wall(circ, 1, True)
    d = (fn - fo) / fo * 100 if fo else 0
    print(f"  {circ:18}{fo/1e6:>9.3f}M{fn/1e6:>10.3f}M{d:>5.0f}%{oo:>5}/{on:<5}{dho:>5}/{dhn:<5}"
          f"{hn-ho:>5}{wo*1e3:>7.1f}/{wn*1e3:<8.1f}", flush=True)
