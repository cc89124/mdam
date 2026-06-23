"""Phase 2 / Step 1 bit-exact verification: strided half-array single-axis-Z rotation
(global-phase-dropped) vs the pre-Step-1 parity kernel.  Same process, toggling
CliftAxisNearClifford._step1_diaghalf.  Step 1 changes only a global phase, so records,
peak rank, and Born p0 must be IDENTICAL; the statevector must match up to a global phase
(|<old|new>| = 1).  Also reports the per-circuit FLOP + wall-clock delta from Step 1.
"""
import sys, time, statistics
sys.path.insert(0, "/home/jung/clifft-paper")
sys.setrecursionlimit(400000)
from collections import defaultdict
import numpy as np
import nearclifford_backend.backend as bk
from nearclifford_backend.clifft_axis import budget as _bud
from nearclifford_backend.clifft_axis.engine import CliftAxisNearClifford as Eng
from nearclifford_backend.clifft_axis.bounded import compile_bounded

COEF = {'rot:offdiag': 12, 'rot:offdiag-scalar': 12, 'rot:diag': 6, 'rot:diag0': 6,
        'rot:diag-scalar': 6, 'rot:diaghalf': 3, 'collapse:offdiag': 12, 'collapse:diag': 6,
        'collapse:diag0': 6, 'meas': 10, 'exp': 10, 'reduce:verify': 10, 'sqnorm': 2,
        'normalize': 2, 'purge:h': 5, 'purge:s': 3, 'purge:cnot': 0, 'reduce:cnot': 0,
        'drop': 0, 'promote': 0, 'reduce:gf2scan': 0, 'init': 0, 'post-reduce': 0}


def run(circ, seed, step1, want_flop=False):
    prog = compile_bounded(open(f"qec_bench/circuits/{circ}.stim").read())
    Eng._step1_diaghalf = step1
    flop = [0.0]
    orig = _bud.DenseMemoryBudget.charge
    if want_flop:
        def charge(self, resident, transient=0, where=""):
            flop[0] += COEF.get(where, 0) * int(resident)
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
        Eng._step1_diaghalf = True
    return rec, pk, p0, flop[0]


def wall(circ, seed, step1, iters=5):
    prog = compile_bounded(open(f"qec_bench/circuits/{circ}.stim").read())
    Eng._step1_diaghalf = step1
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
        Eng._step1_diaghalf = True
    return statistics.median(ts)


CIRCS = [("coherent_ry_d3_r1", 10), ("coherent_ry_d3_r3", 6), ("cultivation_d3", 16),
         ("cultivation_d5", 8), ("coherent_rx_d3_r3", 8), ("coherent_d3_r3", 10),
         ("coherent_rx_d3_r1", 8), ("distillation", 16), ("coherent_d5_r5", 2)]

print("=== Phase 2 Step 1: strided half-array single-Z rotation, bit-exact A/B ===\n")
allok = True
for circ, ns in CIRCS:
    rm = km = pm = 0
    for s in range(1, ns + 1):
        r1, k1, q1, _ = run(circ, s, True)
        r0, k0, q0, _ = run(circ, s, False)
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

print("=== Step 1 FLOP + wall-clock delta (single-Z diagonal rotations) ===")
print(f"{'circuit':20}{'FLOP off':>12}{'FLOP on':>12}{'dFLOP':>8}{'wall off':>11}{'wall on':>11}{'speedup':>9}")
for circ, _ in CIRCS:
    _, _, _, f_off = run(circ, 1, False, want_flop=True)
    _, _, _, f_on = run(circ, 1, True, want_flop=True)
    w_off = wall(circ, 1, False); w_on = wall(circ, 1, True)
    d = (f_on - f_off) / f_off * 100 if f_off else 0
    sp = w_off / w_on if w_on else 0
    print(f"  {circ:18}{f_off/1e6:>10.3f}M{f_on/1e6:>11.3f}M{d:>7.0f}%"
          f"{w_off*1e3:>9.1f}ms{w_on*1e3:>9.1f}ms{sp:>8.2f}x", flush=True)
