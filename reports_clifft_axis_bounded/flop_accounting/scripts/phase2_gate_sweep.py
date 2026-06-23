"""Phase 2 / gate tuning -- sweep the EXISTING _loc_min_size knob (no new heuristic).

The localizer engages only when phi.size >= _loc_min_size. The d5_r5 wall cost is the
localizer's strided sweeps + O(n)=O(72) incremental inverse-frame update per fold, which at
n=72 / low rank (<=2^13) costs more wall than the butterfly. The RY circuits reach rank 16
(phi.size up to 2^16). A threshold between 2^13 and 2^16 disengages the localizer on d5_r5
(low rank) while keeping it for the RY regime (high rank) -- IF the RY FLOP win survives.

For each threshold we report FLOP + wall in P2fold+inv mode; target: smallest threshold where
d5_r5 wall <= P1 (~199 ms) AND ry_d3_r1 FLOP stays ~= clifft-unfused (12.29M).
Reference rows: P1 (localizer fully OFF) and clifft-unfused FLOP.
"""
import sys, time, statistics
sys.path.insert(0, "/home/jung/clifft-paper")
sys.setrecursionlimit(400000)
from collections import defaultdict
import clifft
from clifft import _clifft_core as cc
import nearclifford_backend.backend as bk
from nearclifford_backend.clifft_axis import budget as _bud
from nearclifford_backend.clifft_axis.bounded import compile_bounded, CliftAxisBoundedNearClifford as C

CONV = dict(cmul=6, rcmul=2, cadd=2, sqmag=4, vdot=8)
BW = {'rot:offdiag': 12, 'rot:offdiag-scalar': 12, 'rot:diaghalf': 3, 'rot:diag': 6,
      'rot:diag0': 6, 'rot:diag-scalar': 6, 'collapse:offdiag': 12, 'collapse:diag': 6,
      'collapse:diag0': 6, 'meas': 10, 'exp': 10, 'reduce:verify': 10, 'sqnorm': 2,
      'normalize': 2, 'purge:h': 4, 'purge:s': 2}


def bounded(circ, loc_min, step2=True, seed=1, iters=5):
    prog = compile_bounded(open(f"qec_bench/circuits/{circ}.stim").read())
    C._step2_localize = step2; C._loc_undo = False; C._loc_min_size = loc_min
    tot = [0.0]; nloc = [0]
    orig = _bud.DenseMemoryBudget.charge

    def charge(self, resident, transient=0, where=""):
        N = int(resident); tot[0] += BW.get(where, 0) * N
        if where.startswith('collapse'):
            tot[0] += 6 * N
        return orig(self, resident, transient, where)
    _bud.DenseMemoryBudget.charge = charge
    try:
        be = bk.NearCliffordBackend(clifft_axis_bounded=True, drop_dead=False,
                                    structure_once=False, clifft_axis_enforce=True)
        be.run_shot(prog, seed)
    finally:
        _bud.DenseMemoryBudget.charge = orig
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
        C._step2_localize = True; C._loc_min_size = 1 << 12
    return tot[0], statistics.median(ts)


def clifft_flop(circ, seed=1):
    prog = clifft.compile(open(f"qec_bench/circuits/{circ}.stim").read(), bytecode_passes=None)
    cc.cost_meter_reset(); cc.cost_meter_enable(True)
    clifft.sample(prog, 1, seed)
    cc.cost_meter_enable(False)
    snap = cc.cost_meter_snapshot()
    return sum(sum(CONV[k] * s[k] for k in CONV) for s in snap.values())


def H(x):
    a = abs(x)
    for u, s in ((1e9, 'G'), (1e6, 'M'), (1e3, 'k')):
        if a >= u:
            return f"{x/u:.2f}{s}"
    return f"{x:.0f}"


CIRCS = ["coherent_ry_d3_r1", "coherent_ry_d3_r3", "coherent_rx_d3_r3",
         "coherent_d5_r5", "cultivation_d5"]
THRS = [("2^12(cur)", 1 << 12), ("2^13", 1 << 13), ("2^14", 1 << 14),
        ("2^15", 1 << 15), ("2^16", 1 << 16), ("OFF=P1", 1 << 40)]

print(f"{'circuit':18}{'clf-unf':>9}{'P1 wall':>9} | thresholds (FLOP / wall ms):")
for circ in CIRCS:
    cu = clifft_flop(circ)
    cells = []
    p1w = None
    for nm, T in THRS:
        f, w = bounded(circ, T)
        if nm == "OFF=P1":
            p1w = w
        cells.append((nm, f, w))
    print(f"\n{circ:18}{H(cu):>9}{(p1w*1e3):>8.0f}m")
    for nm, f, w in cells:
        print(f"    {nm:10} FLOP={H(f):>9}  wall={w*1e3:>7.1f}ms", flush=True)
