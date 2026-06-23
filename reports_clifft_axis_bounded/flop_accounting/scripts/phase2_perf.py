"""Phase 2 / section 4 -- performance re-measurement: FLOP / wall / sweeps / traffic /
permutation / pullback-recompute / peak-rank across bounded MODES + clifft fused/unfused.

bounded modes (all bit-exact -- verified in phase2_correctness.py):
  P1        : _step2_localize=False                          (Phase-1 off-diagonal butterfly)
  P2undo    : _step2_localize=True,  _loc_undo=True          (2-H undo, frame untouched)
  P2fold-noI: _step2_localize=True,  _loc_undo=False, inv OFF (1-H frame-fold, GF(2) pullback)  <- old regression
  P2fold+inv: _step2_localize=True,  _loc_undo=False, inv ON  (1-H frame-fold + inverse-frame)   <- OPTIMAL

"pullback recompute" = the EXPENSIVE O(n^2) GF(2) basis eliminations actually performed
  (inv OFF: _pullback_basis cache misses; inv ON: _inv_recompute = AG-measure lazy rebuilds).
CNOT/SWAP cost 0 FLOP but are counted as sweeps + permutations + traffic words (never hidden).
clifft (C++) wall is cross-language and reported separately for reference only.
"""
import sys, time, statistics
sys.path.insert(0, "/home/jung/clifft-paper")
sys.setrecursionlimit(400000)
from collections import defaultdict
import clifft
from clifft import _clifft_core as cc
import nearclifford_backend.backend as bk
from nearclifford_backend.clifft_axis import budget as _bud
from nearclifford_backend.clifft_axis.engine import CliftAxisNearClifford
from nearclifford_backend.clifft_axis.bounded import compile_bounded, CliftAxisBoundedNearClifford as C

CONV = dict(cmul=6, rcmul=2, cadd=2, sqmag=4, vdot=8)
# bounded `where` -> (FLOP/word, traffic words/elt, is_perm, is_rot)
BW = {'rot:offdiag': (12, 2, 0, 1), 'rot:offdiag-scalar': (12, 2, 0, 1),
      'rot:diaghalf': (3, 2, 0, 1), 'rot:diag': (6, 2, 0, 1), 'rot:diag0': (6, 1, 0, 1),
      'rot:diag-scalar': (6, 2, 0, 1),
      'collapse:offdiag': (12, 2, 0, 0), 'collapse:diag': (6, 2, 0, 0), 'collapse:diag0': (6, 1, 0, 0),
      'meas': (10, 2, 0, 0), 'exp': (10, 2, 0, 0), 'reduce:verify': (10, 2, 0, 0),
      'sqnorm': (2, 1, 0, 0), 'normalize': (2, 1, 0, 0),
      'purge:h': (4, 2, 0, 0), 'purge:s': (2, 1, 0, 0),
      'purge:cnot': (0, 2, 1, 0), 'reduce:cnot': (0, 2, 1, 0), 'reduce:gf2scan': (0, 2, 0, 0),
      'drop': (0, 1, 1, 0), 'promote': (0, 1, 1, 0), 'init': (0, 0, 0, 0), 'post-reduce': (0, 0, 0, 0)}
CLF_PERM = {'array_cnot', 'array_swap', 'array_multi_cnot'}
CLF_ROT = {'array_rot', 'expand_rot'}

MODES = [("P1", False, True, True), ("P2undo", True, True, True),
         ("P2fold-noI", True, False, False), ("P2fold+inv", True, False, True)]


def patched_init(inv):
    base = CliftAxisNearClifford.__init__
    def init(self, n):
        base(self, n)
        self._inv_enabled = inv
    return init, base


def gf2_counter():
    """Wrap _pullback_basis to count real GF(2) recomputes (cache misses)."""
    base = CliftAxisNearClifford._pullback_basis
    cnt = [0]
    def pb(self):
        if self._pb_cache is None or self._pb_cache[0] != self._frame_ver:
            cnt[0] += 1
        return base(self)
    return pb, base, cnt


def bounded_run(circ, step2, undo, inv, seed=1, time_iters=5):
    prog = compile_bounded(open(f"qec_bench/circuits/{circ}.stim").read())
    C._step2_localize = step2; C._loc_undo = undo
    agg = defaultdict(lambda: [0, 0.0, 0, 0, 0])
    orig = _bud.DenseMemoryBudget.charge
    init_p, init_b = patched_init(inv)
    pb_p, pb_b, gf2 = gf2_counter()
    CliftAxisNearClifford.__init__ = init_p
    CliftAxisNearClifford._pullback_basis = pb_p

    def charge(self, resident, transient=0, where=""):
        N = int(resident); f, tw, isp, isr = BW.get(where, (0, 1, 0, 0))
        a = agg[where]; a[0] += 1; a[1] += f * N; a[2] += tw * N; a[3] += isp; a[4] += isr
        return orig(self, resident, transient, where)
    _bud.DenseMemoryBudget.charge = charge
    try:
        be = bk.NearCliffordBackend(clifft_axis_bounded=True, drop_dead=False,
                                    structure_once=False, clifft_axis_enforce=True)
        be.run_shot(prog, seed)
        inv_rebuild = be.nc._inv_recompute
        gf2_acct = gf2[0]          # capture BEFORE the timing loop re-runs _pullback_basis
    finally:
        _bud.DenseMemoryBudget.charge = orig
    # wall WITHOUT charge hook (keep init/pb patches -- they are the modes under test)
    try:
        for _ in range(2):
            bk.NearCliffordBackend(clifft_axis_bounded=True, drop_dead=False,
                                   structure_once=False, clifft_axis_enforce=True).run_shot(prog, seed)
        ts = []
        for _ in range(time_iters):
            b2 = bk.NearCliffordBackend(clifft_axis_bounded=True, drop_dead=False,
                                        structure_once=False, clifft_axis_enforce=True)
            t0 = time.perf_counter(); b2.run_shot(prog, seed); ts.append(time.perf_counter() - t0)
    finally:
        CliftAxisNearClifford.__init__ = init_b
        CliftAxisNearClifford._pullback_basis = pb_b
        C._step2_localize = True; C._loc_undo = False
    wall = statistics.median(ts)
    flop = sum(v[1] for v in agg.values())
    sweeps = sum(v[0] for k, v in agg.items() if BW.get(k, (0, 1, 0, 0))[1] > 0)
    traffic = sum(v[2] for v in agg.values())
    perms = sum(v[3] for v in agg.values())
    recompute = inv_rebuild if inv else gf2_acct
    return dict(flop=flop, wall=wall, sweeps=sweeps, traffic=traffic, perms=perms,
                recompute=recompute, peak=be.nc.budget.peak_resident.bit_length() - 1)


def clifft_run(circ, fused, seed=1, time_iters=5):
    src = open(f"qec_bench/circuits/{circ}.stim").read()
    prog = clifft.compile(src) if fused else clifft.compile(src, bytecode_passes=None)
    cc.cost_meter_reset(); cc.cost_meter_enable(True)
    clifft.sample(prog, 1, seed)
    cc.cost_meter_enable(False)
    snap = cc.cost_meter_snapshot()
    flop = sweeps = traffic = perms = 0
    for name, s in snap.items():
        flop += sum(CONV[k] * s[k] for k in CONV)
        sweeps += s['calls']; traffic += 2 * s['processed']
        if name in CLF_PERM:
            perms += s['calls']
    ts = []
    for _ in range(time_iters):
        t0 = time.perf_counter(); clifft.sample(prog, 1, seed); ts.append(time.perf_counter() - t0)
    return dict(flop=flop, wall=statistics.median(ts), sweeps=sweeps, traffic=traffic,
                perms=perms, recompute=0, peak=prog.peak_rank)


def H(x):
    a = abs(x)
    for u, s in ((1e9, 'G'), (1e6, 'M'), (1e3, 'k')):
        if a >= u:
            return f"{x/u:.2f}{s}"
    return f"{x:.0f}"


CIRCS = ["coherent_ry_d3_r1", "coherent_ry_d3_r3", "coherent_rx_d3_r3",
         "coherent_d5_r5", "cultivation_d5"]
print(f"{'circuit':18}{'mode':12}{'FLOP':>9}{'wall(ms)':>10}{'sweeps':>8}{'traffic':>9}"
      f"{'perms':>7}{'pb_recmp':>9}{'peakK':>7}")
print("-" * 89)
for circ in CIRCS:
    cf = clifft_run(circ, fused=True); cu = clifft_run(circ, fused=False)
    rows = [("clifft-fused", cf), ("clifft-unfused", cu)]
    for nm, st2, un, inv in MODES:
        rows.append((nm, bounded_run(circ, st2, un, inv)))
    for i, (nm, d) in enumerate(rows):
        print(f"{circ if i == 0 else '':18}{nm:12}{H(d['flop']):>9}{d['wall']*1e3:>10.2f}"
              f"{d['sweeps']:>8}{H(d['traffic']):>9}{d['perms']:>7}{d['recompute']:>9}"
              f"{d['peak']:>7}", flush=True)
    print()
