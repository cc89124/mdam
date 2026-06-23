"""Phase 2 runtime + cultivation residual-gap accounting.

Two deliverables the user demanded ALONGSIDE implementation:
 (A) cultivation residual gap: for BOTH backends, rotation event count, Sum_rot 2^r,
     T-specific path, localization cost vs other -- attribute the ~2x clifft gap.
 (B) CNOT/SWAP are 0 FLOP but NOT 0 runtime: report array-sweep count, permutation count,
     memory traffic (words touched), and wall-clock -- so the FLOP picture is not mistaken
     for the runtime picture.

bounded: budget.charge hook -> per `where` (FLOP, sweep, traffic words, is_perm) + wall-clock.
clifft : CostMeter snapshot -> per kernel (FLOP, calls=sweeps, processed=traffic, perms) +
         clifft.sample wall-clock (C++ reference; cross-language, reported separately).
"""
import sys, time, statistics
sys.path.insert(0, "/home/jung/clifft-paper")
sys.setrecursionlimit(400000)
from collections import defaultdict
import numpy as np
import clifft
from clifft import _clifft_core as cc
import nearclifford_backend.backend as bk
from nearclifford_backend.clifft_axis import budget as _bud
from nearclifford_backend.clifft_axis.bounded import compile_bounded, CliftAxisBoundedNearClifford as C

CONV = dict(cmul=6, rcmul=2, cadd=2, sqmag=4, vdot=8)

# bounded `where` -> (FLOP/word, traffic words/elt touched, is_permutation, is_rotation)
BW = {'rot:offdiag': (12, 2, 0, 1), 'rot:offdiag-scalar': (12, 2, 0, 1),
      'rot:diaghalf': (3, 2, 0, 1), 'rot:diag': (6, 2, 0, 1), 'rot:diag0': (6, 1, 0, 1), 'rot:diag-scalar': (6, 2, 0, 1),
      'collapse:offdiag': (12, 2, 0, 0), 'collapse:diag': (6, 2, 0, 0), 'collapse:diag0': (6, 1, 0, 0),
      'meas': (10, 2, 0, 0), 'exp': (10, 2, 0, 0), 'reduce:verify': (10, 2, 0, 0),
      'sqnorm': (2, 1, 0, 0), 'normalize': (2, 1, 0, 0),
      'purge:h': (4, 2, 0, 0), 'purge:s': (2, 1, 0, 0),
      'purge:cnot': (0, 2, 1, 0), 'reduce:cnot': (0, 2, 1, 0), 'reduce:gf2scan': (0, 2, 0, 0),
      'drop': (0, 1, 1, 0), 'promote': (0, 1, 1, 0), 'init': (0, 0, 0, 0), 'post-reduce': (0, 0, 0, 0)}

# clifft kernel -> (is_perm, is_rotation)
CLF_PERM = {'array_cnot', 'array_swap', 'array_multi_cnot'}
CLF_ROT = {'array_rot', 'expand_rot'}


def bounded_run(circ, seed=1, time_iters=5):
    prog = compile_bounded(open(f"qec_bench/circuits/{circ}.stim").read())
    agg = defaultdict(lambda: [0, 0.0, 0, 0, 0])    # where -> [n, flop, traffic, perm_n, rot_n]
    rot_pow2 = [0]                                    # Sum_rot 2^r
    rot_events = [0]
    orig = _bud.DenseMemoryBudget.charge

    def charge(self, resident, transient=0, where=""):
        N = int(resident)
        f, tw, isp, isr = BW.get(where, (0, 1, 0, 0))
        a = agg[where]
        a[0] += 1; a[1] += f * N; a[2] += tw * N; a[3] += isp; a[4] += isr
        if isr:
            rot_pow2[0] += N; rot_events[0] += 1
        return orig(self, resident, transient, where)

    # ONE hooked run for accounting only
    _bud.DenseMemoryBudget.charge = charge
    try:
        be = bk.NearCliffordBackend(clifft_axis_bounded=True, drop_dead=False,
                                    structure_once=False, clifft_axis_enforce=True)
        be.run_shot(prog, seed)
    finally:
        _bud.DenseMemoryBudget.charge = orig
    # wall-clock WITHOUT the accounting hook (warmup + median of fresh-engine run_shots)
    for _ in range(2):
        bw = bk.NearCliffordBackend(clifft_axis_bounded=True, drop_dead=False,
                                    structure_once=False, clifft_axis_enforce=True)
        bw.run_shot(prog, seed)
    ts = []
    for _ in range(time_iters):
        be2 = bk.NearCliffordBackend(clifft_axis_bounded=True, drop_dead=False,
                                     structure_once=False, clifft_axis_enforce=True)
        t0 = time.perf_counter(); be2.run_shot(prog, seed); ts.append(time.perf_counter() - t0)
    wall = statistics.median(ts)
    flop = sum(v[1] for v in agg.values())
    sweeps = sum(v[0] for v in agg.values() if BW.get(_, (0, 0, 0, 0)) or True)
    sweeps = sum(v[0] for k, v in agg.items() if BW.get(k, (0, 1, 0, 0))[1] > 0)
    traffic = sum(v[2] for v in agg.values())
    perms = sum(v[3] for v in agg.values())
    rot_sweeps = sum(v[0] for k, v in agg.items() if BW.get(k, (0, 0, 0, 0))[3])
    return dict(flop=flop, sweeps=sweeps, traffic=traffic, perms=perms, wall=wall,
                rot_events=rot_events[0], rot_pow2=rot_pow2[0], rot_sweeps=rot_sweeps,
                peak=prog.peak_rank, agg=dict(agg))


def clifft_run(circ, seed=1, time_iters=5):
    src = open(f"qec_bench/circuits/{circ}.stim").read()
    prog = clifft.compile(src, bytecode_passes=None)
    cc.cost_meter_reset(); cc.cost_meter_enable(True)
    clifft.sample(prog, 1, seed)
    cc.cost_meter_enable(False)
    snap = cc.cost_meter_snapshot()
    flop = sweeps = traffic = perms = rot_events = rot_pow2 = 0
    for name, s in snap.items():
        f = sum(CONV[k] * s[k] for k in CONV)
        flop += f; sweeps += s['calls']; traffic += 2 * s['processed']
        if name in CLF_PERM:
            perms += s['calls']
        if name in CLF_ROT:
            rot_events += s['calls']; rot_pow2 += s['sum_pow2k']
    ts = []
    for _ in range(time_iters):
        t0 = time.perf_counter(); clifft.sample(prog, 1, seed); ts.append(time.perf_counter() - t0)
    wall = statistics.median(ts)
    return dict(flop=flop, sweeps=sweeps, traffic=traffic, perms=perms, wall=wall,
                rot_events=rot_events, rot_pow2=rot_pow2, peak=prog.peak_rank, snap=snap)


def H(x):
    a = abs(x)
    for u, s in ((1e9, 'G'), (1e6, 'M'), (1e3, 'k')):
        if a >= u:
            return f"{x/u:.2f}{s}"
    return f"{x:.0f}"


CIRCS = ["coherent_ry_d3_r1", "cultivation_d3", "cultivation_d5", "coherent_rx_d3_r3",
         "coherent_d5_r5", "distillation"]
print(f"{'circuit':18}{'backend':9}{'FLOP':>9}{'sweeps':>8}{'perms':>7}{'traffic':>9}"
      f"{'rot_ev':>7}{'Srot2^r':>9}{'wall':>10}")
print("-" * 96)
for circ in CIRCS:
    b = bounded_run(circ); c = clifft_run(circ)
    for nm, d in (("clifft-unf", c), ("bounded-P1", b)):
        print(f"{circ if nm=='clifft-unf' else '':18}{nm:9}{H(d['flop']):>9}{d['sweeps']:>8}"
              f"{d['perms']:>7}{H(d['traffic']):>9}{d['rot_events']:>7}{H(d['rot_pow2']):>9}"
              f"{d['wall']*1e3:>8.1f}ms", flush=True)
    print()
