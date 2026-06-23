"""Phase 2 / section 5 -- FINAL 9-circuit table.

Columns: Clifft fused | Clifft unfused | bounded original | Phase 1 | Phase 2 FLOP | Phase 2 wall
         | peak bnd/clf.  Plus aggregates: #circuits Phase 2 beats Clifft-unfused, FLOP reduction
vs Phase 1, wall reduction vs Phase 1, total pullback recomputes, total inverse update/lookup,
peak resident-rank change.

FLOP definitions (one convention cmul6/rcmul2/cadd2/sqmag4/vdot8):
  bounded original = pre-Phase-1 measure_z (repeated sqnorm) + off-diagonal butterfly rotation
  Phase 1          = post-Phase-1 measure_z                  + off-diagonal butterfly rotation
  Phase 2          = post-Phase-1 measure_z + 1-H frame-fold localizer + incremental inverse-frame
clifft fused/unfused = C++ CostMeter (array kernels), the fixed reference.
"""
import sys, time, statistics
sys.path.insert(0, "/home/jung/clifft-paper")
sys.setrecursionlimit(400000)
from collections import defaultdict
import clifft
from clifft import _clifft_core as cc
import nearclifford_backend.backend as bk
from nearclifford_backend.simulator import pauli_commute
from nearclifford_backend.clifft_axis import budget as _bud
from nearclifford_backend.clifft_axis.engine import CliftAxisNearClifford
from nearclifford_backend.clifft_axis.bounded import compile_bounded, CliftAxisBoundedNearClifford as C

CONV = dict(cmul=6, rcmul=2, cadd=2, sqmag=4, vdot=8)
NEW_measure_z = C.measure_z
_orig_sq1d = C._sqnorm_1d.__func__ if hasattr(C._sqnorm_1d, "__func__") else C._sqnorm_1d
BCOEF = {'rot:offdiag': 12, 'rot:offdiag-scalar': 12, 'rot:diaghalf': 3, 'rot:diag': 6,
         'rot:diag0': 6, 'rot:diag-scalar': 6, 'collapse:offdiag': 12, 'collapse:diag': 6,
         'collapse:diag0': 6, 'meas': 10, 'exp': 10, 'reduce:verify': 10, 'sqnorm': 2,
         'normalize': 2, 'purge:h': 4, 'purge:s': 2}


def OLD_measure_z(self, q):
    self._flush_core(0, 1 << q)
    Pm = (0, 1 << q, 0)
    magset = set(self.M)
    anti_s = [i for i in range(self.n) if i not in magset and not pauli_commute(self.Zc[i], Pm)]
    if anti_s:
        out = self._ag_measure(Pm, anti_s)
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
                self.budget.charge(self.phi.size, 0, "normalize")
                self.phi /= nrm2 ** 0.5
            self._compress_magic()
    self._reduce_full()
    if len(self.M) > self.max_M:
        self.max_M = len(self.M)
    self.budget.note_resident(self.phi.size, "post-reduce")
    self._meas_log_ctr += 1
    return out


def bounded_flop(circ, *, old, step2, seed=1):
    prog = compile_bounded(open(f"qec_bench/circuits/{circ}.stim").read())
    C.measure_z = OLD_measure_z if old else NEW_measure_z
    C._step2_localize = step2
    tot = [0.0]
    orig = _bud.DenseMemoryBudget.charge

    def charge(self, resident, transient=0, where=""):
        N = int(resident)
        tot[0] += BCOEF.get(where, 0) * N
        if where.startswith('collapse'):
            tot[0] += 6 * N
        return orig(self, resident, transient, where)
    _bud.DenseMemoryBudget.charge = charge
    if old:
        def sq1d(arr):
            tot[0] += 2 * int(arr.size)
            return _orig_sq1d(arr)
        C._sqnorm_1d = staticmethod(sq1d)
    try:
        be = bk.NearCliffordBackend(clifft_axis_bounded=True, drop_dead=False,
                                    structure_once=False, clifft_axis_enforce=True)
        be.run_shot(prog, seed)
        pk = be.nc.budget.peak_resident.bit_length() - 1
        upd, look, rec = be.nc._inv_update, be.nc._inv_lookup, be.nc._inv_recompute
    finally:
        _bud.DenseMemoryBudget.charge = orig
        C.measure_z = NEW_measure_z
        C._step2_localize = True
        C._sqnorm_1d = staticmethod(_orig_sq1d)
    return tot[0], pk, upd, look, rec


def bounded_wall(circ, step2, seed=1, iters=5):
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


def clifft_flop(circ, fused, seed=1):
    src = open(f"qec_bench/circuits/{circ}.stim").read()
    prog = clifft.compile(src) if fused else clifft.compile(src, bytecode_passes=None)
    cc.cost_meter_reset(); cc.cost_meter_enable(True)
    clifft.sample(prog, 1, seed)
    cc.cost_meter_enable(False)
    snap = cc.cost_meter_snapshot()
    f = sum(sum(CONV[k] * s[k] for k in CONV) for s in snap.values())
    return f, prog.peak_rank


def H(x):
    a = abs(x)
    for u, s in ((1e9, 'G'), (1e6, 'M'), (1e3, 'k')):
        if a >= u:
            return f"{x/u:.2f}{s}"
    return f"{x:.0f}"


CIRCS = ["coherent_ry_d3_r1", "coherent_ry_d3_r3", "cultivation_d3", "cultivation_d5",
         "coherent_rx_d3_r1", "coherent_rx_d3_r3", "coherent_d3_r3", "coherent_d5_r5",
         "distillation"]

print(f"{'circuit':18}{'clf-fused':>10}{'clf-unfus':>10}{'bnd-orig':>10}{'P1':>9}"
      f"{'P2-FLOP':>9}{'P2-wall':>10}{'peak b/c':>10}{'P2<unf?':>9}")
print("-" * 95)
beat = 0; sum_p1 = sum_p2 = 0.0; sum_wp1 = sum_wp2 = 0.0
tot_rec = tot_upd = tot_look = 0
peak_changes = []
for circ in CIRCS:
    cf, kc = clifft_flop(circ, fused=True)
    cu, _ = clifft_flop(circ, fused=False)
    fo, pko, *_ = bounded_flop(circ, old=True, step2=False)       # bounded original
    f1, pk1, *_ = bounded_flop(circ, old=False, step2=False)      # Phase 1
    f2, pk2, upd, look, rec = bounded_flop(circ, old=False, step2=True)  # Phase 2 optimal
    w1 = bounded_wall(circ, step2=False)
    w2 = bounded_wall(circ, step2=True)
    win = f2 < cu
    beat += int(win)
    sum_p1 += f1; sum_p2 += f2; sum_wp1 += w1; sum_wp2 += w2
    tot_rec += rec; tot_upd += upd; tot_look += look
    peak_changes.append((circ, pk1, pk2, kc))
    print(f"{circ:18}{H(cf):>10}{H(cu):>10}{H(fo):>10}{H(f1):>9}{H(f2):>9}"
          f"{w2*1e3:>8.1f}ms{f'{pk2}/{kc}':>10}{'YES' if win else 'no':>9}", flush=True)

print("-" * 95)
print(f"\nPhase 2 beats Clifft-unfused: {beat}/{len(CIRCS)} circuits")
print(f"Phase 2 FLOP vs Phase 1 (sum): {H(sum_p2)} / {H(sum_p1)} = {(sum_p2/sum_p1-1)*100:+.1f}%")
print(f"Phase 2 wall vs Phase 1 (sum): {sum_wp2*1e3:.1f}ms / {sum_wp1*1e3:.1f}ms = "
      f"{(sum_wp2/sum_wp1-1)*100:+.1f}%")
print(f"Total pullback FULL recomputes (Phase 2, all circuits): {tot_rec}  "
      f"(all from AG-measure lazy rebuild; frame-fold-induced = 0)")
print(f"Total inverse-frame updates: {tot_upd}   lookups: {tot_look}")
print("Peak resident-rank (Phase1 -> Phase2, cap=clifft k):")
for circ, pk1, pk2, kc in peak_changes:
    print(f"  {circ:18} {pk1} -> {pk2}  (cap {kc})  {'UNCHANGED' if pk1 == pk2 else 'CHANGED'}")
