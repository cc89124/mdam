"""Step B1 FLOP: Policy-3 persistent-split engine vs bounded butterfly (a05843e) vs clifft-unfused.
Reuses the Phase-3 cost model (BCOEF per `where`, clifft CostMeter). Policy-3 reuses the same kernels
so its diagonal (rot:diaghalf c=3), born-H (purge:h c=4) and fallback (rot:offdiag c=12) are captured
automatically by the `where` labels."""
import sys; sys.path.insert(0, "/home/jung/clifft-paper"); sys.setrecursionlimit(400000)
from collections import defaultdict
import clifft
from clifft import _clifft_core as cc
import nearclifford_backend.backend as bk
from nearclifford_backend.clifft_axis import budget as _bud
from nearclifford_backend.clifft_axis.bounded import compile_bounded

BCOEF = {'rot:offdiag': 12, 'rot:offdiag-scalar': 12, 'rot:diaghalf': 3, 'rot:diag': 6,
         'rot:diag0': 6, 'rot:diag-scalar': 6, 'collapse:offdiag': 12, 'collapse:diag': 6,
         'collapse:diag0': 6, 'meas': 10, 'exp': 10, 'reduce:verify': 10, 'sqnorm': 2,
         'normalize': 2, 'purge:h': 4, 'purge:s': 2, 'purge:cnot': 0, 'reduce:cnot': 0,
         'reduce:gf2scan': 0, 'drop': 0, 'promote': 0, 'init': 0, 'post-reduce': 0}
CONV = {'cmul': 6, 'rcmul': 2, 'cadd': 2, 'sqmag': 4, 'vdot': 8}

CIRCS = ["cultivation_d5", "cultivation_d3", "coherent_d5_r5", "coherent_ry_d3_r1", "distillation"]


def clifft_flop(circ, seed=1):
    prog = clifft.compile(open(f"qec_bench/circuits/{circ}.stim").read(), bytecode_passes=None)
    cc.cost_meter_reset(); cc.cost_meter_enable(True)
    clifft.sample(prog, 1, seed)
    cc.cost_meter_enable(False)
    snap = cc.cost_meter_snapshot()
    return sum(sum(CONV[k] * s[k] for k in CONV) for s in snap.values()), prog.peak_rank


def bounded_flop(circ, policy3, seed=1):
    prog = compile_bounded(open(f"qec_bench/circuits/{circ}.stim").read())
    agg = defaultdict(int)
    orig = _bud.DenseMemoryBudget.charge

    def charge(self, resident, transient=0, where=""):
        agg[where] += int(resident)
        return orig(self, resident, transient, where)
    _bud.DenseMemoryBudget.charge = charge
    try:
        be = bk.NearCliffordBackend(clifft_axis_bounded=True, drop_dead=False, structure_once=False,
                                    clifft_axis_enforce=True, clifft_axis_policy3=policy3)
        be.run_shot(prog, seed)
        pk = be.nc.budget.peak_resident.bit_length() - 1
        diag = getattr(be.nc, "_p3_diag", 0); fb = getattr(be.nc, "_p3_fallback", 0)
        born = getattr(be.nc, "_p3_bornH", 0)
    finally:
        _bud.DenseMemoryBudget.charge = orig
    flop = sum(BCOEF.get(w, 0) * s for w, s in agg.items())
    rot = {w: BCOEF.get(w, 0) * s for w, s in agg.items() if w.startswith("rot") or w == "purge:h"}
    return flop, pk, diag, fb, born, rot


def H(x):
    a = abs(x)
    for u, s in ((1e9, 'G'), (1e6, 'M'), (1e3, 'k')):
        if a >= u:
            return f"{x/u:.2f}{s}"
    return f"{x:.0f}"


print("=" * 96)
print("STEP B1 FLOP -- Policy-3 (born-X + diagonal dispatch) vs bounded butterfly vs clifft-unfused")
print("=" * 96)
print(f"{'circuit':16}{'clifft':>10}{'bounded':>10}{'policy3':>10}{'p3/clf':>8}{'p3/bnd':>8}"
      f"{'diag/flush':>12}{'bornH':>7}")
for circ in CIRCS:
    fc, kc = clifft_flop(circ)
    fb, kb, _, _, _, rb = bounded_flop(circ, False)
    fp, kp, diag, fbk, born, rp = bounded_flop(circ, True)
    tot = diag + fbk
    print(f"{circ:16}{H(fc):>10}{H(fb):>10}{H(fp):>10}{fp/fc:>8.2f}{fp/fb:>8.2f}"
          f"{f'{diag}/{tot}':>12}{born:>7}")
print("-" * 96)
print("rot+bornH breakdown (FLOP) -- bounded vs policy3, last circuit's detail:")
for w in sorted(set(list(rb) + list(rp))):
    print(f"   {w:18} bounded={H(rb.get(w,0)):>9}   policy3={H(rp.get(w,0)):>9}")
