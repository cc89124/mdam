"""FLOP accounting for the clifft_axis_bounded engine, via the EXISTING budget.charge hook
(no engine edits).  Compares bounded FLOPs against a clifft dense-2^k baseline.

Method
------
Every dense magic-register kernel already calls `DenseMemoryBudget.charge(resident, transient,
where)` with `resident = phi.size` and a label.  We monkeypatch `charge` to ALSO accumulate
FLOPs, in the repo convention (flop_meter.py): complex mult/scale = 6, complex add/sub = 2,
vdot = 8, norm = 4 -- per element.

Per-label FLOP/element (derived from each kernel's arithmetic):
  rot/collapse :offdiag(-scalar)  16  (2x2 butterfly: 2 outputs x (alpha*a 6 + bph*sk*b 8 + 2))
  rot/collapse :diag/:diag0/:diag-scalar  6   (one complex scalar mult/element)
  meas        (<phi|P|phi> expectation)   8   (vdot-like conj*gather sum)
  sqnorm                                   4   (norm scan)
  collapse -> implied np.linalg.norm       4   (post-collapse renorm; not separately charged)
  promote                                  0   (zero-fill growth = memory, not arithmetic)
  reduce:gf2scan 2 / reduce:cnot 1         (localize-and-drop parity reduction -- bounded only)
  purge:h 6 / purge:s 6 / purge:cnot 0     (measured-magic purge -- bounded only)
  drop                                     2   (disentangled-axis compress -- bounded only)

clifft baseline: clifft holds its active 2^k array and performs the SAME rotations / Born
collapses / norms on it, but NEVER the localize-and-drop overhead (promote/reduce/purge/drop).
So for every clifft-SHARED event (prefix rot|collapse|meas|sqnorm) we charge `coeff * cap`
(cap = 2^k_clifft, known to the budget object); bounded-only events are excluded from it.
bounded charges `coeff * resident` (the materialised core size = the localize-and-drop win).
"""
import sys
sys.path.insert(0, "/home/jung/clifft-paper")
sys.setrecursionlimit(400000)
import numpy as np
import nearclifford_backend.backend as bk
from nearclifford_backend.clifft_axis import budget as _bud
from nearclifford_backend.clifft_axis.bounded import compile_bounded

# ---- per-element FLOP coeff + bucket by kernel-suffix; clifft-shared by prefix ----
def classify(where):
    """return (coeff_per_element, bucket, clifft_shares)."""
    pre = where.split(":", 1)[0]
    shared = pre in ("rot", "collapse", "meas", "sqnorm")
    if where in ("meas",):                       return 8.0, "vdot_norm", True
    if where in ("sqnorm",):                     return 4.0, "vdot_norm", True
    if where == "promote":                       return 0.0, "elementwise", False
    if where == "drop":                          return 2.0, "elementwise", False
    if where == "reduce:gf2scan":                return 2.0, "elementwise", False
    if where == "reduce:cnot":                   return 1.0, "elementwise", False
    if where == "purge:h":                       return 6.0, "apply_kron", False
    if where == "purge:s":                       return 6.0, "apply_kron", False
    if where == "purge:cnot":                    return 0.0, "apply_kron", False
    suf = where.split(":", 1)[1] if ":" in where else ""
    if suf.startswith("offdiag"):                return 16.0, "apply_kron", shared
    if suf.startswith("diag"):                   return 6.0, "elementwise", shared
    return 0.0, "elementwise", shared            # init / post-reduce / unknown -> 0


class FlopMeter:
    BUCKETS = ("apply_kron", "vdot_norm", "elementwise")

    def __init__(self):
        self.reset()

    def reset(self):
        self.bnd = {b: 0.0 for b in self.BUCKETS}
        self.cl = {b: 0.0 for b in self.BUCKETS}
        self.cat = {}                       # category (rot/collapse/meas/reduce/...) -> bounded flops
        self.n_ev = 0

    def record(self, resident, where, cap):
        coeff, bucket, shared = classify(where)
        f_b = coeff * resident
        self.bnd[bucket] += f_b
        cat = where.split(":", 1)[0]
        self.cat[cat] = self.cat.get(cat, 0.0) + f_b
        if shared:
            self.cl[bucket] += coeff * cap
        if where.startswith("collapse"):    # implied post-collapse np.linalg.norm (4N), not charged
            self.bnd["vdot_norm"] += 4.0 * resident
            self.cat["collapse"] = self.cat.get("collapse", 0.0) + 4.0 * resident
            self.cl["vdot_norm"] += 4.0 * cap
        self.n_ev += 1

    def tot_bounded(self): return sum(self.bnd.values())
    def tot_clifft(self):  return sum(self.cl.values())


METER = FlopMeter()
_orig_charge = _bud.DenseMemoryBudget.charge


def _hook_charge(self, resident, transient=0, where=""):
    METER.record(int(resident), where, self.cap)
    return _orig_charge(self, resident, transient, where)


def enable():
    _bud.DenseMemoryBudget.charge = _hook_charge


def disable():
    _bud.DenseMemoryBudget.charge = _orig_charge


def run_circuit(stim_path, seed=1):
    prog = compile_bounded(open(stim_path).read())
    METER.reset()
    be = bk.NearCliffordBackend(clifft_axis_bounded=True, drop_dead=False,
                                structure_once=False, clifft_axis_enforce=True)
    enable()
    try:
        be.run_shot(prog, seed)
    finally:
        disable()
    return dict(bnd=dict(METER.bnd), cl=dict(METER.cl), cat=dict(METER.cat),
                tot_b=METER.tot_bounded(), tot_c=METER.tot_clifft(),
                k=prog.peak_rank, n_ev=METER.n_ev)


def fmt(x):
    for u, s in ((1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "k")):
        if abs(x) >= u:
            return f"{x/u:.2f}{s}"
    return f"{x:.0f}"


if __name__ == "__main__":
    # ---- sanity: a tiny rotation+measure ----
    print("=== SANITY: per-event FLOP on a tiny rotation+measure ===")
    import tempfile, os
    tmp = tempfile.NamedTemporaryFile("w", suffix=".stim", delete=False)
    tmp.write("R_Y(0.2) 0\nR_Y(0.1) 1\nCX 0 1\nM 0 1\n"); tmp.close()
    s = run_circuit(tmp.name, 1); os.unlink(tmp.name)
    print(f"  tiny RY+CX+M: events={s['n_ev']}  bounded={fmt(s['tot_b'])}  "
          f"clifft(2^{s['k']})={fmt(s['tot_c'])}  by-category={ {k:fmt(v) for k,v in s['cat'].items()} }")

    CIRCS = [("coherent_d3_r1", "R_Z"), ("coherent_rx_d3_r1", "R_X"), ("coherent_ry_d3_r1", "R_Y"),
             ("coherent_d3_r3", "R_Z"), ("coherent_rx_d3_r3", "R_X"), ("coherent_ry_d3_r3", "R_Y"),
             ("cultivation_d3", "T"), ("distillation", "T")]
    print(f"\n=== FLOP: bounded vs clifft dense-2^k baseline (1 shot, seed 1) ===")
    print(f"{'circuit':20}{'noise':6}{'k':>4}{'bounded FLOP':>14}{'clifft 2^k FLOP':>16}"
          f"{'bnd/clifft':>11}{'  buckets(ak/vn/el) bounded'}")
    for c, ax in CIRCS:
        s = run_circuit(f"qec_bench/circuits/{c}.stim", 1)
        ratio = s["tot_b"] / s["tot_c"] if s["tot_c"] else float("nan")
        bk_ = s["bnd"]
        print(f"{c:20}{ax:6}{s['k']:>4}{fmt(s['tot_b']):>14}{fmt(s['tot_c']):>16}{ratio:>11.3f}"
              f"   {fmt(bk_['apply_kron'])}/{fmt(bk_['vdot_norm'])}/{fmt(bk_['elementwise'])}")
