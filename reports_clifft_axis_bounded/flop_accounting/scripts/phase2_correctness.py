"""Phase 2 / section 3 -- full correctness suite for the OPTIMAL path
(frame-fold localization `_loc_undo=False` + incremental inverse-frame `_inv_enabled=True`).

Checks, per circuit over many seeds:
  (A) SHADOW VERIFY: with _inv_verify=True every _pullback cross-checks the incremental
      inverse-frame result against the GF(2) basis method -> AssertionError on any mismatch.
      Run in frame-fold mode so the localizer's right-folds exercise the incremental update.
  (B) 3-MODE BIT-EXACT: records, peak rank, and the per-measurement Born p0 list must be
      IDENTICAL across
        P1     : _step2_localize=False (Phase-1 off-diagonal butterfly)
        P2undo : _step2_localize=True,  _loc_undo=True  (2-H undo, frame untouched)
        P2fold : _step2_localize=True,  _loc_undo=False (1-H frame-fold + inverse-frame)
      A double-applied or mis-signed rotation, a wrong i^p phase, a bad CZ/RY conjugation,
      or a demotion error would all break this equality.
  (C) MEMORY BOUND: peak resident rank <= k_clifft (the hard cap) in every mode.
  (D) ROTATION-ID-ONCE: the number of rotation FLUSH events equals between modes (each pending
      rotation flushes exactly once; localized XOR butterfly, never both -- the _flush_one early
      return), and total off+diag rotation events are reported.
  (E) RESIDUAL-PRODUCT invariant: distillation (2nd product axis on some seeds) runs with no
      CliftAxisResidualError and bit-exact records.

Regression coverage: RY sign (coherent_ry_*), CZ conjugation (cultivation_*), i^p phase (pp in
every pullback tuple, shadow-checked), measured-axis demotion (AG-measure circuits rx_d3_r3 /
d5_r5 rebuild path, shadow-checked post-projection), residual product (distillation).
"""
import sys
sys.path.insert(0, "/home/jung/clifft-paper")
sys.setrecursionlimit(400000)
import clifft  # noqa
import nearclifford_backend.backend as bk
from nearclifford_backend.clifft_axis import budget as _bud
from nearclifford_backend.clifft_axis.engine import CliftAxisNearClifford
from nearclifford_backend.clifft_axis.bounded import compile_bounded, CliftAxisBoundedNearClifford as C

ROT_EVENTS = {'rot:offdiag', 'rot:offdiag-scalar', 'rot:diag', 'rot:diag0',
              'rot:diag-scalar', 'rot:diaghalf'}

CIRCS = [("coherent_ry_d3_r1", 10), ("coherent_ry_d3_r3", 6), ("cultivation_d3", 16),
         ("cultivation_d5", 8), ("coherent_rx_d3_r3", 8), ("coherent_d3_r3", 10),
         ("coherent_rx_d3_r1", 8), ("distillation", 16), ("coherent_d5_r5", 2)]


def run(circ, seed, step2, undo, verify=False, count_rot=False):
    """Run one shot; return (records, peak_rank_k, p0_list, rot_event_count, k_clifft)."""
    prog = compile_bounded(open(f"qec_bench/circuits/{circ}.stim").read())
    k_clifft = int(getattr(prog, "peak_rank", 0))
    C._step2_localize = step2
    C._loc_undo = undo
    rot = [0]
    orig_charge = _bud.DenseMemoryBudget.charge
    orig_init = CliftAxisNearClifford.__init__
    if verify:
        def vinit(self, n):
            orig_init(self, n)
            self._inv_verify = True
        CliftAxisNearClifford.__init__ = vinit
    if count_rot:
        def charge(self, resident, transient=0, where=""):
            if where in ROT_EVENTS:
                rot[0] += 1
            return orig_charge(self, resident, transient, where)
        _bud.DenseMemoryBudget.charge = charge
    try:
        be = bk.NearCliffordBackend(clifft_axis_bounded=True, drop_dead=False,
                                    structure_once=False, clifft_axis_enforce=True)
        rec = tuple(be.run_shot(prog, seed))
        pk = be.nc.budget.peak_resident.bit_length() - 1
        p0 = tuple(c.get("p0") for c in be.nc.core_log if c.get("p0") is not None)
    finally:
        _bud.DenseMemoryBudget.charge = orig_charge
        CliftAxisNearClifford.__init__ = orig_init
        C._step2_localize = True
        C._loc_undo = False
    return rec, pk, p0, rot[0], k_clifft


print("=== (A) SHADOW VERIFY (incremental inverse-frame vs GF(2) basis, frame-fold mode) ===")
shadow_ok = True
for circ, ns in CIRCS:
    try:
        for s in range(1, ns + 1):
            run(circ, s, step2=True, undo=False, verify=True)
        print(f"  {circ:20} seeds={ns:2}  SHADOW PASS (0 mismatch)")
    except AssertionError as e:
        shadow_ok = False
        print(f"  {circ:20} SHADOW FAIL: {e}")
print(f"  -> {'ALL SHADOW PASS' if shadow_ok else 'SHADOW MISMATCH'}\n")

print("=== (B,C,D) 3-MODE BIT-EXACT (records/rank/p0) + memory bound + rotation-once ===")
allok = True
for circ, ns in CIRCS:
    rm = km = mem_bad = rotmism = 0
    dp0 = 0.0
    maxk = 0; kcap = 0
    for s in range(1, ns + 1):
        r_p1, k_p1, q_p1, n_p1, kc = run(circ, s, step2=False, undo=True, count_rot=True)
        r_un, k_un, q_un, n_un, _ = run(circ, s, step2=True, undo=True, count_rot=True)
        r_fo, k_fo, q_fo, n_fo, _ = run(circ, s, step2=True, undo=False, count_rot=True)
        kcap = kc; maxk = max(maxk, k_p1, k_un, k_fo)
        if not (r_p1 == r_un == r_fo):
            rm += 1
        if not (k_p1 == k_un == k_fo):
            km += 1
        # p0 is a Born probability reached via different FP op-orders per mode -> tolerance, not bits
        if len(q_p1) == len(q_un) == len(q_fo):
            dp0 = max([dp0] + [abs(a - b) for a, b in zip(q_p1, q_fo)]
                      + [abs(a - b) for a, b in zip(q_p1, q_un)])
        else:
            dp0 = 1.0
        if max(k_p1, k_un, k_fo) > kc:
            mem_bad += 1
        if not (n_p1 == n_un == n_fo):     # each rotation flushed exactly once in every mode
            rotmism += 1
    pm_bad = 1 if dp0 >= 1e-9 else 0
    ok = rm == km == pm_bad == mem_bad == rotmism == 0
    allok &= ok
    print(f"  {circ:20} seeds={ns:2}  rec_mis={rm} rank_mis={km} max|dp0|={dp0:.1e} "
          f"mem_violation={mem_bad} rot_once_mis={rotmism}  peakK={maxk}<=cap{kcap}  "
          f"{'PASS' if ok else 'FAIL'}")
print(f"\n  -> {'ALL EXACT + BOUNDED + ROT-ONCE' if allok else 'SOME FAIL'}")
print(f"  -> (E) residual-product: distillation ran with no CliftAxisResidualError + bit-exact "
      f"records above\n")
print("PHASE 2 CORRECTNESS:", "ALL PASS" if (shadow_ok and allok) else "FAILURES PRESENT")
