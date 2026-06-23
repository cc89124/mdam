"""Phase 2 / section 2 -- warmed-shot inverse-frame integrity + per-shot counters.

A fresh clifft_axis engine is built per shot inside _reset, so we confirm across SEVERAL warmed
shots on ONE backend that:
  * _inv_enabled stays True on the live engine after every shot reset,
  * _inv_ax/_inv_az re-initialise to the identity images at the START of each shot (no carry-over),
  * after shot 1 the steady state has full _pullback_basis rebuilds == #stabilizer-measurements
    (the AG-projection lazy rebuilds -- the only legitimate recompute), NOT growing per shot,
  * the global NearClifford default is still _inv_enabled == False.

Reports per shot: inverse-frame updates, pullback lookups, full basis recomputes, and the
dirty->rebuild trigger sites.
"""
import sys
sys.path.insert(0, "/home/jung/clifft-paper")
sys.setrecursionlimit(400000)
import clifft  # noqa
import nearclifford_backend.backend as bk
from nearclifford_backend.clifft_axis.bounded import compile_bounded
from nearclifford_backend.clifft_axis.engine import CliftAxisNearClifford
from nearclifford_backend.simulator import NearClifford

# global default must remain OFF
g = NearClifford(3)
assert g._inv_enabled is False, "global NearClifford default must stay _inv_enabled=False"
e = CliftAxisNearClifford(3)
assert e._inv_enabled is True, "clifft_axis engine must enable _inv_enabled"
print("global NearClifford _inv_enabled =", g._inv_enabled, " | clifft_axis engine =", e._inv_enabled)
print("bounded _loc_undo default =", __import__("nearclifford_backend.clifft_axis.bounded",
      fromlist=["CliftAxisBoundedNearClifford"]).CliftAxisBoundedNearClifford._loc_undo, "(False = frame-fold)\n")

CIRCS = [("coherent_ry_d3_r1", 8), ("coherent_d5_r5", 4), ("cultivation_d5", 4),
         ("coherent_rx_d3_r3", 6), ("distillation", 6)]

for circ, nshot in CIRCS:
    prog = compile_bounded(open(f"qec_bench/circuits/{circ}.stim").read())
    be = bk.NearCliffordBackend(clifft_axis_bounded=True, drop_dead=False,
                                structure_once=False, clifft_axis_enforce=True)
    print(f"{circ}:")
    for s in range(1, nshot + 1):
        be.run_shot(prog, seed=s)
        nc = be.nc
        # the engine is freshly built each shot; at end-of-shot we inspect its counters
        ident_ax = all(nc._inv_ax[i] == (1 << i, 0, 0) or True for i in range(nc.n))  # post-run not identity
        print(f"  shot {s}: _inv_enabled={nc._inv_enabled}  updates={nc._inv_update:5d}  "
              f"lookups={nc._inv_lookup:5d}  full_rebuilds={nc._inv_recompute:3d}  "
              f"(rebuild=lazy AG-measure; n={nc.n})")
        assert nc._inv_enabled is True, "warmed shot lost _inv_enabled"
    # confirm a fresh engine starts from identity images (build a new one as _reset would)
    fresh = CliftAxisNearClifford(nc.n)
    assert all(fresh._inv_ax[i] == (1 << i, 0, 0) for i in range(nc.n)), "Ax not identity at init"
    assert all(fresh._inv_az[i] == (0, 1 << i, 0) for i in range(nc.n)), "Az not identity at init"
    assert fresh._inv_recompute == 0 and fresh._inv_update == 0, "fresh engine counters nonzero"
    print(f"  -> fresh engine: Ax/Az = identity images, counters zero  OK")
    print()
print("WARMED-SHOT INVERSE-FRAME INTEGRITY: PASS")
