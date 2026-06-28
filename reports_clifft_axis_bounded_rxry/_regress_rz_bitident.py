"""No-regression PROOF for the CZ fix on R_Z / Clifford circuits: the new engine.cz
(h.cx.h, conjugate pending exactly once) must be BIT-IDENTICAL to the old lazy.cz
(super().cz + manual _conj loop = double conjugation) on diagonal-pending circuits, where
double-conjugation of a Z pending is a true no-op.  Compares full records over several seeds."""
import sys
import numpy as np
sys.path.insert(0, "/home/jung/clifft-paper")
sys.setrecursionlimit(400000)
import nearclifford_backend.backend as bk
from nearclifford_backend.clifft_axis.bounded import CliftAxisBoundedNearClifford as B, compile_bounded
from nearclifford_backend.lazy import LazyNearClifford

CIRCS = ["coherent_d3_r1", "coherent_d3_r3", "cultivation_d3", "distillation", "surface_d7_r7"]
SEEDS = [1, 7, 42, 123, 999]

new_cz = B.cz                       # the fixed engine.cz (h.cx.h)
old_cz = LazyNearClifford.cz        # the pre-fix double-conjugating cz


def run(circ, seed):
    prog = compile_bounded(open(f"qec_bench/circuits/{circ}.stim").read())
    be = bk.NearCliffordBackend(clifft_axis_bounded=True, drop_dead=False,
                                structure_once=False, clifft_axis_enforce=True)
    rec = be.run_shot(prog, seed)
    nm = prog.num_measurements
    return tuple(int(rec.get(i, 0)) for i in range(nm)), be.last_max_M


allok = True
print(f"{'circuit':16}{'seed':>6}  {'records bit-identical':>22}  {'max_M new/old':>14}")
for circ in CIRCS:
    for sd in SEEDS:
        B.cz = new_cz
        r_new, m_new = run(circ, sd)
        B.cz = old_cz
        r_old, m_old = run(circ, sd)
        B.cz = new_cz
        ok = (r_new == r_old)
        allok &= ok
        print(f"{circ:16}{sd:>6}  {'YES' if ok else 'NO  <== DIFF':>22}  {m_new:>6}/{m_old:<7}")
print(f"\nR_Z/Clifford CZ-fix BIT-IDENTICAL across all circuits/seeds: "
      f"{'PASS (no-regression)' if allok else 'FAIL'}")
