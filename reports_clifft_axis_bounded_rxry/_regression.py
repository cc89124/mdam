"""Regression proof for the R_Y flush-phase fix: (A) the fix is a strict no-op when the
pending flush phase is 0 -- instrument every _flush_one and count nonzero-phase flushes per
circuit (R_Z / R_X must be 0 -> bit-identical to pre-fix); (B) R_Z canonical + R_X d3 still
AGREE with clifft within the null baseline; (C) the strict-memory rank trace is unchanged
(max_M per circuit)."""
import sys
sys.path.insert(0, "/home/jung/clifft-paper")
sys.setrecursionlimit(400000)
import numpy as np
import clifft
import nearclifford_backend.backend as bk
from nearclifford_backend.clifft_axis import engine as eng_mod
from nearclifford_backend.clifft_axis.bounded import compile_bounded

# (A) instrument _flush_one to record the phase argument actually used
_orig = eng_mod.CliftAxisNearClifford._flush_one
PHASES = {"nonzero": 0, "total": 0}
def _instr(self, x, z, theta, phase=0):
    PHASES["total"] += 1
    if phase & 3:
        PHASES["nonzero"] += 1
    return _orig(self, x, z, theta, phase)
eng_mod.CliftAxisNearClifford._flush_one = _instr

RZ = ['coherent_d3_r1', 'coherent_d3_r3', 'cultivation_d3', 'distillation', 'surface_d7_r7']
RX = ['coherent_rx_d3_r1', 'coherent_rx_d3_r3']
RY = ['coherent_ry_d3_r1', 'coherent_ry_d3_r3']

def run(circ, compile_fn):
    PHASES["nonzero"] = PHASES["total"] = 0
    prog = compile_fn(open(f'qec_bench/circuits/{circ}.stim').read())
    be = bk.NearCliffordBackend(clifft_axis_bounded=True, drop_dead=False,
                                structure_once=False, clifft_axis_enforce=True)
    be.run_shot(prog, 1)
    return PHASES["nonzero"], PHASES["total"], be.nc.max_M

print("=== (A) nonzero-phase flushes per circuit (R_Z/R_X must be 0 => fix is no-op) ===")
print(f"{'circuit':20}{'compile':10}{'nonzero/total flush':>22}{'max_M':>7}")
for c in RZ:
    nz, tot, mm = run(c, clifft.compile)
    print(f"{c:20}{'default':10}{f'{nz}/{tot}':>22}{mm:>7}   {'NO-OP' if nz==0 else 'CHANGED!'}")
for c in RX:
    nz, tot, mm = run(c, compile_bounded)
    print(f"{c:20}{'bounded':10}{f'{nz}/{tot}':>22}{mm:>7}   {'NO-OP' if nz==0 else 'CHANGED!'}")
for c in RY:
    nz, tot, mm = run(c, compile_bounded)
    print(f"{c:20}{'bounded':10}{f'{nz}/{tot}':>22}{mm:>7}   {'(R_Y uses phase)' if nz else ''}")

# (B) distributional AGREE for R_Z + R_X (fast circuits) with null baseline
print("\n=== (B) distributional regression (R_Z/R_X) vs clifft, null baseline, 5000 shots ===")
SH = 5000
for c, cf in [(x, clifft.compile) for x in ['coherent_d3_r1', 'coherent_d3_r3', 'cultivation_d3', 'distillation']] \
            + [(x, compile_bounded) for x in RX]:
    prog = cf(open(f'qec_bench/circuits/{c}.stim').read())
    g1 = np.asarray(clifft.sample(prog, shots=SH, seed=11).measurements).mean(0)
    g2 = np.asarray(clifft.sample(prog, shots=SH, seed=22).measurements).mean(0)
    null = np.abs(g1 - g2).max()
    be = bk.NearCliffordBackend(clifft_axis_bounded=True, drop_dead=False,
                                structure_once=False, clifft_axis_enforce=True)
    bb = be.sample(prog, shots=SH, seed=33).mean(0)
    d = np.abs(g1 - bb).max()
    print(f"  {c:20} null={null:.4f}  bounded-vs-clifft={d:.4f}  ratio={d/null:.2f}  "
          f"{'PASS' if d <= null*1.6 else 'FAIL'}")

eng_mod.CliftAxisNearClifford._flush_one = _orig
