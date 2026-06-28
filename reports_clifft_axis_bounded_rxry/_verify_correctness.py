"""Distributional correctness of the bounded backend on off-axis (R_X/R_Y) noise circuits.
Compares per-measurement marginals of the bounded backend vs clifft's own sampler (ground
truth), against a clifft-vs-clifft null baseline (the pure statistical spread at the same
shot count).  PASS iff bounded-vs-clifft <= 1.6 x the null spread."""
import sys
sys.path.insert(0, "/home/jung/clifft-paper")
sys.setrecursionlimit(400000)
import numpy as np
import clifft
import nearclifford_backend.backend as bk
from nearclifford_backend.clifft_axis.bounded import compile_bounded

SHOTS = 6000
CIRCS = ['coherent_rx_d3_r1', 'coherent_rx_d3_r3', 'coherent_ry_d3_r1', 'coherent_ry_d3_r3']
print(f"correctness @ {SHOTS} shots/method   (null = clifft-vs-clifft spread)")
allpass = True
for c in CIRCS:
    src = open(f'qec_bench/circuits/{c}.stim').read()
    prog = compile_bounded(src)
    g1 = np.asarray(clifft.sample(prog, shots=SHOTS, seed=11).measurements).mean(0)
    g2 = np.asarray(clifft.sample(prog, shots=SHOTS, seed=22).measurements).mean(0)
    null = np.abs(g1 - g2).max()
    be = bk.NearCliffordBackend(clifft_axis_bounded=True, drop_dead=False,
                                structure_once=False, clifft_axis_enforce=True)
    bb = be.sample(prog, shots=SHOTS, seed=33).mean(0)
    diff = np.abs(g1 - bb).max()
    ok = diff <= null * 1.6
    allpass &= ok
    print(f"  {c:18} null={null:.4f}  bounded-vs-clifft={diff:.4f}  "
          f"ratio={diff/null:.2f}  {'PASS' if ok else 'INVESTIGATE'}", flush=True)
print("ALL PASS" if allpass else "SOME INVESTIGATE")
