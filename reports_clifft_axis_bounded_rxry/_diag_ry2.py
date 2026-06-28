"""Localize the R_Y bias: clifft (truth) vs bounded backend, per-measurement marginals.
(The default unbounded NC backend is omitted -- it blows up on off-axis R_Y.)  We look for
which measurement indices carry a systematic |bounded-clifft| >> the clifft-vs-clifft noise."""
import sys
sys.path.insert(0, "/home/jung/clifft-paper")
sys.setrecursionlimit(400000)
import numpy as np
import clifft
import nearclifford_backend.backend as bk
from nearclifford_backend.clifft_axis.bounded import compile_bounded

SHOTS = 12000
for c in ['coherent_ry_d3_r1', 'coherent_rx_d3_r1']:
    src = open(f'qec_bench/circuits/{c}.stim').read()
    prog = compile_bounded(src)
    g = np.asarray(clifft.sample(prog, shots=SHOTS, seed=11).measurements).mean(0)
    g2 = np.asarray(clifft.sample(prog, shots=SHOTS, seed=99).measurements).mean(0)
    be = bk.NearCliffordBackend(clifft_axis_bounded=True, drop_dead=False,
                                structure_once=False, clifft_axis_enforce=True)
    bb = be.sample(prog, shots=SHOTS, seed=33).mean(0)
    noise = np.abs(g - g2).max()
    print(f"\n=== {c}  ({SHOTS} shots)   clifft-noise-max={noise:.4f} ===")
    print(f"{'meas':>4} {'clifft':>8} {'bounded':>8} {'signed b-cl':>12}")
    for i in range(len(g)):
        d = bb[i] - g[i]
        flag = "  <== BIAS" if abs(d) > 3 * noise + 0.01 else ""
        if abs(d) > 0.02:
            print(f"{i:>4} {g[i]:>8.4f} {bb[i]:>8.4f} {d:>+12.4f}{flag}")
    print(f"bounded-vs-clifft max={np.abs(bb-g).max():.4f}  "
          f"mean={np.abs(bb-g).mean():.4f}  n_meas={len(g)}")
