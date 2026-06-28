"""Isolate the dominant R_Y bias on a PURE-STATE (no stochastic flip) surface-code circuit:
only the coherent R_Y(0.02) after each Clifford, all flip/reset/round Pauli noise = 0.  Then
the circuit is deterministic and the bounded backend must match an EXACT dense statevector
oracle (clifft.sample as ground truth; clifft is exact for this).  Finds the FIRST measurement
whose Born marginal diverges -> the first physical-state divergence."""
import sys
import numpy as np
sys.path.insert(0, "/home/jung/clifft-paper")
sys.setrecursionlimit(400000)
import stim, re
import clifft
import nearclifford_backend.backend as bk
from nearclifford_backend.clifft_axis.bounded import compile_bounded

def make_coherent_only(d, r, ang=0.02):
    """surface code with ONLY after_clifford_depolarization -> R_Y(ang); no flip/reset/round
    noise -> the only non-Clifford content is the coherent R_Y, state stays pure."""
    c = stim.Circuit.generated("surface_code:rotated_memory_z", rounds=r, distance=d,
                               after_clifford_depolarization=1e-3,   # -> R_Y sites
                               after_reset_flip_probability=0.0,
                               before_measure_flip_probability=0.0,
                               before_round_data_depolarization=0.0)
    out = []
    for line in str(c).split("\n"):
        s = line.strip()
        if s.startswith("DEPOLARIZE1(") or s.startswith("DEPOLARIZE2("):
            out.append(f"R_Y({ang}) {s.split(')')[1].strip()}")
        else:
            out.append(line)
    return "\n".join(out)

src = make_coherent_only(3, 1)
prog = compile_bounded(src)
nm = prog.num_measurements
print(f"coherent-only d3_r1: {nm} measurements, pure state (no stochastic noise)")

SH = 60000
cl = np.asarray(clifft.sample(prog, shots=SH, seed=5).measurements).mean(0)
cl2 = np.asarray(clifft.sample(prog, shots=SH, seed=6).measurements).mean(0)
be = bk.NearCliffordBackend(clifft_axis_bounded=True, drop_dead=False,
                            structure_once=False, clifft_axis_enforce=True)
bb = be.sample(prog, shots=SH, seed=7).mean(0)
null = np.abs(cl - cl2).max()
print(f"clifft-vs-clifft null max = {null:.4f}")
print(f"{'meas':>4} {'clifft':>8} {'bounded':>8} {'signed Δ':>10}")
worst = 0.0
for i in range(nm):
    dd = bb[i] - cl[i]
    if abs(dd) > 0.02:
        print(f"{i:>4} {cl[i]:>8.4f} {bb[i]:>8.4f} {dd:>+10.4f}  <== diverges")
    worst = max(worst, abs(dd))
print(f"max|bounded-clifft| = {worst:.4f}   {'BIAS PRESENT' if worst > 3*null+0.01 else 'matches (bias needs stochastic noise)'}")
