"""Bisect bug #2 into rotation-application vs measurement-handling: truncate the coherent-only
d3_r1 BEFORE the first measurement (keep R + R_Y + syndrome Cliffords), append a Z-measurement
on every qubit, and compare bounded vs clifft per-qubit marginals.  If a qubit's <Z> already
diverges here, the bug is in the (active) R_Y APPLICATION before any measurement; if they all
match, the bug is in the MEASUREMENT/collapse handling."""
import sys
import numpy as np
sys.path.insert(0, "/home/jung/clifft-paper")
sys.setrecursionlimit(400000)
import stim
import clifft
import nearclifford_backend.backend as bk
from nearclifford_backend.clifft_axis.bounded import compile_bounded

def make_coherent_only(d, r, ang=0.02):
    c = stim.Circuit.generated("surface_code:rotated_memory_z", rounds=r, distance=d,
                               after_clifford_depolarization=1e-3, after_reset_flip_probability=0.0,
                               before_measure_flip_probability=0.0, before_round_data_depolarization=0.0)
    out = []
    for l in str(c).split("\n"):
        s = l.strip()
        if s.startswith(("DETECTOR", "OBSERVABLE_INCLUDE", "SHIFT_COORDS", "QUBIT_COORDS")):
            continue
        out.append(f"R_Y({ang}) {s.split(')')[1].strip()}"
                   if s.startswith(("DEPOLARIZE1(", "DEPOLARIZE2(")) else l)
    return "\n".join(out)

full = make_coherent_only(3, 1)
lines = full.split("\n")
# find the first measurement instruction (M / MX / MR ...) and truncate before it
def is_meas(s):
    t = s.strip().split("(")[0].split()[0] if s.strip() else ""
    return t in ("M", "MX", "MY", "MZ", "MR", "MRX", "MRY", "MRZ", "MPP")
cut = next(i for i, l in enumerate(lines) if is_meas(l))
print(f"first measurement at line {cut}: {lines[cut].strip()!r}; truncating before it")

# determine qubit set
nq = stim.Circuit("\n".join(l for l in lines if not l.strip().startswith("R_Y"))
                  .replace("R_Y", "DEPOLARIZE1")).num_qubits  # crude qubit count
qubits = sorted({int(t) for l in lines[:cut] for t in l.strip().split()[1:]
                 if t.lstrip("-").isdigit()})
prefix = "\n".join(lines[:cut]) + "\nM " + " ".join(map(str, qubits)) + "\n"

prog = compile_bounded(prefix)
nm = prog.num_measurements
SH = 60000
cl = np.asarray(clifft.sample(prog, shots=SH, seed=5).measurements).mean(0)
cl2 = np.asarray(clifft.sample(prog, shots=SH, seed=6).measurements).mean(0)
be = bk.NearCliffordBackend(clifft_axis_bounded=True, drop_dead=False,
                            structure_once=False, clifft_axis_enforce=True)
bb = be.sample(prog, shots=SH, seed=7).mean(0)
null = np.abs(cl - cl2).max()
print(f"pre-measurement Z-marginals of {nm} qubits; clifft null={null:.4f}")
worst = 0.0
for i in range(nm):
    d = abs(bb[i] - cl[i])
    worst = max(worst, d)
    if d > 0.02:
        print(f"  qubit#{i:2d}: clifft={cl[i]:.4f} bounded={bb[i]:.4f} Δ={bb[i]-cl[i]:+.4f}  <== R_Y APPLICATION already wrong")
print(f"max|Δ| = {worst:.4f}  -> {'ROTATION-APPLICATION bug (pre-measurement)' if worst>3*null+0.01 else 'pre-measurement state CORRECT -> bug is in MEASUREMENT handling'}")
