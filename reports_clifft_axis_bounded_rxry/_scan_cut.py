"""Scan truncation points (TICK boundaries) of the coherent-only d3_r1 unitary prefix; at each,
append M on all qubits and compare bounded vs clifft per-qubit <Z>.  Finds the FIRST circuit
block after which the R_Y application makes the state diverge -> pinpoints the culprit context."""
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

lines = make_coherent_only(3, 1).split("\n")
def is_meas(s):
    t = s.strip().split("(")[0].split()[0] if s.strip() else ""
    return t in ("M", "MX", "MY", "MZ", "MR", "MRX", "MRY", "MRZ", "MPP")
first_meas = next(i for i, l in enumerate(lines) if is_meas(l))

# candidate cut points: after each line up to the first measurement
SH = 40000
prev = 0.0
print(f"{'cut':>4} {'last op':40} {'max|Δ<Z>|':>10}")
for cut in range(1, first_meas + 1):
    seg = lines[:cut]
    # only test cuts that end right after an R_Y or an entangling op, and have >=1 R_Y
    if not any(l.strip().startswith("R_Y") for l in seg):
        continue
    qubits = sorted({int(t) for l in seg for t in l.strip().split()[1:] if t.lstrip("-").isdigit()})
    prog = compile_bounded("\n".join(seg) + "\nM " + " ".join(map(str, qubits)) + "\n")
    cl = np.asarray(clifft.sample(prog, shots=SH, seed=5).measurements).mean(0)
    be = bk.NearCliffordBackend(clifft_axis_bounded=True, drop_dead=False,
                                structure_once=False, clifft_axis_enforce=True)
    bb = be.sample(prog, shots=SH, seed=7).mean(0)
    d = float(np.abs(cl - bb).max())
    jump = "  <== JUMP" if d - prev > 0.015 else ""
    print(f"{cut:>4} {lines[cut-1].strip()[:40]:40} {d:>10.4f}{jump}")
    prev = d
