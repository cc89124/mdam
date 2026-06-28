"""Find the SMALLEST circuit where the bounded backend's R_Y marginal diverges from clifft,
exercising syndrome-style readouts (X-stabiliser / interference) of an R_Y'd qubit.  Uses
clifft as the trusted reference (exact-ish at 40k shots)."""
import sys
import numpy as np
sys.path.insert(0, "/home/jung/clifft-paper")
sys.setrecursionlimit(400000)
import clifft
import nearclifford_backend.backend as bk
from nearclifford_backend.clifft_axis.bounded import compile_bounded

SH = 40000
TH = 0.6                       # use a not-tiny angle so any O(theta) loss is visible

CIRCS = {
 # Z-stabiliser readout of RY'd data (ancilla copies data Z-parity)
 "Z-stab":  f"R 0 1\nR_Y({TH}) 0\nCX 0 1\nM 1\n",
 # X-stabiliser readout: H anc, CX(anc->data), H anc, M anc  (measures data X-parity)
 "X-stab(CX)": f"R 0 1\nR_Y({TH}) 0\nH 1\nCX 1 0\nH 1\nM 1\n",
 # X-stabiliser via CZ
 "X-stab(CZ)": f"R 0 1\nR_Y({TH}) 0\nH 1\nCZ 1 0\nH 1\nM 1\n",
 # measure the RY'd qubit itself in X basis (interfere)
 "self-X":   f"R 0\nR_Y({TH}) 0\nH 0\nM 0\n",
 # measure RY'd qubit in Y basis (S^dag H then Z)
 "self-Y":   f"R 0\nR_Y({TH}) 0\nS_DAG 0\nH 0\nM 0\n",
 # two RY'd data into one ancilla X-parity
 "XX-stab":  f"R 0 1 2\nR_Y({TH}) 0\nR_Y({TH}) 1\nH 2\nCX 2 0\nCX 2 1\nH 2\nM 2\n",
 # RY between two syndrome rounds (data measured at end in Z after a round)
 "round+final": f"R 0 1\nR_Y({TH}) 0\nH 1\nCZ 1 0\nH 1\nM 1\nR_Y({TH}) 0\nM 0\n",
}

for name, src in CIRCS.items():
    try:
        prog = compile_bounded(src)
        nm = prog.num_measurements
        cl = np.asarray(clifft.sample(prog, shots=SH, seed=5).measurements).mean(0)
        be = bk.NearCliffordBackend(clifft_axis_bounded=True, drop_dead=False,
                                    structure_once=False, clifft_axis_enforce=True)
        bb = be.sample(prog, shots=SH, seed=7).mean(0)
        d = np.abs(cl - bb)
        tag = "  <== REPRODUCES" if d.max() > 0.02 else ""
        print(f"{name:14} nm={nm}  clifft={np.array2string(cl,precision=4)}  "
              f"bounded={np.array2string(bb,precision=4)}  max|Δ|={d.max():.4f}{tag}")
    except Exception as e:
        print(f"{name:14} ERROR {type(e).__name__}: {str(e)[:50]}")
