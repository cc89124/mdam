"""Find a minimal circuit where R_Y is applied to an ACTIVE qubit (compiles to OP_ARRAY_S/H/ROT,
not the FRAME path) and the backend diverges from clifft -> isolates the backend-layer R_Y bug
(self.frame / _rot sign), since the engine itself is exact (proved by _collapse_test.py)."""
import sys
import numpy as np
sys.path.insert(0, "/home/jung/clifft-paper")
sys.setrecursionlimit(400000)
import clifft
import nearclifford_backend.backend as bk
from nearclifford_backend.backend import _opname
from nearclifford_backend.clifft_axis.bounded import compile_bounded

SH = 50000
TH = 0.5

def ops_of(prog):
    from collections import Counter
    return Counter(_opname(prog[i].opcode) for i in range(len(prog)))

CIRCS = {
 # T makes q0 active(magic); then R_Y on the ACTIVE q0 -> OP_ARRAY path
 "T,RY,M":        f"R 0\nT 0\nR_Y({TH}) 0\nM 0\n",
 # two RY: 2nd on active q0
 "RY,T,RY,M":     f"R 0\nR_Y({TH}) 0\nT 0\nR_Y({TH}) 0\nM 0\n",
 # active RY then X-basis measure
 "T,RY,H,M":      f"R 0\nT 0\nR_Y({TH}) 0\nH 0\nM 0\n",
 # active RY on q0, entangle, measure ancilla in X (syndrome)
 "T,RY,synX":     f"R 0 1\nT 0\nR_Y({TH}) 0\nH 1\nCX 1 0\nH 1\nM 1\n",
 # T on data, RY, syndrome, RY again, measure data
 "full-ish":      f"R 0 1\nT 0\nR_Y({TH}) 0\nH 1\nCZ 1 0\nH 1\nM 1\nR_Y({TH}) 0\nM 0\n",
 # RY on two active(magic) data into one ancilla X-parity
 "TT,RYRY,synX":  f"R 0 1 2\nT 0\nT 1\nR_Y({TH}) 0\nR_Y({TH}) 1\nH 2\nCX 2 0\nCX 2 1\nH 2\nM 2\n",
}

for name, src in CIRCS.items():
    try:
        prog = compile_bounded(src)
        oc = ops_of(prog)
        active = oc.get("OP_ARRAY_S", 0) + oc.get("OP_ARRAY_H", 0)
        nm = prog.num_measurements
        cl = np.asarray(clifft.sample(prog, shots=SH, seed=5).measurements).mean(0)
        be = bk.NearCliffordBackend(clifft_axis_bounded=True, drop_dead=False,
                                    structure_once=False, clifft_axis_enforce=True)
        bb = be.sample(prog, shots=SH, seed=7).mean(0)
        d = np.abs(cl - bb)
        tag = "  <== REPRODUCES (backend R_Y bug)" if d.max() > 0.02 else ""
        print(f"{name:14} ARRAY_S/H={active:2d} nm={nm}  clifft={np.array2string(cl,precision=3)} "
              f"bounded={np.array2string(bb,precision=3)} max|Δ|={d.max():.4f}{tag}")
    except Exception as e:
        print(f"{name:14} ERROR {type(e).__name__}: {str(e)[:60]}")
