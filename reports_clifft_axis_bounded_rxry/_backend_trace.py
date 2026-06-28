"""Trace the ACTUAL backend path for R_Y (compile_bounded + run_shot), not direct engine drive.
(1) dump the exact compiled instruction sequence for a 1-qubit R_Y; (2) test the backend
marginal vs clifft vs exact on minimal R_Y circuits; (3) if minimal passes, escalate."""
import sys
import numpy as np
sys.path.insert(0, "/home/jung/clifft-paper")
sys.setrecursionlimit(400000)
import clifft
import nearclifford_backend.backend as bk
from nearclifford_backend.backend import _opname
from nearclifford_backend.clifft_axis.bounded import compile_bounded
import ttn_backend.frame_layer as ds_mod

print("=== (1) compiled instruction sequence for  R 0 ; R_Y(1.0) 0 ; M 0 ===")
src = "R 0\nR_Y(1.0) 0\nM 0\n"
prog = compile_bounded(src)
for i in range(len(prog)):
    inst = prog[i]; nm = _opname(inst.opcode)
    if nm in ds_mod.IGNORE_OPS:
        continue
    extra = ""
    try:
        d = ds_mod._d(inst)
        if "weight_re" in d:
            import cmath
            extra = f"  angle={cmath.phase(complex(d['weight_re'], d['weight_im'])):.4f}"
    except Exception:
        pass
    print(f"  [{i}] {nm}  a1={int(inst.axis_1)} a2={int(inst.axis_2)} "
          f"flags={int(getattr(inst,'flags',0))}{extra}")

print("\n=== (2) backend marginal vs clifft vs exact, minimal R_Y ===")
def exact_p1_RY(theta):    # P(measure 1) for RY(theta)|0> = sin^2(theta/2)
    return np.sin(theta / 2) ** 2

SH = 40000
for theta in (0.3, 1.0, 2.0):
    src = f"R 0\nR_Y({theta}) 0\nM 0\n"
    prog = compile_bounded(src)
    cl = np.asarray(clifft.sample(prog, shots=SH, seed=5).measurements).mean(0)[0]
    be = bk.NearCliffordBackend(clifft_axis_bounded=True, drop_dead=False,
                                structure_once=False, clifft_axis_enforce=True)
    bb = be.sample(prog, shots=SH, seed=7).mean(0)[0]
    ex = exact_p1_RY(theta)
    print(f"  RY({theta})  exact p1={ex:.4f}  clifft={cl:.4f}  bounded={bb:.4f}  "
          f"|bnd-exact|={abs(bb-ex):.4f}  {'OK' if abs(bb-ex) < 0.01 else 'BIASED'}")

print("\n=== (3) two-qubit: R_Y(q0) then CNOT(q0->q1), measure q1 (syndrome-style) ===")
# RY(theta) on q0, CNOT(0,1), measure q1: P(q1=1) = sin^2(theta/2)  (q1 copies q0 in Z basis)
for theta in (0.3, 1.0):
    src = f"R 0 1\nR_Y({theta}) 0\nCX 0 1\nM 1\n"
    prog = compile_bounded(src)
    cl = np.asarray(clifft.sample(prog, shots=SH, seed=5).measurements).mean(0)[0]
    be = bk.NearCliffordBackend(clifft_axis_bounded=True, drop_dead=False,
                                structure_once=False, clifft_axis_enforce=True)
    bb = be.sample(prog, shots=SH, seed=7).mean(0)[0]
    ex = np.sin(theta / 2) ** 2
    print(f"  RY({theta})·CX meas q1  exact={ex:.4f}  clifft={cl:.4f}  bounded={bb:.4f}  "
          f"|bnd-exact|={abs(bb-ex):.4f}  {'OK' if abs(bb-ex) < 0.01 else 'BIASED'}")
