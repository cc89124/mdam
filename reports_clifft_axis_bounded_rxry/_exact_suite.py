"""EXACT (non-sampling) post-fix validation of R_Y in clifft_axis_bounded, <1e-12 tolerances.
Dense statevector oracle vs the bounded engine driven by primitives; Born p0 captured exactly
from the engine via log_cores.  Covers: single-qubit RY, 2-qubit RY+CNOT/CZ propagation, Born,
post-measurement reconstructed state."""
import sys
import numpy as np
sys.path.insert(0, "/home/jung/clifft-paper")
from nearclifford_backend.clifft_axis.bounded import CliftAxisBoundedNearClifford

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
S = np.array([[1, 0], [0, 1j]], dtype=complex)
def RY(t): return np.cos(t/2)*I2 - 1j*np.sin(t/2)*Y
def RZ(t): return np.cos(t/2)*I2 - 1j*np.sin(t/2)*Z

def op(n, g, q):                      # 1-qubit gate g on wire q of n (q=0 is LSB)
    m = [I2]*n; m[q] = g
    out = np.array([[1]], dtype=complex)
    for k in range(n-1, -1, -1): out = np.kron(out, m[k])
    return out
def cnot(n, c, t):
    d = 1 << n; M = np.zeros((d, d), dtype=complex)
    for i in range(d):
        j = i ^ (1 << t) if (i >> c) & 1 else i
        M[j, i] = 1
    return M
def cz(n, a, b):
    d = 1 << n; M = np.eye(d, dtype=complex)
    for i in range(d):
        if (i >> a) & 1 and (i >> b) & 1: M[i, i] = -1
    return M

# bounded engine driver: apply RY via the compiler's S·H·RZ·H·S† (time order S†,H,RZ,H,S)
def eng_ry(e, q, t): e.s(q, dag=True); e.h(q); e.apply_rotation(0, 1 << q, t); e.h(q); e.s(q, dag=False)

PASS = True
def check(name, err, tol=1e-12):
    global PASS; ok = err < tol; PASS &= ok
    print(f"  {'OK ' if ok else 'FAIL'} {name:48} err={err:.2e}")

print("=== 1. single-qubit RY: bounded.statevector vs RY|0> ===")
for t in (0.02, -0.02, np.pi/7, 1.3):
    e = CliftAxisBoundedNearClifford(1); e.set_clifft_budget(4, enforce=False)
    eng_ry(e, 0, t)
    psi = e.statevector(); true = RY(t) @ np.array([1, 0], dtype=complex)
    check(f"RY({t:+.3f})|0>", np.abs(psi - true).max())

print("=== 2. 2-qubit propagation: RY then entangling Clifford ===")
cases = [
    ("RY(q0)·CNOT(0,1)", lambda e: (eng_ry(e,0,0.7), e.cx(0,1)),      lambda: cnot(2,0,1) @ op(2,RY(0.7),0)),
    ("RY(q1)·CNOT(0,1)", lambda e: (eng_ry(e,1,0.7), e.cx(0,1)),      lambda: cnot(2,0,1) @ op(2,RY(0.7),1)),
    ("RY(q0)·CZ(0,1)",   lambda e: (eng_ry(e,0,0.7), e.cz(0,1)),      lambda: cz(2,0,1)   @ op(2,RY(0.7),0)),
    ("RY(q1)·CZ(0,1)",   lambda e: (eng_ry(e,1,-0.5),e.cz(0,1)),      lambda: cz(2,0,1)   @ op(2,RY(-0.5),1)),
    ("RY(q0)·RY(q1)·CNOT",lambda e:(eng_ry(e,0,0.3),eng_ry(e,1,0.9),e.cx(0,1)), lambda: cnot(2,0,1)@op(2,RY(0.9),1)@op(2,RY(0.3),0)),
]
for name, drive, oracle in cases:
    e = CliftAxisBoundedNearClifford(2); e.set_clifft_budget(6, enforce=False)
    drive(e)
    psi = e.statevector(); true = oracle() @ (np.eye(4, dtype=complex)[:, 0])
    check(name, np.abs(psi - true).max())

print("=== 3. Born p0 (engine, via log_cores) vs exact |<...0|psi>|^2 ===")
for t in (0.02, np.pi/7, 1.3):
    e = CliftAxisBoundedNearClifford(1); e.set_clifft_budget(4, enforce=False)
    e.log_cores = True; e.core_log = []
    eng_ry(e, 0, t)
    psi = RY(t) @ np.array([1, 0], dtype=complex)
    p0_exact = abs(psi[0])**2
    _ = e.measure_z(0)
    p0_eng = e.core_log[-1]["p0"]
    check(f"Born p0 RY({t:+.3f})  (exact {p0_exact:.6f})", abs(p0_eng - p0_exact))

print("=== 4. post-measurement state (2-qubit, NON-degenerate): measure q0, q1 keeps RY) ===")
class _Rfix:                                      # force the q0 outcome to test both branches
    def __init__(s, b): s.b = b
    def random(s): return 0.0 if s.b == 0 else 1.0
for forced in (0, 1):
    # state = RY(0.9)_q1 RY(1.2)_q0 |00>; measure q0 (out forced); q1 still carries RY -> register
    # non-empty after the drop, so the oracle reconstruction is well-posed.
    e = CliftAxisBoundedNearClifford(2); e.set_clifft_budget(6, enforce=False)
    eng_ry(e, 0, 1.2); eng_ry(e, 1, 0.9)
    e.rng = _Rfix(forced)
    out = e.measure_z(0)
    full = op(2, RY(0.9), 1) @ op(2, RY(1.2), 0) @ np.eye(4, dtype=complex)[:, 0]
    # exact projector onto q0=out
    P = np.zeros((4, 4), dtype=complex)
    for i in range(4):
        if (i >> 0) & 1 == out: P[i, i] = 1
    proj = P @ full; proj = proj / np.linalg.norm(proj)
    post = e.statevector(); post = post / (np.linalg.norm(post) + 1e-300)
    fid = abs(np.vdot(proj, post))**2
    check(f"post-meas state (2q) measured q0={out}", abs(1 - fid))

print("=== 5. R_Z |1>-collapse sanity (no R_Y): X|0>=|1> then measure Z forced out=1 ===")
e = CliftAxisBoundedNearClifford(2); e.set_clifft_budget(6, enforce=False)
e.h(0); e.s(0); e.s(0); e.h(0)                   # H S S H = H Z H = X  -> |1> on q0
eng_ry(e, 1, 0.5)
e.rng = _Rfix(1)
out = e.measure_z(0)
check(f"R_Z-path |1> collapse outcome==1 (got {out})", 0.0 if out == 1 else 1.0)

print("\nRESULT:", "ALL EXACT PASS" if PASS else "SOME FAIL")
sys.exit(0 if PASS else 1)
