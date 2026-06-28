"""EXACT-ORACLE root-cause trace for the R_Y bounded-engine bias.  No sampling: 1-qubit
statevector oracles, < 1e-12 tolerances.  Proves WHERE bounded first diverges from the exact
R_Y unitary and isolates compiler-decomp vs lazy-pending-conjugation vs flush.

Convention (read from simulator.py:_apply_magic_pauli + lazy.py:_conj_*):
  Pauli P(x,z,p) = i^p * X^x * Z^z   (X left, Z right; verified below)
  rotation R_P(theta) = exp(-i theta P / 2) = cos(t/2) I - i sin(t/2) P   (_flush_one)
  pending entry = [x, z, p(hase), theta, uid]   (lazy.py:142, apply_rotation stores p=0)
  conjugation in h()/s() : pending P -> C P C^dag   (verified HYH^dag=-Y below)
"""
import sys
import numpy as np
sys.path.insert(0, "/home/jung/clifft-paper")
np.set_printoptions(precision=5, suppress=True)

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
S = np.array([[1, 0], [0, 1j]], dtype=complex)
Sd = S.conj().T


def Pmat(x, z, p):
    """i^p X^x Z^z on 1 qubit (the engine's convention)."""
    m = np.linalg.matrix_power(X, x) @ np.linalg.matrix_power(Z, z)
    return (1j ** p) * m


def RP(x, z, p, theta):
    """exp(-i theta P/2) = cos I - i sin P, P=i^p X^x Z^z (the _flush_one formula)."""
    P = Pmat(x, z, p)
    return np.cos(theta / 2) * I2 - 1j * np.sin(theta / 2) * P


def RY_exact(theta):
    return np.cos(theta / 2) * I2 - 1j * np.sin(theta / 2) * Y


def signed_name(x, z, p):
    base = {(0, 0): "I", (1, 0): "X", (0, 1): "Z", (1, 1): "XZ"}[(x, z)]
    return f"i^{p}·{base}"


print("=" * 70)
print("STEP 2/3  convention + H/S conjugation table (matrix oracle)")
print("=" * 70)
# verify +Y == P(1,1,1)
print("P(1,1,1) == Y ?", np.allclose(Pmat(1, 1, 1), Y),
      "   P(1,1,0)=XZ == -iY ?", np.allclose(Pmat(1, 1, 0), -1j * Y))

from nearclifford_backend.lazy import _conj_h, _conj_s

def check_conj(name, Cmat, conj_fn):
    print(f"\n  {name}:   pending-conj  vs   C P C^dag   (P->CPC^dag)")
    ok = True
    for (x, z, p, lbl) in [(1, 0, 0, "+X"), (0, 1, 0, "+Z"), (1, 1, 1, "+Y"), (1, 1, 3, "-Y")]:
        nx, nz, npp = conj_fn((x, z, p), 0)
        got = Pmat(nx, nz, npp)
        exp = Cmat @ Pmat(x, z, p) @ Cmat.conj().T
        m = np.allclose(got, exp, atol=1e-12)
        ok &= m
        print(f"    {lbl}={signed_name(x,z,p):8} -> {signed_name(nx,nz,npp):10} "
              f"{'OK' if m else 'FAIL'}")
    return ok

ok_h = check_conj("H", H, lambda P, q: _conj_h(P, q))
ok_s = check_conj("S", S, lambda P, q: _conj_s(P, q, False))
print(f"\n  conj table: H {'OK' if ok_h else 'FAIL'}   S {'OK' if ok_s else 'FAIL'}")

print("\n" + "=" * 70)
print("STEP 4B  compiler decomposition: does S·H·RZ(θ)·H·S† == RY(θ) ?  (matrix)")
print("=" * 70)
for th in (0.02, -0.02, np.pi / 7):
    Ucomp = S @ H @ RP(0, 1, 0, th) @ H @ Sd          # S H RZ(t) H S^dag
    err = np.abs(Ucomp - RY_exact(th)).max()
    # also global-phase-adjusted
    print(f"  theta={th:+.4f}   max|S·H·RZ·H·S† - RY| = {err:.2e}   "
          f"{'OK' if err < 1e-12 else 'FAIL (decomp wrong)'}")

print("\n" + "=" * 70)
print("STEP 4C  lazy pending trace: Z deferred, conjugated by trailing H then S")
print("=" * 70)
# time order S^dag, H, RZ(theta), H, S  => after RZ the trailing gates are H, S.
P = (0, 1, 0)                      # RZ deferred as physical Z (apply_rotation stores p=0)
print(f"  deferred RZ        pending = {signed_name(*P):10}  (should be +Z)")
P = _conj_h(P, 0); print(f"  after trailing H   pending = {signed_name(*P):10}  (should be +X)")
P = _conj_s(P, 0, False); print(f"  after trailing S   pending = {signed_name(*P):10}  (should be +Y, i.e. i^1·XZ)")
print(f"  => pending phase field p = {P[2]}  (NONZERO: the +Y phase the flush must keep)")

print("\n" + "=" * 70)
print("STEP 5/8  what _flush_one actually applies (drops pending phase p) vs correct")
print("=" * 70)
x, z, p = P                                    # (1,1,1) = +Y
theta = 0.02
# what the engine does: _flush_one(x,z,theta) -> _pullback(x,z) gives pp for BARE XZ (=0 here),
# pending phase p is NOT passed (lazy.py:242).  So it applies RP(x,z, pp=0, theta):
U_buggy = RP(x, z, 0, theta)                   # cos I - i sin (XZ)
U_fixed = RP(x, z, p, theta)                   # cos I - i sin (i^p XZ) = cos I - i sin Y
U_true = RY_exact(theta)
print(f"  buggy flush  RP(1,1,p=0): max|·-RY| = {np.abs(U_buggy-U_true).max():.4e}  "
      f"(unitary? {np.allclose(U_buggy.conj().T@U_buggy, I2)})")
print(f"  fixed flush  RP(1,1,p=1): max|·-RY| = {np.abs(U_fixed-U_true).max():.2e}  "
      f"(unitary? {np.allclose(U_fixed.conj().T@U_fixed, I2)})")
print(f"  buggy applies  cos·I - sin·Y  (Y term REAL, not imaginary) -> non-unitary, loses coherence")

print("\n" + "=" * 70)
print("STEP 4A/5  end-to-end: bounded engine statevector vs exact RY|0>")
print("=" * 70)
from nearclifford_backend.clifft_axis.bounded import CliftAxisBoundedNearClifford
for theta in (0.02, -0.02, np.pi / 7):
    e = CliftAxisBoundedNearClifford(1)
    e.set_clifft_budget(4, enforce=False)
    # drive the S·H·RZ·H·S† decomposition in TIME order: S^dag, H, RZ, H, S
    e.s(0, dag=True); e.h(0); e.apply_rotation(0, 1 << 0, theta); e.h(0); e.s(0, dag=False)
    psi = e.statevector()
    true = RY_exact(theta) @ np.array([1, 0], dtype=complex)
    fid = abs(np.vdot(true, psi)) ** 2
    err = np.abs(psi - true).max()
    print(f"  theta={theta:+.4f}  bounded statevector vs RY|0>:  max|Δ|={err:.4e}  fid={fid:.10f}  "
          f"{'OK' if err < 1e-12 else 'DIVERGES'}")
