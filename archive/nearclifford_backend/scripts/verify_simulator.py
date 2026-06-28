"""Verify the near-Clifford CORE (Clifford frame + magic register) against a dense
statevector reference: statevector fidelity for Clifford + Pauli-rotation circuits,
measurement-distribution TVD for Z-measured circuits, and -- new here -- the ZXZ
arbitrary-single-qubit-unitary decomposition used by the backend's U2/U4 de-fusion.
"""
from __future__ import annotations
import numpy as np
from nearclifford_backend.simulator import NearClifford
from nearclifford_backend.backend import _zxz_angles

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
S = np.array([[1, 0], [0, 1j]], dtype=complex)


def op1(n, q, m):
    full = np.array([[1.0 + 0j]])
    for i in range(n):
        full = np.kron(m if i == q else I2, full)
    return full


def cx_mat(n, c, t):
    dim = 1 << n; U = np.zeros((dim, dim), dtype=complex)
    for b in range(dim):
        U[b ^ (((b >> c) & 1) << t), b] = 1.0
    return U


def cz_mat(n, a, b):
    dim = 1 << n; U = np.eye(dim, dtype=complex)
    for k in range(dim):
        if ((k >> a) & 1) and ((k >> b) & 1):
            U[k, k] = -1.0
    return U


def pauli_mat(n, x, z):
    op = np.array([[1.0 + 0j]])
    for q in range(n):
        xq = (x >> q) & 1; zq = (z >> q) & 1
        m = (X @ Z) if (xq and zq) else (X if xq else (Z if zq else I2))
        op = np.kron(m, op)
    return op


def rot_mat(n, x, z, th):
    return np.cos(th / 2) * np.eye(1 << n) - 1j * np.sin(th / 2) * pauli_mat(n, x, z)


def fidelity(a, b):
    return abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-18)


def run_case(name, n, ops):
    nc = NearClifford(n)
    ref = np.zeros(1 << n, dtype=complex); ref[0] = 1.0
    for op in ops:
        if op[0] == 'h':   nc.h(op[1]);            ref = op1(n, op[1], H) @ ref
        elif op[0] == 's': nc.s(op[1]);            ref = op1(n, op[1], S) @ ref
        elif op[0] == 'cx':nc.cx(op[1], op[2]);    ref = cx_mat(n, op[1], op[2]) @ ref
        elif op[0] == 'cz':nc.cz(op[1], op[2]);    ref = cz_mat(n, op[1], op[2]) @ ref
        elif op[0] == 'rot':
            _, x, z, th = op
            nc.apply_rotation(x, z, th); ref = rot_mat(n, x, z, th) @ ref
    f = fidelity(nc.statevector(), ref)
    ok = f > 1 - 1e-9
    print(f"[{'OK ' if ok else 'FAIL'}] {name:34s} fidelity={f:.12f}  |M|={len(nc.M)}")
    return ok


def measure_case(name, n, ops, shots=20000, seed=0):
    ref = np.zeros(1 << n, dtype=complex); ref[0] = 1.0
    for op in ops:
        if op[0] == 'h': ref = op1(n, op[1], H) @ ref
        elif op[0] == 's': ref = op1(n, op[1], S) @ ref
        elif op[0] == 'cx': ref = cx_mat(n, op[1], op[2]) @ ref
        elif op[0] == 'cz': ref = cz_mat(n, op[1], op[2]) @ ref
        elif op[0] == 'rot': ref = rot_mat(n, op[1], op[2], op[3]) @ ref
    probs = np.abs(ref) ** 2; probs /= probs.sum()
    counts = np.zeros(1 << n); rng = np.random.default_rng(seed)
    for _ in range(shots):
        nc = NearClifford(n); nc.rng = np.random.default_rng(int(rng.integers(0, 2**60)))
        for op in ops:
            if op[0] == 'h': nc.h(op[1])
            elif op[0] == 's': nc.s(op[1])
            elif op[0] == 'cx': nc.cx(op[1], op[2])
            elif op[0] == 'cz': nc.cz(op[1], op[2])
            elif op[0] == 'rot': nc.apply_rotation(op[1], op[2], op[3])
        b = 0
        for q in range(n):
            if nc.measure_z(q): b |= 1 << q
        counts[b] += 1
    tvd = 0.5 * np.sum(np.abs(counts / shots - probs))
    ok = tvd < 0.02
    print(f"[{'OK ' if ok else 'FAIL'}] meas {name:28s} TVD={tvd:.4f}")
    return ok


def zxz_case(name, U, seed):
    """Verify the ZXZ decomposition used by U2/U4 de-fusion: apply U (via Rz,Rx,Rz
    Pauli rotations) to a random entangled 2-qubit state on qubit 0, compare to the
    dense U(x)I action (up to global phase)."""
    from nearclifford_backend.backend import NearCliffordBackend
    # build a fixed entangling prelude so qubit 0 is genuinely entangled
    nc = NearClifford(2); ref = np.zeros(4, dtype=complex); ref[0] = 1.0
    nc.h(0); ref = op1(2, 0, H) @ ref
    nc.cx(0, 1); ref = cx_mat(2, 0, 1) @ ref
    nc.apply_rotation(0, 0b01, 0.6); ref = rot_mat(2, 0, 0b01, 0.6) @ ref
    # apply U on qubit 0 via ZXZ
    b, c, d = _zxz_angles(U)
    nc.apply_rotation(0, 0b01, d); ref = rot_mat(2, 0, 0b01, d) @ ref
    nc.apply_rotation(0b01, 0, c); ref2 = rot_mat(2, 0b01, 0, c) @ ref
    ref = ref2
    nc.apply_rotation(0, 0b01, b); ref = rot_mat(2, 0, 0b01, b) @ ref
    f = fidelity(nc.statevector(), ref)
    # also check that ZXZ truly reconstructs U (up to global phase) as a 2x2
    Rz = lambda t: np.array([[np.exp(-1j*t/2), 0], [0, np.exp(1j*t/2)]], dtype=complex)
    Rx = lambda t: np.cos(t/2)*I2 - 1j*np.sin(t/2)*X
    Urec = Rz(b) @ Rx(c) @ Rz(d)
    i, j = np.unravel_index(np.argmax(np.abs(U)), U.shape)
    ph = U[i, j] / Urec[i, j]
    derr = np.linalg.norm(U - ph * Urec)
    ok = f > 1 - 1e-9 and derr < 1e-9
    print(f"[{'OK ' if ok else 'FAIL'}] zxz {name:29s} self-consist={f:.12f}  |U-ZXZ|={derr:.2e}")
    return ok


def main():
    allok = True
    print("--- statevector (Clifford + Pauli rotations) ---")
    allok &= run_case("H", 1, [('h', 0)])
    allok &= run_case("RZ on |0>", 1, [('rot', 0, 1, 0.7)])
    allok &= run_case("H,RZ,H (=RX)", 1, [('h', 0), ('rot', 0, 1, 0.5), ('h', 0)])
    allok &= run_case("RX directly (promote)", 1, [('rot', 1, 0, 0.9)])
    allok &= run_case("S then RX", 1, [('s', 0), ('rot', 1, 0, 0.3)])
    allok &= run_case("Bell + RZ", 2, [('h', 0), ('cx', 0, 1), ('rot', 0, 1, 0.4)])
    allok &= run_case("Bell + RX (promote)", 2, [('h', 0), ('cx', 0, 1), ('rot', 1, 0, 0.6)])
    allok &= run_case("CZ + rots", 2, [('h', 0), ('h', 1), ('cz', 0, 1), ('rot', 0, 1, 0.3), ('rot', 1, 0, 0.4)])
    allok &= run_case("3q mixed", 3, [
        ('h', 0), ('cx', 0, 1), ('cx', 1, 2), ('rot', 0, 0b010, 0.3),
        ('h', 2), ('rot', 0b100, 0, 0.5), ('s', 1), ('rot', 0b011, 0b100, 0.2)])
    allok &= run_case("commuting Z-rots (k=0)", 3, [
        ('cx', 0, 1), ('cx', 1, 2), ('rot', 0, 0b001, 0.3),
        ('rot', 0, 0b010, 0.4), ('rot', 0, 0b100, 0.5)])

    print("--- ZXZ arbitrary-unitary decomposition (used by U2/U4 de-fusion) ---")
    rng = np.random.default_rng(11)
    for i in range(6):
        # random 2x2 unitary via QR
        A = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
        Q, R = np.linalg.qr(A)
        Q = Q @ np.diag(np.exp(1j * np.angle(np.diag(R))))
        allok &= zxz_case(f"random#{i}", Q, i)
    allok &= zxz_case("H", H, 99)
    allok &= zxz_case("T", np.array([[1, 0], [0, np.exp(1j*np.pi/4)]], dtype=complex), 98)

    print("--- measurement distribution vs dense ---")
    allok &= measure_case("Bell+RX", 2, [('h', 0), ('cx', 0, 1), ('rot', 1, 0, 0.9)])
    allok &= measure_case("3q H/CX/rot", 3, [
        ('h', 0), ('cx', 0, 1), ('cx', 1, 2), ('rot', 0, 0b010, 0.5),
        ('h', 2), ('rot', 0b100, 0, 0.6)])
    allok &= measure_case("syndrome-like", 3, [
        ('h', 2), ('cx', 2, 0), ('cx', 2, 1), ('rot', 0, 0b001, 0.3),
        ('rot', 0, 0b010, 0.4), ('h', 2)])
    print("\nALL", "PASS" if allok else "FAIL")
    return allok


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
