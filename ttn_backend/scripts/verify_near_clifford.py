"""Verify NearClifford (Clifford frame + magic register) against a dense
statevector reference, on small circuits with Clifford gates + Pauli rotations +
measurement. Compares statevectors up to global phase."""
from __future__ import annotations
import numpy as np
from ttn_backend.near_clifford import NearClifford

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
S = np.array([[1, 0], [0, 1j]], dtype=complex)


def op1(n, q, m):
    full = np.array([[1.0 + 0j]])
    for i in range(n):
        full = np.kron(m if i == q else I2, full)   # qubit q at bit q (LSB=0)
    return full


def cx_mat(n, c, t):
    dim = 1 << n
    U = np.zeros((dim, dim), dtype=complex)
    for b in range(dim):
        bc = (b >> c) & 1
        nb = b ^ ((bc) << t)
        U[nb, b] = 1.0
    return U


def cz_mat(n, a, b):
    dim = 1 << n
    U = np.eye(dim, dtype=complex)
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
    P = pauli_mat(n, x, z)
    return np.cos(th / 2) * np.eye(1 << n) - 1j * np.sin(th / 2) * P


def fidelity(a, b):
    return abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-18)


def run_case(name, n, ops):
    """ops: list of ('h',q)/('s',q)/('cx',c,t)/('cz',a,b)/('rot',x,z,theta)."""
    nc = NearClifford(n)
    ref = np.zeros(1 << n, dtype=complex); ref[0] = 1.0
    for op in ops:
        if op[0] == 'h':
            nc.h(op[1]); ref = op1(n, op[1], H) @ ref
        elif op[0] == 's':
            nc.s(op[1]); ref = op1(n, op[1], S) @ ref
        elif op[0] == 'cx':
            nc.cx(op[1], op[2]); ref = cx_mat(n, op[1], op[2]) @ ref
        elif op[0] == 'cz':
            nc.cz(op[1], op[2]); ref = cz_mat(n, op[1], op[2]) @ ref
        elif op[0] == 'rot':
            _, x, z, th = op
            nc.apply_rotation(x, z, th); ref = rot_mat(n, x, z, th) @ ref
    sv = nc.statevector()
    f = fidelity(sv, ref)
    ok = f > 1 - 1e-9
    print(f"[{'OK ' if ok else 'FAIL'}] {name:30s} fidelity={f:.12f}  |M|={len(nc.M)} (magic dim {2**len(nc.M)})")
    return ok


def measure_case(name, n, ops, shots=20000, seed=0):
    """Build state via ops, then measure ALL qubits in Z; compare empirical
    distribution (NearClifford) to exact |amp|^2 (dense)."""
    # exact probabilities
    ref = np.zeros(1 << n, dtype=complex); ref[0] = 1.0
    for op in ops:
        if op[0] == 'h': ref = op1(n, op[1], H) @ ref
        elif op[0] == 's': ref = op1(n, op[1], S) @ ref
        elif op[0] == 'cx': ref = cx_mat(n, op[1], op[2]) @ ref
        elif op[0] == 'cz': ref = cz_mat(n, op[1], op[2]) @ ref
        elif op[0] == 'rot': ref = rot_mat(n, op[1], op[2], op[3]) @ ref
    probs = np.abs(ref) ** 2
    probs /= probs.sum()
    # empirical from NearClifford
    counts = np.zeros(1 << n)
    rng = np.random.default_rng(seed)
    for sh in range(shots):
        nc = NearClifford(n); nc.rng = np.random.default_rng(int(rng.integers(0, 2**60)))
        for op in ops:
            if op[0] == 'h': nc.h(op[1])
            elif op[0] == 's': nc.s(op[1])
            elif op[0] == 'cx': nc.cx(op[1], op[2])
            elif op[0] == 'cz': nc.cz(op[1], op[2])
            elif op[0] == 'rot': nc.apply_rotation(op[1], op[2], op[3])
        b = 0
        for q in range(n):
            if nc.measure_z(q):
                b |= 1 << q
        counts[b] += 1
    emp = counts / shots
    tvd = 0.5 * np.sum(np.abs(emp - probs))
    ok = tvd < 0.02
    print(f"[{'OK ' if ok else 'FAIL'}] meas {name:24s} TVD={tvd:.4f}  (shots={shots})")
    return ok


def main():
    allok = True
    allok &= run_case("H", 1, [('h', 0)])
    allok &= run_case("RZ on |0>", 1, [('rot', 0, 1, 0.7)])
    allok &= run_case("H,RZ,H (=RX)", 1, [('h', 0), ('rot', 0, 1, 0.5), ('h', 0)])
    allok &= run_case("RX directly (promote)", 1, [('rot', 1, 0, 0.9)])
    allok &= run_case("S then RX", 1, [('s', 0), ('rot', 1, 0, 0.3)])
    allok &= run_case("Bell", 2, [('h', 0), ('cx', 0, 1)])
    allok &= run_case("Bell + RZ", 2, [('h', 0), ('cx', 0, 1), ('rot', 0, 1, 0.4)])
    allok &= run_case("Bell + RX (promote)", 2, [('h', 0), ('cx', 0, 1), ('rot', 1, 0, 0.6)])
    allok &= run_case("2-qubit rot X0Z1", 2, [('rot', 0b01, 0b10, 0.55)])
    allok &= run_case("CZ + rots", 2, [('h', 0), ('h', 1), ('cz', 0, 1), ('rot', 0, 1, 0.3), ('rot', 1, 0, 0.4)])
    # deeper: 3 qubits, mixed
    allok &= run_case("3q mixed", 3, [
        ('h', 0), ('cx', 0, 1), ('cx', 1, 2), ('rot', 0, 0b010, 0.3),
        ('h', 2), ('rot', 0b100, 0, 0.5), ('s', 1), ('rot', 0b011, 0b100, 0.2)])
    # commuting rotations should NOT grow magic beyond needed
    allok &= run_case("commuting Z-rots (k=0 magic)", 3, [
        ('cx', 0, 1), ('cx', 1, 2), ('rot', 0, 0b001, 0.3),
        ('rot', 0, 0b010, 0.4), ('rot', 0, 0b100, 0.5)])
    print("--- measurement distribution vs dense ---")
    allok &= measure_case("H", 1, [('h', 0)])
    allok &= measure_case("RX(theta)", 1, [('rot', 1, 0, 1.1)])
    allok &= measure_case("Bell", 2, [('h', 0), ('cx', 0, 1)])
    allok &= measure_case("Bell+RZ", 2, [('h', 0), ('cx', 0, 1), ('rot', 0, 1, 0.7)])
    allok &= measure_case("Bell+RX", 2, [('h', 0), ('cx', 0, 1), ('rot', 1, 0, 0.9)])
    allok &= measure_case("3q H/CX/rot", 3, [
        ('h', 0), ('cx', 0, 1), ('cx', 1, 2), ('rot', 0, 0b010, 0.5),
        ('h', 2), ('rot', 0b100, 0, 0.6)])
    allok &= measure_case("syndrome-like", 3, [
        ('h', 2), ('cx', 2, 0), ('cx', 2, 1), ('rot', 0, 0b001, 0.3),
        ('rot', 0, 0b010, 0.4), ('h', 2)])
    print("\nALL", "PASS" if allok else "FAIL")


if __name__ == "__main__":
    main()
