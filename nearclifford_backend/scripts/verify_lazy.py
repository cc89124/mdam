"""Verify the LAZY near-Clifford simulator (deferred Pauli rotations + anticommuting
-core flush) against a dense statevector reference, and check the deferral actually
keeps the magic register empty for measurement-irrelevant phases.
"""
from __future__ import annotations
import numpy as np
from nearclifford_backend.lazy import LazyNearClifford
from nearclifford_backend.scripts.verify_simulator import (
    op1, cx_mat, cz_mat, rot_mat, fidelity, H, S, I2)


def apply_ops(nc, ops):
    for op in ops:
        if op[0] == 'h':   nc.h(op[1])
        elif op[0] == 's': nc.s(op[1])
        elif op[0] == 'cx':nc.cx(op[1], op[2])
        elif op[0] == 'cz':nc.cz(op[1], op[2])
        elif op[0] == 'rot':nc.apply_rotation(op[1], op[2], op[3])


def dense_ops(n, ops):
    ref = np.zeros(1 << n, dtype=complex); ref[0] = 1.0
    for op in ops:
        if op[0] == 'h':   ref = op1(n, op[1], H) @ ref
        elif op[0] == 's': ref = op1(n, op[1], S) @ ref
        elif op[0] == 'cx':ref = cx_mat(n, op[1], op[2]) @ ref
        elif op[0] == 'cz':ref = cz_mat(n, op[1], op[2]) @ ref
        elif op[0] == 'rot':ref = rot_mat(n, op[1], op[2], op[3]) @ ref
    return ref


def run_case(name, n, ops):
    nc = LazyNearClifford(n); apply_ops(nc, ops)
    sv = nc.statevector(); ref = dense_ops(n, ops)
    f = fidelity(sv, ref); ok = f > 1 - 1e-9
    print(f"[{'OK ' if ok else 'FAIL'}] {name:34s} fidelity={f:.12f}")
    return ok


def measure_case(name, n, ops, meas_basis=None, shots=20000, seed=0):
    """meas_basis: list of 'z'/'x' per qubit (default all 'z'). Compares the lazy
    sampled distribution to the dense Born distribution, and reports peak live |M|."""
    if meas_basis is None:
        meas_basis = ['z'] * n
    ref = dense_ops(n, ops)
    # dense probabilities in the chosen per-qubit basis: rotate X-measured qubits by H
    rot = np.array([[1.0]], dtype=complex)
    for q in range(n):
        m = H if meas_basis[q] == 'x' else I2
        rot = np.kron(m, rot)
    probs = np.abs(rot @ ref) ** 2; probs /= probs.sum()
    counts = np.zeros(1 << n); rng = np.random.default_rng(seed); peakM = 0
    for _ in range(shots):
        nc = LazyNearClifford(n); nc.rng = np.random.default_rng(int(rng.integers(0, 2**60)))
        apply_ops(nc, ops)
        b = 0
        for q in range(n):
            if meas_basis[q] == 'x':
                nc.h(q)
            if nc.measure_z(q):
                b |= 1 << q
            peakM = max(peakM, nc.live_magic())
        counts[b] += 1
    tvd = 0.5 * np.sum(np.abs(counts / shots - probs)); ok = tvd < 0.02
    print(f"[{'OK ' if ok else 'FAIL'}] meas {name:28s} TVD={tvd:.4f}  peak|M|={peakM}")
    return ok


def main():
    allok = True
    print("--- statevector (flush-all) vs dense ---")
    allok &= run_case("RZ on |0>", 1, [('rot', 0, 1, 0.7)])
    allok &= run_case("H,RZ,H (=RX)", 1, [('h', 0), ('rot', 0, 1, 0.5), ('h', 0)])
    allok &= run_case("Bell + RZ", 2, [('h', 0), ('cx', 0, 1), ('rot', 0, 1, 0.4)])
    allok &= run_case("CZ + rots", 2, [('h', 0), ('h', 1), ('cz', 0, 1), ('rot', 0, 1, 0.3), ('rot', 1, 0, 0.4)])
    allok &= run_case("3q mixed", 3, [
        ('h', 0), ('cx', 0, 1), ('cx', 1, 2), ('rot', 0, 0b010, 0.3),
        ('h', 2), ('rot', 0b100, 0, 0.5), ('s', 1), ('rot', 0b011, 0b100, 0.2)])
    allok &= run_case("two RZ + CNOT chain", 4, [
        ('h', 0), ('h', 1), ('rot', 0, 0b0001, 0.3), ('rot', 0, 0b0010, 0.5),
        ('cx', 0, 2), ('cx', 1, 3), ('rot', 0, 0b0100, 0.2)])

    print("--- measurement distribution vs dense (+ peak live |M|) ---")
    # equatorial phase measured in Z: phase irrelevant -> peak|M| must be 0
    allok &= measure_case("|+>,RZ, meas Z (free)", 1, [('h', 0), ('rot', 0, 1, 1.1)], ['z'])
    # same state measured in X: phase matters -> must flush, peak|M|=1
    allok &= measure_case("|+>,RZ, meas X (flush)", 1, [('h', 0), ('rot', 0, 1, 1.1)], ['x'])
    # Bell + RZ measured Z
    allok &= measure_case("Bell+RZ measZ", 2, [('h', 0), ('cx', 0, 1), ('rot', 0, 1, 0.7)], ['z', 'z'])
    # syndrome-like: ancilla H/CX/RZ/H then Z-measure
    allok &= measure_case("syndrome-like", 3, [
        ('h', 2), ('cx', 2, 0), ('cx', 2, 1), ('rot', 0, 0b001, 0.3),
        ('rot', 0, 0b010, 0.4), ('h', 2)], ['z', 'z', 'z'])
    # 3q: data qubits RZ then read Z (free), ancilla read X (flush)
    allok &= measure_case("data-Z/anc-X", 3, [
        ('h', 0), ('h', 1), ('h', 2), ('rot', 0, 0b001, 0.6), ('rot', 0, 0b010, 0.7),
        ('cx', 0, 2), ('cx', 1, 2)], ['z', 'z', 'x'])
    print("\nALL", "PASS" if allok else "FAIL")
    return allok


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
