"""Verify BlockLazyNearClifford (block-factored magic register): correctness vs
dense, AND that the factoring keeps the live max-block small where the monolithic
register would accumulate.
"""
from __future__ import annotations
import numpy as np
from nearclifford_backend.block_magic import BlockLazyNearClifford
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
    nc = BlockLazyNearClifford(n); apply_ops(nc, ops)
    f = fidelity(nc.statevector(), dense_ops(n, ops)); ok = f > 1 - 1e-9
    print(f"[{'OK ' if ok else 'FAIL'}] {name:34s} fidelity={f:.12f}")
    return ok


def measure_case(name, n, ops, meas_basis=None, shots=20000, seed=0):
    if meas_basis is None:
        meas_basis = ['z'] * n
    ref = dense_ops(n, ops)
    rot = np.array([[1.0]], dtype=complex)
    for q in range(n):
        rot = np.kron(H if meas_basis[q] == 'x' else I2, rot)
    probs = np.abs(rot @ ref) ** 2; probs /= probs.sum()
    counts = np.zeros(1 << n); rng = np.random.default_rng(seed); peak = 0
    for _ in range(shots):
        nc = BlockLazyNearClifford(n); nc.rng = np.random.default_rng(int(rng.integers(0, 2**60)))
        apply_ops(nc, ops)
        b = 0
        for q in range(n):
            if meas_basis[q] == 'x':
                nc.h(q)
            if nc.measure_z(q):
                b |= 1 << q
        peak = max(peak, nc.max_M)
        counts[b] += 1
    tvd = 0.5 * np.sum(np.abs(counts / shots - probs)); ok = tvd < 0.02
    print(f"[{'OK ' if ok else 'FAIL'}] meas {name:26s} TVD={tvd:.4f}  peak max-block={peak}")
    return ok


def purge_case(name, n, ops, meas_seq, seeds=16):
    """Exercise the measured-magic purge (_purge_redundant / W_M peel): build a
    genuinely multi-qubit magic block, measure qubits on the magic path, and check
    the POST-measurement statevector still equals the exact dense Z-projection (so the
    purge -- block-local Clifford W on the vector + W^dag folded into the frame -- is
    state-exact), while the block shrinks. Worst-case fidelity over random outcomes."""
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    worst = 1.0; max_blk = 0
    for seed in range(seeds):
        nc = BlockLazyNearClifford(n); nc.rng = np.random.default_rng(seed)
        apply_ops(nc, ops); ref = dense_ops(n, ops)
        for q in meas_seq:
            b = nc.measure_z(q)                 # flush+measure+purge happen in here
            max_blk = max(max_blk, nc.max_M)    # transient peak block the purge saw
            P = op1(n, q, (np.eye(2) + (-1 if b else 1) * Z) / 2)
            ref = P @ ref; nrm = np.linalg.norm(ref)
            if nrm < 1e-12:
                worst = 0.0; break
            ref = ref / nrm
            worst = min(worst, fidelity(nc.statevector(), ref))
    ok = worst > 1 - 1e-9
    print(f"[{'OK ' if ok else 'FAIL'}] purge {name:25s} worst-fidelity={worst:.12f}  "
          f"peak block={max_blk}")
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

    print("--- measurement vs dense (+ peak max-block) ---")
    allok &= measure_case("|+>,RZ measZ (free)", 1, [('h', 0), ('rot', 0, 1, 1.1)], ['z'])
    allok &= measure_case("|+>,RZ measX (flush)", 1, [('h', 0), ('rot', 0, 1, 1.1)], ['x'])
    allok &= measure_case("Bell+RZ measZ", 2, [('h', 0), ('cx', 0, 1), ('rot', 0, 1, 0.7)], ['z', 'z'])
    allok &= measure_case("syndrome-like", 3, [
        ('h', 2), ('cx', 2, 0), ('cx', 2, 1), ('rot', 0, 0b001, 0.3),
        ('rot', 0, 0b010, 0.4), ('h', 2)], ['z', 'z', 'z'])
    # MANY equatorial data qubits read in Z + one ancilla read in X: block factoring
    # must keep peak max-block SMALL (data qubits peel into dim-1/2 blocks).
    n = 6
    ops = [('h', q) for q in range(n)]
    ops += [('rot', 0, 1 << q, 0.3 + 0.1 * q) for q in range(n - 1)]   # RZ on data 0..n-2
    ops += [('cx', q, n - 1) for q in range(n - 1)]                    # entangle into ancilla n-1
    allok &= measure_case("5 data-Z + 1 anc-X (factor!)", n, ops,
                          ['z'] * (n - 1) + ['x'])

    print("--- measured-magic purge: post-measurement statevector vs dense ---")
    allok &= purge_case("3q magic block", 3, [
        ('h', 0), ('cx', 0, 1), ('cx', 1, 2), ('rot', 0, 0b001, 0.7),
        ('rot', 0, 0b010, 0.5), ('rot', 0, 0b100, 0.3)], [0, 1, 2])
    allok &= purge_case("4q entangled, mid-out", 4, [
        ('h', 0), ('h', 1), ('rot', 0, 0b0001, 0.6), ('rot', 0, 0b0010, 0.9),
        ('cx', 0, 2), ('cx', 1, 3), ('rot', 0b0100, 0, 0.4), ('cx', 2, 3),
        ('rot', 0, 0b1000, 0.5)], [2, 0, 3, 1])
    # S+H frame + measure-middle drive multi-qubit pullbacks, so the purge's W_M peel
    # collapses a genuine >=2-qubit Z-string (peak block 2-3) -- the CNOT-collapse path.
    allok &= purge_case("X/Y-type pullback (S+H)", 3, [
        ('h', 0), ('cx', 0, 1), ('rot', 0, 0b001, 0.8), ('s', 0), ('h', 0),
        ('cx', 0, 2), ('rot', 0, 0b100, 0.4)], [0, 2, 1])
    print("\nALL", "PASS" if allok else "FAIL")
    return allok


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
