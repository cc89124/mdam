"""Deterministic (noise-free) proof that _reduce_full is STATE-EXACT: it applies a
Clifford W to phi and folds W into the frame, so the physical state U_C(|0..0> x phi)
is invariant -- only the representation (|M|) shrinks. We construct parity-slaved
magic states directly, snapshot the full statevector, reduce, and require fidelity 1
AND |M| strictly dropped. This isolates the reduction from sampling-convergence noise
(cultivation_d5's marginals need many shots; this test needs none)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np

from nearclifford_backend.virtual_axis.virtual_axis_runtime import VirtualAxisNearClifford


def _fidelity(a, b):
    return abs(complex(np.vdot(a, b))) / (np.linalg.norm(a) * np.linalg.norm(b))


def _set_state(n, M, phi):
    nc = VirtualAxisNearClifford(n)
    nc.M = list(M)
    nc.phi = np.asarray(phi, dtype=complex)
    nc.phi /= np.linalg.norm(nc.phi)
    return nc


def case(name, n, M, phi, expect_drop):
    nc = _set_state(n, M, phi)
    before = nc.statevector()
    m0 = len(nc.M)
    nc._reduce_full()
    after = nc.statevector()
    m1 = len(nc.M)
    f = _fidelity(before, after)
    ok = f > 1 - 1e-9 and (m1 < m0 if expect_drop else m1 == m0)
    print(f"[{'OK ' if ok else 'FAIL'}] {name:34s} |M| {m0}->{m1}  fidelity={f:.12f}")
    return ok


def main():
    rng = np.random.default_rng(7)
    allok = True
    # Bell-like parity slave: a|00>+b|11> over M=[0,1] (Z0Z1 stabiliser) -> peel to 1
    a, b = 0.6, 0.8
    allok &= case("a|00>+b|11> (Z0Z1 slave)", 2, [0, 1], [a, 0, 0, b], True)
    # 3-qubit GHZ-magic a|000>+b|111> : two parity slaves -> peel to 1
    allok &= case("a|000>+b|111> (2 slaves)", 3, [0, 1, 2],
                  [a, 0, 0, 0, 0, 0, 0, b], True)
    # parity slave on a subset: (a|00>+b|11>) x (magic qubit) -> peel 1 of 3
    g = np.array([0.5, 0.3 + 0.4j])           # genuine magic single qubit
    phi = np.kron(g, np.array([a, 0, 0, b]))  # qubit2 = g, qubits0,1 slaved
    allok &= case("slave pair (x) magic qubit", 3, [0, 1, 2], phi, True)
    # NO redundancy: random full-rank 2-qubit magic state -> must NOT drop
    rphi = rng.standard_normal(4) + 1j * rng.standard_normal(4)
    allok &= case("random 2q magic (no slave)", 2, [0, 1], rphi, False)
    # random 3q magic with one planted parity slave (q2 = q0 XOR q1 parity)
    #   build a|even>+... only on even-parity basis so Z0Z1Z2 stabilises -> peel
    v = np.zeros(8, dtype=complex)
    for idx in range(8):
        if bin(idx).count("1") % 2 == 0:      # even parity -> Z0Z1Z2 = +1
            v[idx] = rng.standard_normal() + 1j * rng.standard_normal()
    allok &= case("3q even-parity (Z0Z1Z2 slave)", 3, [0, 1, 2], v, True)
    print("ALL PASS" if allok else "SOME FAILED")
    return allok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
