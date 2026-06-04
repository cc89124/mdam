"""Unit tests for the RASL symplectic engine."""

from __future__ import annotations

import itertools
import random
import numpy as np

from ttn_backend.rasl.candidate import CliffordOp, LocalizationCandidate, verify_candidate
from ttn_backend.rasl.symplectic import PauliVec, apply_H, apply_S, apply_CNOT, apply_CZ, apply_ops


I = np.eye(2, dtype=np.complex128)
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=np.complex128)
S = np.array([[1, 0], [0, 1j]], dtype=np.complex128)
CNOT = np.array(
    [[1, 0, 0, 0],
     [0, 1, 0, 0],
     [0, 0, 0, 1],
     [0, 0, 1, 0]], dtype=np.complex128)
CZ = np.diag([1, 1, 1, -1]).astype(np.complex128)
PAULIS = {"I": I, "X": X, "Y": Y, "Z": Z}


def _kron_all(mats):
    out = np.array([[1]], dtype=np.complex128)
    for m in mats:
        out = np.kron(out, m)
    return out


def _pauli_matrix(types):
    return _kron_all([PAULIS[t] for t in reversed(types)])


def _embed_1q(U, q, n):
    mats = [I] * n
    mats[q] = U
    return _kron_all(list(reversed(mats)))


def _embed_2q(U4, a, b, n):
    dim = 1 << n
    out = np.zeros((dim, dim), dtype=np.complex128)
    lo = min(a, b)
    hi = max(a, b)
    for x in range(dim):
        local = (((x >> hi) & 1) << 1) | ((x >> lo) & 1)
        rest = x & ~(1 << lo) & ~(1 << hi)
        for ylocal in range(4):
            y = rest | ((ylocal & 1) << lo) | (((ylocal >> 1) & 1) << hi)
            out[y, x] = U4[ylocal, local]
    if a > b and U4 is CNOT:
        # Rebuild directional CNOT(a -> b) explicitly.
        out[:] = 0
        for x in range(dim):
            y = x ^ ((1 << b) if ((x >> a) & 1) else 0)
            out[y, x] = 1
    elif a < b and U4 is CNOT:
        out[:] = 0
        for x in range(dim):
            y = x ^ ((1 << b) if ((x >> a) & 1) else 0)
            out[y, x] = 1
    return out


def _types_from_matrix(M, n):
    for types in itertools.product("IXYZ", repeat=n):
        P = _pauli_matrix(types)
        tr = np.trace(P.conj().T @ M) / (1 << n)
        if abs(abs(tr) - 1) < 1e-9:
            return "".join(types)
    raise AssertionError("matrix is not a Pauli up to phase")


def _types_phase_from_matrix(M, n):
    for types in itertools.product("IXYZ", repeat=n):
        P = _pauli_matrix(types)
        tr = np.trace(P.conj().T @ M) / (1 << n)
        if abs(abs(tr) - 1) < 1e-9:
            phases = [1 + 0j, 0 + 1j, -1 + 0j, 0 - 1j]
            phase = min(range(4), key=lambda i: abs(tr - phases[i]))
            return "".join(types), phase
    raise AssertionError("matrix is not a Pauli up to phase")


def test_single_gate_truth_tables():
    for t in "IXYZ":
        p = PauliVec.from_types(t)
        apply_H(p, 0)
        M = H @ PAULIS[t] @ H.conj().T
        typ, phase = _types_phase_from_matrix(M, 1)
        assert (p.types(), p.phase) == (typ, phase)

        p = PauliVec.from_types(t)
        apply_S(p, 0)
        M = S @ PAULIS[t] @ S.conj().T
        typ, phase = _types_phase_from_matrix(M, 1)
        assert (p.types(), p.phase) == (typ, phase)

    for types in itertools.product("IXYZ", repeat=2):
        s = "".join(types)
        p = PauliVec.from_types(s)
        apply_CNOT(p, 0, 1)
        U = _embed_2q(CNOT, 0, 1, 2)
        M = U @ _pauli_matrix(s) @ U.conj().T
        typ, phase = _types_phase_from_matrix(M, 2)
        assert (p.types(), p.phase) == (typ, phase)

        p = PauliVec.from_types(s)
        apply_CZ(p, 0, 1)
        M = CZ @ _pauli_matrix(s) @ CZ.conj().T
        typ, phase = _types_phase_from_matrix(M, 2)
        assert (p.types(), p.phase) == (typ, phase)


def test_random_clifford_sequences_against_bruteforce():
    rng = random.Random(1234)
    for n in range(1, 5):
        for _ in range(50):
            types = "".join(rng.choice("IXYZ") for _ in range(n))
            p = PauliVec.from_types(types)
            M = _pauli_matrix(types)
            ops = []
            for _ in range(20):
                if n >= 2 and rng.random() < 0.5:
                    a, b = rng.sample(range(n), 2)
                    if rng.random() < 0.5:
                        ops.append(CliffordOp("CNOT", a, b))
                        U = _embed_2q(CNOT, a, b, n)
                    else:
                        ops.append(CliffordOp("CZ", a, b))
                        U = _embed_2q(CZ, a, b, n)
                else:
                    q = rng.randrange(n)
                    if rng.random() < 0.5:
                        ops.append(CliffordOp("H", q))
                        U = _embed_1q(H, q, n)
                    else:
                        ops.append(CliffordOp("S", q))
                        U = _embed_1q(S, q, n)
                M = U @ M @ U.conj().T
            out = apply_ops(p, ops)
            typ, phase = _types_phase_from_matrix(M, n)
            assert (out.types(), out.phase) == (typ, phase)


def test_candidate_verification_weight_one():
    p = PauliVec.from_types("ZZZ")
    cand = LocalizationCandidate(
        step_id=0,
        kind="active_z_route_star",
        target_axis=0,
        ops=[CliffordOp("CNOT", 1, 0), CliffordOp("CNOT", 2, 0)],
    )
    assert verify_candidate(p, cand)
    assert cand.final_pauli_type == "Z"


if __name__ == "__main__":
    test_single_gate_truth_tables()
    test_random_clifford_sequences_against_bruteforce()
    test_candidate_verification_weight_one()
    print("RASL symplectic tests passed")
