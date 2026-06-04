"""Small binary symplectic engine for RASL candidate validation."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


_PHASE_TO_COMPLEX = (1 + 0j, 0 + 1j, -1 + 0j, 0 - 1j)
_COMPLEX_TO_PHASE = {
    (1, 0): 0,
    (0, 1): 1,
    (-1, 0): 2,
    (0, -1): 3,
}

_I = np.eye(2, dtype=np.complex128)
_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
_H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=np.complex128)
_S = np.array([[1, 0], [0, 1j]], dtype=np.complex128)
_CNOT = np.array(
    [[1, 0, 0, 0],
     [0, 0, 0, 1],
     [0, 0, 1, 0],
     [0, 1, 0, 0]], dtype=np.complex128)
_CZ = np.diag([1, 1, 1, -1]).astype(np.complex128)
_PMATS = {"I": _I, "X": _X, "Y": _Y, "Z": _Z}
_PTYPES = "IXYZ"


def _phase_from_scalar(z: complex) -> int:
    vals = [1 + 0j, 0 + 1j, -1 + 0j, 0 - 1j]
    k = min(range(4), key=lambda i: abs(z - vals[i]))
    if abs(z - vals[k]) > 1e-8:
        raise ValueError(f"not a Pauli phase: {z}")
    return k


def _single_phase_table(U):
    tab = {}
    for old in _PTYPES:
        M = U @ _PMATS[old] @ U.conj().T
        for new in _PTYPES:
            tr = np.trace(_PMATS[new].conj().T @ M) / 2
            if abs(abs(tr) - 1) < 1e-9:
                tab[old] = (new, _phase_from_scalar(tr))
                break
    return tab


def _kron2(lo_type, hi_type):
    return np.kron(_PMATS[hi_type], _PMATS[lo_type])


def _two_phase_table(U):
    tab = {}
    for lo_old in _PTYPES:
        for hi_old in _PTYPES:
            M = U @ _kron2(lo_old, hi_old) @ U.conj().T
            for lo_new in _PTYPES:
                for hi_new in _PTYPES:
                    P = _kron2(lo_new, hi_new)
                    tr = np.trace(P.conj().T @ M) / 4
                    if abs(abs(tr) - 1) < 1e-9:
                        tab[(lo_old, hi_old)] = (lo_new, hi_new, _phase_from_scalar(tr))
                        break
                if (lo_old, hi_old) in tab:
                    break
    return tab


_H_TABLE = _single_phase_table(_H)
_S_TABLE = _single_phase_table(_S)
_CNOT_TABLE = _two_phase_table(_CNOT)
_CZ_TABLE = _two_phase_table(_CZ)


@dataclass
class PauliVec:
    x: np.ndarray
    z: np.ndarray
    phase: int = 0

    @classmethod
    def zeros(cls, n: int) -> "PauliVec":
        return cls(np.zeros(n, dtype=np.bool_), np.zeros(n, dtype=np.bool_), 0)

    @classmethod
    def from_types(cls, types: str) -> "PauliVec":
        p = cls.zeros(len(types))
        for q, t in enumerate(types.upper()):
            if t == "I":
                continue
            if t == "X":
                p.x[q] = True
            elif t == "Z":
                p.z[q] = True
            elif t == "Y":
                p.x[q] = True
                p.z[q] = True
            else:
                raise ValueError(f"unknown Pauli type: {t}")
        return p

    def copy(self) -> "PauliVec":
        return PauliVec(self.x.copy(), self.z.copy(), self.phase)

    @property
    def n(self) -> int:
        return int(self.x.size)

    def support(self) -> set[int]:
        return {int(i) for i in np.flatnonzero(self.x | self.z)}

    def weight(self) -> int:
        return len(self.support())

    def pauli_type(self, q: int) -> str:
        xb = bool(self.x[q])
        zb = bool(self.z[q])
        if xb and zb:
            return "Y"
        if xb:
            return "X"
        if zb:
            return "Z"
        return "I"

    def is_single_axis(self) -> bool:
        return self.weight() == 1

    def types(self) -> str:
        return "".join(self.pauli_type(i) for i in range(self.n))

    def phase_complex(self) -> complex:
        return _PHASE_TO_COMPLEX[self.phase % 4]

    def single_axis_result(self):
        support = sorted(self.support())
        if len(support) != 1:
            return None
        q = support[0]
        return q, self.pauli_type(q), self.phase % 4


def _set_type(p: PauliVec, q: int, typ: str) -> None:
    p.x[q] = typ in ("X", "Y")
    p.z[q] = typ in ("Z", "Y")


def apply_H(p: PauliVec, q: int) -> None:
    new_type, delta = _H_TABLE[p.pauli_type(q)]
    _set_type(p, q, new_type)
    p.phase = (p.phase + delta) % 4


def apply_S(p: PauliVec, q: int) -> None:
    new_type, delta = _S_TABLE[p.pauli_type(q)]
    _set_type(p, q, new_type)
    p.phase = (p.phase + delta) % 4


def apply_CNOT(p: PauliVec, c: int, t: int) -> None:
    lo = min(c, t)
    hi = max(c, t)
    old = (p.pauli_type(lo), p.pauli_type(hi))
    p.x[t] ^= p.x[c]
    p.z[c] ^= p.z[t]
    new = (p.pauli_type(lo), p.pauli_type(hi))
    expected_new = _CNOT_TABLE[old]
    if c > t:
        # The precomputed matrix is CNOT(lo -> hi). For the opposite direction,
        # derive the phase by explicit local conjugation on this ordered pair.
        U = np.zeros((4, 4), dtype=np.complex128)
        for x in range(4):
            lo_bit = x & 1
            hi_bit = (x >> 1) & 1
            y_lo = lo_bit ^ hi_bit
            y_hi = hi_bit
            U[(y_hi << 1) | y_lo, x] = 1
        expected_new = _two_phase_table(U)[old]
    if new != expected_new[:2]:
        raise AssertionError(f"CNOT symplectic mismatch: {old}->{new}, expected {expected_new}")
    p.phase = (p.phase + expected_new[2]) % 4


def apply_CZ(p: PauliVec, a: int, b: int) -> None:
    lo = min(a, b)
    hi = max(a, b)
    old = (p.pauli_type(lo), p.pauli_type(hi))
    xa = bool(p.x[a])
    xb = bool(p.x[b])
    p.z[b] ^= xa
    p.z[a] ^= xb
    new = (p.pauli_type(lo), p.pauli_type(hi))
    expected_new = _CZ_TABLE[old]
    if new != expected_new[:2]:
        raise AssertionError(f"CZ symplectic mismatch: {old}->{new}, expected {expected_new}")
    p.phase = (p.phase + expected_new[2]) % 4


def apply_op(p: PauliVec, op) -> None:
    name = op.name
    if name == "H":
        apply_H(p, op.a)
    elif name == "S":
        apply_S(p, op.a)
    elif name == "CNOT":
        apply_CNOT(p, op.a, op.b)
    elif name == "CZ":
        apply_CZ(p, op.a, op.b)
    else:
        raise ValueError(f"unknown Clifford op: {op}")


def apply_ops(p: PauliVec, ops) -> PauliVec:
    out = p.copy()
    for op in ops:
        apply_op(out, op)
    return out
