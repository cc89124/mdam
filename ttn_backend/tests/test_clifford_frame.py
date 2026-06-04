from __future__ import annotations

import numpy as np

from ttn_backend.clifford_frame import (
    RegionLinearFrame,
    apply_cnot_tensor,
    apply_swap_tensor,
)


def _seq_bits(bits, ops, idents):
    pos = {int(x): i for i, x in enumerate(idents)}
    x = np.array(bits, dtype=np.uint8)
    for op in ops:
        if op[0] == "cnot":
            c = pos[op[1]]
            t = pos[op[2]]
            x[t] ^= x[c]
        elif op[0] == "swap":
            a = pos[op[1]]
            b = pos[op[2]]
            x[[a, b]] = x[[b, a]]
        elif op[0] == "multi":
            target = pos[op[1]]
            for c_ident in op[2]:
                c = pos[c_ident]
                if c != target:
                    x[target] ^= x[c]
        else:
            raise ValueError(op)
    return x


def _seq_tensor(tensor, ops, idents):
    pos = {int(x): i for i, x in enumerate(idents)}
    T = tensor
    for op in ops:
        if op[0] == "cnot":
            T = apply_cnot_tensor(T, pos[op[1]], pos[op[2]])
        elif op[0] == "swap":
            T = apply_swap_tensor(T, pos[op[1]], pos[op[2]])
        elif op[0] == "multi":
            for c_ident in op[2]:
                if int(c_ident) != int(op[1]):
                    T = apply_cnot_tensor(T, pos[c_ident], pos[op[1]])
        else:
            raise ValueError(op)
    return T


def test_region_linear_frame_bitstrings():
    idents = [10, 11, 12, 13]
    ops = [
        ("cnot", 10, 12),
        ("multi", 13, [10, 11, 12]),
        ("swap", 11, 12),
        ("cnot", 13, 10),
    ]
    frame = RegionLinearFrame(idents)
    for op in ops:
        if op[0] == "cnot":
            frame.compose_cnot(op[1], op[2])
        elif op[0] == "multi":
            frame.compose_multicnot(op[1], op[2])
        elif op[0] == "swap":
            frame.compose_swap(op[1], op[2])

    for idx in range(2 ** len(idents)):
        bits = np.array([(idx >> i) & 1 for i in range(len(idents))], dtype=np.uint8)
        expected = _seq_bits(bits, ops, idents)
        actual = frame.apply_bits(bits)
        assert np.array_equal(actual, expected)


def test_region_linear_frame_tensor_materialization():
    rng = np.random.default_rng(123)
    idents = [5, 7, 9, 12]
    ops = [
        ("cnot", 5, 9),
        ("swap", 7, 12),
        ("multi", 9, [5, 7, 12]),
        ("cnot", 9, 5),
    ]
    tensor = rng.normal(size=(2, 2, 2, 2, 3)) + 1j * rng.normal(size=(2, 2, 2, 2, 3))
    frame = RegionLinearFrame(idents)
    for op in ops:
        if op[0] == "cnot":
            frame.compose_cnot(op[1], op[2])
        elif op[0] == "multi":
            frame.compose_multicnot(op[1], op[2])
        elif op[0] == "swap":
            frame.compose_swap(op[1], op[2])

    expected = _seq_tensor(tensor, ops, idents)
    actual = frame.materialize_to_tensor(tensor, {ident: i for i, ident in enumerate(idents)})
    diff = np.linalg.norm((expected - actual).ravel())
    assert diff < 1e-10


if __name__ == "__main__":
    test_region_linear_frame_bitstrings()
    test_region_linear_frame_tensor_materialization()
    print("ok")
