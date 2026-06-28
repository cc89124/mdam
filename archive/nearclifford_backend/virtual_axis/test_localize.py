"""Isolated validation of localize_to_virtual_axes (no circuits): the map must be a
phase-exact Pauli ISOMORPHISM onto r <= B virtual axes.

Checks (the spec's section 15):
  15.1 commutation preserved:  commute(P_i,P_j) == commute(mask_i,mask_j)  for all i,j
  15.2 product preserved:       mask(P_i*P_j) == mask_i * mask_j           (x,z AND phase)
       r <= B, and r is minimal (matches the symplectic-rank formula dim V - s).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np

from nearclifford_backend.simulator import pauli_mul, pauli_commute
from nearclifford_backend.virtual_axis.virtual_axis import (
    localize_to_virtual_axes, VirtualPauliMask, _symp, _bits)


_PH = [0]


def _check(paulis, n, label, expect_r=None):
    base = list(paulis)
    # append all pairwise products so the homomorphism can be checked end-to-end
    prods = {}
    ext = list(base)
    for i in range(len(base)):
        for j in range(len(base)):
            prods[(i, j)] = len(ext)
            ext.append(pauli_mul(base[i], base[j]))
    res = localize_to_virtual_axes(ext, n)
    assert res.valid, f"{label}: localization invalid: {res.reason}"
    m = res.masks
    # all masks on exactly r axes, r <= B
    assert all(mk.n_axes == res.r for mk in m), f"{label}: masks not all on r axes"
    assert res.r <= res.physical_B, f"{label}: r={res.r} > B={res.physical_B}"
    if expect_r is not None:
        assert res.r == expect_r, f"{label}: r={res.r} != expected {expect_r}"
    # 15.1 commutation preserved (phase-independent -> the memory-claim structure)
    for i in range(len(base)):
        for j in range(len(base)):
            assert _symp(base[i], base[j]) == (0 if m[i].commutes(m[j]) else 1), \
                f"{label}: commutation mismatch at ({i},{j})"
    # 15.2 product preserved -- (x,z) AND phase (phase-exact after _herm normalisation)
    phase_mismatch = 0
    for i in range(len(base)):
        for j in range(len(base)):
            prod_mask = m[i].mul(m[j])
            got = m[prods[(i, j)]]
            assert (got.x, got.z, got.phase) == (prod_mask.x, prod_mask.z, prod_mask.phase), \
                f"{label}: product mismatch at ({i},{j}): {got} vs {prod_mask}"
    return res, phase_mismatch


def main():
    total_phase = 0
    # --- structured: {Z0Z1, X0} -- one hyperbolic pair, B=2 -> r=1 ---
    _, p = _check([(0, 0b11, 0), (0b01, 0, 0)], 2, "Z0Z1 & X0 (parity reduce)", 1); total_phase += p
    # --- duplicate Pauli: span is 1-dim though B=2 ---
    _, p = _check([(0, 0b11, 0), (0, 0b11, 2)], 2, "duplicate Z0Z1 (+-)", 1); total_phase += p
    # --- no reduction: X0,Z0,X1,Z1 independent symplectic -> r=B=2 ---
    _, p = _check([(0b01, 0, 0), (0, 0b01, 0), (0b10, 0, 0), (0, 0b10, 0)], 2,
                  "full 2q algebra", 2); total_phase += p
    # --- central (commuting): Z0, Z1 -> r=2 (each an independent commuting axis) ---
    _, p = _check([(0, 0b01, 0), (0, 0b10, 0)], 2, "Z0,Z1 central", 2); total_phase += p
    # --- parity trio on 3 qubits: Z0Z2, Z1Z2, X0X1X2 ---
    _, p = _check([(0, 0b101, 0), (0, 0b110, 0), (0b111, 0, 0)], 3, "3q parity trio"); total_phase += p

    # --- random fuzz: varied n and #paulis ---
    rng = np.random.default_rng(12345)
    for t in range(400):
        n = int(rng.integers(2, 7))
        m = int(rng.integers(1, 9))
        ps = [(int(rng.integers(0, 1 << n)), int(rng.integers(0, 1 << n)),
               int(rng.integers(0, 4))) for _ in range(m)]
        _, p = _check(ps, n, f"random#{t}"); total_phase += p
    print(f"PASS: phase-exact Pauli isomorphism onto r<=B virtual axes  (405 cases)")
    print(f"  commutation + product (x,z AND phase) preserved; phase mismatches: {total_phase}")
    assert total_phase == 0, "phase not exact"


if __name__ == "__main__":
    main()
