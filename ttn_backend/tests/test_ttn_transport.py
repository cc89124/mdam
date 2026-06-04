"""Small exact tests for TTN adjacent transport sweep."""
from __future__ import annotations

import math
import numpy as np

from ttn_backend import TTNState, _CNOT, INV_SQRT2


def _canonical(psi, order):
    if not order:
        return psi
    target = sorted(order)
    perm = [order.index(i) for i in target]
    return np.transpose(psi, perm)


def _assert_same_state(a, b, tol=1e-12):
    psi_a, order_a = a.contract_into_one()
    psi_b, order_b = b.contract_into_one()
    ca = _canonical(psi_a, order_a)
    cb = _canonical(psi_b, order_b)
    diff = float(np.max(np.abs(ca - cb)))
    assert diff < tol, f"state mismatch diff={diff:.3e}, orders={order_a},{order_b}"


def _add_zero_ident(state, bag_id, ident):
    bag = state.bags[bag_id]
    new_shape = bag.tensor.shape[:bag.n_own()] + (2,) + bag.tensor.shape[bag.n_own():]
    new_t = np.zeros(new_shape, dtype=np.complex128)
    sl = [slice(None)] * len(new_shape)
    sl[bag.n_own()] = 0
    new_t[tuple(sl)] = bag.tensor
    bag.tensor = new_t
    bag.own_idents.append(ident)
    state._record_metrics()


def test_transport_roundtrip_2bag():
    state = TTNState(bag_neighbors=[[1], [0]], home={0: 0, 1: 1})
    state.bags[0].own_idents = [0]
    state.bags[0].tensor = np.zeros((2, 2), dtype=np.complex128)
    state.bags[0].tensor[0, 0] = INV_SQRT2
    state.bags[0].tensor[1, 1] = INV_SQRT2
    state.bags[1].own_idents = [1]
    state.bags[1].tensor = np.eye(2, dtype=np.complex128)
    state.center_bag = 0

    before = TTNState(bag_neighbors=[[1], [0]], home={0: 0, 1: 1})
    before.bags[0].own_idents = [0]
    before.bags[0].tensor = state.bags[0].tensor.copy()
    before.bags[1].own_idents = [1]
    before.bags[1].tensor = state.bags[1].tensor.copy()
    before.center_bag = 0

    state.transport_ident_across_edge(0, 0, 1)
    state.check_invariant_I1()
    state.check_invariant_I3()
    assert 0 in state.bags[1].own_idents

    state.transport_ident_across_edge(0, 1, 0)
    state.check_all_invariants()
    _assert_same_state(before, state)
    assert state.metrics["n_transports"] == 2
    assert state.metrics["peak_pair_workspace_bytes"] > 0


def test_class_b_two_bag_bell():
    state = TTNState(bag_neighbors=[[1], [0]], home={0: 0, 1: 1})
    state.expand(0)
    _add_zero_ident(state, 1, 1)
    state.apply_2q_class_B_path(0, 1, _CNOT, [0, 1])
    state.check_all_invariants()
    psi, order = state.contract_into_one()
    psi = _canonical(psi, order)
    expected = np.zeros((2, 2), dtype=np.complex128)
    expected[0, 0] = INV_SQRT2
    expected[1, 1] = INV_SQRT2
    diff = float(np.max(np.abs(psi - expected)))
    assert diff < 1e-12, diff


def test_class_b_three_bag_path():
    state = TTNState(bag_neighbors=[[1], [0, 2], [1]], home={0: 0, 1: 1, 2: 2})
    state.expand(0)
    state.expand(1)
    _add_zero_ident(state, 2, 2)
    state.apply_2q_class_B_path(0, 2, _CNOT, [0, 1, 2])
    state.check_all_invariants()
    psi, order = state.contract_into_one()
    psi = _canonical(psi, order)
    expected = np.zeros((2, 2, 2), dtype=np.complex128)
    expected[0, 0, 0] = 0.5
    expected[0, 1, 0] = 0.5
    expected[1, 0, 1] = 0.5
    expected[1, 1, 1] = 0.5
    diff = float(np.max(np.abs(psi - expected)))
    assert diff < 1e-12, diff


if __name__ == "__main__":
    test_transport_roundtrip_2bag()
    test_class_b_two_bag_bell()
    test_class_b_three_bag_path()
    print("transport tests: PASS")
