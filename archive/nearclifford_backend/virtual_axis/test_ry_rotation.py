"""Correctness tests for off-axis `R_Y` coherent rotations on the fused single-frame backend.

`R_Y(theta) = exp(-i theta Y/2)` with `Y = i XZ` is a single-qubit Pauli rotation, so it must
flow through the deferred Pauli-rotation ledger as the mask `(x=1<<q, z=1<<q)` -- NOT a general
2-qubit unitary.  Two independent checks:

  A. ENGINE-DIRECT (forced-outcome statevector fidelity): drive `FusedSingleFrame` with explicit
     Y/X/Z `apply_rotation` calls + Cliffords, force each outcome, and compare the reconstructed
     dense state to a dense reference.  Exercises the x&z!=0 path through `apply_rotation` (phase
     via `_herm`), `_gate` conjugation, `_core_entries`, `_mask_for`, `_pauli_sum`,
     `_contract_single`, `statevector`.

  B. END-TO-END vs clifft's EXACT reference (`clifft.record_probabilities`): compile a circuit
     containing `R_Y` with `compile_circuit` (bytecode fusion skipped so R_Y stays a Pauli
     rotation, no U4 -> no NotImplementedError), run the backend, and compare the measurement
     record distribution to clifft's own exact probabilities.

Run: clifft_env/bin/python -m nearclifford_backend.virtual_axis.test_ry_rotation
"""
import itertools
import sys

sys.path.insert(0, "/home/jung/clifft-paper")
import numpy as np

import clifft
import nearclifford_backend.backend as bk
from nearclifford_backend.block_magic import _apply_pauli_local
from nearclifford_backend.virtual_axis.virtual_axis import _herm
from nearclifford_backend.virtual_axis.fused_single_frame import (
    FusedSingleFrame, compile_circuit)

I2 = np.eye(2, dtype=complex)
Xm = np.array([[0, 1], [1, 0]], complex)
Ym = np.array([[0, -1j], [1j, 0]], complex)
Hm = np.array([[1, 1], [1, -1]], complex) / np.sqrt(2)
Sm = np.array([[1, 0], [0, 1j]], complex)


def _kron(lst):
    M = lst[0]
    for o in lst[1:]:
        M = np.kron(o, M)               # qubit 0 = LSB (innermost), matches _apply_pauli_local
    return M


def _g1(U, q, n):
    a = [I2] * n
    a[q] = U
    return _kron(a)


def _gcx(c, t, n):
    P0 = np.array([[1, 0], [0, 0]], complex)
    P1 = np.array([[0, 0], [0, 1]], complex)
    return (_kron([P0 if i == c else I2 for i in range(n)]) +
            _kron([P1 if i == c else (Xm if i == t else I2) for i in range(n)]))


def _dense_rot(psi, n, x, z, theta):
    _, _, ph = _herm((x, z, 0))                         # Hermitian Pauli i^ph X^x Z^z
    Pv = _apply_pauli_local(list(range(n)), psi, x, z, ph)
    return np.cos(theta / 2) * psi - 1j * np.sin(theta / 2) * Pv


def _fid(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 1.0 if na < 1e-12 and nb < 1e-12 else 0.0
    return abs(complex(np.vdot(a, b))) / (na * nb)


# ---------------------------------------------------------------- A. engine-direct
def _engine_case(n, ops, mq):
    """ops: ('h',q)/('s',q,dag)/('cx',c,t)/('ry',q,th)/('rx',q,th)/('rz',q,th); measure Z_mq.
    For each forced outcome compare engine statevector (+ leftover pending) to the dense ref."""
    psi = np.zeros(1 << n, complex); psi[0] = 1.0
    for op in ops:
        if op[0] == 'h':   psi = _g1(Hm, op[1], n) @ psi
        elif op[0] == 's': psi = _g1(Sm.conj().T if op[2] else Sm, op[1], n) @ psi
        elif op[0] == 'cx':psi = _gcx(op[1], op[2], n) @ psi
        elif op[0] == 'ry':psi = _dense_rot(psi, n, 1 << op[1], 1 << op[1], op[2])
        elif op[0] == 'rx':psi = _dense_rot(psi, n, 1 << op[1], 0, op[2])
        elif op[0] == 'rz':psi = _dense_rot(psi, n, 0, 1 << op[1], op[2])
    ok = True
    for b in (0, 1):
        Pv = _apply_pauli_local(list(range(n)), psi, 0, 1 << mq, 0)
        proj = 0.5 * (psi + (1 if b == 0 else -1) * Pv)
        nb = np.linalg.norm(proj)
        ref = proj / nb if nb > 1e-12 else proj
        e = FusedSingleFrame(n)
        for op in ops:
            if op[0] == 'h':   e.h(op[1])
            elif op[0] == 's': e.s(op[1], dag=op[2])
            elif op[0] == 'cx':e.cx(op[1], op[2])
            elif op[0] == 'ry':e.apply_rotation(1 << op[1], 1 << op[1], op[2])
            elif op[0] == 'rx':e.apply_rotation(1 << op[1], 0, op[2])
            elif op[0] == 'rz':e.apply_rotation(0, 1 << op[1], op[2])
        e._forced = [b]
        e.measure_z(mq)
        sv = e.statevector()
        for uid in sorted(e.pending):                   # apply still-deferred (commuting) rots
            r = e.pending[uid]
            Pv = _apply_pauli_local(list(range(n)), sv, r[0], r[1], r[2])
            sv = np.cos(r[3] / 2) * sv - 1j * np.sin(r[3] / 2) * Pv
        nrm = np.linalg.norm(sv); sv = sv / nrm if nrm > 1e-12 else sv
        f = _fid(sv, ref)
        ok = ok and f > 1 - 1e-7
    return ok


def test_engine_direct():
    cases = [
        ('1q Y', 1, [('ry', 0, 0.7)], 0),
        ('1q Y then H', 1, [('ry', 0, 1.3), ('h', 0)], 0),
        ('1q X via S-conj', 1, [('s', 0, False), ('rx', 0, 0.9), ('s', 0, True)], 0),
        ('2q Y,Y,CX', 2, [('ry', 0, 0.7), ('ry', 1, 1.3), ('cx', 0, 1)], 0),
        ('2q H,Y,CX,Y,S', 2, [('h', 0), ('ry', 0, 1.1), ('cx', 0, 1), ('ry', 1, 0.9), ('s', 1, False)], 1),
        ('3q mixed', 3, [('ry', 0, 0.5), ('rx', 1, 1.2), ('cx', 0, 1), ('ry', 2, 0.8),
                         ('cx', 1, 2), ('s', 2, False), ('ry', 1, 0.6)], 2),
    ]
    allok = True
    for nm, n, ops, mq in cases:
        ok = _engine_case(n, ops, mq)
        allok &= ok
        print(f"  [{'OK' if ok else 'FAIL'}] engine-direct: {nm}")
    return allok


# ---------------------------------------------------------------- B. end-to-end vs clifft
def _backend_dist(stim, nshots):
    prog = compile_circuit(stim)                         # bytecode fusion skipped for R_Y
    nm = prog.num_measurements
    orig = bk.LazyNearClifford
    bk.LazyNearClifford = FusedSingleFrame
    d = {}
    try:
        for s in range(nshots):
            be = bk.NearCliffordBackend(lazy=True, drop_dead=False, structure_once=False)
            be.run_shot(prog, s)
            key = tuple(int(be.record.get(i, 0)) for i in range(nm))
            d[key] = d.get(key, 0) + 1
    finally:
        bk.LazyNearClifford = orig
    return {k: v / nshots for k, v in d.items()}, prog, nm


def _ref_dist(prog, nm):
    out = {}
    for bits in itertools.product([0, 1], repeat=nm):
        p = float(clifft.record_probabilities(prog, np.array([list(bits)], dtype=np.uint8))[0])
        if p > 1e-9:
            out[bits] = p
    return out


def test_end_to_end(nshots=8000, tol=0.03):
    # No reset gates (record_probabilities cannot marginalise hidden meas slots); |0> init.
    circuits = [
        ('1q RY(1.0) -> MZ', 'QUBIT_COORDS(0,0) 0\nR_Y(1.0) 0\nM 0\n'),
        ('1q RY(0.7) -> MX', 'QUBIT_COORDS(0,0) 0\nR_Y(0.7) 0\nMX 0\n'),
        ('2q RY,RY,CX -> M', 'QUBIT_COORDS(0,0) 0\nQUBIT_COORDS(1,0) 1\n'
                             'R_Y(0.7) 0\nR_Y(1.3) 1\nCX 0 1\nM 0 1\n'),
        ('3q RY+CX chain', 'QUBIT_COORDS(0,0) 0\nQUBIT_COORDS(1,0) 1\nQUBIT_COORDS(2,0) 2\n'
                           'R_Y(0.9) 0\nCX 0 1\nR_Y(1.1) 1\nCX 1 2\nR_Y(0.6) 2\nM 0 1 2\n'),
    ]
    allok = True
    for nm, stim in circuits:
        emp, prog, num = _backend_dist(stim, nshots)
        ref = _ref_dist(prog, num)
        keys = set(emp) | set(ref)
        err = max(abs(emp.get(k, 0) - ref.get(k, 0)) for k in keys)
        ok = err < tol
        allok &= ok
        print(f"  [{'OK' if ok else 'FAIL'}] e2e vs clifft-ref: {nm}  max|emp-ref|={err:.4f}")
    return allok


def test_d3r3_loud_fail():
    """coherent_d3_r3 with R_Y: no NotImplementedError (the U4 crash is fixed); instead the
    off-axis core (L=48 -> ~2^34 fused Pauli-sum terms) must LOUD-FAIL fast with
    LargeCoreNeedsProjectedTN (a COMPUTE wall) -- NOT hang, NOT crash, NOT silently build 2^W."""
    from nearclifford_backend.virtual_axis.fused_integrate import LargeCoreNeedsProjectedTN
    txt = open("qec_bench/circuits/coherent_d3_r3.stim").read().replace("R_Z(0.02)", "R_Y(0.02)")
    prog = compile_circuit(txt)
    orig = bk.LazyNearClifford
    bk.LazyNearClifford = FusedSingleFrame
    raised = None
    try:
        be = bk.NearCliffordBackend(lazy=True, drop_dead=False, structure_once=False)
        be.run_shot(prog, 7)
    except LargeCoreNeedsProjectedTN as e:
        raised = e
    except NotImplementedError:
        pass                                            # the OLD bug -> leave raised None (FAIL)
    finally:
        bk.LazyNearClifford = orig
    ok = raised is not None and raised.L >= 20
    msg = (f"L={raised.L}, ~2^{raised.estimated_terms.bit_length()-1} terms" if raised
           else "did NOT raise LargeCoreNeedsProjectedTN")
    print(f"  [{'OK' if ok else 'FAIL'}] coherent_d3_r3 R_Y loud-fails (projected-TN required): {msg}")
    return ok


if __name__ == "__main__":
    print("A. engine-direct (forced-outcome statevector fidelity):")
    a = test_engine_direct()
    print("B. end-to-end vs clifft exact reference (record_probabilities):")
    b = test_end_to_end()
    print("C. original-failure circuit (now loud-fail, not hang):")
    c = test_d3r3_loud_fail()
    allok = a and b and c
    print("R_Y ROTATION TESTS", "PASS" if allok else "FAIL")
    sys.exit(0 if allok else 1)
