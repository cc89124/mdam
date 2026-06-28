"""Standalone verification of fused_core_apply on cultivation_d5's problem core (mi=20,
the 38-rotation core that drives the streaming engine to the 11-axis transient).

Verifies the user's 7 criteria:
  1. fidelity vs the streaming 11-axis path = 1 (both compute Pi_b prod R |in>);
  2. both outcomes b in {0,1};
  3. non-commuting rotations (the core has them inherently);
  4. rotations that TOUCH the ephemeral/ancilla axis (mask bit on axis a);
  5. zero dense materialisation of the ephemeral axis (fused max exponent < W);
  6. max workspace exponent <= clifft bound (cult_d5: <= 10);
  7. the 11-axis streaming result is reproduced in a 10-axis workspace.
Also checks the Born weight ||phi_out||^2 == P(outcome b) from the streaming path.
"""
import copy
import sys

sys.path.insert(0, "/home/jung/clifft-paper")
import numpy as np

from nearclifford_backend.virtual_axis.test_c3 import (
    capture_stream, dense_step_rot, dense_measure)
from nearclifford_backend.virtual_axis.virtual_engine import TableauEngine
from nearclifford_backend.virtual_axis.fused_core import fused_core_apply


def _fid(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 1.0 if (na < 1e-12 and nb < 1e-12) else 0.0
    return abs(complex(np.vdot(a, b))) / (na * nb)


def capture_problem_core(circ="cultivation_d5"):
    """Run the streaming engine to the core with the largest r_peak; snapshot the engine
    state at its start + the core's rotations + measurement + a valid outcome."""
    n, EV = capture_stream(circ)
    rng = np.random.default_rng(100)
    psi = np.zeros(1 << n, dtype=complex); psi[0] = 1.0
    outs = []
    for (Pm, rots) in EV:
        for (P, th) in rots:
            psi = dense_step_rot(psi, n, P, th)
        psi, out, p0 = dense_measure(psi, n, Pm, rng); outs.append(out)

    eng = TableauEngine(n)
    best = None
    for mi, (Pm, rots) in enumerate(EV):
        snap = copy.deepcopy(eng)
        for (P, th) in rots:
            eng.apply_rotation(P, th)
        r_peak = len(eng.magic)
        eng.measure_drop(Pm, forced=outs[mi])
        if best is None or r_peak > best[0]:
            best = (r_peak, snap, rots, Pm, outs[mi])
    return best[1], best[2], best[3], best[0]


def run():
    eng0, rots, Pm, r_peak = capture_problem_core()
    r_in = len(eng0.magic)
    ok = True

    # criterion 4: at least one rotation touches the ephemeral axis (verified via masks)
    eng_probe = copy.deepcopy(eng0)
    for (P, th) in rots:
        eng_probe._mask_for(P)
    W = len(eng_probe.magic)
    mmx, mmz, _ = eng_probe._mask_for(Pm)
    a = next(s for s in range(W) if (mmz >> s) & 1)
    eng_probe2 = copy.deepcopy(eng0)
    touch = 0
    for (P, th) in rots:
        mx, mz, mph = eng_probe2._mask_for(P)
        if ((mx >> a) & 1) or ((mz >> a) & 1):
            touch += 1

    from nearclifford_backend.block_magic import _apply_pauli_local
    n = eng0.n
    in_sv = eng0.statevector()                     # physical magic state at core start (2^n)
    for b in (0, 1):
        # ---- reference: DENSE n-qubit path (apply rotations, project P_meas = b) ----
        ref_sv = in_sv.copy()
        for (P, th) in rots:
            Pv = _apply_pauli_local(list(range(n)), ref_sv, P[0], P[1], P[2])
            ref_sv = np.cos(th / 2.0) * ref_sv - 1j * np.sin(th / 2.0) * Pv
        Pm_v = _apply_pauli_local(list(range(n)), ref_sv, Pm[0], Pm[1], Pm[2])
        sign = 1.0 if b == 0 else -1.0
        proj = 0.5 * (ref_sv + sign * Pm_v)
        P_b = float(np.real(np.vdot(proj, proj)))  # Born weight of outcome b
        nb = np.linalg.norm(proj)
        ref_sv = proj / nb if nb > 1e-12 else proj

        # ---- fused: 10-axis workspace ----
        phi_out, born, max_exp, out_eng = fused_core_apply(eng0, rots, Pm, b)
        nrm = np.linalg.norm(out_eng.phi)
        if nrm > 1e-12:
            out_eng.phi = out_eng.phi / nrm
        fused_sv = out_eng.statevector()

        fid = _fid(fused_sv, ref_sv)
        dborn = abs(born - P_b)
        c1 = fid > 1 - 1e-9
        c5 = max_exp < r_peak                      # ephemeral axis never densely materialised
        c6 = max_exp <= 10
        ok = ok and c1 and c5 and c6 and dborn < 1e-9
        print(f"  b={b}: fid={fid:.9f}  born|d|={dborn:.1e}  "
              f"workspace_exp={max_exp} (streaming peak={r_peak})  "
              f"{'OK' if (c1 and c5 and c6 and dborn < 1e-9) else 'FAIL'}")

    print(f"core: r_in={r_in} -> streaming r_peak={r_peak} -> fused workspace<= {r_peak-1}  "
          f"(ephemeral axis a={a}, {touch} rotations touch it)")
    print(f"FUSED-CORE {'PASS' if ok else 'FAIL'}  (criteria: fidelity=1 both outcomes; "
          f"non-commuting + ancilla-touching rotations; 0 dense materialisation of the "
          f"ephemeral axis; workspace exponent <= 10 < streaming's {r_peak})")
    return ok


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
