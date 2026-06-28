"""Find the FIRST core where flush_core_virtual diverges from the streaming engine, and
dump its classification so the buggy contraction path is isolated."""
import copy, os, sys
os.chdir("/home/jung/clifft-paper"); sys.path.insert(0, "/home/jung/clifft-paper")
sys.setrecursionlimit(50000)
import numpy as np
from nearclifford_backend.virtual_axis.virtual_engine import TableauEngine
from nearclifford_backend.virtual_axis.fused_integrate import flush_core_virtual, classify_core
from nearclifford_backend.virtual_axis.test_c3 import capture_stream, dense_step_rot, dense_measure, _fid


def run(circ, seed=100):
    n, EV = capture_stream(circ)
    rng = np.random.default_rng(seed)
    psi = np.zeros(1 << n, dtype=complex); psi[0] = 1.0
    outs = []
    for (Pm, rots) in EV:
        for (P, th) in rots:
            psi = dense_step_rot(psi, n, P, th)
        psi, out, p0 = dense_measure(psi, n, Pm, rng); outs.append(out)

    eng_s = TableauEngine(n)          # streaming ground truth
    eng_f = TableauEngine(n)          # fused
    psi2 = np.zeros(1 << n, dtype=complex); psi2[0] = 1.0
    for mi, (Pm, rots) in enumerate(EV):
        kind, info = classify_core(eng_f, rots, Pm)
        # dense reference for this core
        for (P, th) in rots:
            psi2 = dense_step_rot(psi2, n, P, th)
        Pv = np.zeros_like(psi2)
        from nearclifford_backend.block_magic import _apply_pauli_local
        Pv = _apply_pauli_local(list(range(n)), psi2, Pm[0], Pm[1], Pm[2])
        sign = 1.0 if outs[mi] == 0 else -1.0
        proj = 0.5 * (psi2 + sign * Pv); nrm = np.linalg.norm(proj)
        psi2 = proj / nrm if nrm > 1e-12 else proj
        # streaming
        for (P, th) in rots:
            eng_s.apply_rotation(P, th)
        eng_s.measure_drop(Pm, forced=outs[mi])
        # fused
        flush_core_virtual(eng_f, rots, Pm, forced=outs[mi])
        fF = _fid(eng_f.statevector(), psi2)
        fS = _fid(eng_s.statevector(), psi2)
        if fF < 1 - 1e-7:
            print(f"{circ}: DIVERGE at mi={mi}  kind={kind} info={info}  "
                  f"fid_fused={fF:.6f} fid_stream={fS:.6f}  out={outs[mi]}")
            return mi, kind, info
    print(f"{circ}: all {len(EV)} cores OK")
    return None


if __name__ == "__main__":
    for c in (sys.argv[1:] or ["cultivation_d3", "cultivation_d5"]):
        run(c)
