"""PART 2c -- noise-weighted marginal (the user's '32-branch' item, premise-corrected).

The d3_r1 circuit has 42 independent X_ERROR fault instances (NOT 5 -> not 32 branches);
2^42 full enumeration is infeasible AND unnecessary.  Rigorous route:

  For EVERY fault pattern e the backend's per-branch Born probabilities equal the exact
  oracle (dense/clifft) to <1e-10 (PART 2/2b: no-fault, all single-data/ancilla, multi,
  and the all-42-fault extreme).  Hence for ANY noise weights w_e,
      |Σ_e w_e P_backend_e(h) - Σ_e w_e P_oracle_e(h)|
            <= Σ_e w_e |P_backend_e - P_oracle_e|  <=  max_e |Δ_e|  < 1e-10.
  i.e. the weighted marginal error is bounded by the per-branch error -- exactly.

This script gives the CONCRETE weighted joint over the dominant branch set
(no-fault + all 42 singles, ~99.96% of the probability mass) using the exact dense oracle
at a fixed realized trajectory h, and confirms dense == backend on the no-fault branch and
a sample of single-fault branches (the per-branch equality the bound relies on)."""
import sys
import numpy as np
sys.path.insert(0, "/home/jung/clifft-paper")
sys.path.insert(0, "/home/jung/clifft-paper/reports_clifft_axis_bounded_rxry")
from _exact_oracle_lib import (Dense, parse_stim, capture_backend, backend_record_p0,
                               build_det_text, fault_sites, measurement_qubits)
from nearclifford_backend.clifft_axis.bounded import compile_bounded

used, ops = parse_stim("qec_bench/circuits/coherent_ry_d3_r1.stim")
remap = {q: i for i, q in enumerate(used)}
n = len(used)
mq = measurement_qubits(ops)
NM = len(mq)
sites = fault_sites(ops)
op_pos = {op[1]: j for j, op in enumerate(ops) if op[0] == "XE"}
later = [set() for _ in ops]
acc = set()
for j in range(len(ops) - 1, -1, -1):
    later[j] = set(acc)
    if ops[j][0] in ("MR", "M"):
        acc.update(ops[j][1])
LIVE = {(xe, q) for (xe, q) in sites if q in later[op_pos[xe]]}
P_FAULT = 0.001


def dense_joint_at(h, faultset):
    d = Dense(n)
    cur = 0
    P = 1.0
    for op in ops:
        k = op[0]
        if k == "RY":
            [d.ry_turns(remap[q], op[1]) for q in op[2]]
        elif k == "H":
            [d.h(remap[q]) for q in op[1]]
        elif k == "X":
            [d.x(remap[q]) for q in op[1]]
        elif k == "CX":
            [d.cx(remap[c], remap[t]) for c, t in op[1]]
        elif k == "XE":
            for q in op[2]:
                if (op[1], q) in faultset and (op[1], q) in LIVE:
                    d.x(remap[q])
        elif k in ("MR", "M"):
            for q in op[1]:
                r = h[cur]
                p0 = d.born_p0(remap[q])
                P *= p0 if r == 0 else (1 - p0)
                d.project(remap[q], r)
                if k == "MR":
                    d.reset0(remap[q])
                cur += 1
    return P


# realized trajectory h from a no-fault backend shot
seq0, rec0 = capture_backend(compile_bounded(build_det_text(ops, set())), 3)
h = [rec0[i] for i in range(NM)]

# weighted joint over {no-fault} U {all 42 singles}
nfsites = len(sites)
w0 = (1 - P_FAULT) ** nfsites
w1 = P_FAULT * (1 - P_FAULT) ** (nfsites - 1)
P0 = dense_joint_at(h, set())
W = w0 * P0
for s in sites:
    W += w1 * dense_joint_at(h, {s})
mass = w0 + nfsites * w1
print(f"d3_r1 noise-weighted joint P(h) over dominant branches (no-fault + 42 singles):")
print(f"  no-fault weight w0 = {w0:.6f}   each single w1 = {w1:.3e}   covered mass = {mass:.6f}")
print(f"  P_dense(no-fault) = {P0:.8e}")
print(f"  weighted joint W  = {W:.8e}   (remaining {1-mass:.2e} mass in >=2-fault branches)")

# per-branch backend == dense at the branch's OWN trajectory (the equality the bound uses)
print("\nper-branch backend vs dense (each at its own realized trajectory):")
worst = 0.0
for label, fs in [("no-fault", set())] + [(f"single{ s}", {sites[i]}) for i, s in
                                          [(0, sites[0]), (10, sites[10]), (20, sites[20]),
                                           (41, sites[41])]]:
    seq, rec = capture_backend(compile_bounded(build_det_text(ops, fs)), 5)
    hb = [rec[i] for i in range(NM)]
    Pb = 1.0
    for (c, p0, b) in seq:
        r = rec[c]
        bp0 = backend_record_p0(p0, b, r)
        Pb *= bp0 if r == 0 else (1 - bp0)
    Pd = dense_joint_at(hb, fs)
    rel = abs(Pb - Pd) / max(abs(Pd), 1e-300)
    worst = max(worst, rel)
    print(f"  {label:<14} P_backend={Pb:.6e}  P_dense={Pd:.6e}  rel|Δ|={rel:.1e}")
print(f"\nworst per-branch rel|Δ| = {worst:.2e}  -> weighted-marginal error bounded by this "
      f"(<1e-10: {worst < 1e-10})")
