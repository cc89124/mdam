"""PART 2b -- exact validation on the MULTI-ROUND coherent_ry_d3_r3 circuit.

clifft.record_probabilities cannot be used here (the ancilla are RESET and reused each
round; clifft rejects resets and MR->M would corrupt later rounds).  The exact oracle is
the dense 2^17 statevector run in STRICT OP ORDER (correct fault timing + ancilla reset),
which was independently validated against BOTH clifft and the backend to <1e-15 on d3_r1
and all calibration cases.

Test (order-independent, so dense's op-order and the backend's exec-order are comparable):
  realized-trajectory JOINT probability  P_dense(h) == P_backend(h)  < 1e-10 (relative).
A single per-measurement error anywhere on the path would blow up this product.
"""
import sys
import numpy as np
sys.path.insert(0, "/home/jung/clifft-paper")
sys.path.insert(0, "/home/jung/clifft-paper/reports_clifft_axis_bounded_rxry")
from _exact_oracle_lib import (Dense, parse_stim, capture_backend, backend_record_p0,
                               build_det_text, fault_sites, measurement_qubits)
from nearclifford_backend.clifft_axis.bounded import compile_bounded

CIRC = "qec_bench/circuits/coherent_ry_d3_r3.stim"
used, ops = parse_stim(CIRC)
remap = {q: i for i, q in enumerate(used)}
n = len(used)
mq = measurement_qubits(ops)
sites = fault_sites(ops)
NM = len(mq)
op_pos = {op[1]: j for j, op in enumerate(ops) if op[0] == "XE"}
later = [set() for _ in ops]
acc = set()
for j in range(len(ops) - 1, -1, -1):
    later[j] = set(acc)
    if ops[j][0] in ("MR", "M"):
        acc.update(ops[j][1])
LIVE = {(xe, q) for (xe, q) in sites if q in later[op_pos[xe]]}
print(f"circuit={CIRC}  qubits={n}  measurements={NM}  fault_sites={len(sites)}  live={len(LIVE)}")


def dense_joint(record, faultset):
    """dense in STRICT op order; faults at their op position (live only -- dead faults are
    on qubits never measured again and cannot affect any record); measurements at their op
    position in cidx order, projecting onto r, MR ancilla reset.  Returns P_dense(h)."""
    d = Dense(n)
    cur = 0
    P_d = 1.0
    cond = []
    for op in ops:
        k = op[0]
        if k == "RY":
            for q in op[2]:
                d.ry_turns(remap[q], op[1])
        elif k == "H":
            for q in op[1]:
                d.h(remap[q])
        elif k == "X":
            for q in op[1]:
                d.x(remap[q])
        elif k == "CX":
            for c, t in op[1]:
                d.cx(remap[c], remap[t])
        elif k == "XE":
            for q in op[2]:
                if (op[1], q) in faultset and (op[1], q) in LIVE:
                    d.x(remap[q])
        elif k in ("MR", "M"):
            for q in op[1]:
                r = record[cur]
                dp0 = d.born_p0(remap[q])
                cond.append(dp0 if r == 0 else (1 - dp0))
                P_d *= dp0 if r == 0 else (1 - dp0)
                d.project(remap[q], r)
                if k == "MR":
                    d.reset0(remap[q])
                cur += 1
    return P_d, cond


def single(idx):
    return {sites[idx]}

ry_data = [i for i, (xe, q) in enumerate(sites) if xe == 0]
PATTERNS = [("no-fault", set())]
for i in ry_data[:4]:
    PATTERNS.append((f"single {sites[i]}", single(i)))
for i in (1, len(sites) // 3, len(sites) // 2, 2 * len(sites) // 3, len(sites) - 4):
    PATTERNS.append((f"single {sites[i]}", single(i)))
PATTERNS.append(("multi A", {sites[0], sites[5], sites[len(sites) // 2]}))
PATTERNS.append(("multi B", {sites[2], sites[9], sites[len(sites) - 3], sites[len(sites) - 1]}))
PATTERNS.append(("all-fault", set(sites)))

SEEDS = (3, 17, 29)
worst_joint = 0.0
print(f"\n{'pattern':<26}{'seed':>5}{'P_backend':>13}{'P_dense':>13}{'rel|Δ|':>11}")
for (pname, fs) in PATTERNS:
    prog_b = compile_bounded(build_det_text(ops, fs))
    for sd in SEEDS:
        seq, record = capture_backend(prog_b, sd)
        # backend joint = product of backend conditionals along exec order
        P_b = 1.0
        for (c, p0, b) in seq:
            r = record[c]
            bp0 = backend_record_p0(p0, b, r)
            P_b *= bp0 if r == 0 else (1 - bp0)
        P_d, cond = dense_joint(record, fs)
        rel = abs(P_b - P_d) / max(abs(P_d), 1e-300)
        worst_joint = max(worst_joint, rel)
        flag = "  <== CHECK" if rel > 1e-9 else ""
        print(f"{pname:<26}{sd:>5}{P_b:>13.4e}{P_d:>13.4e}{rel:>11.1e}{flag}")

print(f"\n########## d3_r3: worst relative JOINT |P_backend - P_dense| = {worst_joint:.2e}  "
      f"(PASS<1e-9: {worst_joint < 1e-9}) ##########")
