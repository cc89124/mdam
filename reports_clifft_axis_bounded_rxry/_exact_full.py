"""PART 2 -- exact deterministic Born validation on a FULL R_Y QEC circuit
(coherent_ry_d3_r1 or _r3), parametrized by argv[1] in {r1, r3}.

Per FIXED fault pattern (X_ERROR frozen to explicit X) and several backend trajectories:
  * per-measurement |dense_phys_p0 - backend_record_p0| < 1e-10
    (phase-grouped dense replay: gates in op order with correct fault timing; measurements
     within a gate-separated phase deferred and taken in the backend's EXACT exec order,
     projecting onto the realized record bit r; MR ancilla reset between phases);
  * realized-trajectory JOINT probability  P_dense == P_backend == P_clifft
    (clifft.record_probabilities single row -> independent exact cross-check).
dense = exact 2^17 statevector, independent of the backend.  No sampling.
"""
import sys
import numpy as np
sys.path.insert(0, "/home/jung/clifft-paper")
sys.path.insert(0, "/home/jung/clifft-paper/reports_clifft_axis_bounded_rxry")
import clifft
from _exact_oracle_lib import (Dense, parse_stim, capture_backend, backend_record_p0,
                               build_det_text, build_clean_det_text, fault_sites,
                               measurement_qubits)
from nearclifford_backend.clifft_axis.bounded import compile_bounded

AX = sys.argv[1] if len(sys.argv) > 1 else "ry"
TAG = sys.argv[2] if len(sys.argv) > 2 else "r1"
CIRC = f"qec_bench/circuits/coherent_{AX}_d3_{TAG}.stim"
used, ops = parse_stim(CIRC)
remap = {q: i for i, q in enumerate(used)}
n = len(used)
mq = measurement_qubits(ops)
mres = []
for op in ops:
    if op[0] in ("MR", "M"):
        mres.extend(op[0] == "MR" for _ in op[1])
sites = fault_sites(ops)
NM = len(mq)

# live/dead fault classification: live iff qubit measured AFTER the X_ERROR op
op_pos = {op[1]: j for j, op in enumerate(ops) if op[0] == "XE"}
later = [set() for _ in ops]
acc = set()
for j in range(len(ops) - 1, -1, -1):
    later[j] = set(acc)
    if ops[j][0] in ("MR", "M"):
        acc.update(ops[j][1])
LIVE = {(xe, q) for (xe, q) in sites if q in later[op_pos[xe]]}
print(f"circuit={CIRC}  qubits={n}  measurements={NM}  fault_sites={len(sites)}  "
      f"live={len(LIVE)} dead={len(sites)-len(LIVE)}")

GATES = {"RY", "H", "CX", "X"}


def dense_and_joint(seq, record, faultset):
    d = Dense(n)
    exec_order = [c for (c, _, _) in seq]
    cap = {c: (p0, b) for (c, p0, b) in seq}
    cur = 0
    worst = 0.0
    P_d = 1.0
    P_b = 1.0
    rows = []
    i = 0
    while i < len(ops):
        op = ops[i]
        k = op[0]
        if k == "RY":
            for q in op[2]:
                d.ry_turns(remap[q], op[1])
            i += 1
        elif k == "RX":
            for q in op[2]:
                d.rx_turns(remap[q], op[1])
            i += 1
        elif k == "H":
            for q in op[1]:
                d.h(remap[q])
            i += 1
        elif k == "X":
            for q in op[1]:
                d.x(remap[q])
            i += 1
        elif k == "CX":
            for c, t in op[1]:
                d.cx(remap[c], remap[t])
            i += 1
        elif k == "XE":
            for q in op[2]:
                if (op[1], q) in faultset and (op[1], q) in LIVE:
                    d.x(remap[q])
            i += 1
        elif k in ("MR", "M"):
            # collect a measurement PHASE: consecutive MR/M/XE/X (no gate between)
            phase = []          # (cidx, is_reset)
            j = i
            while j < len(ops) and ops[j][0] in ("MR", "M", "XE", "X"):
                oj = ops[j]
                if oj[0] in ("MR", "M"):
                    for _q in oj[1]:
                        phase.append((cur, oj[0] == "MR")); cur += 1
                elif oj[0] == "XE":
                    for q in oj[2]:
                        if (oj[1], q) in faultset and (oj[1], q) in LIVE:
                            d.x(remap[q])
                elif oj[0] == "X":
                    for q in oj[1]:
                        d.x(remap[q])
                j += 1
            pset = {c for c, _ in phase}
            resetf = {c: rf for c, rf in phase}
            block = [c for c in exec_order if c in pset]
            assert len(block) == len(phase), "phase/exec mismatch"
            for c in block:
                q = mq[c]
                r = record[c]
                p0, b = cap[c]
                dp0 = d.born_p0(remap[q])
                bp0 = backend_record_p0(p0, b, r)
                worst = max(worst, abs(dp0 - bp0))
                rows.append((c, q, dp0, bp0, abs(dp0 - bp0)))
                P_d *= dp0 if r == 0 else (1 - dp0)
                P_b *= bp0 if r == 0 else (1 - bp0)
                d.project(remap[q], r)
                if resetf[c]:
                    d.reset0(remap[q])
            i = j
        else:
            i += 1
    return worst, P_d, P_b, rows


def single(idx):
    return {sites[idx]}

ry_data = [i for i, (xe, q) in enumerate(sites) if xe == 0]
PATTERNS = [("no-fault", set())]
for i in ry_data[:4]:
    PATTERNS.append((f"single {sites[i]}", single(i)))
# a few faults from later layers / ancilla
for i in (1, len(sites) // 3, len(sites) // 2, len(sites) - 5):
    PATTERNS.append((f"single {sites[i]}", single(i)))
PATTERNS.append(("multi A", {sites[0], sites[5], sites[len(sites) // 2]}))
PATTERNS.append(("multi B", {sites[2], sites[9], sites[len(sites) - 3], sites[len(sites) - 1]}))
PATTERNS.append(("all-fault", set(sites)))

SEEDS = (3, 17)
overall = 0.0
joint_worst = 0.0
print(f"\n{'pattern':<26}{'seed':>5}{'per-step|Δ|':>13}{'P_clifft':>11}{'  rel d/b   d/c   b/c'}")
for (pname, fs) in PATTERNS:
    prog_b = compile_bounded(build_det_text(ops, fs))
    prog_c = compile_bounded(build_clean_det_text(ops, fs))
    for sd in SEEDS:
        seq, record = capture_backend(prog_b, sd)
        worst, P_d, P_b, rows = dense_and_joint(seq, record, fs)
        h = np.array([[record[i] for i in range(NM)]], dtype=np.uint8)
        P_c = float(np.asarray(clifft.record_probabilities(prog_c, h))[0])
        ref = max(P_c, 1e-300)
        rel_db = abs(P_d - P_b) / max(abs(P_b), 1e-300)
        rel_dc = abs(P_d - P_c) / ref
        rel_bc = abs(P_b - P_c) / ref
        overall = max(overall, worst)
        joint_worst = max(joint_worst, rel_db, rel_dc, rel_bc)
        flag = "" if (worst < 1e-10 and max(rel_dc, rel_bc) < 1e-8) else "  <== CHECK"
        print(f"{pname:<26}{sd:>5}{worst:>13.2e}{P_c:>11.3e}"
              f"   {rel_db:.0e} {rel_dc:.0e} {rel_bc:.0e}{flag}")

print(f"\n########## FULL d3_{TAG}: worst per-step |dense-backend| = {overall:.2e}  "
      f"(PASS<1e-10: {overall < 1e-10})")
print(f"########## worst relative JOINT disagreement = {joint_worst:.2e}  "
      f"(PASS<1e-8: {joint_worst < 1e-8}) ##########")
