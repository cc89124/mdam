"""PART 1 -- calibration (CORRECTED): three independent exact objects must agree to
<1e-12 on tiny circuits, in the RECORD-bit convention:
   dense statevector  vs  bounded backend  vs  clifft.record_probabilities
Comparison is deterministic Born P(record_i = 0 | exec-prefix); dense follows the
backend's execution order and projects onto the realized record bit r."""
import sys, itertools
import numpy as np
sys.path.insert(0, "/home/jung/clifft-paper")
sys.path.insert(0, "/home/jung/clifft-paper/reports_clifft_axis_bounded_rxry")
import clifft
from _exact_oracle_lib import (Dense, parse_stim, capture_backend, backend_record_p0,
                               build_clean_det_text)
from nearclifford_backend.clifft_axis.bounded import compile_bounded


def meas_info(ops):
    """program-order list of (qubit, is_reset) for each measurement (cidx)."""
    out = []
    for op in ops:
        if op[0] in ("MR", "M"):
            out.extend((q, op[0] == "MR") for q in op[1])
    return out


def clifft_cond_exec(prog_c, nm, h, exec_order):
    """exact clifft P(record_c = 0 | {record_p = h_p for p measured before c in exec order}).
       Conditions on the SAME prefix set the backend used (exec order), not cidx order.
       cheap here (nm tiny): enumerate the full joint once. Returns dict cidx -> p0."""
    recs = np.array(list(itertools.product((0, 1), repeat=nm)), dtype=np.uint8)
    P = np.asarray(clifft.record_probabilities(prog_c, recs))
    out = {}
    seen = []
    for c in exec_order:
        if seen:
            pref = np.all(recs[:, seen] == np.array([h[p] for p in seen], dtype=np.uint8), axis=1)
        else:
            pref = np.ones(len(recs), bool)
        denom = P[pref].sum()
        num = P[pref & (recs[:, c] == 0)].sum()
        out[c] = (num / denom) if denom > 1e-300 else None
        seen.append(c)
    return out


def run_case(name, text, seed):
    used, ops = parse_stim(text, is_text=True)
    remap = {q: i for i, q in enumerate(used)}
    n = len(used)
    mi = meas_info(ops)
    prog = compile_bounded(text)
    nm = prog.num_measurements
    seq, record = capture_backend(prog, seed)
    prog_c = compile_bounded(build_clean_det_text(ops, set()))
    h = [record[i] for i in range(nm)]
    exec_order = [cidx for (cidx, _, _) in seq]
    cl = clifft_cond_exec(prog_c, nm, h, exec_order)

    # dense: apply all non-measurement ops (orig order), then measure in EXEC order
    d = Dense(n)
    for op in ops:
        if op[0] == "RY":
            for q in op[2]:
                d.ry_turns(remap[q], op[1])
        elif op[0] == "H":
            for q in op[1]:
                d.h(remap[q])
        elif op[0] == "X":
            for q in op[1]:
                d.x(remap[q])
        elif op[0] == "CX":
            for c, t in op[1]:
                d.cx(remap[c], remap[t])
        # R / XE(none in calib) / measurements handled below

    worst = 0.0
    rows = []
    for (cidx, p0, b) in seq:
        q, is_reset = mi[cidx]
        r = record[cidx]
        dp0 = d.born_p0(remap[q])
        bp0 = backend_record_p0(p0, b, r)
        cp0 = cl[cidx]
        dd_b = abs(dp0 - bp0)
        dd_c = abs(dp0 - cp0) if cp0 is not None else 0.0
        worst = max(worst, dd_b, dd_c)
        rows.append((cidx, q, dp0, bp0, cp0, dd_b, dd_c))
        d.project(remap[q], r)
        if is_reset:
            d.reset0(remap[q])
    print(f"\n=== {name} ===  (exec order; cidx shown)")
    for (cidx, q, dp0, bp0, cp0, dd_b, dd_c) in rows:
        flag = "  <== DIVERGES" if max(dd_b, dd_c) > 1e-12 else ""
        cps = f"{cp0:.12f}" if cp0 is not None else "  (det)   "
        print(f"  cidx{cidx:>2} q={q:<2} dense={dp0:.12f} backend={bp0:.12f} clifft={cps} "
              f"|d-b|={dd_b:.1e} |d-c|={dd_c:.1e}{flag}")
    print(f"  worst |Δ| = {worst:.2e}  {'PASS (<1e-12)' if worst < 1e-12 else 'FAIL'}")
    return worst


CASES = [
    ("1q RY",        "R_Y(0.02) 0\nM 0"),
    ("1q RY x2",     "R_Y(0.13) 0\nR_Y(0.07) 0\nM 0"),
    ("2q RY+CX",     "R_Y(0.11) 0 1\nCX 0 1\nR_Y(0.09) 0 1\nM 0 1"),
    ("2q H+RY+CX",   "H 0\nR_Y(0.2) 0 1\nCX 0 1\nH 1\nR_Y(0.05) 0 1\nM 0 1"),
    ("3q deep",      "R_Y(0.15) 0 1 2\nCX 0 1\nCX 1 2\nR_Y(0.08) 0 1 2\nH 1\nCX 2 0\nR_Y(0.04) 0 1 2\nM 0 1 2"),
    ("2q +X(fault)", "X 0\nR_Y(0.11) 0 1\nCX 0 1\nR_Y(0.09) 0 1\nM 0 1"),
]
allw = 0.0
for nm, c in CASES:
    for sd in (3, 7, 11, 19):
        allw = max(allw, run_case(f"{nm} seed={sd}", c, sd))
print(f"\n########## CALIBRATION worst |Δ| (dense/backend/clifft) = {allw:.2e} "
      f"{'-> ALL PASS' if allw < 1e-12 else '-> FAIL'} ##########")
