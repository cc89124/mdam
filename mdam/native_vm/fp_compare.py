#!/usr/bin/env python
"""Op-stream-aligned comparison of the Python (FPP) and native (FPN) inverse-frame fingerprints.

Why op-stream alignment: Python's run_shot iterates PROG indices (recorder fires BEFORE the
IGNORE_OPS skip, so FPP step == prog index, INCLUDING ignored/dropped ops).  The native VM iterates
the TRANSLATED stream (ignored ops + drop-only ops removed).  So FPP step != FPN step; the indices
drift.  We instead align by OP CONTENT.

Semantics:
  FPP[s]  = inverse-frame state BEFORE prog instruction s executes (op named on the line).
  FPN[i]  = inverse-frame state AFTER translated op i executes (op named on the line).

So native FPN[i] (state after its op) corresponds to Python FPP[s+1] where prog instruction s is the
SOURCE of translated op i.  We recover src(i) by greedily matching the native op stream against the
python op stream (skipping python ops that translate drops), then compare FPN[i] to FPP[src(i)+1].

Native prints two 64-bit words comma-separated; we fold to word0|(word1<<64) to match python big-ints.
Phase is ignored on both sides (canonical fingerprint).
"""
import sys

# MO opcode number -> set of acceptable prog opcode NAMEs (from verify_mdam_oneshot.translate).
MO2NAME = {
    0: {"OP_FRAME_H"}, 1: {"OP_FRAME_CNOT"}, 2: {"OP_FRAME_CZ"}, 3: {"OP_FRAME_SWAP"},
    4: {"OP_FRAME_S", "OP_FRAME_S_DAG"}, 5: {"OP_APPLY_PAULI"}, 6: {"OP_NOISE"}, 7: {"OP_NOISE_BLOCK"},
    8: {"OP_READOUT_NOISE"}, 9: {"OP_MEAS_DORMANT_STATIC"}, 10: {"OP_MEAS_DORMANT_RANDOM"},
    11: {"OP_ARRAY_CNOT"}, 12: {"OP_ARRAY_CZ"}, 13: {"OP_ARRAY_MULTI_CNOT"}, 14: {"OP_ARRAY_MULTI_CZ"},
    15: {"OP_ARRAY_T"}, 16: {"OP_ARRAY_T_DAG"}, 17: {"OP_ARRAY_S"}, 18: {"OP_EXPAND_T"},
    19: {"OP_EXPAND_T_DAG"}, 20: {"OP_SWAP_MEAS_INTERFERE"}, 21: {"OP_ARRAY_ROT"}, 22: {"OP_EXPAND_ROT"},
    23: {"OP_ARRAY_SWAP"}, 24: {"OP_MEAS_ACTIVE_DIAGONAL"}, 25: {"OP_MEAS_ACTIVE_INTERFERE"},
    26: {"OP_EXPAND"}, 27: {"OP_ARRAY_H"}, 28: {"OP_ARRAY_U2"}, 29: {"OP_ARRAY_U4"}, 30: {"OP_END"},
}

def fold(s):
    if "," in s:
        v = 0
        for w, hx in enumerate(s.split(",")):
            v |= int(hx, 16) << (64 * w)
        return v
    return int(s, 16)

def parse_py(path):
    """Return list of (step, name, a1, a2, {q:(x,z)}), n."""
    rows = []; n = None
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("FPMETA"):
                for tok in line.split()[1:]:
                    if tok.startswith("n="): n = int(tok[2:])
                continue
            if not line.startswith("FPP"): continue
            parts = line.split()
            step = int(parts[1]); name = parts[2][5:]; a1 = int(parts[3][3:]); a2 = int(parts[4][3:])
            m = {}
            for p in parts[5:]:
                qi, xs, zs = p.split(":"); m[int(qi)] = (fold(xs), fold(zs))
            rows.append((step, name, a1, a2, m))
    return rows, n

def parse_nat(path):
    rows = []; n = None
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("FPMETA"):
                for tok in line.split()[1:]:
                    if tok.startswith("n="): n = int(tok[2:])
                continue
            if not line.startswith("FPN"): continue
            parts = line.split()
            step = int(parts[1]); op = int(parts[2][3:]); a1 = int(parts[3][3:]); a2 = int(parts[4][3:])
            m = {}
            for p in parts[5:]:
                qi, xs, zs = p.split(":"); m[int(qi)] = (fold(xs), fold(zs))
            rows.append((step, op, a1, a2, m))
    return rows, n

if __name__ == "__main__":
    pyf = sys.argv[1] if len(sys.argv) > 1 else "/tmp/fp_py.txt"
    natf = sys.argv[2] if len(sys.argv) > 2 else "/tmp/fp_nat.err"
    py, npy = parse_py(pyf)
    nat, nnat = parse_nat(natf)
    print(f"python n={npy} FPP rows={len(py)} (step {py[0][0]}..{py[-1][0]})")
    print(f"native n={nnat} FPN rows={len(nat)} (step {nat[0][0]}..{nat[-1][0]})")

    # index python rows by step for FPP[s+1] lookup
    py_by_step = {r[0]: r for r in py}

    # Greedy op-stream alignment: for each native op i (in order), advance the python pointer to the
    # next FPP whose opcode NAME is in MO2NAME[op] and whose a1/a2 match.  That FPP's step is src(i).
    pj = 0; aligned = []   # list of (i, native_row, src_step)
    for (i, op, na1, na2, nm) in nat:
        names = MO2NAME.get(op, set())
        found = None
        scan = pj
        while scan < len(py):
            (ps, pname, pa1, pa2, pmm) = py[scan]
            if pname in names and pa1 == na1 and (pa2 == na2 or op in (5, 6, 7, 9, 13, 14, 21, 22, 26)):
                found = scan; break
            scan += 1
        if found is None:
            print(f"!! could not align native op i={i} op={op} a1={na1} a2={na2} (python ptr {pj})")
            break
        aligned.append((i, op, na1, na2, nm, py[found][0]))
        pj = found + 1

    print(f"aligned {len(aligned)}/{len(nat)} native ops to python prog indices")

    # Now compare each native FPN[i] (state AFTER op i) to python FPP[src+1] (state after prog op src).
    first = None
    for (i, op, na1, na2, nm, src) in aligned:
        tgt = src + 1
        prow = py_by_step.get(tgt)
        if prow is None:
            continue
        pmm = prow[4]
        diffs = []
        for q in sorted(set(nm) | set(pmm)):
            nv = nm.get(q); pv = pmm.get(q)
            if nv != pv:
                diffs.append((q, nv, pv))
        if diffs and first is None:
            first = (i, op, na1, na2, src, tgt, diffs)
            break

    if first is None:
        print("\nNO DIVERGENCE: every aligned native op matches python (state-after) on every qubit.")
        sys.exit(0)
    i, op, na1, na2, src, tgt, diffs = first
    print(f"\n*** FIRST DIVERGENT STEP ***")
    print(f"    native translated step i={i}  (op={op} {sorted(MO2NAME.get(op,{'?'}))} a1={na1} a2={na2})")
    print(f"    = state AFTER python prog instruction src={src}; compared to python FPP[{tgt}]")
    print(f"    differing qubits: {len(diffs)}")
    for (q, nv, pv) in diffs[:20]:
        nx = hex(nv[0]) if nv else None; nz = hex(nv[1]) if nv else None
        px = hex(pv[0]) if pv else None; pz = hex(pv[1]) if pv else None
        print(f"      qubit i={q}: native (x,z)=({nx},{nz})  python (x,z)=({px},{pz})")
