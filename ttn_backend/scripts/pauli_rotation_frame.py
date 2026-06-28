"""Pauli-rotation frame analysis (EXACT combinatorics, not a proxy, not a TTN run).

Generalises the broken Z-only (A,f): push EVERY Clifford (CNOT/H/S/CZ/SWAP) into a
symplectic Clifford frame; each non-Clifford RZ(theta) on physical qubit q becomes
a PAULI-STRING rotation R_P(theta), P = U_C^dag Z_q U_C, stored as (P, theta).

The real boundary is NOT H -- it is the ANTICOMMUTATION structure of the pending
Pauli rotations. Commuting rotations are simultaneously Clifford-diagonalizable
(handled as one diagonal f-block); only the anticommuting core is irreducibly
active. The decisive number is the GF(2) rank of the anticommutation (symplectic
Gram) matrix: rank = 2k, where k = number of conjugate pairs = the dimension of
the genuinely-active magic. k small => tractable; k ~ n => hard.

We compute, exactly:
  total RZ count, unique Pauli rotations after merge, commutation-graph edges,
  connected components, greedy commuting-block coloring, anticommutation rank,
  and the per-measurement-round segmented ranks (measurements segment the analysis).
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict

import clifft
from ttn_backend import treewidth as T_mod
from ttn_backend import frame_layer as ds_mod

NONCLIFF_ROT = {"OP_ARRAY_T", "OP_ARRAY_T_DAG", "OP_ARRAY_ROT"}
EXPAND_ROT = {"OP_EXPAND_T", "OP_EXPAND_T_DAG", "OP_EXPAND_ROT"}
CLIFF_S = {"OP_ARRAY_S", "OP_ARRAY_S_DAG"}
THETA = {"OP_ARRAY_ROT": 0.02, "OP_ARRAY_T": 0.7853981633974483,
         "OP_ARRAY_T_DAG": -0.7853981633974483,
         "OP_EXPAND_ROT": 0.02, "OP_EXPAND_T": 0.7853981633974483,
         "OP_EXPAND_T_DAG": -0.7853981633974483}


def anticommute(pi, pj):
    xi, zi = pi; xj, zj = pj
    return (bin(xi & zj).count("1") + bin(zi & xj).count("1")) & 1


def gf2_rank(rows):
    basis = []
    for r in rows:
        x = r
        for b in basis:
            x = min(x, x ^ b)
        if x:
            basis.append(x); basis.sort(reverse=True)
    return len(basis)


def anticomm_rank(paulis):
    """GF(2) rank of the antisymmetric anticommutation Gram matrix."""
    m = len(paulis)
    rows = []
    for i in range(m):
        r = 0
        for j in range(m):
            if i != j and anticommute(paulis[i], paulis[j]):
                r |= 1 << j
        rows.append(r)
    return gf2_rank(rows)


def greedy_color_anticomm(paulis):
    """Proper coloring of the anticommutation graph; each color = a commuting block
    (simultaneously diagonalizable). Returns number of colors and max block size."""
    m = len(paulis)
    adj = [set() for _ in range(m)]
    for i in range(m):
        for j in range(i + 1, m):
            if anticommute(paulis[i], paulis[j]):
                adj[i].add(j); adj[j].add(i)
    order = sorted(range(m), key=lambda i: -len(adj[i]))
    color = {}
    for v in order:
        used = {color[u] for u in adj[v] if u in color}
        c = 0
        while c in used:
            c += 1
        color[v] = c
    ncolors = (max(color.values()) + 1) if color else 0
    sizes = Counter(color.values())
    edges = sum(len(a) for a in adj) // 2
    return ncolors, (max(sizes.values()) if sizes else 0), edges


def collect(prog):
    """Replay; push Cliffords into the symplectic frame; collect (P, theta) for
    every non-Clifford rotation. Returns list of rotation records with a segment
    id incremented at each active-measurement layer."""
    slot2id = {}; nextid = 0
    xcol = {}; zcol = {}

    def new_id(slot):
        nonlocal nextid
        if slot in slot2id:
            return slot2id[slot]
        i = nextid; nextid += 1
        slot2id[slot] = i; xcol[i] = (1 << i, 0); zcol[i] = (0, 1 << i)
        return i

    def pmul(a, b):
        return (a[0] ^ b[0], a[1] ^ b[1])

    def cnot(a, b):
        zcol[b] = pmul(zcol[a], zcol[b]); xcol[a] = pmul(xcol[a], xcol[b])

    def hgate(q):
        xcol[q], zcol[q] = zcol[q], xcol[q]

    def sgate(q):
        xcol[q] = pmul(xcol[q], zcol[q])

    def cz(a, b):
        xcol[a] = pmul(xcol[a], zcol[b]); xcol[b] = pmul(xcol[b], zcol[a])

    rots = []          # (P=(x,z), theta, segment)
    seg = 0
    prev_meas = False
    MEAS = {"OP_MEAS_ACTIVE_DIAGONAL", "OP_MEAS_ACTIVE_DIAGONAL_FORCED",
            "OP_MEAS_ACTIVE_INTERFERE", "OP_MEAS_ACTIVE_INTERFERE_FORCED",
            "OP_SWAP_MEAS_INTERFERE", "OP_SWAP_MEAS_INTERFERE_FORCED"}
    for k in range(len(prog)):
        inst = prog[k]; name = T_mod._opname(inst.opcode)
        a1 = int(inst.axis_1); a2 = int(inst.axis_2)
        # round boundary: a measurement BATCH ends (meas -> non-meas transition)
        is_meas = name in MEAS
        if prev_meas and not is_meas:
            seg += 1
        prev_meas = is_meas
        if name.startswith("OP_EXPAND"):
            i = new_id(a1)
            if name in EXPAND_ROT:
                rots.append((zcol[i], THETA[name], seg))
        elif name == "OP_ARRAY_CNOT":
            new_id(a1); new_id(a2); cnot(a1, a2)
        elif name == "OP_ARRAY_MULTI_CNOT":
            d = ds_mod._d(inst); new_id(a1)
            for c in ds_mod._bits(int(d["mask"])):
                if c != a1:
                    new_id(c); cnot(c, a1)
        elif name == "OP_ARRAY_SWAP":
            slot2id[a1], slot2id[a2] = slot2id.get(a2), slot2id.get(a1)
        elif name == "OP_ARRAY_CZ":
            new_id(a1); new_id(a2); cz(a1, a2)
        elif name in CLIFF_S:
            i = slot2id.get(a1)
            if i is not None: sgate(i)
        elif name in NONCLIFF_ROT:
            i = slot2id.get(a1) or new_id(a1)
            i = slot2id[a1]
            rots.append((zcol[i], THETA.get(name, 0.02), seg))
        elif name == "OP_ARRAY_H":
            i = slot2id.get(a1)
            if i is not None: hgate(i)
        elif name == "OP_ARRAY_U2":           # de-fuse H*RZ
            i = slot2id.get(a1)
            if i is not None:
                rots.append((zcol[i], 0.02, seg)); hgate(i)
        elif name == "OP_ARRAY_U4":           # de-fuse (H*RZ)_lo . CNOT(lo->hi)
            lo = slot2id.get(a1); hi = slot2id.get(a2)
            if lo is not None and hi is not None:
                cnot(lo, hi); rots.append((zcol[lo], 0.02, seg)); hgate(lo)
        elif name in ("OP_MEAS_ACTIVE_DIAGONAL", "OP_MEAS_ACTIVE_DIAGONAL_FORCED",
                      "OP_MEAS_ACTIVE_INTERFERE", "OP_MEAS_ACTIVE_INTERFERE_FORCED"):
            slot2id.pop(a1, None)
        elif name in ("OP_SWAP_MEAS_INTERFERE", "OP_SWAP_MEAS_INTERFERE_FORCED"):
            slot2id[a1] = slot2id.get(a2); slot2id.pop(a2, None)
    return rots, nextid, seg + 1


def merge(records):
    """Merge identical Pauli strings (angle add); drop angle==0 mod 2pi.
    Returns list of unique Pauli strings P=(x,z)."""
    d = defaultdict(float)
    for rec in records:
        P, th = rec[0], rec[1]
        d[P] += th
    twopi = 2 * 3.141592653589793
    return [P for P, th in d.items() if abs(((th + 1e-12) % twopi)) > 1e-6]


def analyze_set(paulis, label):
    if not paulis:
        print(f"  [{label}] empty"); return
    weights = [bin(x | z).count("1") for (x, z) in paulis]
    ncol, maxblock, edges = greedy_color_anticomm(paulis)
    arank = anticomm_rank(paulis)
    print(f"  [{label}] unique Pauli rotations = {len(paulis)}")
    print(f"      weight hist: {dict(sorted(Counter(weights).items()))}")
    print(f"      anticommute-graph edges = {edges}")
    print(f"      greedy commuting-block coloring = {ncol} blocks  (max block {maxblock})")
    print(f"      *** ANTICOMMUTATION RANK = {arank}  (k = conjugate pairs = {arank//2}) ***")
    return arank


def _selftest():
    # X0,Z0 anticommute -> rank 2 (k=1)
    assert anticomm_rank([(1, 0), (0, 1)]) == 2
    # Z0,Z1 commute -> rank 0
    assert anticomm_rank([(0, 1), (0, 2)]) == 0
    # X0,Z0,Z1: only X0/Z0 anticommute -> rank 2
    assert anticomm_rank([(1, 0), (0, 1), (0, 2)]) == 2
    # two independent conjugate pairs X0Z0, X1Z1 -> rank 4 (k=2)
    assert anticomm_rank([(1, 0), (0, 1), (2, 0), (0, 2)]) == 4
    # all-Z (commuting) -> rank 0
    assert anticomm_rank([(0, 1), (0, 2), (0, 3), (0, 5)]) == 0
    print("[selftest] anticomm_rank OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("circuit", nargs="?", default="coherent_d5_r5")
    args = ap.parse_args()
    _selftest()
    prog = clifft.compile(open(f"qec_bench/circuits/{args.circuit}.stim").read())
    rots, nid, nseg = collect(prog)
    print(f"=== {args.circuit} : Pauli-rotation frame (exact) ===")
    print(f"idents={nid}  measurement-segments={nseg}")
    print(f"total non-Clifford rotations (incl. de-fused U2/U4) = {len(rots)}")

    # ---- GLOBAL (all rotations, full ident space) ----
    glob = merge(rots)
    print("\nGLOBAL (all magic, single commuting analysis -- upper bound on active dim):")
    analyze_set(glob, "global")

    # ---- PER-SEGMENT (measurements segment the analysis; active dim per round) ----
    by_seg = defaultdict(list)
    for P, th, s in rots:
        by_seg[s].append((P, th, s))
    seg_ranks = []
    print("\nPER-SEGMENT (between active measurements -- the realizable active dim):")
    for s in sorted(by_seg):
        ps = merge(by_seg[s])
        if not ps:
            continue
        r = anticomm_rank(ps)
        seg_ranks.append(r)
    if seg_ranks:
        print(f"  segments with magic = {len(seg_ranks)}")
        print(f"  anticommutation rank per segment: hist {dict(sorted(Counter(seg_ranks).items()))}")
        print(f"  max per-segment rank = {max(seg_ranks)} (k={max(seg_ranks)//2});  "
              f"mean = {sum(seg_ranks)/len(seg_ranks):.1f}")
    print("\nINTERPRETATION:")
    print("  active register dimension ~ 2^k where k = (anticommutation rank)/2.")
    print("  small k  -> Pauli-rotation frame collapses the problem (WIN).")
    print("  k ~ log2(observed chi)=11 (d5_r5) -> matches the entanglement floor (no escape).")


if __name__ == "__main__":
    main()
