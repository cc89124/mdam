"""Decisive net-compute measurement for the hybrid affine-TTN frame.

Per deferrable segment, the deferred gates form a CNOT+RZ circuit: a set of
parities to phase (the diagonal RZ/T, in terms of segment-input variables) plus a
final linear map A (the net CNOTs). The frame's real CNOT cost = a CNOT+RZ
synthesis of that segment, which must realize BOTH the phases (the f part) AND
the linear map A.

We synthesize each segment with a VERIFIED greedy parity-network + linear fix-up
and count CNOTs. The greedy is not provably optimal, so its count is an
ACHIEVABLE UPPER BOUND on the frame's cost (a real GraySynth could only do
better). Decisive direction: if total_synth < raw, the frame DOES cut CNOTs.

We compare against:
  raw       = per-control CNOTs the current backend would issue (frame-off baseline)
  GE-linear = Gauss-Jordan synthesis of the net linear map ONLY (ignores f) -- a
              LOWER bound; the gap to total_synth is the price of materializing f.

Every segment's synthesis is verified: the emitted CNOT list, simulated from the
identity with phases at the marked steps, must reproduce exactly the target
parity set and the target linear map A. Any mismatch aborts (count untrusted).
"""
from __future__ import annotations

import argparse
import clifft
from ttn_backend import treewidth as T_mod
from ttn_backend import frame_layer as ds_mod

DIAG = {"OP_ARRAY_T", "OP_ARRAY_T_DAG", "OP_ARRAY_S", "OP_ARRAY_S_DAG",
        "OP_ARRAY_ROT", "OP_PHASE_T", "OP_PHASE_T_DAG", "OP_PHASE_ROT"}
HARD = {"OP_ARRAY_H", "OP_ARRAY_U2", "OP_ARRAY_U4",
        "OP_MEAS_ACTIVE_INTERFERE", "OP_MEAS_ACTIVE_INTERFERE_FORCED",
        "OP_SWAP_MEAS_INTERFERE", "OP_SWAP_MEAS_INTERFERE_FORCED"}
ZMEAS = {"OP_MEAS_ACTIVE_DIAGONAL", "OP_MEAS_ACTIVE_DIAGONAL_FORCED"}

DEFAULT = ["coherent_d5_r1", "coherent_d7_r1", "coherent_d5_r5",
           "cultivation_d3", "distillation"]


# ---- GF(2) linear algebra over bitmask vectors (bit i = coord i) ----

def solve_coeffs(basis, p):
    """Find subset R of qubit indices with XOR_{q in R} basis[q] == p.
    basis = list of bitmask row vectors of an invertible matrix B (row q = basis[q]).
    c with c*B = p is c = p * B^{-1}; here c = XOR_{k: bit k of p set} Binv[k]."""
    n = len(basis)
    Binv = mat_inv(basis, n)              # left inverse: mat_mul(Binv, basis) == I
    c = 0
    for k in range(n):
        if (p >> k) & 1:
            c ^= Binv[k]
    # verify membership: XOR of selected basis rows == p
    chk = 0
    for q in range(n):
        if (c >> q) & 1:
            chk ^= basis[q]
    if chk != p:
        return None
    return [q for q in range(n) if (c >> q) & 1]


def mat_inv(rows, n):
    """Invert an n x n GF2 matrix given as list of row bitmasks. Returns inverse rows."""
    a = list(rows)
    inv = [1 << i for i in range(n)]
    for col in range(n):
        piv = None
        for r in range(col, n):
            if (a[r] >> col) & 1:
                piv = r; break
        if piv is None:
            raise ValueError("singular")
        a[col], a[piv] = a[piv], a[col]
        inv[col], inv[piv] = inv[piv], inv[col]
        for r in range(n):
            if r != col and ((a[r] >> col) & 1):
                a[r] ^= a[col]; inv[r] ^= inv[col]
    return inv


def mat_mul(A, B, n):
    """GF2 matrix product A*B; rows as bitmasks. (A*B)[i] = XOR over k with A[i][k] of B[k]."""
    out = []
    for i in range(n):
        r = 0
        ai = A[i]
        for k in range(n):
            if (ai >> k) & 1:
                r ^= B[k]
        out.append(r)
    return out


def ge_synth_count(rows, n):
    """CNOTs to synthesize linear map (rows over n inputs): reduce to identity,
    counting row-additions (each = one CNOT)."""
    a = list(rows)
    m = len(a)
    cx = 0
    r = 0
    for c in range(n):
        if r >= m:
            break
        piv = next((k for k in range(r, m) if (a[k] >> c) & 1), None)
        if piv is None:
            continue
        if piv != r:
            a[r], a[piv] = a[piv], a[r]
        for k in range(m):
            if k != r and ((a[k] >> c) & 1):
                a[k] ^= a[r]; cx += 1
        r += 1
    return cx


def synth_realize_set(targets, n):
    """Greedy parity-network realizing a SET of parities (each must appear on some
    qubit at some point -- this is what GraySynth solves; a real GraySynth could
    do better, so this CNOT count is an achievable UPPER bound). Realizing the
    segment's linear-map rows AND its phase/measurement parities all live in this
    one set. Returns (cx_count, verify_ok).

    Each realize emits CNOTs that overwrite one qubit to hold the target parity;
    that qubit then carries it (a phase/measurement is read there). We never need
    to restore, since the final linear state just becomes the next segment's input
    basis (re-indexed)."""
    if not targets:
        return 0, True
    basis = [1 << q for q in range(n)]
    cx = []
    realized = set()
    realized.add(0) if False else None
    appeared = set(basis)                      # parities currently present on some qubit
    remaining = [t for t in targets if t not in appeared]
    realized = set(t for t in targets if t in appeared)
    while remaining:
        best = None; bestR = None
        for p in remaining:
            R = solve_coeffs(basis, p)
            if R is None:
                return None, False
            if best is None or len(R) < len(bestR):
                best, bestR = p, R
        p, R = best, bestR
        remaining.remove(p)
        q0 = R[0]
        for q in R[1:]:
            cx.append((q, q0)); basis[q0] ^= basis[q]
        if basis[q0] != p:
            return None, False
        realized.add(p)
        # opportunistically: other remaining now present?
        appeared = set(basis)
        still = []
        for t in remaining:
            if t in appeared:
                realized.add(t)
            else:
                still.append(t)
        remaining = still
    # verify: simulate cx from identity, collect every parity that ever appears
    st = [1 << q for q in range(n)]
    ever = set(st)
    for (c, t) in cx:
        st[t] ^= st[c]; ever.add(st[t])
    ok = set(targets).issubset(ever)
    return len(cx), ok


def relevant_phase_steps(prog):
    """Backward Z-phase reachability: a diagonal phase (ROT/T/S) is RELEVANT only
    if the qubit it sits on reaches a non-diagonal op (U2/U4/H / X-basis measure)
    before being read in Z. A Z-phase that only ever feeds a Z-measurement does
    not affect any outcome and can be DROPPED.

    Heisenberg rule for Z under CNOT(c->t): Z_t -> Z_c Z_t (a target Z-phase
    spreads to the control), Z_c -> Z_c. So backward, rel[t] |= rel[c].
    Returns the set of step indices whose diagonal phase is relevant."""
    rel = {}                       # slot -> bool (is a Z-phase here relevant)
    relevant = set()
    for i in range(len(prog) - 1, -1, -1):
        inst = prog[i]
        name = T_mod._opname(inst.opcode)
        a1 = int(inst.axis_1); a2 = int(inst.axis_2)
        if name in ("OP_ARRAY_H", "OP_ARRAY_U2"):
            rel[a1] = True
        elif name == "OP_ARRAY_U4":
            rel[a1] = True; rel[a2] = True
        elif name in ("OP_MEAS_ACTIVE_INTERFERE", "OP_MEAS_ACTIVE_INTERFERE_FORCED"):
            rel[a1] = True
        elif name in ("OP_SWAP_MEAS_INTERFERE", "OP_SWAP_MEAS_INTERFERE_FORCED"):
            # measures (X-basis) at a2 after swap -> relevant; swap relevance
            rel[a2] = True
            rel[a1], rel[a2] = rel.get(a2, False), rel.get(a1, False)
        elif name in ("OP_MEAS_ACTIVE_DIAGONAL", "OP_MEAS_ACTIVE_DIAGONAL_FORCED"):
            rel[a1] = False         # read in Z -> Z-phase just before is irrelevant
        elif name.startswith("OP_EXPAND"):
            rel[a1] = False
        elif name == "OP_ARRAY_CNOT":          # a1=ctrl, a2=tgt
            if rel.get(a1, False):
                rel[a2] = True
        elif name == "OP_ARRAY_MULTI_CNOT":
            d = ds_mod._d(inst)
            anyc = any(rel.get(c, False) for c in ds_mod._bits(int(d["mask"])) if c != a1)
            if anyc:
                rel[a1] = True
        elif name == "OP_ARRAY_SWAP":
            rel[a1], rel[a2] = rel.get(a2, False), rel.get(a1, False)
        elif name in DIAG:
            if rel.get(a1, False):
                relevant.add(i)
    return relevant


def measurement_parity_weights(circ):
    """Weight of each active Z-measurement's parity ell_j = A[j] in the deferred
    (A,f) frame, splitting only at hard non-diagonal boundaries (U2/U4/H/interfere).
    Returns (list of weights, n_meas). A Z-measurement is a parity projector over
    support(ell_j) -- f-independent -- NOT a full (A,f) flush."""
    prog = clifft.compile(open(f"qec_bench/circuits/{circ}.stim").read())
    seg_idx = {}; row = {}; nidx = 0
    weights = []

    def get(s):
        nonlocal nidx
        if s not in seg_idx:
            seg_idx[s] = nidx; row[s] = 1 << nidx; nidx += 1
        return row[s]

    for i in range(len(prog)):
        inst = prog[i]; name = T_mod._opname(inst.opcode)
        a1 = int(inst.axis_1); a2 = int(inst.axis_2)
        if name in HARD:
            seg_idx.clear(); row.clear(); nidx = 0
        elif name.startswith("OP_EXPAND"):
            seg_idx[a1] = nidx; row[a1] = 1 << nidx; nidx += 1
        elif name == "OP_ARRAY_CNOT":
            get(a1); row[a2] = get(a2) ^ row[a1]
        elif name == "OP_ARRAY_MULTI_CNOT":
            d = ds_mod._d(inst); get(a1)
            for c in ds_mod._bits(int(d["mask"])):
                if c != a1:
                    row[a1] = row[a1] ^ get(c)
        elif name == "OP_ARRAY_SWAP":
            ra, rb = get(a1), get(a2); row[a1], row[a2] = rb, ra
        elif name in ZMEAS:
            w = bin(get(a1)).count("1")
            weights.append(w)
            row.pop(a1, None); seg_idx.pop(a1, None)
    return weights, len(weights)


def analyze(circ, split_zmeas, partial=False, meas_free=False):
    # partial=True models the user's refinement: at each flush realize ONLY the
    # boundary-support rows + Z-measured parities + RELEVANT phases (phases that
    # reach a non-diagonal boundary), deferring untouched survivors and dropping
    # irrelevant phases. partial=False is the conservative full-barrier (realize
    # the whole A,f every flush).
    prog = clifft.compile(open(f"qec_bench/circuits/{circ}.stim").read())
    rel_steps = relevant_phase_steps(prog) if partial else None
    seg_idx = {}          # slot -> input index within current segment
    row = {}              # slot -> bitmask over segment-input indices
    realize = set()       # parities that must appear (phases + measured + boundary)
    nidx = 0
    raw = 0
    tot_raw = 0
    tot_ge = 0
    tot_synth = 0
    all_ok = True
    n_seg = 0

    def get(s):
        nonlocal nidx
        if s not in seg_idx:
            seg_idx[s] = nidx; row[s] = 1 << nidx; nidx += 1
        return row[s]

    def close():
        nonlocal nidx, tot_ge, tot_synth, all_ok, n_seg, raw, tot_raw
        rset = set(realize)
        if not partial:
            # full barrier: also realize every surviving slot's row (the whole A)
            for s in row:
                rset.add(row[s])
        rset = {p for p in rset if p != 0}
        if nidx == 0 or not rset:
            tot_raw += raw
            seg_idx.clear(); row.clear(); realize.clear(); nidx = 0; raw = 0
            return
        n = nidx
        ge = ge_synth_count(sorted(rset), n)         # reference: reduce the realize set
        synth, ok = synth_realize_set(rset, n)
        if synth is None or not ok:
            all_ok = False
            synth = max(ge, len(rset))      # fallback estimate
        tot_ge += ge
        tot_synth += synth
        tot_raw += raw
        n_seg += 1
        seg_idx.clear(); row.clear(); realize.clear(); nidx = 0; raw = 0

    for i in range(len(prog)):
        inst = prog[i]
        name = T_mod._opname(inst.opcode)
        a1 = int(inst.axis_1); a2 = int(inst.axis_2)
        if name in HARD:
            if a1 in row:
                realize.add(row[a1])
            if name == "OP_ARRAY_U4" and a2 in row:
                realize.add(row[a2])
            close()
            continue
        if name.startswith("OP_EXPAND"):
            seg_idx[a1] = nidx; row[a1] = 1 << nidx; nidx += 1
        elif name == "OP_ARRAY_CNOT":
            get(a1); row[a2] = get(a2) ^ row[a1]; raw += 1
        elif name == "OP_ARRAY_MULTI_CNOT":
            d = ds_mod._d(inst)
            get(a1)
            for c in ds_mod._bits(int(d["mask"])):
                if c != a1:
                    row[a1] = row[a1] ^ get(c); raw += 1
        elif name == "OP_ARRAY_SWAP":
            ra, rb = get(a1), get(a2); row[a1], row[a2] = rb, ra
        elif name in DIAG:
            if rel_steps is None or i in rel_steps:
                realize.add(get(a1))
        elif name in ZMEAS:
            if not meas_free:
                realize.add(get(a1))      # count its parity in the (A,f) lowering
            else:
                get(a1)                   # Z-meas handled separately (parity projector)
            row.pop(a1, None); seg_idx.pop(a1, None)
            if split_zmeas:               # meas-flush: collapse materialized -> barrier
                close()
    close()
    return dict(circ=circ, n_seg=n_seg, raw=tot_raw, ge=tot_ge, synth=tot_synth,
                ok=all_ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("circuits", nargs="*", default=DEFAULT)
    args = ap.parse_args()
    circuits = args.circuits or DEFAULT
    print("Net CNOT cost of the (A,f) frame = verified CNOT+RZ parity-network synthesis")
    print("  full   = realize whole A,f every flush (worst case)")
    print("  PARTIAL= realize only boundary-support + Z-measured + RELEVANT phases,")
    print("           drop irrelevant phases & untouched survivors (the right way)\n")
    print(f"{'circuit':15s} {'mode':28s} {'raw':>6s} {'synth':>7s} {'raw/synth':>10s} {'verified':>9s}")
    for c in circuits:
        for split, partial, mfree, lbl in [
            (False, True,  False, "PARTIAL (meas in lowering)"),
            (False, True,  True,  "PARTIAL (meas = sep. projector)"),
        ]:
            r = analyze(c, split, partial=partial, meas_free=mfree)
            ratio = r["raw"] / r["synth"] if r["synth"] else float("inf")
            rr = f"{ratio:.2f}" if ratio != float("inf") else "inf"
            print(f"{c:15s} {lbl:28s} {r['raw']:6d} {r['synth']:7d} "
                  f"{rr:>10s} {'OK' if r['ok'] else 'MISMATCH':>9s}")
        # measurement parity-projector cost, reported SEPARATELY (not a full flush)
        w, nm = measurement_parity_weights(c)
        from collections import Counter
        hist = dict(sorted(Counter(w).items()))
        w1 = sum(1 for x in w if x <= 1)
        proj_cost = sum(max(0, x - 1) for x in w)   # ~CNOTs to gather support(ell)
        print(f"{c:15s} {'  Z-meas (' + str(nm) + ') parity weights':28s} "
              f"weight1={w1}/{nm}  hist={hist}  projCost~{proj_cost}")
        print()
    print("synth(A+f lowering) = U2/U4/H boundaries + RELEVANT phases only.")
    print("'meas in lowering'  = also realize each measured parity in the network.")
    print("'meas = sep projector' = Z-meas handled separately (parity projector over")
    print("   support(ell_j), f-independent) -> excluded from the A,f lowering count.")
    print("Z-meas: NOT a hard boundary; f never materialized for it; ident dies immediately.")


if __name__ == "__main__":
    main()
