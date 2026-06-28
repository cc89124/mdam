"""Honest QR-count comparison on the REAL carving tree, in the correct
per-segment full-defer regime (NOT the no-reset model I sloppily used before).

Both a CNOT transport and a diagonal parity-phase apply cost ~ (tree distance)
QRs. So compare, in the same unit (tree edges):

  eager_cnot_pathsum  = sum over every CNOT/MULTI_CNOT control of
                        tree_path_len(home[control], home[target])
                        ~ the transport/QR work eager pays (what (A,f) tries to cut)

  diag_steiner_sum    = sum over f_touch phases (per-segment reset) of
                        |Steiner subtree of the phase's physical support|
                        ~ the QR work the diagonal-direct apply ADDS

If diag_steiner_sum is comparable to or larger than eager_cnot_pathsum, then even
ignoring the linear-map synthesis savings, the diagonal phase apply alone costs
about as much as ALL eager CNOT transport -> no win. If it is much smaller, the
(A,f) frame can plausibly win.

Per-segment full-defer = split at hard boundaries (U2/U4/H/interfere). Within a
segment, CNOTs accumulate into the affine map; a ROT records a phase over the
CURRENT parity; at the boundary the phases overlapping the boundary qubit are the
f_touch that must be diagonal-applied; then the segment resets (rebase).
"""
from __future__ import annotations

import argparse
from collections import Counter, deque

import clifft
from ttn_backend import treewidth as T_mod
from ttn_backend import frame_layer as ds_mod
from ttn_backend.scripts.diag_phase_ttn_cost import build_and_run

DIAG = {"OP_ARRAY_T", "OP_ARRAY_T_DAG", "OP_ARRAY_S", "OP_ARRAY_S_DAG", "OP_ARRAY_ROT"}
EXPAND_DIAG = {"OP_EXPAND_T", "OP_EXPAND_T_DAG", "OP_EXPAND_ROT"}
HARD_1Q = {"OP_ARRAY_H", "OP_ARRAY_U2"}
ZMEAS = {"OP_MEAS_ACTIVE_DIAGONAL", "OP_MEAS_ACTIVE_DIAGONAL_FORCED"}
HARDMEAS = {"OP_MEAS_ACTIVE_INTERFERE", "OP_MEAS_ACTIVE_INTERFERE_FORCED",
            "OP_SWAP_MEAS_INTERFERE", "OP_SWAP_MEAS_INTERFERE_FORCED"}


def tree_path_len(adj, a, b):
    if a == b:
        return 0
    seen = {a}; dq = deque([(a, 0)])
    while dq:
        u, d = dq.popleft()
        for v in adj[u]:
            if v == b:
                return d + 1
            if v not in seen:
                seen.add(v); dq.append((v, d + 1))
    return None  # disconnected


def steiner_edges(adj, bags):
    bags = set(bags)
    if len(bags) <= 1:
        return 0
    cur = {b: set(adj[b]) for b in adj}
    keep = set(adj)
    changed = True
    while changed:
        changed = False
        for u in list(keep):
            if len(cur[u] & keep) <= 1 and u not in bags:
                keep.discard(u); changed = True
    # edges within keep
    edges = 0
    for u in keep:
        for v in cur[u]:
            if v in keep and u < v:
                edges += 1
    return edges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("circuit", nargs="?", default="coherent_d5_r5")
    ap.add_argument("--chi-cache", default=None)
    args = ap.parse_args()

    R = build_and_run(args.circuit, chi_cache=args.chi_cache)
    prog, home, adj = R["prog"], R["home"], R["adj"]

    # --- slot -> spec ident replay (to map physical qubits to home bags) ---
    slot2id = {}; nextid = 0

    def new_id(slot):
        nonlocal nextid
        if slot in slot2id:
            return slot2id[slot]
        slot2id[slot] = nextid; nextid += 1
        return slot2id[slot]

    def home_of_slot(slot):
        i = slot2id.get(slot)
        return home.get(i) if i is not None else None

    # --- per-segment frame: row[slot] = set of "vars"; var_home[var]=bag ---
    row = {}; var_home = {}; nvar = 0
    live = []  # (phase parity as frozenset of vars)

    def fresh(slot):
        nonlocal nvar
        v = nvar; nvar += 1
        var_home[v] = home_of_slot(slot)
        row[slot] = {v}
        return v

    diag_steiner_sizes = []     # Steiner edges per f_touch phase
    eager_cnot_pathsum = 0
    eager_cnot_count = 0
    nohome_phase = 0

    def support_homes(parity):
        return {var_home[v] for v in parity if var_home.get(v) is not None}

    def do_boundary(slots):
        nonlocal live, nohome_phase
        support = set()
        for s in slots:
            support |= row.get(s, set())
        touched, keep = [], []
        for p in live:
            (touched if (p & support) else keep).append(p)
        live = keep
        for p in touched:
            sh = support_homes(p)
            if len(sh) != len(p):
                nohome_phase += 1
            diag_steiner_sizes.append(steiner_edges(adj, sh))

    for i in range(len(prog)):
        inst = prog[i]; name = T_mod._opname(inst.opcode)
        a1 = int(inst.axis_1); a2 = int(inst.axis_2)
        if name.startswith("OP_EXPAND"):
            new_id(a1); fresh(a1)
            if name in EXPAND_DIAG:
                live.append(frozenset(row[a1]))
        elif name == "OP_ARRAY_CNOT":
            new_id(a1); new_id(a2)
            if a1 not in row: fresh(a1)
            if a2 not in row: fresh(a2)
            # eager path cost: control a1 -> target a2
            ha, hb = home_of_slot(a1), home_of_slot(a2)
            if ha is not None and hb is not None:
                pl = tree_path_len(adj, ha, hb)
                if pl: eager_cnot_pathsum += pl
            eager_cnot_count += 1
            row[a2] = row[a2] ^ row[a1]
        elif name == "OP_ARRAY_MULTI_CNOT":
            d = ds_mod._d(inst); new_id(a1)
            if a1 not in row: fresh(a1)
            hb = home_of_slot(a1)
            for c in ds_mod._bits(int(d["mask"])):
                if c == a1: continue
                new_id(c)
                if c not in row: fresh(c)
                ha = home_of_slot(c)
                if ha is not None and hb is not None:
                    pl = tree_path_len(adj, ha, hb)
                    if pl: eager_cnot_pathsum += pl
                eager_cnot_count += 1
                row[a1] = row[a1] ^ row[c]
        elif name == "OP_ARRAY_SWAP":
            slot2id[a1], slot2id[a2] = slot2id.get(a2), slot2id.get(a1)
            row[a1], row[a2] = row.get(a2, set()), row.get(a1, set())
        elif name in DIAG:
            if a1 not in row: new_id(a1); fresh(a1)
            live.append(frozenset(row[a1]))
        elif name in HARD_1Q:
            do_boundary([a1]); fresh(a1)
        elif name == "OP_ARRAY_U4":
            do_boundary([a1, a2]); fresh(a1); fresh(a2)
        elif name in ZMEAS:
            row.pop(a1, None); slot2id.pop(a1, None)   # f cancels, no materialize
        elif name in HARDMEAS:
            do_boundary([a1]); row.pop(a1, None); slot2id.pop(a1, None)

    n = len(diag_steiner_sizes)
    diag_sum = sum(diag_steiner_sizes)
    hist = dict(sorted(Counter(diag_steiner_sizes).items()))
    print(f"=== {args.circuit}  (real carving tree, per-segment full-defer) ===")
    print(f"bags={len(adj)}  observed maxχ={R['max_bond']}")
    print(f"\nADDED by diagonal-direct apply:")
    print(f"  f_touch phases (per-segment) = {n}")
    print(f"  Steiner-subtree size per phase (= QRs to diag-apply it):")
    print(f"     hist {hist}")
    print(f"  mean {diag_sum/max(n,1):.2f} edges/phase   TOTAL diag QRs = {diag_sum}")
    if nohome_phase:
        print(f"  (note: {nohome_phase} phases touched a home=None ident; Steiner uses placed bits only)")
    print(f"\nREDUCED (eager CNOT transport, same unit = tree-path QRs):")
    print(f"  eager CNOT/MULTI_CNOT controls = {eager_cnot_count}")
    print(f"  eager_cnot_pathsum (Σ control->target tree distance) = {eager_cnot_pathsum}")
    print(f"\nVERDICT:")
    print(f"  diag-apply ADDS ~{diag_sum} QRs;  eager CNOT transport TOTAL ~{eager_cnot_pathsum} QRs.")
    if diag_sum >= eager_cnot_pathsum:
        print(f"  => diagonal apply alone (>= all eager CNOT transport) => (A,f) does NOT win.")
    else:
        frac = 100 * diag_sum / max(eager_cnot_pathsum, 1)
        print(f"  => diag apply = {frac:.0f}% of eager CNOT transport; net depends on how much")
        print(f"     of that transport the linear-map synthesis (8.33x) actually removes.")


if __name__ == "__main__":
    main()
