"""Measure the *TTN* cost of applying the deferred diagonal parity-phases of the
(A,f) frame DIRECTLY (as diagonal tree-operators), instead of via gather-CNOTs.

The user's correction: counting high-weight phases as gather-CNOTs (2(w-1) each)
is too pessimistic, BUT treating their TTN cost as ~0 is also wrong. A diagonal
parity-phase does not permute the basis, yet applied to a factorized TTN it can
still GROW BONDS. The decisive quantity is therefore NOT a CNOT count but:

  for each tree cut e, the operator Schmidt rank that the *batch* of deferred
  phases crossing e induces across e.

A diagonal phase e^{iθ ℓ(x)} = e^{iθ(ℓ_L ⊕ ℓ_R)} across a cut depends on x only
through the 1-bit left-restriction ℓ_L and right-restriction ℓ_R, so it is
rank-2 across e. A SET of such phases depends on x through the values of all
their left-restrictions; the operator Schmidt rank across e is exactly

        2 ^ ( GF(2)-rank of { ℓ_L : ℓ crosses e } ) .

That 2^{r_e} is the worst-case factor the diagonal-phase batch can multiply the
bond χ_e by. We compare it to the OBSERVED max χ_e from a real eager run:

  * 2^{r_e} <= χ_e (observed) on every cut  =>  direct diagonal apply never
    pushes a bond above the entanglement floor the eager run already paid =>
    the phases are genuinely cheap (the bottleneck does NOT move to them).
  * 2^{r_e} >> χ_e on some cut             =>  the deferred batch would inflate
    that bond beyond the floor => the bottleneck just MOVED to the phase apply.

This is a conservative (worst-case, no-reset / defer-everything) upper bound:
parities are tracked over the spec's persistent identities WITHOUT rebasing at
boundaries, which only OVER-states r_e. If even this bound stays under the
observed χ, the optimistic reading is safe.

Coordinates: identities are exactly the spec's lifecycle ids (same assignment as
backend_spec Pass 1), so home[ident] -> bag maps each parity bit to a tree leaf.
"""
from __future__ import annotations

import argparse
import math
import os
from collections import deque, Counter

import clifft
from ttn_backend import treewidth as T_mod
from ttn_backend import frame_layer as ds_mod
from ttn_backend.backend_spec import export_backend_spec, assign_homes_and_classify

POLICY = dict(
    TTN_FUSE_MULTICNOT="1", TTN_PERSISTENT_MULTICNOT="1",
    TTN_PERSISTENT_MULTICNOT_MIN_MULTIS="2", TTN_DESTRUCTIVE_OPEN="1",
    TTN_FUSE_MULTICNOT_BATCH="1",
    TTN_FUSE_MULTICNOT_CAP_BYTES=str(64 * 1024 * 1024),
    TTN_FUSE_MULTICNOT_TOTAL_CAP_BYTES=str(64 * 1024 * 1024),
    TTN_PREFISSION_TRANSPORT_CAP_BYTES=str(64 * 1024 * 1024),
    TTN_PREFISSION_MIN_GAIN="1.01",
)

DIAG = {"OP_ARRAY_T", "OP_ARRAY_T_DAG", "OP_ARRAY_S", "OP_ARRAY_S_DAG",
        "OP_ARRAY_ROT"}
EXPAND_DIAG = {"OP_EXPAND_T", "OP_EXPAND_T_DAG", "OP_EXPAND_ROT"}


# ----------------------------------------------------------------------------
# build + run the real backend to get tree topology, homes, observed max chi
# ----------------------------------------------------------------------------
def build_and_run(circuit, layout="carving", timeout=600.0, seed=42,
                  chi_cache=None):
    """Build the static carving tree + homes (cheap; NO run_shot). The observed
    per-edge max χ is read from a cached carving_leaf_metrics.json (run_shot on
    d5_r5 is minutes-long; the tree topology is static so the cached χ profile
    applies as long as the edge sets agree -- which we assert)."""
    from ttn_backend import TTNBackend
    import json
    src = open(f"qec_bench/circuits/{circuit}.stim").read()
    prog = clifft.compile(src)
    base_spec = export_backend_spec(prog, strict=False)
    if layout == "union":
        spec, homing = base_spec, assign_homes_and_classify(base_spec)
    else:
        from temporal_carving.pipeline import run as run_pipeline
        from ttn_backend.scripts.qec_temporal_carving_report import trace_from_program
        from ttn_backend.scripts.qec_temporal_carving_runtime import build_carving_executable_spec
        trace = trace_from_program(prog, strict=False)
        carving = run_pipeline(trace, seeder="recursive_balanced_mincut",
                               refine_moves=("nni",), seed=0,
                               partitioner="networkx", exact=False)
        spec, homing = build_carving_executable_spec(base_spec, carving["tree"])
    backend = TTNBackend(spec, homing)              # construct only (no run_shot)
    adj = {i: set(int(n) for n in nbrs) for i, nbrs in enumerate(backend.bag_neighbors)}

    edge_chi = {}
    max_bond = 1
    n_qr = peak = 0
    if chi_cache:
        m = json.load(open(chi_cache))
        for key, dim in m.get("edge_max_bond_dim", {}).items():
            a, b = key.split("-"); edge_chi[(int(a), int(b))] = int(dim)
        max_bond = int(m.get("max_bond_dim_observed", m.get("max_bond_dim", max_bond)))
        n_qr = int(m.get("n_qr", 0)); peak = int(m.get("peak_stored_bytes", 0))
        # consistency: static-tree edges must be covered by the cached edge keys
        tree_edges = {(min(a, b), max(a, b)) for a in adj for b in adj[a]}
        cached_edges = set(edge_chi)
        missing = tree_edges - cached_edges
        cov = 100.0 * (len(tree_edges) - len(missing)) / max(1, len(tree_edges))
        print(f"[cache] tree edges={len(tree_edges)} cached edges={len(cached_edges)} "
              f"coverage={cov:.1f}%  (missing {len(missing)} treated as χ=1)")
    return dict(prog=prog, home=dict(homing["home"]), adj=adj, edge_chi=edge_chi,
                n_qr=n_qr, peak=peak, max_bond=max_bond)


# ----------------------------------------------------------------------------
# replay over SPEC identities (no rebasing) -> collect diagonal-phase parities
# ----------------------------------------------------------------------------
def collect_phase_parities(prog):
    """Return list of parity bitmasks (over spec ident ids) for every diagonal
    phase op, plus n_idents. Ident assignment replicates backend_spec Pass 1."""
    slot2id = {}
    next_id = 0
    parity = {}        # slot -> bitmask over idents

    def new_node(slot):
        nonlocal next_id
        if slot in slot2id:
            return slot2id[slot]
        i = next_id; next_id += 1
        slot2id[slot] = i
        parity[slot] = 1 << i
        return i

    phases = []
    for step in range(len(prog)):
        inst = prog[step]
        name = T_mod._opname(inst.opcode)
        a1 = int(inst.axis_1); a2 = int(inst.axis_2)
        if name in EXPAND_DIAG:
            new_node(a1)
            phases.append(parity[a1])                 # weight-1 phase on fresh ident
        elif name == "OP_EXPAND":
            new_node(a1)
        elif name == "OP_ARRAY_CNOT":                 # a1=control, a2=target
            if a1 not in slot2id: new_node(a1)
            if a2 not in slot2id: new_node(a2)
            parity[a2] ^= parity[a1]
        elif name == "OP_ARRAY_MULTI_CNOT":           # a1=target, mask=controls
            d = ds_mod._d(inst)
            if a1 not in slot2id: new_node(a1)
            for c in ds_mod._bits(int(d["mask"])):
                if c == a1:
                    continue
                if c not in slot2id: new_node(c)
                parity[a1] ^= parity[c]
        elif name == "OP_ARRAY_SWAP":
            slot2id[a1], slot2id[a2] = slot2id.get(a2), slot2id.get(a1)
            parity[a1], parity[a2] = parity.get(a2, 0), parity.get(a1, 0)
        elif name in DIAG:
            if a1 not in slot2id: new_node(a1)
            phases.append(parity[a1])
        elif name in ("OP_MEAS_ACTIVE_DIAGONAL", "OP_MEAS_ACTIVE_DIAGONAL_FORCED",
                      "OP_MEAS_ACTIVE_INTERFERE", "OP_MEAS_ACTIVE_INTERFERE_FORCED"):
            slot2id.pop(a1, None); parity.pop(a1, None)
        elif name in ("OP_SWAP_MEAS_INTERFERE", "OP_SWAP_MEAS_INTERFERE_FORCED"):
            # swap a1<-a2 then measure; conservative: move then drop
            slot2id[a1] = slot2id.get(a2); parity[a1] = parity.get(a2, 0)
            slot2id.pop(a2, None); parity.pop(a2, None)
        # U2/U4/H boundaries: realized unitary on the state; in the no-reset
        # worst-case model we keep tracking the same ident parities (overstates r).
    return phases, next_id


# ----------------------------------------------------------------------------
# GF(2) rank of a list of bitmasks
# ----------------------------------------------------------------------------
def per_boundary_touched_rank(prog):
    """REALISTIC (per-segment reset) model. Defer CNOTs+phases within a segment;
    at each H/U2/U4 boundary, the phases that MUST be materialized are those whose
    parity overlaps the boundary qubit's line ℓ_j (they don't commute with the
    boundary's non-diagonal action). The GF(2) rank of that touched batch (over
    fresh per-segment vars -- basis-independent) UPPER-BOUNDS the operator Schmidt
    rank the diagonal-phase apply induces across ANY cut: 2^rank. So 2^rank is the
    worst-case bond-growth factor *from one boundary's batch*. Small rank => the
    diagonal apply is cheap and does NOT inflate bonds.

    Z-measurements do NOT materialize phases (f cancels for the Z probability);
    they just drop the qubit. Data qubits never hit an H boundary, so their
    accumulated phases are never materialized here -- which is the whole point."""
    row = {}
    nvar = 0
    live = []                     # deferred phase parities (bitmasks over fresh vars)
    ranks = []                    # GF(2) rank of touched batch at each boundary
    ntouch = []
    HARD_1Q = {"OP_ARRAY_H", "OP_ARRAY_U2"}
    ZMEAS = {"OP_MEAS_ACTIVE_DIAGONAL", "OP_MEAS_ACTIVE_DIAGONAL_FORCED"}
    HARDMEAS = {"OP_MEAS_ACTIVE_INTERFERE", "OP_MEAS_ACTIVE_INTERFERE_FORCED",
                "OP_SWAP_MEAS_INTERFERE", "OP_SWAP_MEAS_INTERFERE_FORCED"}

    def fresh():
        nonlocal nvar
        v = 1 << nvar; nvar += 1
        return v

    def get(s):
        if s not in row:
            row[s] = fresh()
        return row[s]

    def do_boundary(support):
        nonlocal live
        touched, keep = [], []
        for p in live:
            (touched if (p & support) else keep).append(p)
        live = keep
        ranks.append(gf2_rank(touched))
        ntouch.append(len(touched))

    for i in range(len(prog)):
        inst = prog[i]
        name = T_mod._opname(inst.opcode)
        a1 = int(inst.axis_1); a2 = int(inst.axis_2)
        if name.startswith("OP_EXPAND"):
            row[a1] = fresh()
            if name in EXPAND_DIAG:
                live.append(row[a1])
        elif name == "OP_ARRAY_CNOT":
            get(a1); row[a2] = get(a2) ^ row[a1]
        elif name == "OP_ARRAY_MULTI_CNOT":
            d = ds_mod._d(inst); get(a1)
            for c in ds_mod._bits(int(d["mask"])):
                if c != a1:
                    row[a1] = row[a1] ^ get(c)
        elif name == "OP_ARRAY_SWAP":
            ra, rb = get(a1), get(a2); row[a1], row[a2] = rb, ra
        elif name in DIAG:
            live.append(get(a1))
        elif name in HARD_1Q:
            do_boundary(get(a1)); row[a1] = fresh()
        elif name == "OP_ARRAY_U4":
            do_boundary(get(a1) | get(a2)); row[a1] = fresh(); row[a2] = fresh()
        elif name in ZMEAS:
            row.pop(a1, None)                       # f cancels; no phase materialize
        elif name in HARDMEAS:
            do_boundary(get(a1)); row.pop(a1, None)
    return ranks, ntouch


def gf2_rank(vectors):
    basis = []
    for v in vectors:
        x = v
        for b in basis:
            x = min(x, x ^ b)
        if x:
            basis.append(x)
            basis.sort(reverse=True)
    return len(basis)


def bag_side(adj, a, b):
    """Bags reachable from a without using edge (a,b)."""
    seen = {a}; dq = deque([a])
    while dq:
        u = dq.popleft()
        for w in adj[u]:
            if (u == a and w == b) or (u == b and w == a):
                continue
            if w not in seen:
                seen.add(w); dq.append(w)
    return seen


def steiner_edges(adj, bags):
    """Minimal subtree (edge set) connecting `bags` in the tree `adj`."""
    bags = set(bags)
    if len(bags) <= 1:
        return set()
    # tree: prune leaves not in bags repeatedly -> remaining edges form Steiner tree
    deg = {u: len(adj[u]) for u in adj}
    keep = set(adj)
    # iteratively remove degree-1 nodes that are not terminals
    changed = True
    cur_adj = {u: set(adj[u]) for u in adj}
    while changed:
        changed = False
        for u in list(cur_adj):
            if u not in keep:
                continue
            if len(cur_adj[u]) <= 1 and u not in bags:
                for w in cur_adj[u]:
                    cur_adj[w].discard(u)
                cur_adj[u].clear(); keep.discard(u); changed = True
    edges = set()
    for u in keep:
        for w in cur_adj[u]:
            if u < w:
                edges.add((u, w))
    return edges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("circuit", nargs="?", default="coherent_d5_r5")
    ap.add_argument("--layout", choices=["carving", "union"], default="carving")
    ap.add_argument("--chi-cache", default=None,
                    help="carving_leaf_metrics.json with observed edge_max_bond_dim")
    args = ap.parse_args()

    R = build_and_run(args.circuit, layout=args.layout, chi_cache=args.chi_cache)
    prog, home, adj, edge_chi = R["prog"], R["home"], R["adj"], R["edge_chi"]
    phases, n_id = collect_phase_parities(prog)

    print(f"=== {args.circuit}  (layout={args.layout}) ===")
    print(f"identities={n_id}  bags={len(adj)}  tree-edges={sum(len(v) for v in adj.values())//2}")
    print(f"observed: max χ={R['max_bond']}  peak={R['peak']/2**20:.1f} MiB  n_qr={R['n_qr']}")
    print(f"diagonal phases collected={len(phases)}")

    # ids with no home -> can't place on the tree; report
    nohome = sorted({i for i in range(n_id) if home.get(i) is None})
    if nohome:
        touch_nohome = sum(1 for p in phases if any((p >> i) & 1 for i in nohome))
        print(f"WARNING: {len(nohome)} idents have home=None; {touch_nohome} phases touch them "
              f"(excluded from those bits).")

    def support_bags(p):
        bags = set()
        i = 0; x = p
        while x:
            if x & 1:
                h = home.get(i)
                if h is not None:
                    bags.add(h)
            x >>= 1; i += 1
        return bags

    # per-phase support / Steiner stats
    supp_sizes, steiner_sizes = [], []
    for p in phases:
        sb = support_bags(p)
        supp_sizes.append(len(sb))
        steiner_sizes.append(len(steiner_edges(adj, sb)))

    def hist(xs):
        return dict(sorted(Counter(xs).items()))
    print(f"\nphase support (#distinct home-bags) hist: {hist(supp_sizes)}")
    print(f"phase Steiner-subtree size (#tree edges)   hist: {hist(steiner_sizes)}")

    # ---- REALISTIC per-segment-reset per-boundary batch rank ----
    ranks, ntouch = per_boundary_touched_rank(prog)
    if ranks:
        mx = max(ranks)
        print(f"\n[PER-SEGMENT RESET] boundaries={len(ranks)}  touched-phases/boundary: "
              f"mean {sum(ntouch)/len(ntouch):.2f} max {max(ntouch)}")
        print(f"  per-boundary touched-batch GF(2) rank (= log2 of max bond-growth factor):")
        print(f"     rank hist: {hist(ranks)}   max rank={mx} -> max bond-growth 2^{mx}={2**mx}")
        floor_log2 = int(math.log2(R['max_bond'])) if R['max_bond'] > 1 else 0
        n_over = sum(1 for r in ranks if r > floor_log2)
        print(f"  observed max χ = {R['max_bond']} = 2^{floor_log2};  "
              f"{n_over}/{len(ranks)} boundaries have rank > {floor_log2} (batch could exceed global floor)")

    # ---- the decisive per-cut measurement ----
    # for each edge, left-restrict every crossing phase, GF(2) rank -> 2^r vs χ
    print(f"\n{'edge':>10s} {'obsχ':>7s} {'r_e':>4s} {'2^r_e':>8s} {'verdict':>14s}")
    worst = []
    edges = sorted({(min(a, b), max(a, b)) for a in adj for b in adj[a]})
    for (a, b) in edges:
        sideA = bag_side(adj, a, b)
        # bit i is on side A iff home[i] in sideA
        maskA = 0
        for i in range(n_id):
            h = home.get(i)
            if h is not None and h in sideA:
                maskA |= 1 << i
        left_parts = []
        for p in phases:
            lp = p & maskA
            rp = p & ~maskA
            if lp and rp:            # crosses the cut
                left_parts.append(lp)
        r = gf2_rank(left_parts) if left_parts else 0
        chi = edge_chi.get((a, b), 1)
        two_r = 2 ** r if r < 40 else float("inf")
        if r == 0:
            verdict = "no-cross"
        elif two_r <= chi:
            verdict = "<= floor OK"
        else:
            verdict = "ABOVE floor!"
        worst.append((r, chi, two_r, a, b, verdict, len(left_parts)))

    # show edges with live χ (>1) or any crossing phase, sorted by r desc
    worst.sort(key=lambda t: (t[0], t[1]), reverse=True)
    shown = 0
    for r, chi, two_r, a, b, verdict, ncross in worst:
        if chi <= 1 and r == 0:
            continue
        tr = "inf" if two_r == float("inf") else str(int(two_r))
        print(f"{a:4d}-{b:<5d} {chi:7d} {r:4d} {tr:>8s} {verdict:>14s}   (crossing phases={ncross})")
        shown += 1
        if shown >= 40:
            print("  ... (truncated)")
            break

    # summary verdict
    n_above = sum(1 for r, chi, two_r, *_ in worst if r > 0 and two_r > chi)
    n_cross = sum(1 for r, *_ in worst if r > 0)
    print(f"\nSUMMARY: {n_cross} cuts have crossing phases; "
          f"{n_above} of them have 2^r_e ABOVE observed χ.")
    if n_above == 0:
        print("=> worst-case diagonal-phase batch stays <= entanglement floor on EVERY cut:")
        print("   applying f directly does NOT inflate any bond beyond what the eager run paid.")
    else:
        print("=> on some cuts the deferred phase batch would exceed the floor (no-reset model);")
        print("   per-boundary reset would be needed there (bottleneck partially moves to phases).")
    print("\nNOTE: 2^r_e is a worst-case (no-reset, defer-everything) upper bound; the true")
    print("transient χ is min(2^r_e · base, true Schmidt rank). r_e here OVER-states reality.")


if __name__ == "__main__":
    main()
