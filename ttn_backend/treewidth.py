"""
treewidth.py -- Active-state treewidth floor measurement for tensor-active Clifft.
(Revised: exact peak over ALL snapshots w/ UB pruning; slot->identity tracking
for ARRAY_SWAP relabeling; strict edge validation; softened verdict.)

WHAT THIS MEASURES (and its epistemic limits)
----------------------------------------------
We replay the compiled Clifft VM bytecode and reconstruct, per snapshot, the
ACTIVE INTERACTION GRAPH G_t = (A_t, E_t): nodes are active virtual axes, edges
are "these two were entangled by an array 2-axis op while both active". We then
report peak k = max|A_t| vs peak tau = max tw(G_t).

This sits in a chain of upper bounds on the TRUE tensor cost:

    min-fill(G_t)  >=  exact tw(G_t)  >=  true tensor-cost exponent

The second inequality holds because the graph keeps the UNION of past
interaction edges (cancellations are NOT reflected) and ignores the actual
entanglement values. Consequences:

  * SMALL value (even cheap min-fill) -> true cost small -> GO is SOLID.
  * LARGE value (even exact tw)       -> does NOT prove true cost is large.
                                         NO-GO is NOT establishable here.
    To argue NO-GO you need a real lower bound on the STATE (cut Schmidt ranks,
    or cancellation-resolved contraction width via cotengra/quimb).

So exact tw only earns its keep when min-fill is loose and near k: it tightens
the GO threshold. It never unlocks a NO-GO.

GRAPH MODEL
-----------
node-add (k+1):  OP_EXPAND, OP_EXPAND_T, OP_EXPAND_T_DAG, OP_EXPAND_ROT
edge (pair):     OP_ARRAY_CNOT, OP_ARRAY_CZ, OP_ARRAY_U4   -> edge(a1,a2)
edge (star):     OP_ARRAY_MULTI_CNOT, OP_ARRAY_MULTI_CZ    -> hub=a1, spokes=mask bits
relabel:         OP_ARRAY_SWAP  -> swaps virtual-axis DATA (verified in svm kernel),
                                   so we swap slot->identity, NOT ignore it.
node-del (k-1):  OP_MEAS_ACTIVE_DIAGONAL, OP_MEAS_ACTIVE_INTERFERE (+_FORCED)
fused swap+meas: OP_SWAP_MEAS_INTERFERE(axis_1=swap_from, axis_2=swap_to):
                 swap, then remove the node that ends up at swap_to (= original
                 swap_from occupant); survivor lives at slot swap_from afterward.
no graph change: single-axis ARRAY ops, all FRAME ops, NOISE, DETECTOR, ...

Node IDENTITY is tracked separately from axis SLOT (slot2id), so axis-index
reuse after a measurement and SWAP relabeling are both handled correctly.
"""

from __future__ import annotations
from dataclasses import dataclass, field

NODE_ADD = {"OP_EXPAND", "OP_EXPAND_T", "OP_EXPAND_T_DAG", "OP_EXPAND_ROT"}
EDGE_PAIR = {"OP_ARRAY_CNOT", "OP_ARRAY_CZ", "OP_ARRAY_U4"}
EDGE_STAR = {"OP_ARRAY_MULTI_CNOT", "OP_ARRAY_MULTI_CZ"}
NODE_DEL = {"OP_MEAS_ACTIVE_DIAGONAL", "OP_MEAS_ACTIVE_INTERFERE",
            "OP_MEAS_ACTIVE_DIAGONAL_FORCED", "OP_MEAS_ACTIVE_INTERFERE_FORCED"}
SWAP_PAIR = {"OP_ARRAY_SWAP"}
SWAP_MEAS = {"OP_SWAP_MEAS_INTERFERE", "OP_SWAP_MEAS_INTERFERE_FORCED"}


def _opname(op):
    return op if isinstance(op, str) else getattr(op, "name", str(op))


def _bits(mask):
    while mask:
        b = mask & (-mask)
        yield b.bit_length() - 1
        mask ^= b


@dataclass
class Snapshot:
    step: int
    opcode: str
    k: int
    adj: dict  # identity -> set(identity), among currently-active nodes


@dataclass
class TraceResult:
    snapshots: list = field(default_factory=list)
    peak_k: int = 0
    peak_k_step: int = -1


def replay(program, strict: bool = True, record_every: bool = False) -> TraceResult:
    """Replay a compiled Clifft Program -> per-snapshot active interaction graphs.

    slot2id maps an active axis SLOT to a stable node IDENTITY (monotonic int),
    so SWAP relabeling and axis-index reuse are handled. strict=True raises if an
    edge op references an inactive axis (catches missing node-add coverage or a
    broken replay model); set strict=False to degrade with a warning count.
    """
    slot2id: dict[int, int] = {}
    adj: dict[int, set[int]] = {}
    next_id = 0
    res = TraceResult()
    warnings = {"inactive_edge": 0}

    def new_node(slot):
        nonlocal next_id
        ident = next_id
        next_id += 1
        slot2id[slot] = ident
        adj[ident] = set()

    def resolve(slot, ctx):
        ident = slot2id.get(slot)
        if ident is None:
            if strict:
                raise ValueError(f"{ctx}: axis {slot} not active; "
                                 f"active slots={sorted(slot2id)}")
            warnings["inactive_edge"] += 1
        return ident

    def add_edge(s1, s2, ctx):
        if s1 == s2:
            return
        u = resolve(s1, ctx); v = resolve(s2, ctx)
        if u is None or v is None or u == v:
            return
        adj[u].add(v); adj[v].add(u)

    def del_slot(slot):
        ident = slot2id.pop(slot, None)
        if ident is not None:
            for w in adj.get(ident, ()):
                adj[w].discard(ident)
            adj.pop(ident, None)

    def snap(step, name):
        res.snapshots.append(Snapshot(
            step=step, opcode=name, k=len(slot2id),
            adj={i: set(adj[i]) for i in slot2id.values()}))

    n = len(program)
    for step in range(n):
        inst = program[step]
        name = _opname(inst.opcode)
        a1 = int(inst.axis_1); a2 = int(inst.axis_2)
        changed = False

        if name in NODE_ADD:
            # EXPAND* promotes a DORMANT axis -> it must be inactive beforehand.
            if a1 in slot2id:
                if strict:
                    raise ValueError(f"{name}: axis {a1} already active "
                                     f"(EXPAND on active axis)")
                # non-strict: keep idempotent (no-op)
            else:
                new_node(a1)
            changed = True
        elif name in EDGE_PAIR:
            add_edge(a1, a2, name); changed = True
        elif name in EDGE_STAR:
            mask = int(inst.as_dict().get("mask", 0))
            for spoke in _bits(mask):
                add_edge(a1, spoke, name)
            changed = True
        elif name in SWAP_PAIR:
            # physical data swap -> relabel identities (verified: exec_array_swap)
            i1 = slot2id.get(a1); i2 = slot2id.get(a2)
            if (i1 is None) != (i2 is None) and strict:
                raise ValueError(f"{name}: swap between active/inactive ({a1},{a2})")
            slot2id[a1], slot2id[a2] = i2, i1
            if slot2id[a1] is None:
                slot2id.pop(a1, None)
            if slot2id[a2] is None:
                slot2id.pop(a2, None)
            # relabel changes no edges among identities -> tw unchanged; no snap needed
        elif name in NODE_DEL:
            del_slot(a1); changed = True
        elif name in SWAP_MEAS:
            # axis_1=swap_from, axis_2=swap_to(measured). After the data swap the
            # ORIGINAL swap_from occupant sits at swap_to and is measured out; the
            # ORIGINAL swap_to occupant (if active) survives at slot swap_from.
            # Valid Clifft bytecode guarantees BOTH axes active here (the fused
            # ARRAY_SWAP kernel asserts both < active_k); the other branches are
            # defensive hardening for strict bug-catching, not normal paths.
            i_from = slot2id.get(a1)
            i_to = slot2id.get(a2)
            if i_from is None:
                # swap_from inactive: anomaly. Silently mis-handling could delete
                # the survivor (original a2), so fail loud in strict mode.
                if strict:
                    raise ValueError(f"{name}: swap_from axis {a1} not active; "
                                     f"active slots={sorted(slot2id)}")
                warnings["inactive_edge"] += 1
            elif i_to is None:
                # only swap_from active: its data swaps to a2 and is measured out;
                # no active survivor from a2.
                del_slot(a1)
            else:
                # both active (normal): a2's node survives at slot a1; a1's removed.
                slot2id[a1] = i_to
                slot2id.pop(a2, None)
                for w in adj.get(i_from, ()):
                    adj[w].discard(i_from)
                adj.pop(i_from, None)
            changed = True
        # else: single-axis / frame / noise -> no graph change

        if len(slot2id) > res.peak_k:
            res.peak_k = len(slot2id); res.peak_k_step = step
        if record_every or changed:
            snap(step, name)

    if not strict and warnings["inactive_edge"]:
        print(f"[replay] WARNING: {warnings['inactive_edge']} edge ops on inactive "
              f"axes were skipped (possible missing node-add coverage).")
    return res


# ---------------- treewidth: exact (subset DP) + bracket -------------------

def _to_bitadj(nodes, adj):
    order = sorted(nodes)
    idx = {v: i for i, v in enumerate(order)}
    n = len(order)
    badj = [0] * n
    for v in order:
        for w in adj.get(v, ()):
            if w in idx:
                badj[idx[v]] |= 1 << idx[w]
    return n, badj


def _pc(x):
    return bin(x).count("1")


def exact_treewidth(nodes, adj, max_n: int = 24):
    n, badj = _to_bitadj(nodes, adj)
    if n == 0:
        return 0
    if n > max_n:
        return None
    full = (1 << n) - 1

    def r(S, v):
        closure = 0; stack = [v]; seen = 1 << v
        while stack:
            u = stack.pop(); closure |= 1 << u
            m = badj[u] & S & ~seen
            while m:
                b = m & (-m); w = b.bit_length() - 1
                seen |= b; stack.append(w); m ^= b
        bnd = 0; reach = closure | (1 << v); m = reach
        while m:
            b = m & (-m); bnd |= badj[b.bit_length() - 1]; m ^= b
        return _pc(bnd & ~S & ~(1 << v) & full)

    f = [0] * (1 << n)
    for T in range(1, 1 << n):
        best = n; m = T
        while m:
            b = m & (-m); v = b.bit_length() - 1; S = T ^ b
            rv = r(S, v); cur = f[S] if f[S] > rv else rv
            if cur < best:
                best = cur
            m ^= b
        f[T] = best
    return f[full]


def minfill_upper(nodes, adj):
    g = {v: set(adj.get(v, ())) & set(nodes) for v in nodes}
    width = 0; rem = set(nodes)
    while rem:
        bv, bf = None, None
        for v in rem:
            nb = list(g[v] & rem); fill = 0
            for i in range(len(nb)):
                for j in range(i + 1, len(nb)):
                    if nb[j] not in g[nb[i]]:
                        fill += 1
            if bf is None or fill < bf:
                bf, bv = fill, v
        nb = list(g[bv] & rem); width = max(width, len(nb))
        for i in range(len(nb)):
            for j in range(i + 1, len(nb)):
                g[nb[i]].add(nb[j]); g[nb[j]].add(nb[i])
        rem.discard(bv)
        for w in nb:
            g[w].discard(bv)
    return width


def mmd_lower(nodes, adj):
    """Graph-treewidth LOWER bound (NOT a lower bound on true tensor cost)."""
    g = {v: set(adj.get(v, ())) & set(nodes) for v in nodes}
    if not g:
        return 0
    lb = 0; verts = set(nodes)
    while len(verts) > 1:
        v = min(verts, key=lambda x: len(g[x] & verts))
        lb = max(lb, len(g[v] & verts))
        nb = g[v] & verts
        if nb:
            u = min(nb, key=lambda x: len(g[x] & verts))
            for w in (nb - {u}):
                g[u].add(w); g[w].add(u)
        verts.discard(v)
    return lb


# -------------------------- top-level report -------------------------------

def analyze(program, strict: bool = True, exact_max_n: int = 22,
            peak_rank: int | None = None, verbose: bool = True):
    trace = replay(program, strict=strict)
    snaps = trace.snapshots

    ks = [s.k for s in snaps]
    ub = [minfill_upper(set(s.adj), s.adj) for s in snaps]
    lb = [mmd_lower(set(s.adj), s.adj) for s in snaps]
    peak_k = trace.peak_k
    peak_ub = max(ub) if ub else 0
    peak_lb = max(lb) if lb else 0

    # EXACT peak over ALL snapshots, pruned by descending UB.
    # Correct because exact(G_i) <= ub(G_i): once ub <= current exact peak,
    # no remaining (smaller-ub) snapshot can beat it.
    exact_peak = 0; all_exact = True
    for i in sorted(range(len(snaps)), key=lambda i: ub[i], reverse=True):
        if ub[i] <= exact_peak:
            break
        te = exact_treewidth(set(snaps[i].adj), snaps[i].adj, max_n=exact_max_n)
        if te is None:
            all_exact = False; break
        exact_peak = max(exact_peak, te)

    peak_tau = exact_peak if all_exact else peak_ub
    exact_flag = all_exact

    if peak_k == 0:
        verdict = "empty / no active state"
    elif peak_tau <= max(1, peak_k // 2):
        verdict = ("GO (solid): peak tau << peak k. Upper bound is small, so the "
                   "TRUE cost is small. Proceed to tau-aware localization.")
    else:
        verdict = ("WEAK on this proxy: tau ~ k for the interaction graph. This is "
                   "NOT a no-go (graph tw upper-bounds true cost). Resolve with "
                   "cancellation-aware contraction width (cotengra/quimb) before rejecting.")

    # peak-width snapshot structure (backend blueprint)
    peak_struct = None
    if snaps:
        pi = max(range(len(snaps)),
                 key=lambda i: (ub[i], sum(len(v) for v in snaps[i].adj.values()) // 2))
        peak_struct = analyze_structure(set(snaps[pi].adj), snaps[pi].adj)
        peak_struct["snapshot_step"] = snaps[pi].step
        peak_struct["snapshot_opcode"] = snaps[pi].opcode

    rep = dict(peak_k=peak_k, peak_tau=peak_tau, peak_tau_exact=exact_flag,
               peak_tau_minfill_ub=peak_ub, peak_tau_mmd_lb=peak_lb,
               k_trace=ks, tau_ub_trace=ub, tau_lb_trace=lb, verdict=verdict,
               peak_struct=peak_struct)

    if peak_rank is not None:
        rep["clifft_peak_rank"] = peak_rank
        rep["replay_peak_k_matches"] = (peak_rank == peak_k)

    if verbose:
        print(f"peak k                 = {peak_k}   (dense exponent 2^{peak_k})")
        if peak_rank is not None and peak_rank != peak_k:
            print(f"  !! prog.peak_rank={peak_rank} != replay peak_k={peak_k} "
                  f"-> missing node-add opcode coverage; investigate.")
        kind = "EXACT (all snapshots)" if exact_flag else "min-fill upper bound (exact infeasible)"
        print(f"peak tau               = {peak_tau}   ({kind}; 2^{peak_tau})")
        print(f"graph-proxy tw bracket = [{peak_lb} MMD+ lb .. {peak_ub} min-fill ub]"
              f"   (graph treewidth, NOT true tensor cost)")
        print(f"verdict                = {rep['verdict']}")
    return rep


# ----------------- STEP 0: peak-snapshot structure extraction --------------
# Backend blueprint: from the peak interaction graph, build the junction-tree
# (bag) tensor network. It is ALWAYS a tree (no loops) -> canonical form exists
# -> measurement marginals are local. MPS is the special case where the
# junction tree is a path. Cost knobs: max bag (=tau+1) -> largest tensor
# 2^(tau+1); max separator -> largest bond 2^sep; sum_B 2^|B| -> total memory.

def minfill_order(nodes, adj):
    """min-fill elimination -> (width, order). The order is the backend's
    contraction/sweep order; the chordal completion defines the bags."""
    g = {v: set(adj.get(v, ())) & set(nodes) for v in nodes}
    width = 0; rem = set(nodes); order = []
    while rem:
        bv, bf = None, None
        for v in rem:
            nb = list(g[v] & rem); fill = 0
            for i in range(len(nb)):
                for j in range(i + 1, len(nb)):
                    if nb[j] not in g[nb[i]]:
                        fill += 1
            if bf is None or fill < bf:
                bf, bv = fill, v
        nb = list(g[bv] & rem); width = max(width, len(nb)); order.append(bv)
        for i in range(len(nb)):
            for j in range(i + 1, len(nb)):
                g[nb[i]].add(nb[j]); g[nb[j]].add(nb[i])
        rem.discard(bv)
        for w in nb:
            g[w].discard(bv)
    return width, order


def _chordal_bags(nodes, adj, order):
    g = {v: set(adj.get(v, ())) & set(nodes) for v in nodes}
    rem = set(nodes); bags = {}
    for v in order:
        nb = g[v] & rem
        bags[v] = frozenset({v} | nb)
        nb = list(nb)
        for i in range(len(nb)):
            for j in range(i + 1, len(nb)):
                g[nb[i]].add(nb[j]); g[nb[j]].add(nb[i])
        rem.discard(v)
        for w in nb:
            g[w].discard(v)
    return bags


def _maximal_bags(bags_dict):
    uniq = list(set(bags_dict.values()))
    maximal = [b for b in uniq if not any(b < other for other in uniq)]
    return sorted(maximal, key=lambda b: (-len(b), tuple(sorted(b))))


def _junction_tree(bags):
    """Max-weight spanning tree on |Bi ∩ Bj| (Jensen: a valid junction tree).
    Returns (edges=[(i,j,separator)], degrees)."""
    L = len(bags)
    if L <= 1:
        return [], [0] * L
    in_tree = [False] * L; in_tree[0] = True
    best_w = [-1] * L; best_src = [-1] * L
    for j in range(1, L):
        best_w[j] = len(bags[0] & bags[j]); best_src[j] = 0
    edges = []; deg = [0] * L
    for _ in range(L - 1):
        u = max((j for j in range(L) if not in_tree[j]), key=lambda j: best_w[j])
        in_tree[u] = True
        s = best_src[u]
        edges.append((s, u, bags[u] & bags[s])); deg[s] += 1; deg[u] += 1
        for j in range(L):
            if not in_tree[j]:
                w = len(bags[u] & bags[j])
                if w > best_w[j]:
                    best_w[j] = w; best_src[j] = u
    return edges, deg


def analyze_structure(nodes, adj):
    """Full junction-tree structure of one interaction graph (backend blueprint)."""
    nodes = set(nodes)
    if not nodes:
        return dict(n=0, tau=0, bags=[], n_bags=0, max_bag=0, sum2=0,
                    max_sep=0, max_deg=0, shape="empty", order=[], edges=[])
    _, order = minfill_order(nodes, adj)
    mbags = _maximal_bags(_chordal_bags(nodes, adj, order))
    edges, deg = _junction_tree(mbags)
    max_bag = max(len(b) for b in mbags)
    max_sep = max((len(s) for _, _, s in edges), default=0)
    max_deg = max(deg, default=0)
    return dict(n=len(nodes), tau=max_bag - 1, bags=mbags, n_bags=len(mbags),
                max_bag=max_bag, sum2=sum(2 ** len(b) for b in mbags),
                max_sep=max_sep, max_deg=max_deg,
                shape=("path" if max_deg <= 2 else "branching"),
                order=order, edges=edges)


def peak_structure(program, strict: bool = False):
    """Extract the junction-tree structure of the peak-width snapshot."""
    trace = replay(program, strict=strict)
    if not trace.snapshots:
        return analyze_structure(set(), {})
    def _ec(s):
        return sum(len(v) for v in s.adj.values()) // 2
    ub = [minfill_upper(set(s.adj), s.adj) for s in trace.snapshots]
    i = max(range(len(trace.snapshots)), key=lambda i: (ub[i], _ec(trace.snapshots[i])))
    s = trace.snapshots[i]
    st = analyze_structure(set(s.adj), s.adj)
    st["snapshot_step"] = s.step
    st["snapshot_opcode"] = s.opcode
    return st


def backend_hint(st):
    """One-line backend recommendation from a structure dict."""
    if st["n"] == 0:
        return "no active state (pure Clifford) -- no tensor backend needed"
    if st["n_bags"] == 1:
        return (f"SINGLE dense bag 2^{st['max_bag']} (clique) -- NO decomposition "
                f"benefit at peak; this snapshot is effectively dense")
    if st["tau"] <= 1:
        return f"FOREST/MPS trivial (bond dim 2); {st['n_bags']} bags"
    if st["shape"] == "path":
        return (f"MPS-shaped junction tree: bond dim 2^{st['max_sep']}, "
                f"max tensor 2^{st['max_bag']}, {st['n_bags']} bags")
    return (f"BRANCHING junction tree -> TTN (bond dim 2^{st['max_sep']}, "
            f"max tensor 2^{st['max_bag']}, {st['n_bags']} bags); "
            f"MPS not ruled out but TTN is the safe structure")