"""
backend_spec.py -- A1+B1 backend specification exporter for tensor-active Clifft.

A1 = gate application in bytecode TIME ORDER.
B1 = state stored on a STATIC junction-tree bag layout built from the UNION
     interaction graph (NOT the per-step front A_t).

Outputs everything a tensor backend needs to be CONSTRUCTED (not just analyzed):
  union tau, max bag, sum 2^|B|, max separator, bags, bag tree (separators),
  op_to_bag, measurement_spec, lifecycle, SWAP slot mapping, and INVARIANT
  checks. The key invariant (raises if violated):

      for every two-axis op (u,v): exists a bag B with {u,v} subseteq B.

Provides:
  - export_backend_spec(prog)           : JT layout + op_to_bag + invariants
  - compute_memory_estimates(spec)      : bond-aware static memory estimates
  - assign_homes_and_classify(spec)     : home heuristic + A/B/C classification
  - analyze_sweep_grouping(spec, homing): Phase 1 lazy refactor
  - analyze_lifetime_regions(spec,homing): Phase 2 region scheduling
  - analyze_memory_capped_region(...)   : Phase 2 with rho budget
"""

from __future__ import annotations
from collections import deque

from . import treewidth as T

SINGLE_AXIS = {"OP_ARRAY_H", "OP_ARRAY_S", "OP_ARRAY_S_DAG", "OP_ARRAY_T",
               "OP_ARRAY_T_DAG", "OP_ARRAY_ROT", "OP_ARRAY_U2"}


# ==========================================================================
# Pass 1: bytecode replay -> identity graph + lifecycle
# ==========================================================================

def _instrumented_replay(program, strict=True):
    """Replay bytecode capturing identities, union edges, lifecycle, swaps."""
    slot2id = {}
    next_id = 0
    union_adj = {}
    lifecycle = {}
    two_axis_ops = []
    single_ops = []
    meas_ops = []
    swap_events = []
    warn = {"inactive": 0}

    def new_node(slot, step, opn):
        nonlocal next_id
        if slot in slot2id:
            if strict:
                raise ValueError(f"step {step} {opn}: axis {slot} already active")
            return slot2id[slot]
        i = next_id; next_id += 1
        slot2id[slot] = i
        union_adj[i] = set()
        lifecycle[i] = dict(id=i, init_slot=slot, promote_step=step,
                            promote_op=opn, demote_step=None, demote_op=None)
        return i

    def resolve(slot, step, opn):
        i = slot2id.get(slot)
        if i is None:
            if strict:
                raise ValueError(f"step {step} {opn}: axis {slot} not active; "
                                 f"active={sorted(slot2id)}")
            warn["inactive"] += 1
        return i

    def uedge(a, b):
        if a != b:
            union_adj[a].add(b); union_adj[b].add(a)

    def demote(i, step, opn):
        if i in lifecycle and lifecycle[i]["demote_step"] is None:
            lifecycle[i]["demote_step"] = step
            lifecycle[i]["demote_op"] = opn

    for step in range(len(program)):
        inst = program[step]
        name = T._opname(inst.opcode)
        a1 = int(inst.axis_1); a2 = int(inst.axis_2)

        if name in T.NODE_ADD:
            new_node(a1, step, name)
        elif name in T.EDGE_PAIR:
            iu = resolve(a1, step, name); iv = resolve(a2, step, name)
            if iu is not None and iv is not None and iu != iv:
                uedge(iu, iv); two_axis_ops.append((step, name, iu, iv))
        elif name in T.EDGE_STAR:
            hub = resolve(a1, step, name)
            mask = int(inst.as_dict().get("mask", 0))
            for b in T._bits(mask):
                iv = resolve(b, step, name)
                if hub is not None and iv is not None and hub != iv:
                    uedge(hub, iv)
                    two_axis_ops.append((step, name, hub, iv))
        elif name in T.SWAP_PAIR:
            i1 = slot2id.get(a1); i2 = slot2id.get(a2)
            if (i1 is None) != (i2 is None) and strict:
                raise ValueError(f"step {step} {name}: swap active/inactive ({a1},{a2})")
            slot2id[a1], slot2id[a2] = i2, i1
            if slot2id[a1] is None: slot2id.pop(a1, None)
            if slot2id[a2] is None: slot2id.pop(a2, None)
            swap_events.append((step, a1, a2, i1, i2))
        elif name in T.NODE_DEL:
            i = slot2id.pop(a1, None)
            if i is not None:
                demote(i, step, name); meas_ops.append((step, name, i))
        elif name in T.SWAP_MEAS:
            i_from = slot2id.get(a1); i_to = slot2id.get(a2)
            if i_from is None:
                if strict:
                    raise ValueError(f"step {step} {name}: swap_from {a1} inactive")
                warn["inactive"] += 1
            elif i_to is None:
                demote(i_from, step, name); slot2id.pop(a1, None)
                meas_ops.append((step, name, i_from))
            else:
                slot2id[a1] = i_to; slot2id.pop(a2, None)
                demote(i_from, step, name); meas_ops.append((step, name, i_from))
        elif name in SINGLE_AXIS:
            i = resolve(a1, step, name)
            if i is not None:
                single_ops.append((step, name, i))
        # else: frame / noise / detector -> no active-state structure

    return dict(union_adj=union_adj, lifecycle=lifecycle, two_axis_ops=two_axis_ops,
                single_ops=single_ops, meas_ops=meas_ops, swap_events=swap_events,
                warn=warn)


# ==========================================================================
# Pass 2: JT layout export + invariants
# ==========================================================================

def _find_bag(bags, ids):
    s = set(ids)
    for i, B in enumerate(bags):
        if s <= B:
            return i
    return None


def _running_intersection_ok(bags, edges):
    L = len(bags)
    nbr = {i: set() for i in range(L)}
    for i, j, _ in edges:
        nbr[i].add(j); nbr[j].add(i)
    allnodes = set().union(*bags) if bags else set()
    for v in allnodes:
        Sv = [i for i in range(L) if v in bags[i]]
        if len(Sv) <= 1:
            continue
        seen = {Sv[0]}; stack = [Sv[0]]; target = set(Sv)
        while stack:
            x = stack.pop()
            for y in nbr[x]:
                if y in target and y not in seen:
                    seen.add(y); stack.append(y)
        if seen != target:
            return False, v
    return True, None


def export_backend_spec(program, strict=True, peak_rank=None, exact_max_n=22):
    rec = _instrumented_replay(program, strict=strict)
    union_adj = rec["union_adj"]

    # UNION static junction-tree layout
    st = T.analyze_structure(set(union_adj), union_adj)
    bags = st["bags"]
    bag_edges = st["edges"]

    # op_to_bag + invariant A
    op_to_bag = []
    violations = []
    for (step, name, iu, iv) in rec["two_axis_ops"]:
        bi = _find_bag(bags, (iu, iv))
        if bi is None:
            violations.append((step, name, iu, iv))
        op_to_bag.append(dict(step=step, op=name, axes=(iu, iv), kind="two", bag=bi))
    for (step, name, i) in rec["single_ops"]:
        bi = _find_bag(bags, (i,))
        op_to_bag.append(dict(step=step, op=name, axes=(i,), kind="single", bag=bi))

    if violations:
        msg = "; ".join(f"step {s} {n} ({u},{v}) not covered by any bag"
                        for s, n, u, v in violations[:5])
        raise ValueError(f"INVARIANT VIOLATION: {len(violations)} two-axis op(s) "
                         f"not contained in a single bag: {msg}")

    # measurement_spec
    measurement_spec = []
    for (step, name, i) in rec["meas_ops"]:
        bi = _find_bag(bags, (i,))
        measurement_spec.append(dict(step=step, op=name, axis=i, marginal_bag=bi))

    # invariants B, C, D
    ri_ok, ri_bad = _running_intersection_ok(bags, bag_edges)
    if not ri_ok:
        raise ValueError(f"INVARIANT VIOLATION: running intersection fails at node {ri_bad}")

    lc = rec["lifecycle"]
    for (step, name, iu, iv) in rec["two_axis_ops"]:
        for i in (iu, iv):
            L = lc.get(i)
            if L is None or step < L["promote_step"] or \
               (L["demote_step"] is not None and step > L["demote_step"]):
                raise ValueError(f"INVARIANT VIOLATION: op step {step} on id {i} "
                                 f"outside its lifecycle {L}")

    for i, j, sep in bag_edges:
        if sep != (bags[i] & bags[j]):
            raise ValueError(f"INVARIANT VIOLATION: bond {i}-{j} != bag intersection")

    # PEAK snapshot structure for over-allocation comparison
    rep = T.analyze(program, strict=False, exact_max_n=exact_max_n,
                    peak_rank=peak_rank, verbose=False)
    peak = rep.get("peak_struct") or dict(tau=0, max_bag=0, sum2=0, max_sep=0, n_bags=0, n=0)

    return dict(
        union=dict(n_ids=len(union_adj), tau=st["tau"], max_bag=st["max_bag"],
                   sum2=st["sum2"], max_sep=st["max_sep"], n_bags=st["n_bags"],
                   shape=st["shape"], bags=[sorted(b) for b in bags],
                   bag_edges=[(i, j, sorted(s)) for i, j, s in bag_edges]),
        peak=dict(tau=peak["tau"], max_bag=peak["max_bag"], sum2=peak["sum2"],
                  max_sep=peak["max_sep"], n_bags=peak["n_bags"], n=peak["n"]),
        peak_k=rep["peak_k"], peak_tau=rep["peak_tau"], peak_tau_exact=rep["peak_tau_exact"],
        op_to_bag=op_to_bag, measurement_spec=measurement_spec,
        lifecycle=lc, swap_events=rec["swap_events"],
        invariants=dict(two_axis_coverage=True, running_intersection=True,
                        lifecycle=True, bond_consistency=True),
        warnings=rec["warn"],
    )


def print_spec(spec, name="circuit", full=False):
    u, p = spec["union"], spec["peak"]
    print(f"=== backend spec: {name} ===")
    print(f"UNION layout: ids={u['n_ids']} tau={u['tau']} max_bag=2^{u['max_bag']} "
          f"sum2^|B|={u['sum2']:,} max_sep=2^{u['max_sep']} bags={u['n_bags']} shape={u['shape']}")
    print(f"PEAK snapshot: n={p['n']} tau={p['tau']} max_bag=2^{p['max_bag']} "
          f"sum2^|B|={p['sum2']:,} max_sep=2^{p['max_sep']} bags={p['n_bags']}")
    over_mem = (u['sum2'] / p['sum2']) if p['sum2'] else float('inf')
    over_bag = u['max_bag'] - p['max_bag']
    print(f"OVER-ALLOCATION (union vs peak): mem x{over_mem:.2f}, max_bag +{over_bag} "
          f"(tau {p['tau']}->{u['tau']})")
    print(f"invariants: {spec['invariants']}")
    if spec["warnings"]["inactive"]:
        print(f"  WARNING: {spec['warnings']['inactive']} ops on inactive axes (non-strict)")
    print(f"#ops mapped={len(spec['op_to_bag'])}  #measurements={len(spec['measurement_spec'])}  "
          f"#swaps={len(spec['swap_events'])}  #identities={len(spec['lifecycle'])}")
    if full:
        print("bags:")
        for i, b in enumerate(u["bags"]):
            print(f"  B{i} = {b}")
        print("bag_edges (separators):")
        for i, j, s in u["bag_edges"]:
            print(f"  B{i} -- B{j}  sep={s}")


def compute_memory_estimates(spec, homing=None):
    """Return structural lower-bound and separator-saturated memory estimates.

    `union["sum2"] * 16` assumes every bond dimension is 1. It is useful as a
    structural lower bound, but it is not an executable TTN memory prediction.
    The separator estimates assume exact QR can saturate every tree edge to the
    corresponding separator Hilbert-space dimension.
    """
    if homing is None:
        homing = assign_homes_and_classify(spec)

    bags = [set(b) for b in spec["union"]["bags"]]
    n_bags = len(bags)
    owned = homing.get("owned_phys", {})
    adj = {i: [] for i in range(n_bags)}
    s_max = 0
    for i, j, sep in spec["union"]["bag_edges"]:
        s = len(sep)
        adj[i].append((j, s))
        adj[j].append((i, s))
        s_max = max(s_max, s)

    per_bag = []
    for bid in range(n_bags):
        own_count = len(owned.get(bid, []))
        bond_exp = sum(s for _, s in adj[bid])
        nbytes = 16 * (2 ** (own_count + bond_exp))
        per_bag.append(dict(
            bag=bid,
            degree=len(adj[bid]),
            bag_size=len(bags[bid]),
            own_count=own_count,
            separator_exp=bond_exp,
            bytes=nbytes,
        ))

    return dict(
        M_static=16 * int(spec["union"]["sum2"]),
        M_separator_worst=sum(x["bytes"] for x in per_bag),
        M_separator_max_bag=max((x["bytes"] for x in per_bag), default=0),
        D_max=max((len(v) for v in adj.values()), default=0),
        S_max=s_max,
        per_bag=per_bag,
    )


# ==========================================================================
# Pass 3: home assignment + A/B/C op classification
# ==========================================================================

def assign_homes_and_classify(spec):
    """Heuristic v2: each identity's home is the bag with the most two-axis ops on it."""
    bags = spec["union"]["bags"]
    bag_edges = spec["union"]["bag_edges"]
    op_to_bag = spec["op_to_bag"]
    lifecycle = spec["lifecycle"]
    n_bags = len(bags)

    bag_set = [set(b) for b in bags]
    candidates = {ident: [] for ident in lifecycle}
    for bid, S in enumerate(bag_set):
        for ident in S:
            candidates[ident].append(bid)

    op_count = {ident: {} for ident in lifecycle}
    for r in op_to_bag:
        if r["kind"] != "two" or r["bag"] is None:
            continue
        cb = r["bag"]
        for ax in r["axes"]:
            op_count[ax][cb] = op_count[ax].get(cb, 0) + 1

    home = {}
    for ident, cands in candidates.items():
        if not cands:
            home[ident] = None
            continue
        counts = op_count.get(ident, {})
        best_count = -1
        best_bag = sorted(cands)[0]
        for cb in sorted(cands):
            c = counts.get(cb, 0)
            if c > best_count:
                best_count = c
                best_bag = cb
        home[ident] = best_bag

    owned_phys = {bid: [] for bid in range(n_bags)}
    for ident, bid in home.items():
        if bid is not None:
            owned_phys[bid].append(ident)
    for bid in owned_phys:
        owned_phys[bid].sort()

    bag_adj = {i: [] for i in range(n_bags)}
    for i, j, _ in bag_edges:
        bag_adj[i].append(j)
        bag_adj[j].append(i)

    def tree_path(src, dst):
        if src == dst:
            return [src]
        parent = {src: None}
        q = deque([src])
        while q:
            u = q.popleft()
            if u == dst:
                break
            for v in bag_adj[u]:
                if v not in parent:
                    parent[v] = u
                    q.append(v)
        if dst not in parent:
            return None
        out = []
        cur = dst
        while cur is not None:
            out.append(cur)
            cur = parent[cur]
        return list(reversed(out))

    bag_size_pow2 = [2 ** len(b) for b in bags]

    op_classes = []
    n_A = n_B = n_C = 0
    path_lens = []
    refactor_costs = []
    classB_costs = []
    classC_costs = []

    for r in op_to_bag:
        if r["kind"] == "single":
            ident = r["axes"][0]
            op_classes.append(dict(
                step=r["step"], op=r["op"], kind="single", cls="-",
                home=home[ident], compute_bag=r["bag"],
                path_bags=None, path_len=None, refactor_cost=None,
            ))
            continue

        u, v = r["axes"]
        cb = r["bag"]
        hu = home[u]; hv = home[v]

        if hu == cb and hv == cb:
            cls = "A"; n_A += 1
            path = [cb]; rc = bag_size_pow2[cb]
        elif (hu == cb) != (hv == cb):
            cls = "B"; n_B += 1
            far_home = hv if hu == cb else hu
            path = tree_path(cb, far_home)
            rc = sum(bag_size_pow2[b] for b in path) if path else None
            if path is not None:
                path_lens.append(len(path) - 1)
                refactor_costs.append(rc)
                classB_costs.append(rc)
        else:
            cls = "C"; n_C += 1
            path = tree_path(hu, hv)
            rc = sum(bag_size_pow2[b] for b in path) if path else None
            if path is not None:
                path_lens.append(len(path) - 1)
                refactor_costs.append(rc)
                classC_costs.append(rc)

        op_classes.append(dict(
            step=r["step"], op=r["op"], kind="two",
            axes=(u, v), home_u=hu, home_v=hv, compute_bag=cb,
            cls=cls, path_bags=path,
            path_len=(len(path) - 1) if path else None,
            refactor_cost=rc,
        ))

    total = n_A + n_B + n_C
    stats = dict(
        n_two_axis=total, n_A=n_A, n_B=n_B, n_C=n_C,
        pctA=(100.0 * n_A / total) if total else 0.0,
        pctB=(100.0 * n_B / total) if total else 0.0,
        pctC=(100.0 * n_C / total) if total else 0.0,
        avg_path_len=(sum(path_lens) / len(path_lens)) if path_lens else 0.0,
        max_path_len=max(path_lens) if path_lens else 0,
        max_refactor_cost=max(refactor_costs) if refactor_costs else 0,
        sum_refactor_cost=sum(refactor_costs) if refactor_costs else 0,
        max_B_cost=max(classB_costs) if classB_costs else 0,
        max_C_cost=max(classC_costs) if classC_costs else 0,
    )

    return dict(home=home, owned_phys=owned_phys,
                op_classes=op_classes, stats=stats)


def print_homing(spec, homing, name="circuit", full=False):
    s = homing["stats"]
    print(f"=== homing & classification: {name} ===")
    print(f"two-axis ops total = {s['n_two_axis']}  "
          f"(A={s['n_A']}  B={s['n_B']}  C={s['n_C']})")
    print(f"  class A (local, no refactor)     : {s['pctA']:5.1f}%")
    print(f"  class B (one-sided path)         : {s['pctB']:5.1f}%   max cost = {s['max_B_cost']:,}")
    print(f"  class C (two-sided path)         : {s['pctC']:5.1f}%   max cost = {s['max_C_cost']:,}")
    print(f"path length  avg = {s['avg_path_len']:.2f}  max = {s['max_path_len']}")
    print(f"refactor sum 2^|V(B)|  worst-op = {s['max_refactor_cost']:,}  "
          f"summed-over-all-ops = {s['sum_refactor_cost']:,}")


# ==========================================================================
# Phase 1: sweep grouping (lazy refactor)
# ==========================================================================

def analyze_sweep_grouping(spec, homing):
    op_classes = homing["op_classes"]
    bags = spec["union"]["bags"]
    bag_size_pow2 = [2 ** len(b) for b in bags]

    ops_sorted = sorted(op_classes, key=lambda o: (o["step"], 0 if o["kind"] == "two" else 1))

    sweeps = []
    current_region = None
    current_ops = []
    standalone_A = 0
    standalone_single = 0

    def flush():
        nonlocal current_region, current_ops
        if current_region is not None and current_ops:
            sweeps.append({
                "region_bags": sorted(current_region),
                "region_size": sum(bag_size_pow2[b] for b in current_region),
                "n_ops": len(current_ops),
                "n_A_absorbed": sum(1 for o in current_ops if o.get("cls") == "A"),
                "n_B_absorbed": sum(1 for o in current_ops if o.get("cls") == "B"),
                "n_C_absorbed": sum(1 for o in current_ops if o.get("cls") == "C"),
                "n_single_absorbed": sum(1 for o in current_ops if o["kind"] == "single"),
                "first_step": current_ops[0]["step"],
                "last_step": current_ops[-1]["step"],
            })
        current_region = None
        current_ops = []

    for op in ops_sorted:
        if op["kind"] == "single":
            if current_region is not None and op["home"] in current_region:
                current_ops.append(op)
            else:
                standalone_single += 1
            continue

        if op["cls"] == "A":
            cb = op["compute_bag"]
            if current_region is not None and cb in current_region:
                current_ops.append(op)
            else:
                standalone_A += 1
        else:
            path = set(op["path_bags"]) if op.get("path_bags") else None
            if path is None:
                continue
            if current_region is not None and path <= current_region:
                current_ops.append(op)
            else:
                flush()
                current_region = path
                current_ops = [op]
    flush()

    n_two = sum(1 for o in op_classes if o["kind"] == "two")
    n_BC_total = sum(1 for o in op_classes if o["kind"] == "two" and o.get("cls") in ("B", "C"))

    n_sweeps = len(sweeps)
    sum_contract_cost = sum(s["region_size"] for s in sweeps)
    max_contract_cost = max((s["region_size"] for s in sweeps), default=0)
    avg_ops_per_sweep = (sum(s["n_ops"] for s in sweeps) / n_sweeps) if n_sweeps else 0
    max_ops_per_sweep = max((s["n_ops"] for s in sweeps), default=0)
    n_BC_absorbed = sum(s["n_B_absorbed"] + s["n_C_absorbed"] for s in sweeps)
    n_A_absorbed = sum(s["n_A_absorbed"] for s in sweeps)

    naive_refactor_cost = sum(
        o["refactor_cost"] for o in op_classes
        if o["kind"] == "two" and o.get("cls") in ("B", "C")
        and o.get("refactor_cost") is not None
    )

    return dict(
        sweeps=sweeps, n_sweeps=n_sweeps,
        n_BC_total=n_BC_total, n_BC_absorbed=n_BC_absorbed,
        n_A_absorbed=n_A_absorbed,
        standalone_A=standalone_A, standalone_single=standalone_single,
        naive_refactor_count=n_BC_total,
        sweep_refactor_count=n_sweeps,
        refactor_count_reduction=(n_BC_total / n_sweeps) if n_sweeps else float("inf"),
        avg_ops_per_sweep=avg_ops_per_sweep,
        max_ops_per_sweep=max_ops_per_sweep,
        sum_contract_cost=sum_contract_cost,
        max_contract_cost=max_contract_cost,
        naive_refactor_cost=naive_refactor_cost,
        cost_ratio=(sum_contract_cost / naive_refactor_cost) if naive_refactor_cost else float("inf"),
    )


def print_sweep_stats(spec, homing, sw, name="circuit"):
    print(f"=== sweep grouping (lazy refactor): {name} ===")
    print(f"naive  refactor count  = {sw['naive_refactor_count']}")
    print(f"sweep  refactor count  = {sw['sweep_refactor_count']}")
    if sw["sweep_refactor_count"]:
        print(f"  reduction in COUNT   = {sw['refactor_count_reduction']:.1f}x")
    print(f"ops per sweep          = avg {sw['avg_ops_per_sweep']:.1f}  max {sw['max_ops_per_sweep']}")
    print(f"contract cost          = max {sw['max_contract_cost']:,}  sum {sw['sum_contract_cost']:,}")
    print(f"naive refactor cost    = {sw['naive_refactor_cost']:,}")
    if sw["naive_refactor_cost"] > 0:
        print(f"  cost ratio sweep/naive = {sw['cost_ratio']:.3f}x")


# ==========================================================================
# Phase 2: overlap-aware lifetime-region scheduling
# ==========================================================================

def _build_region_per_ident(op_classes, lifecycle, home_of):
    region_per_ident = {}
    for ident in lifecycle:
        h = home_of.get(ident)
        region_per_ident[ident] = set([h]) if h is not None else set()
    for op in op_classes:
        if op["kind"] != "two" or op.get("path_bags") is None:
            continue
        u, v = op["axes"]
        for b in op["path_bags"]:
            region_per_ident[u].add(b)
            region_per_ident[v].add(b)
    return region_per_ident


def _build_event_timeline(op_classes, lifecycle):
    """priority: promote(0) < op(1) < demote(2)"""
    events = []
    for ident, info in lifecycle.items():
        events.append((info["promote_step"], 0, "promote", ident))
        if info["demote_step"] is not None:
            events.append((info["demote_step"], 2, "demote", ident))
    for op in op_classes:
        events.append((op["step"], 1, op["kind"] + "_op", op))
    events.sort(key=lambda e: (e[0], e[1]))
    return events


def analyze_lifetime_regions(spec, homing):
    op_classes = homing["op_classes"]
    lifecycle = spec["lifecycle"]
    bags = spec["union"]["bags"]
    n_bags = len(bags)
    bag_size_pow2 = [2 ** len(b) for b in bags]
    home_of = homing["home"]

    region_per_ident = _build_region_per_ident(op_classes, lifecycle, home_of)
    events = _build_event_timeline(op_classes, lifecycle)

    bag_count = [0] * n_bags
    current_size = 0
    peak_size = 0
    peak_step = -1
    peak_active_bags = []
    peak_n_active_idents = 0

    n_open = 0
    n_close = 0
    sum_open_cost = 0
    sum_close_cost = 0
    max_event_cost = 0

    absorbed_BC = 0
    remaining_BC = 0
    absorbed_A = 0
    standalone_A = 0
    n_active_idents = 0

    ident_region_sizes = {ident: sum(bag_size_pow2[b] for b in R)
                          for ident, R in region_per_ident.items()}

    for step, prio, kind, payload in events:
        if kind == "promote":
            R = region_per_ident.get(payload, set())
            for b in R:
                if bag_count[b] == 0:
                    n_open += 1
                    current_size += bag_size_pow2[b]
                    sum_open_cost += current_size
                    if current_size > max_event_cost:
                        max_event_cost = current_size
                bag_count[b] += 1
            n_active_idents += 1
            if current_size > peak_size:
                peak_size = current_size
                peak_step = step
                peak_active_bags = [b for b in range(n_bags) if bag_count[b] > 0]
                peak_n_active_idents = n_active_idents

        elif kind == "demote":
            R = region_per_ident.get(payload, set())
            for b in R:
                bag_count[b] -= 1
                if bag_count[b] == 0:
                    n_close += 1
                    sum_close_cost += current_size
                    if current_size > max_event_cost:
                        max_event_cost = current_size
                    current_size -= bag_size_pow2[b]
            n_active_idents -= 1

        elif kind == "two_op":
            op = payload
            if op["cls"] == "A":
                cb = op["compute_bag"]
                if cb is not None and bag_count[cb] > 0:
                    absorbed_A += 1
                else:
                    standalone_A += 1
            else:
                path = op.get("path_bags")
                if path is None:
                    continue
                if all(bag_count[b] > 0 for b in path):
                    absorbed_BC += 1
                else:
                    remaining_BC += 1

    n_BC = sum(1 for o in op_classes if o["kind"] == "two" and o.get("cls") in ("B", "C"))
    naive_refactor_cost = sum(
        o["refactor_cost"] for o in op_classes
        if o["kind"] == "two" and o.get("cls") in ("B", "C")
        and o.get("refactor_cost") is not None
    )

    phase2_refactor_count = n_open + n_close
    phase2_total_cost = sum_open_cost + sum_close_cost

    return dict(
        n_BC=n_BC,
        naive_refactor_count=n_BC,
        naive_refactor_cost=naive_refactor_cost,
        absorbed_BC=absorbed_BC,
        remaining_BC=remaining_BC,
        absorbed_A=absorbed_A,
        standalone_A=standalone_A,
        absorption_rate=(absorbed_BC / n_BC * 100) if n_BC else 0.0,
        n_open=n_open, n_close=n_close,
        phase2_refactor_count=phase2_refactor_count,
        phase2_total_cost=phase2_total_cost,
        max_event_cost=max_event_cost,
        peak_region_size=peak_size,
        peak_step=peak_step,
        peak_active_bags=peak_active_bags,
        peak_n_active_idents=peak_n_active_idents,
        count_reduction=(n_BC / phase2_refactor_count) if phase2_refactor_count else float("inf"),
        cost_ratio=(phase2_total_cost / naive_refactor_cost) if naive_refactor_cost else float("inf"),
        max_ident_region_size=max(ident_region_sizes.values()) if ident_region_sizes else 0,
        median_ident_region_size=sorted(ident_region_sizes.values())[len(ident_region_sizes)//2]
            if ident_region_sizes else 0,
    )


def print_lifetime_stats(spec, homing, lt, name="circuit"):
    print(f"=== Phase 2: overlap-aware lifetime-region analysis: {name} ===")
    print(f"naive B/C refactor count     = {lt['naive_refactor_count']}")
    print(f"phase2 refactor count        = {lt['phase2_refactor_count']}  "
          f"(opens={lt['n_open']}, closes={lt['n_close']})")
    if lt["phase2_refactor_count"]:
        print(f"  COUNT reduction            = {lt['count_reduction']:.1f}x")
    print(f"absorbed B/C ops into region = {lt['absorbed_BC']} / {lt['n_BC']}  "
          f"({lt['absorption_rate']:.1f}%)")
    print(f"remaining B/C ops            = {lt['remaining_BC']}")
    print(f"absorbed A ops               = {lt['absorbed_A']}   standalone A = {lt['standalone_A']}")
    print(f"peak active region size      = {lt['peak_region_size']:,}  "
          f"at step {lt['peak_step']}  (active idents at peak = {lt['peak_n_active_idents']})")
    print(f"max single-event cost        = {lt['max_event_cost']:,}")
    print(f"identity region size         = max {lt['max_ident_region_size']:,}, "
          f"median {lt['median_ident_region_size']:,}")
    print(f"phase2 total open/close cost = {lt['phase2_total_cost']:,}")
    print(f"naive total refactor cost    = {lt['naive_refactor_cost']:,}")
    if lt["naive_refactor_cost"]:
        print(f"  COST ratio phase2/naive    = {lt['cost_ratio']:.3f}x")


# ==========================================================================
# Phase 2 with memory budget (rho-curve)
# ==========================================================================

def analyze_memory_capped_region(spec, homing, rho=2.0):
    op_classes = homing["op_classes"]
    lifecycle = spec["lifecycle"]
    bags = spec["union"]["bags"]
    n_bags = len(bags)
    bag_size_pow2 = [2 ** len(b) for b in bags]
    home_of = homing["home"]

    peak_sum2 = spec["peak"]["sum2"]
    M = rho * peak_sum2 if (peak_sum2 > 0 and rho != float("inf")) else float("inf")

    region_per_ident = _build_region_per_ident(op_classes, lifecycle, home_of)
    events = _build_event_timeline(op_classes, lifecycle)

    bag_count = [0] * n_bags
    current_size = 0
    protected = set()

    n_open = 0
    n_close = 0
    sum_open_cost = 0
    sum_close_cost = 0
    peak_size = 0
    peak_step = -1

    n_protected_total = 0
    n_unprotected_total = 0

    absorbed_BC = 0
    fallback_BC = 0
    fallback_BC_cost = 0
    absorbed_A = 0
    standalone_A = 0

    for step, prio, kind, payload in events:
        if kind == "promote":
            R = region_per_ident.get(payload, set())
            new_bags = [b for b in R if bag_count[b] == 0]
            extra_size = sum(bag_size_pow2[b] for b in new_bags)
            if current_size + extra_size <= M:
                for b in R:
                    if bag_count[b] == 0:
                        n_open += 1
                        current_size += bag_size_pow2[b]
                        sum_open_cost += current_size
                    bag_count[b] += 1
                protected.add(payload)
                n_protected_total += 1
            else:
                n_unprotected_total += 1
            if current_size > peak_size:
                peak_size = current_size
                peak_step = step

        elif kind == "demote":
            if payload in protected:
                R = region_per_ident.get(payload, set())
                for b in R:
                    bag_count[b] -= 1
                    if bag_count[b] == 0:
                        n_close += 1
                        sum_close_cost += current_size
                        current_size -= bag_size_pow2[b]
                protected.discard(payload)

        elif kind == "two_op":
            op = payload
            if op["cls"] == "A":
                cb = op["compute_bag"]
                if cb is not None and bag_count[cb] > 0:
                    absorbed_A += 1
                else:
                    standalone_A += 1
            else:
                path = op.get("path_bags")
                if path is None:
                    continue
                if all(bag_count[b] > 0 for b in path):
                    absorbed_BC += 1
                else:
                    fallback_BC += 1
                    if op.get("refactor_cost") is not None:
                        fallback_BC_cost += op["refactor_cost"]

    n_BC = sum(1 for o in op_classes if o["kind"] == "two" and o.get("cls") in ("B", "C"))
    naive_refactor_cost = sum(
        o["refactor_cost"] for o in op_classes
        if o["kind"] == "two" and o.get("cls") in ("B", "C")
        and o.get("refactor_cost") is not None
    )

    phase2_refactor_count = n_open + n_close
    phase2_total_cost = sum_open_cost + sum_close_cost
    total_cost = phase2_total_cost + fallback_BC_cost

    return dict(
        rho=rho, budget=M,
        n_BC=n_BC,
        absorbed_BC=absorbed_BC,
        fallback_BC=fallback_BC,
        absorption_rate=(absorbed_BC / n_BC * 100) if n_BC else 0.0,
        n_protected=n_protected_total,
        n_unprotected=n_unprotected_total,
        absorbed_A=absorbed_A,
        standalone_A=standalone_A,
        n_open=n_open, n_close=n_close,
        phase2_refactor_count=phase2_refactor_count,
        phase2_total_cost=phase2_total_cost,
        fallback_BC_cost=fallback_BC_cost,
        total_cost=total_cost,
        naive_refactor_cost=naive_refactor_cost,
        peak_region_size=peak_size,
        peak_step=peak_step,
        cost_ratio=(total_cost / naive_refactor_cost) if naive_refactor_cost else float("inf"),
    )


def print_memcap_sweep(spec, homing,
                       rhos=(1.0, 1.2, 1.5, 2.0, 3.0, 5.0, 10.0, float("inf")),
                       name="circuit"):
    peak_sum2 = spec["peak"]["sum2"]
    union_sum2 = spec["union"]["sum2"]
    print(f"=== memory-capped region scheduling: {name} ===")
    print(f"  peak snapshot sum2 = {peak_sum2:,}")
    print(f"  union layout sum2  = {union_sum2:,}")
    print(f"{'rho':>5} {'budget':>11} | {'abs%':>5} {'#prot':>5} {'#unp':>4} | "
          f"{'#open':>5} {'#cls':>4} {'p2$':>11} {'fb$':>11} {'tot$':>11} {'tot/N':>5} "
          f"{'peakMem':>10}")
    print("-" * 120)
    results = []
    for rho in rhos:
        r = analyze_memory_capped_region(spec, homing, rho=rho)
        results.append(r)
        rho_str = "inf" if r["rho"] == float("inf") else f"{r['rho']:.1f}"
        bud_str = "inf" if r["budget"] == float("inf") else f"{r['budget']:,.0f}"
        print(f"{rho_str:>5} {bud_str:>11} | {r['absorption_rate']:>4.1f}% "
              f"{r['n_protected']:>5} {r['n_unprotected']:>4} | "
              f"{r['n_open']:>5} {r['n_close']:>4} "
              f"{r['phase2_total_cost']:>11,} {r['fallback_BC_cost']:>11,} "
              f"{r['total_cost']:>11,} {r['cost_ratio']:>4.2f}x "
              f"{r['peak_region_size']:>10,}")
    return results
