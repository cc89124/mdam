"""Layout transforms for backend specs."""
from __future__ import annotations

from copy import deepcopy


def _find_bag(bags, ids):
    s = set(ids)
    for i, bag in enumerate(bags):
        if s <= set(bag):
            return i
    return None


def _running_intersection_ok(bags, edges):
    nbr = {i: set() for i in range(len(bags))}
    for i, j, _ in edges:
        nbr[i].add(j)
        nbr[j].add(i)
    all_ids = set().union(*(set(b) for b in bags)) if bags else set()
    for ident in all_ids:
        containing = [i for i, bag in enumerate(bags) if ident in set(bag)]
        if len(containing) <= 1:
            continue
        target = set(containing)
        seen = {containing[0]}
        stack = [containing[0]]
        while stack:
            u = stack.pop()
            for v in nbr[u]:
                if v in target and v not in seen:
                    seen.add(v)
                    stack.append(v)
        if seen != target:
            return False, ident
    return True, None


def _is_tree(n_bags, edges):
    if n_bags == 0:
        return True
    if len(edges) != n_bags - 1:
        return False
    nbr = {i: set() for i in range(n_bags)}
    for i, j, _ in edges:
        nbr[i].add(j)
        nbr[j].add(i)
    seen = {0}
    stack = [0]
    while stack:
        u = stack.pop()
        for v in nbr[u]:
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return len(seen) == n_bags


def reduce_hub_degree(spec, D_threshold=3):
    """Replace high-degree bags by same-vertex chain copies.

    This transform is intentionally conservative: each splitter copy has the
    same vertex set as the original hub bag. Therefore operation coverage and
    running-intersection validity are preserved. The transform only reduces
    tensor degree; it does not claim to reduce separator size.
    """
    if D_threshold < 3:
        raise ValueError("D_threshold must be >= 3 for chain hub reduction")

    old_bags = [list(b) for b in spec["union"]["bags"]]
    old_edges = [(i, j, list(sep)) for i, j, sep in spec["union"]["bag_edges"]]
    old_adj = {i: [] for i in range(len(old_bags))}
    for i, j, sep in old_edges:
        old_adj[i].append((j, sep))
        old_adj[j].append((i, sep))

    new_bags = []
    bag_map = {}
    edge_copy = {}
    for old_id, bag in enumerate(old_bags):
        degree = len(old_adj[old_id])
        n_copies = degree if degree > D_threshold else 1
        ids = []
        for _ in range(n_copies):
            ids.append(len(new_bags))
            new_bags.append(list(bag))
        bag_map[old_id] = ids
        if n_copies == 1:
            for nb, _ in old_adj[old_id]:
                edge_copy[(old_id, nb)] = ids[0]
        else:
            for idx, (nb, _) in enumerate(sorted(old_adj[old_id], key=lambda x: x[0])):
                edge_copy[(old_id, nb)] = ids[idx]

    new_edges = []
    seen_edges = set()

    def add_edge(a, b, sep):
        if a == b:
            return
        key = tuple(sorted((a, b)))
        if key in seen_edges:
            return
        seen_edges.add(key)
        new_edges.append((a, b, sorted(set(sep))))

    for old_id, ids in bag_map.items():
        if len(ids) > 1:
            sep = old_bags[old_id]
            for a, b in zip(ids, ids[1:]):
                add_edge(a, b, sep)

    for i, j, sep in old_edges:
        ni = edge_copy[(i, j)]
        nj = edge_copy[(j, i)]
        add_edge(ni, nj, sep)

    if not _is_tree(len(new_bags), new_edges):
        raise ValueError("hub reduction produced a non-tree layout")
    ri_ok, bad_ident = _running_intersection_ok(new_bags, new_edges)
    if not ri_ok:
        raise ValueError(f"hub reduction broke running intersection at ident {bad_ident}")

    out = deepcopy(spec)
    op_to_bag = []
    for row in out["op_to_bag"]:
        r = dict(row)
        r["bag"] = _find_bag(new_bags, r["axes"])
        if r["bag"] is None:
            raise ValueError(f"operation no longer covered after transform: {row}")
        op_to_bag.append(r)

    measurement_spec = []
    for row in out["measurement_spec"]:
        r = dict(row)
        r["marginal_bag"] = _find_bag(new_bags, (r["axis"],))
        measurement_spec.append(r)

    max_bag = max((len(b) for b in new_bags), default=0)
    max_sep = max((len(sep) for _, _, sep in new_edges), default=0)
    out["union"].update(dict(
        tau=max_bag - 1 if max_bag else 0,
        max_bag=max_bag,
        sum2=sum(2 ** len(b) for b in new_bags),
        max_sep=max_sep,
        n_bags=len(new_bags),
        shape=f"hub_reduced_D{D_threshold}",
        bags=[sorted(b) for b in new_bags],
        bag_edges=[(i, j, sorted(sep)) for i, j, sep in new_edges],
    ))
    out["op_to_bag"] = op_to_bag
    out["measurement_spec"] = measurement_spec
    out.setdefault("transforms", []).append(dict(
        name="reduce_hub_degree",
        D_threshold=D_threshold,
        old_n_bags=len(old_bags),
        new_n_bags=len(new_bags),
    ))
    return out
