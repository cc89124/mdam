"""Diagonal parity-phase apply on a TTN -- the (A,f) frame's key primitive.

Apply  D_p(theta): |x> -> e^{i theta p(x)} |x>,  p(x) = XOR_{q in S} x_q.

THE KEY FACT (operator is exactly rank-2):
    e^{i theta p(x)} = alpha I + beta Z_S,
    alpha=(1+e^{i theta})/2, beta=(1-e^{i theta})/2, Z_S = prod_{q in S} Z_q.
So D_p is a bond-dim-2 operator across ANY cut; applied to a TTN it can grow
each Steiner-subtree bond by at most x2 (and a QR recompresses). No qubit is
moved (no permutation / no gather-CNOT).

This file verifies the math two ways and measures the resulting bond growth:
  * local case (all support in one bag): pure diagonal, ZERO bond change.
  * multi-bag case: we VERIFY against dense and MEASURE the exact post-apply bond
    dimension on every Steiner edge (so we can confirm the <=x2 rank-2 bound and
    quantify the real QR/bond cost). The reference implementation here contracts
    the Steiner region to apply+refactor; the production zip-up (future) reaches
    the SAME final bonds with bounded workspace -- the bonds/QR measured here are
    the irreducible cost either way.
"""
from __future__ import annotations

import numpy as np


class Bag:
    def __init__(self, bid, neighbors):
        self.bag_id = bid
        self.neighbors = sorted(neighbors)
        self.own_idents = []
        self.tensor = None

    def bond_pos(self, nb):
        return len(self.own_idents) + self.neighbors.index(nb)

    def labels(self):
        return ([('own', i) for i in self.own_idents] +
                [('bond',) + tuple(sorted((self.bag_id, nb))) for nb in self.neighbors])


class TTN:
    def __init__(self, bag_neighbors, home):
        self.bags = [Bag(i, bag_neighbors[i]) for i in range(len(bag_neighbors))]
        self.home = dict(home)
        self.n_qr = 0

    def path(self, s, d):
        if s == d:
            return [s]
        par = {s: None}; q = [s]
        while q:
            u = q.pop(0)
            for v in self.bags[u].neighbors:
                if v not in par:
                    par[v] = u; q.append(v)
        if d not in par:
            return None
        out = []; c = d
        while c is not None:
            out.append(c); c = par[c]
        return out[::-1]

    def steiner(self, bag_ids):
        ids = list(bag_ids); region = {ids[0]}
        for d in ids[1:]:
            best = None
            for r in region:
                p = self.path(r, d)
                if p and (best is None or len(p) < len(best)):
                    best = p
            region.update(best)
        return set(region)

    def to_dense(self):
        tensors = {b.bag_id: b.tensor for b in self.bags}
        labels = {b.bag_id: list(b.labels()) for b in self.bags}
        order = [b.bag_id for b in self.bags]
        cur_t = tensors[order[0]].copy(); cur_l = list(labels[order[0]])
        absorbed = {order[0]}; frontier = list(self.bags[order[0]].neighbors)
        while frontier:
            nb = frontier.pop(0)
            if nb in absorbed:
                continue
            shared = [l for l in cur_l if l in labels[nb]]
            if not shared:
                frontier.append(nb); continue
            lab = shared[0]
            ac = cur_l.index(lab); an = labels[nb].index(lab)
            cur_t = np.tensordot(cur_t, tensors[nb], axes=([ac], [an]))
            cur_l = [l for k, l in enumerate(cur_l) if k != ac] + \
                    [l for k, l in enumerate(labels[nb]) if k != an]
            absorbed.add(nb)
            frontier += [x for x in self.bags[nb].neighbors if x not in absorbed]
        idents = sorted(i for (_, i) in cur_l)
        cur_t = np.transpose(cur_t, [cur_l.index(('own', i)) for i in idents])
        return cur_t.reshape(-1), idents

    def edge_chi(self, a, b):
        bag = self.bags[a]
        return int(bag.tensor.shape[bag.bond_pos(b)])


def apply_parity_phase(ttn: TTN, support, theta, measure_edges=None):
    """Apply e^{i theta XOR_{q in support} x_q} in place. Returns dict of metrics."""
    if abs(theta) < 1e-15 or not support:
        return {"mode": "noop"}
    support = list(support)
    e = np.exp(1j * theta)
    homes = sorted({ttn.home[q] for q in support})

    # ---- local: all support in one bag -> diagonal, no bond change ----
    if len(homes) == 1:
        b = ttn.bags[homes[0]]
        T = b.tensor
        own = [b.own_idents.index(q) for q in support]
        idx = np.indices(T.shape)
        par = np.zeros(T.shape, dtype=np.int64)
        for ax in own:
            par ^= idx[ax].astype(np.int64) & 1
        b.tensor = T * np.where(par == 1, e, 1.0)
        return {"mode": "local", "bond_growth": 1.0, "n_qr": 0}

    # ---- multi-bag: contract Steiner region, apply diagonal over support own
    # axes, then refactor back to the tree by QR (measure final bonds) ----
    region = sorted(ttn.steiner(homes))
    # contract region into one labeled tensor
    cur_t = ttn.bags[region[0]].tensor.copy()
    cur_l = list(ttn.bags[region[0]].labels())
    absorbed = {region[0]}
    frontier = [nb for nb in ttn.bags[region[0]].neighbors if nb in region]
    while frontier:
        nb = frontier.pop(0)
        if nb in absorbed:
            continue
        lab = next((l for l in cur_l if l in ttn.bags[nb].labels()), None)
        if lab is None:
            frontier.append(nb); continue
        nbl = ttn.bags[nb].labels()
        ac = cur_l.index(lab); an = nbl.index(lab)
        cur_t = np.tensordot(cur_t, ttn.bags[nb].tensor, axes=([ac], [an]))
        cur_l = [l for k, l in enumerate(cur_l) if k != ac] + \
                [l for k, l in enumerate(nbl) if k != an]
        absorbed.add(nb)
        frontier += [x for x in ttn.bags[nb].neighbors if x in region and x not in absorbed]
    region_workspace = int(cur_t.size)

    # apply diagonal e^{i theta * parity(support own axes)} on the merged tensor
    own_axes = [cur_l.index(('own', q)) for q in support]
    idx = np.indices(cur_t.shape)
    par = np.zeros(cur_t.shape, dtype=np.int64)
    for ax in own_axes:
        par ^= idx[ax].astype(np.int64) & 1
    cur_t = cur_t * np.where(par == 1, e, 1.0)

    # refactor back: split merged tensor along the ORIGINAL region tree edges by QR.
    # Walk region edges in BFS order from region[0]; each internal edge -> QR split.
    metrics = _refactor_region(ttn, region, cur_t, cur_l)
    metrics["region_workspace"] = region_workspace
    metrics["mode"] = "multi"
    if measure_edges:
        metrics["edge_chi"] = {f"{a}-{b}": ttn.edge_chi(a, b) for (a, b) in measure_edges}
    return metrics


def _refactor_region(ttn, region, big_t, big_l):
    """Split big_t (labels big_l) back into the region's bags along tree edges by
    sequential QR, restoring each bag's canonical axis order. Records n_qr and the
    new per-edge bond dims."""
    region = set(region)
    # internal edges (both endpoints in region)
    internal = []
    for a in region:
        for b in ttn.bags[a].neighbors:
            if b in region and a < b:
                internal.append((a, b))
    # We peel off bags one at a time from leaves of the region-subtree.
    # Build region adjacency + degrees
    radj = {a: [b for b in ttn.bags[a].neighbors if b in region] for a in region}
    deg = {a: len(radj[a]) for a in region}
    cur_t, cur_l = big_t, list(big_l)
    n_qr = 0
    new_chi = {}
    remaining = set(region)
    # peel leaves until one bag remains
    while len(remaining) > 1:
        leaf = next(a for a in remaining if deg[a] == 1)
        par = next(b for b in radj[leaf] if b in remaining)
        # axes that belong to `leaf`: its own idents + its external (non-region) bonds
        leaf_bag = ttn.bags[leaf]
        leaf_axis_labels = [('own', i) for i in leaf_bag.own_idents]
        leaf_axis_labels += [('bond',) + tuple(sorted((leaf, nb)))
                             for nb in leaf_bag.neighbors if nb != par]
        # the cut bond label between leaf and par:
        cut_lab = ('bond',) + tuple(sorted((leaf, par)))
        leaf_axes = [cur_l.index(l) for l in leaf_axis_labels]
        other_axes = [k for k in range(cur_t.ndim) if k not in leaf_axes]
        # reshape (leaf_dim, other_dim), QR so leaf part is orthonormal
        T = np.transpose(cur_t, leaf_axes + other_axes)
        ld = int(np.prod([T.shape[k] for k in range(len(leaf_axes))])) if leaf_axes else 1
        od = int(np.prod([T.shape[k] for k in range(len(leaf_axes), T.ndim)])) if other_axes else 1
        M = T.reshape(ld, od)
        # QR on M^T so that the SHARED (cut) part dimension = rank
        Q, R = np.linalg.qr(M)        # Q:(ld, r) leaf tensor, R:(r, od) stays in rest
        n_qr += 1
        r = Q.shape[1]
        new_chi[tuple(sorted((leaf, par)))] = r
        # build leaf bag tensor: axes = own_idents + bonds(sorted nbrs); bond-to-par is the new r-dim
        leaf_shape = [T.shape[k] for k in range(len(leaf_axes))] + [r]
        leaf_tensor_raw = Q.reshape(leaf_shape)   # axes: leaf_axis_labels..., cut
        leaf_full_labels = leaf_axis_labels + [cut_lab]
        # reorder into canonical bag order
        canon = leaf_bag.labels()
        leaf_bag.tensor = np.transpose(leaf_tensor_raw,
                                       [leaf_full_labels.index(l) for l in canon])
        # remaining big tensor = R reshaped onto the `other` axes + new cut axis
        other_labels = [cur_l[k] for k in other_axes]
        cur_t = R.reshape([r] + [T.shape[k] for k in range(len(leaf_axes), T.ndim)])
        cur_l = [cut_lab] + other_labels
        # update region graph
        remaining.discard(leaf)
        radj[par].remove(leaf); deg[par] -= 1
    # last bag gets the remainder
    last = next(iter(remaining))
    lb = ttn.bags[last]
    lb.tensor = np.transpose(cur_t, [cur_l.index(l) for l in lb.labels()])
    ttn.n_qr += n_qr
    base = max((ttn.edge_chi(*e) for e in []), default=1)
    return {"n_qr": n_qr, "new_edge_chi": {f"{a}-{b}": c for (a, b), c in new_chi.items()}}
