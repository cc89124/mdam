"""Static numerical-rank TTN compression experiment for one peak bag tensor.

This is a feasibility experiment only. It re-decomposes a fixed peak tensor by
recursive SVD splits and reports memory/error tradeoffs. It does not modify the
runtime TTN backend or circuit execution schedule.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

sys.path.insert(0, ".")

import clifft
import numpy as np
from scipy import linalg

from ttn_backend import TTNBackend
from ttn_backend.backend_spec import assign_homes_and_classify, export_backend_spec


SUMMARY_FIELDS = [
    "circuit",
    "step",
    "bag",
    "mode",
    "rank_rule",
    "tol",
    "old_numel",
    "old_bytes",
    "old_log2_numel",
    "new_total_numel",
    "new_total_bytes",
    "new_total_log2_numel",
    "new_peak_numel",
    "new_peak_bytes",
    "new_peak_log2_numel",
    "compression_ratio_total",
    "compression_ratio_peak",
    "num_tensors",
    "num_bonds",
    "max_bond_rank",
    "max_bond_log2",
    "tree_depth",
    "recon_error_abs",
    "recon_error_rel",
    "discarded_energy_bound",
    "elapsed_s",
    "status",
    "notes",
]

CANDIDATE_FIELDS = [
    "node_id",
    "depth",
    "candidate_id",
    "rank_rule",
    "tol",
    "A_legs",
    "B_legs",
    "logdim_A",
    "logdim_B",
    "matrix_shape",
    "rank",
    "rank_log2",
    "discarded_energy",
    "discarded_relative",
    "old_numel",
    "split_total_numel",
    "split_peak_numel",
    "split_total_bytes",
    "split_peak_bytes",
    "score_peak_log2",
    "score_total_log2",
    "accepted",
    "reject_reason",
    "elapsed_s_svd",
]


@dataclass(frozen=True)
class LegMeta:
    name: str
    kind: str
    dim: int
    original_axis: int | None
    edge_id: str | None
    log2_dim: float


@dataclass
class SplitEval:
    candidate_id: int
    A_idx: tuple[int, ...]
    B_idx: tuple[int, ...]
    logdim_A: float
    logdim_B: float
    matrix_shape: tuple[int, int]
    rank: int
    discarded_energy: float
    discarded_relative: float
    split_total_numel: int
    split_peak_numel: int
    elapsed_s_svd: float
    U: np.ndarray
    s: np.ndarray
    Vh: np.ndarray


class TreeNode:
    def __init__(self, node_id, depth, tensor, legs):
        self.node_id = str(node_id)
        self.depth = int(depth)
        self.tensor = tensor
        self.legs = list(legs)
        self.split = None
        self.children = []

    @property
    def is_leaf(self):
        return not self.children


def _load_prog(name):
    with open(os.path.join("qec_bench/circuits", name + ".stim")) as f:
        return clifft.compile(f.read())


def _log2(x):
    x = float(x)
    return float(math.log2(x)) if x > 0 else 0.0


def _prod(xs):
    out = 1
    for x in xs:
        out *= int(x)
    return int(out)


def _leg_logdim(legs, idxs):
    return float(sum(legs[i].log2_dim for i in idxs))


def _leg_names(legs, idxs):
    return [legs[i].name for i in idxs]


def _rank_from_singular_values(s, rank_rule, tol, abs_tol):
    if s.size == 0:
        return 1, 0.0, 0.0
    total = float(np.sum(s * s))
    if rank_rule == "rel":
        threshold = max(float(abs_tol), float(tol) * float(s[0]))
        r = int(np.count_nonzero(s > threshold))
        r = max(1, r)
    elif rank_rule == "energy":
        if total <= 0:
            r = 1
        else:
            tail = np.cumsum((s[::-1] * s[::-1]))[::-1]
            r = len(s)
            limit = (float(tol) ** 2) * total
            for cand in range(1, len(s) + 1):
                discarded = float(tail[cand]) if cand < len(s) else 0.0
                if discarded <= limit:
                    r = cand
                    break
    else:
        raise ValueError(f"unknown rank rule: {rank_rule}")
    discarded = float(np.sum(s[r:] * s[r:]))
    rel = math.sqrt(discarded / total) if total > 0 else 0.0
    return r, discarded, rel


def _snapshot_cache_path(out_dir, circuit, step, bag):
    return Path(out_dir) / f"static_ttn_peak_{circuit}_step{step}_{bag}.npz"


def _save_snapshot(path, tensor, legs):
    os.makedirs(path.parent, exist_ok=True)
    meta = [
        dict(
            name=l.name,
            kind=l.kind,
            dim=int(l.dim),
            original_axis=l.original_axis,
            edge_id=l.edge_id,
            log2_dim=float(l.log2_dim),
        )
        for l in legs
    ]
    np.savez_compressed(path, tensor=tensor, legs_json=json.dumps(meta))


def _load_snapshot(path):
    data = np.load(path, allow_pickle=False)
    tensor = data["tensor"]
    meta = json.loads(str(data["legs_json"]))
    legs = [LegMeta(**row) for row in meta]
    return tensor, legs


def load_peak_bag_tensor(circuit, step, bag_name, out_dir, seed, timeout_s,
                         force_refresh=False, snapshot_cache_dir=None):
    cache_dir = snapshot_cache_dir or out_dir
    cache = _snapshot_cache_path(cache_dir, circuit, step, bag_name)
    if cache.exists() and not force_refresh:
        return _load_snapshot(cache)

    bag_id = int(str(bag_name).lstrip("Bb"))
    prog = _load_prog(circuit)
    spec = export_backend_spec(prog, strict=False)
    homing = assign_homes_and_classify(spec)
    backend = TTNBackend(spec, homing, capture_peak_snapshot=True)
    backend.run_shot(prog, seed, runtime_timeout=timeout_s, check_interval=1)
    metrics = backend.last_metrics or {}
    snapshot = metrics.get("peak_snapshot")
    if not snapshot:
        raise RuntimeError("no peak snapshot captured; increase timeout or enable capture")
    if int(snapshot.get("step_id")) != int(step):
        # Use the requested bag from the captured peak anyway; the exact peak
        # step can move slightly with timeout/checkpoint changes.
        pass
    row = next((r for r in snapshot["bags"] if int(r["bag_id"]) == bag_id), None)
    if row is None:
        raise RuntimeError(f"bag B{bag_id} not present in peak snapshot")
    tensor = np.asarray(row["tensor"])
    legs = []
    axis = 0
    for ident in row["own_idents"]:
        dim = int(tensor.shape[axis])
        legs.append(LegMeta(
            name=f"phys:{int(ident)}",
            kind="physical",
            dim=dim,
            original_axis=axis,
            edge_id=None,
            log2_dim=_log2(dim),
        ))
        axis += 1
    for nb in row["neighbors"]:
        dim = int(tensor.shape[axis])
        eid = f"{min(bag_id, int(nb))}-{max(bag_id, int(nb))}"
        legs.append(LegMeta(
            name=f"bond:{eid}",
            kind="bond",
            dim=dim,
            original_axis=axis,
            edge_id=eid,
            log2_dim=_log2(dim),
        ))
        axis += 1
    _save_snapshot(cache, tensor, legs)
    return tensor, legs


def _canonical_partition(A, n):
    A = tuple(sorted(set(int(x) for x in A)))
    B = tuple(i for i in range(n) if i not in set(A))
    if not A or not B:
        return None
    # Canonicalize complement duplicates.
    if A > B:
        A, B = B, A
    return A, B


def _valid_partition(legs, A, B):
    if not A or not B:
        return False
    if not any(legs[i].dim > 1 for i in A):
        return False
    if not any(legs[i].dim > 1 for i in B):
        return False
    return True


def generate_partitions(legs, random_candidates, balance_tol, rng):
    n = len(legs)
    parts = {}

    def add(A):
        p = _canonical_partition(A, n)
        if p is None:
            return
        A0, B0 = p
        if _valid_partition(legs, A0, B0):
            parts[p] = None

    # Greedy log-dimension balanced split.
    order = sorted(range(n), key=lambda i: legs[i].log2_dim, reverse=True)
    A, B = [], []
    la = lb = 0.0
    for i in order:
        if la <= lb:
            A.append(i); la += legs[i].log2_dim
        else:
            B.append(i); lb += legs[i].log2_dim
    add(A)

    # Largest legs separated across sides.
    nontriv = [i for i in order if legs[i].dim > 1]
    if len(nontriv) >= 2:
        A = [nontriv[0]]
        Bset = {nontriv[1]}
        la = legs[nontriv[0]].log2_dim
        lb = legs[nontriv[1]].log2_dim
        for i in order:
            if i in A or i in Bset:
                continue
            if la <= lb:
                A.append(i); la += legs[i].log2_dim
            else:
                Bset.add(i); lb += legs[i].log2_dim
        add(A)

    # Physical-vs-bond and mixed variants.
    physical = [i for i, l in enumerate(legs) if l.kind == "physical"]
    bonds = [i for i, l in enumerate(legs) if l.kind in ("bond", "internal")]
    if physical and bonds:
        add(physical)
        A = []
        if physical:
            A.append(max(physical, key=lambda i: legs[i].log2_dim))
        if bonds:
            A.append(max(bonds, key=lambda i: legs[i].log2_dim))
        la = _leg_logdim(legs, A)
        for i in order:
            if i in A:
                continue
            if la <= 0.5 * sum(l.log2_dim for l in legs):
                A.append(i); la += legs[i].log2_dim
        add(A)

    # Random balanced partitions.
    total_log = sum(l.log2_dim for l in legs)
    idxs = list(range(n))
    for _ in range(int(random_candidates)):
        rng.shuffle(idxs)
        A = []
        cur = 0.0
        target = total_log / 2.0
        for i in idxs:
            if cur < target or not A:
                A.append(i)
                cur += legs[i].log2_dim
        p = _canonical_partition(A, n)
        if p is None:
            continue
        A0, B0 = p
        if abs(_leg_logdim(legs, A0) - _leg_logdim(legs, B0)) <= float(balance_tol):
            if _valid_partition(legs, A0, B0):
                parts[p] = None
    return list(parts.keys())


def _proxy_sort_partitions(legs, parts):
    rows = []
    for A, B in parts:
        la = _leg_logdim(legs, A)
        lb = _leg_logdim(legs, B)
        rows.append((max(la, lb), abs(la - lb), min(la, lb), A, B))
    rows.sort(key=lambda x: (x[0], x[1], -x[2]))
    return [(A, B) for *_rest, A, B in rows]


def evaluate_split(tensor, legs, A, B, candidate_id, rank_rule, tol, abs_tol):
    t0 = time.perf_counter()
    perm = list(A) + list(B)
    T = np.transpose(tensor, perm)
    dim_A = _prod(legs[i].dim for i in A)
    dim_B = _prod(legs[i].dim for i in B)
    M = T.reshape(dim_A, dim_B)
    U, s, Vh = linalg.svd(M, full_matrices=False, lapack_driver="gesdd")
    elapsed = time.perf_counter() - t0
    r, discarded, rel = _rank_from_singular_values(s, rank_rule, tol, abs_tol)
    split_total = int((dim_A + dim_B) * r)
    split_peak = int(max(dim_A * r, dim_B * r))
    return SplitEval(
        candidate_id=candidate_id,
        A_idx=tuple(A),
        B_idx=tuple(B),
        logdim_A=_leg_logdim(legs, A),
        logdim_B=_leg_logdim(legs, B),
        matrix_shape=(dim_A, dim_B),
        rank=r,
        discarded_energy=discarded,
        discarded_relative=rel,
        split_total_numel=split_total,
        split_peak_numel=split_peak,
        elapsed_s_svd=elapsed,
        U=U,
        s=s,
        Vh=Vh,
    )


def materialize_split(node, ev):
    A = list(ev.A_idx)
    B = list(ev.B_idx)
    r = int(ev.rank)
    sqrt_s = np.sqrt(ev.s[:r])
    left_m = ev.U[:, :r] * sqrt_s[None, :]
    right_m = sqrt_s[:, None] * ev.Vh[:r, :]
    bond_name = f"internal:{node.node_id}"
    left_bond = LegMeta(bond_name, "internal", r, None, bond_name, _log2(r))
    right_bond = LegMeta(bond_name, "internal", r, None, bond_name, _log2(r))
    A_legs = [node.legs[i] for i in A]
    B_legs = [node.legs[i] for i in B]
    left_shape = tuple(l.dim for l in A_legs) + (r,)
    right_shape = (r,) + tuple(l.dim for l in B_legs)
    left = left_m.reshape(left_shape)
    right = right_m.reshape(right_shape)
    left_node = TreeNode(f"{node.node_id}L", node.depth + 1, left, A_legs + [left_bond])
    right_node = TreeNode(f"{node.node_id}R", node.depth + 1, right, [right_bond] + B_legs)
    return left_node, right_node


def candidate_row(node, ev, rank_rule, tol, accepted=False, reject_reason=""):
    old_numel = int(node.tensor.size)
    return dict(
        node_id=node.node_id,
        depth=node.depth,
        candidate_id=ev.candidate_id,
        rank_rule=rank_rule,
        tol=tol,
        A_legs=" ".join(_leg_names(node.legs, ev.A_idx)),
        B_legs=" ".join(_leg_names(node.legs, ev.B_idx)),
        logdim_A=ev.logdim_A,
        logdim_B=ev.logdim_B,
        matrix_shape=f"{ev.matrix_shape[0]}x{ev.matrix_shape[1]}",
        rank=ev.rank,
        rank_log2=_log2(ev.rank),
        discarded_energy=ev.discarded_energy,
        discarded_relative=ev.discarded_relative,
        old_numel=old_numel,
        split_total_numel=ev.split_total_numel,
        split_peak_numel=ev.split_peak_numel,
        split_total_bytes=ev.split_total_numel * 16,
        split_peak_bytes=ev.split_peak_numel * 16,
        score_peak_log2=_log2(ev.split_peak_numel),
        score_total_log2=_log2(ev.split_total_numel),
        accepted=bool(accepted),
        reject_reason=reject_reason,
        elapsed_s_svd=ev.elapsed_s_svd,
    )


def find_best_split(node, rank_rule, tol, rng, args, candidate_rows):
    splits = find_top_splits(node, rank_rule, tol, rng, args, candidate_rows)
    if not splits:
        return None
    best = splits[0]
    candidate_rows.append(candidate_row(node, best, rank_rule, tol, True, ""))
    return best


def find_top_splits(node, rank_rule, tol, rng, args, candidate_rows, limit=None):
    if node.tensor.size <= int(args.min_node_numel):
        return []
    if len(node.legs) <= int(args.min_legs):
        return []
    parts = generate_partitions(
        node.legs,
        args.random_candidates,
        args.balance_tol,
        rng,
    )
    parts = _proxy_sort_partitions(node.legs, parts)
    parts = parts[:int(args.max_proxy_candidates_per_node)]
    parts = parts[:int(args.top_svd)]
    old_numel = int(node.tensor.size)
    evals = []
    for cid, (A, B) in enumerate(parts):
        try:
            ev = evaluate_split(node.tensor, node.legs, A, B, cid, rank_rule, tol, args.abs_tol)
            if ev.split_peak_numel < old_numel or ev.split_total_numel < old_numel:
                reason = ""
            else:
                reason = "no_memory_gain"
            candidate_rows.append(candidate_row(node, ev, rank_rule, tol, False, reason))
            evals.append(ev)
            print(
                f"    node={node.node_id} cand={cid} M={ev.matrix_shape[0]}x{ev.matrix_shape[1]} "
                f"rank={ev.rank} peak_log2={_log2(ev.split_peak_numel):.3f} "
                f"total_log2={_log2(ev.split_total_numel):.3f}",
                flush=True,
            )
        except Exception as exc:
            candidate_rows.append(dict(
                node_id=node.node_id,
                depth=node.depth,
                candidate_id=cid,
                rank_rule=rank_rule,
                tol=tol,
                A_legs=" ".join(_leg_names(node.legs, A)),
                B_legs=" ".join(_leg_names(node.legs, B)),
                logdim_A=_leg_logdim(node.legs, A),
                logdim_B=_leg_logdim(node.legs, B),
                matrix_shape="",
                rank="",
                rank_log2="",
                discarded_energy="",
                discarded_relative="",
                old_numel=old_numel,
                split_total_numel="",
                split_peak_numel="",
                split_total_bytes="",
                split_peak_bytes="",
                score_peak_log2="",
                score_total_log2="",
                accepted=False,
                reject_reason=f"svd_error:{exc!r}",
                elapsed_s_svd=0.0,
            ))
    useful = [ev for ev in evals if ev.split_peak_numel < old_numel]
    if not useful and getattr(args, "allow_plateau_splits", False):
        max_total = float(old_numel) * float(args.plateau_total_factor)
        useful = [
            ev for ev in evals
            if ev.split_peak_numel <= old_numel and ev.split_total_numel <= max_total
        ]
    if not useful:
        return []
    useful.sort(key=lambda ev: (
        _log2(ev.split_peak_numel),
        _log2(ev.split_total_numel),
        ev.discarded_relative,
        abs(ev.logdim_A - ev.logdim_B),
    ))
    out = []
    for ev in useful:
        gain = old_numel / float(ev.split_peak_numel)
        if gain >= float(args.min_gain) or (
            getattr(args, "allow_plateau_splits", False)
            and ev.split_peak_numel <= old_numel
        ):
            out.append(ev)
    if limit is None:
        limit = 1
    return out[:int(limit)]


def decompose_greedy(root, rank_rule, tol, rng, args, candidate_rows):
    discarded = 0.0
    max_rank = 1

    def rec(node):
        nonlocal discarded, max_rank
        if node.depth >= int(args.max_depth):
            return
        best = find_best_split(node, rank_rule, tol, rng, args, candidate_rows)
        if best is None:
            return
        left, right = materialize_split(node, best)
        node.split = best
        node.children = [left, right]
        node.tensor = None
        discarded += float(best.discarded_energy)
        max_rank = max(max_rank, int(best.rank))
        rec(left)
        rec(right)

    rec(root)
    return discarded, max_rank


def clone_tree(node):
    new = TreeNode(node.node_id, node.depth, None if node.tensor is None else node.tensor.copy(), node.legs)
    new.split = node.split
    new.children = [clone_tree(c) for c in node.children]
    return new


def find_node(node, node_id):
    if node.node_id == node_id:
        return node
    for child in node.children:
        out = find_node(child, node_id)
        if out is not None:
            return out
    return None


def eligible_leaves(node, args):
    out = []

    def walk(n):
        if n.is_leaf:
            if (
                n.tensor is not None
                and n.depth < int(args.max_depth)
                and n.tensor.size > int(args.min_node_numel)
                and len(n.legs) > int(args.min_legs)
            ):
                out.append(n)
            return
        for c in n.children:
            walk(c)

    walk(node)
    return out


def _tree_score(root):
    st = tree_stats(root)
    return (
        _log2(st["peak_numel"]),
        _log2(st["total_numel"]),
        int(st["num_tensors"]),
        int(st["tree_depth"]),
    )


def decompose_beam(root, rank_rule, tol, rng, args, candidate_rows):
    """Beam search over partial TTN decompositions.

    Each beam expansion splits one currently eligible leaf. The state score is
    lexicographic: minimize peak leaf tensor size first, then total leaf tensor
    size. This keeps the experiment aligned with the static compression
    objective instead of blindly splitting depth-first like greedy recursion.
    """
    beam = [dict(root=root, discarded=0.0, max_rank=1)]
    beam_width = max(1, int(args.beam_width))
    node_splits = max(1, int(args.beam_node_splits))
    max_rounds = max(0, int(args.beam_max_rounds))
    if max_rounds == 0:
        max_rounds = max(1, (2 ** int(args.max_depth)) - 1)

    for round_idx in range(max_rounds):
        expanded = []
        any_split = False
        for state in beam:
            leaves = eligible_leaves(state["root"], args)
            if not leaves:
                expanded.append(state)
                continue
            state_split = False
            for leaf in leaves:
                splits = find_top_splits(
                    leaf,
                    rank_rule,
                    tol,
                    rng,
                    args,
                    candidate_rows,
                    limit=node_splits,
                )
                for ev in splits:
                    any_split = True
                    state_split = True
                    new_root = clone_tree(state["root"])
                    new_leaf = find_node(new_root, leaf.node_id)
                    if new_leaf is None:
                        continue
                    left, right = materialize_split(new_leaf, ev)
                    candidate_rows.append(
                        candidate_row(new_leaf, ev, rank_rule, tol, True,
                                      f"beam_round_{round_idx}")
                    )
                    new_leaf.split = ev
                    new_leaf.children = [left, right]
                    new_leaf.tensor = None
                    expanded.append(dict(
                        root=new_root,
                        discarded=float(state["discarded"]) + float(ev.discarded_energy),
                        max_rank=max(int(state["max_rank"]), int(ev.rank)),
                    ))
            if not getattr(args, "beam_prune_parent", False) or not state_split:
                expanded.append(state)

        expanded.sort(key=lambda s: (
            _tree_score(s["root"]),
            float(s["discarded"]),
        ))
        beam = expanded[:beam_width]
        if not any_split:
            break

    best = min(beam, key=lambda s: (_tree_score(s["root"]), float(s["discarded"])))
    root.node_id = best["root"].node_id
    root.depth = best["root"].depth
    root.tensor = best["root"].tensor
    root.legs = best["root"].legs
    root.split = best["root"].split
    root.children = best["root"].children
    return float(best["discarded"]), int(best["max_rank"])


def reconstruct(node):
    if node.is_leaf:
        return node.tensor, list(node.legs)
    left_t, left_legs = reconstruct(node.children[0])
    right_t, right_legs = reconstruct(node.children[1])
    left_names = [l.name for l in left_legs]
    right_names = [l.name for l in right_legs]
    shared = [name for name in left_names if name in set(right_names)]
    if len(shared) != 1:
        raise RuntimeError(f"expected one shared internal leg at {node.node_id}, got {shared}")
    s = shared[0]
    li = left_names.index(s)
    ri = right_names.index(s)
    out = np.tensordot(left_t, right_t, axes=([li], [ri]))
    out_legs = [l for i, l in enumerate(left_legs) if i != li]
    out_legs += [l for i, l in enumerate(right_legs) if i != ri]
    target = [l.name for l in node.legs]
    cur = [l.name for l in out_legs]
    perm = [cur.index(name) for name in target]
    return np.transpose(out, perm), list(node.legs)


def tree_stats(node):
    leaves = []
    ranks = []
    max_depth = 0

    def walk(n):
        nonlocal max_depth
        max_depth = max(max_depth, n.depth)
        if n.is_leaf:
            leaves.append(n)
        else:
            ranks.append(int(n.split.rank))
            for c in n.children:
                walk(c)

    walk(node)
    numels = [int(n.tensor.size) for n in leaves]
    return dict(
        leaves=leaves,
        total_numel=int(sum(numels)),
        peak_numel=int(max(numels, default=0)),
        num_tensors=len(leaves),
        num_bonds=len(ranks),
        max_bond_rank=max(ranks, default=1),
        tree_depth=max_depth,
    )


def tree_to_json(node):
    if node.is_leaf:
        return dict(
            node_id=node.node_id,
            kind="leaf",
            legs=[l.name for l in node.legs],
            shape=list(map(int, node.tensor.shape)),
            numel=int(node.tensor.size),
            bytes=int(node.tensor.nbytes),
            children=[],
        )
    ev = node.split
    return dict(
        node_id=node.node_id,
        kind="internal",
        legs=[l.name for l in node.legs],
        shape=[l.dim for l in node.legs],
        numel=0,
        bytes=0,
        split=dict(
            A_legs=_leg_names(node.legs, ev.A_idx),
            B_legs=_leg_names(node.legs, ev.B_idx),
            rank=int(ev.rank),
            discarded_energy=float(ev.discarded_energy),
        ),
        children=[tree_to_json(c) for c in node.children],
    )


def run_config(tensor, legs, circuit, step, bag, mode, rank_rule, tol, args):
    t0 = time.perf_counter()
    rng = np.random.default_rng(int(args.seed))
    root = TreeNode("root", 0, np.asarray(tensor), legs)
    candidate_rows = []
    status = "ok"
    notes = ""
    discarded = 0.0
    max_rank = 1
    try:
        if mode == "depth1":
            best = find_best_split(root, rank_rule, tol, rng, args, candidate_rows)
            if best is not None:
                left, right = materialize_split(root, best)
                root.split = best
                root.children = [left, right]
                root.tensor = None
                discarded = float(best.discarded_energy)
                max_rank = int(best.rank)
            else:
                notes = "no beneficial depth-1 split"
        elif mode == "recursive":
            discarded, max_rank = decompose_greedy(root, rank_rule, tol, rng, args, candidate_rows)
            if root.is_leaf:
                notes = "no beneficial recursive split"
        elif mode == "beam":
            discarded, max_rank = decompose_beam(root, rank_rule, tol, rng, args, candidate_rows)
            if root.is_leaf:
                notes = "no beneficial beam split"
        else:
            raise ValueError(f"unknown mode: {mode}")
        recon, recon_legs = reconstruct(root)
        err_abs = float(np.linalg.norm((np.asarray(tensor) - recon).ravel()))
        norm = float(np.linalg.norm(np.asarray(tensor).ravel()))
        err_rel = err_abs / norm if norm > 0 else 0.0
    except Exception as exc:
        status = "error"
        notes = repr(exc)
        err_abs = ""
        err_rel = ""
        norm = float(np.linalg.norm(np.asarray(tensor).ravel()))
    stats = tree_stats(root)
    old_numel = int(np.asarray(tensor).size)
    old_bytes = int(np.asarray(tensor).nbytes)
    total = int(stats["total_numel"])
    peak = int(stats["peak_numel"])
    elapsed = time.perf_counter() - t0
    summary = dict(
        circuit=circuit,
        step=step,
        bag=bag,
        mode=mode,
        rank_rule=rank_rule,
        tol=tol,
        old_numel=old_numel,
        old_bytes=old_bytes,
        old_log2_numel=_log2(old_numel),
        new_total_numel=total,
        new_total_bytes=total * 16,
        new_total_log2_numel=_log2(total),
        new_peak_numel=peak,
        new_peak_bytes=peak * 16,
        new_peak_log2_numel=_log2(peak),
        compression_ratio_total=(old_numel / total if total else ""),
        compression_ratio_peak=(old_numel / peak if peak else ""),
        num_tensors=int(stats["num_tensors"]),
        num_bonds=int(stats["num_bonds"]),
        max_bond_rank=int(stats["max_bond_rank"]),
        max_bond_log2=_log2(stats["max_bond_rank"]),
        tree_depth=int(stats["tree_depth"]),
        recon_error_abs=err_abs,
        recon_error_rel=err_rel,
        discarded_energy_bound=(math.sqrt(discarded) / norm if norm > 0 else 0.0),
        elapsed_s=elapsed,
        status=status,
        notes=notes,
    )
    return summary, candidate_rows, tree_to_json(root)


def _write_csv(path, fields, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _write_md(path, summaries):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("# Static TTN Compression Experiment\n\n")
        f.write("이 실험은 고정된 peak bag tensor에 대한 tolerance-controlled static TTN decomposition feasibility test다.\n\n")
        for r in summaries:
            f.write(f"## {r['circuit']} {r['bag']} step {r['step']} / {r['mode']} / {r['rank_rule']} tol={r['tol']}\n\n")
            f.write(f"- status: `{r['status']}` {r['notes']}\n")
            f.write(f"- old log2 numel: `{float(r['old_log2_numel']):.3f}`, old bytes: `{r['old_bytes']}`\n")
            f.write(f"- new peak log2 numel: `{float(r['new_peak_log2_numel']):.3f}`, peak ratio: `{r['compression_ratio_peak']}`\n")
            f.write(f"- new total log2 numel: `{float(r['new_total_log2_numel']):.3f}`, total ratio: `{r['compression_ratio_total']}`\n")
            f.write(f"- tensors: `{r['num_tensors']}`, bonds: `{r['num_bonds']}`, max rank: `{r['max_bond_rank']}`\n")
            f.write(f"- recon error rel: `{r['recon_error_rel']}`\n\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--circuit", default="coherent_d5_r5")
    p.add_argument("--step", type=int, default=977)
    p.add_argument("--bag", default="B0")
    p.add_argument("--rank-rules", nargs="+", default=["rel", "energy"])
    p.add_argument("--tols", nargs="+", type=float,
                   default=[1e-12, 1e-10, 1e-8, 1e-6, 1e-4])
    p.add_argument("--mode", nargs="+", default=["depth1", "recursive"])
    p.add_argument("--random-candidates", type=int, default=300)
    p.add_argument("--balance-tol", type=float, default=4.0)
    p.add_argument("--max-proxy-candidates-per-node", type=int, default=500)
    p.add_argument("--top-svd", type=int, default=16)
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--min-node-numel", type=int, default=4096)
    p.add_argument("--min-legs", type=int, default=2)
    p.add_argument("--min-gain", type=float, default=1.05)
    p.add_argument("--allow-plateau-splits", action="store_true")
    p.add_argument("--plateau-total-factor", type=float, default=2.0)
    p.add_argument("--beam-width", type=int, default=4)
    p.add_argument("--beam-node-splits", type=int, default=2)
    p.add_argument("--beam-max-rounds", type=int, default=0,
                   help="0 means derive a conservative limit from max-depth")
    p.add_argument("--beam-prune-parent", action="store_true",
                   help="drop an expandable parent state from the next beam frontier")
    p.add_argument("--abs-tol", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--runtime-timeout", type=float, default=70.0)
    p.add_argument("--out-dir", default="reports")
    p.add_argument("--snapshot-cache-dir", default=None)
    p.add_argument("--force-refresh-snapshot", action="store_true")
    args = p.parse_args()

    tensor, legs = load_peak_bag_tensor(
        args.circuit,
        args.step,
        args.bag,
        args.out_dir,
        args.seed,
        args.runtime_timeout,
        args.force_refresh_snapshot,
        args.snapshot_cache_dir,
    )
    print(
        f"loaded tensor shape={tensor.shape} numel={tensor.size} "
        f"bytes={tensor.nbytes} log2={_log2(tensor.size):.3f}",
        flush=True,
    )
    print("legs:")
    for l in legs:
        if l.dim > 1:
            print(f"  {l.name} kind={l.kind} dim={l.dim} log2={l.log2_dim}", flush=True)

    summaries = []
    candidates = []
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    for mode in args.mode:
        for rank_rule in args.rank_rules:
            for tol in args.tols:
                print(f"\n[static-ttn] mode={mode} rule={rank_rule} tol={tol}", flush=True)
                summary, cand_rows, tree = run_config(
                    tensor, legs, args.circuit, args.step, args.bag,
                    mode, rank_rule, tol, args,
                )
                summaries.append(summary)
                candidates.extend(cand_rows)
                tag = f"{mode}_{rank_rule}_{tol:.0e}".replace("-", "m")
                tree_path = Path(args.out_dir) / f"static_ttn_b0_compression_tree_{tag}.json"
                with open(tree_path, "w") as f:
                    json.dump(tree, f, indent=2)
                print(
                    f"  status={summary['status']} peak_log2={summary['new_peak_log2_numel']} "
                    f"total_log2={summary['new_total_log2_numel']} "
                    f"err={summary['recon_error_rel']}",
                    flush=True,
                )

    out_dir = Path(args.out_dir)
    _write_csv(out_dir / "static_ttn_b0_compression_summary.csv", SUMMARY_FIELDS, summaries)
    _write_csv(out_dir / "static_ttn_b0_compression_candidates.csv", CANDIDATE_FIELDS, candidates)
    _write_md(out_dir / "static_ttn_b0_compression_report.md", summaries)

    print("\nsummary:")
    for r in summaries:
        print(
            f"{r['mode']:9s} {r['rank_rule']:6s} tol={r['tol']:<9g} "
            f"peak_log2={float(r['new_peak_log2_numel']):6.3f} "
            f"total_log2={float(r['new_total_log2_numel']):6.3f} "
            f"peak_ratio={r['compression_ratio_peak']} err={r['recon_error_rel']}"
        )


if __name__ == "__main__":
    main()
