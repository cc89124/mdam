"""Multi-snapshot global TTN skeleton search with lazy live allocation.

This is an offline/profile-time topology experiment. It searches one binary
tree over the union of B0 tensor legs from several critical snapshots, then
evaluates that fixed topology on each snapshot using only the live legs present
at that step.

The final selection objective is lexicographic actual evaluation:

    (max_t peak_log2, max_t total_log2, max_t reconstruction_error,
     num_tensors, tree_depth)

Candidate split generators are deliberately diverse, but they are only proposal
mechanisms. They are not the final score.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np
from scipy import linalg

from ttn_backend.scripts.fixed_topology_reuse_experiment import (
    LegMeta,
    _load_snapshot,
    _log2,
    _snapshot_path,
    capture_snapshots,
    select_critical_steps,
)


SUMMARY_FIELDS = [
    "method",
    "rank_rule",
    "tol",
    "num_steps",
    "steps",
    "worst_peak_log2",
    "worst_peak_step",
    "worst_total_log2",
    "worst_total_step",
    "max_recon_error_rel",
    "max_error_step",
    "median_peak_compression_vs_old",
    "min_peak_compression_vs_old",
    "max_internal_rank",
    "max_internal_rank_log2",
    "num_tensors",
    "tree_depth",
    "elapsed_s",
    "status",
    "notes",
]

PER_STEP_FIELDS = [
    "method",
    "step",
    "old_log2_numel",
    "old_bytes",
    "peak_log2",
    "peak_bytes",
    "total_log2",
    "total_bytes",
    "peak_compression_ratio_vs_old",
    "total_compression_ratio_vs_old",
    "recon_error_rel",
    "max_internal_rank",
    "max_internal_rank_log2",
    "num_tensors",
    "tree_depth_live",
    "status",
    "notes",
]

BEAM_HISTORY_FIELDS = [
    "round",
    "beam_rank",
    "tree_id",
    "worst_peak_log2",
    "worst_peak_step",
    "worst_total_log2",
    "worst_total_step",
    "max_recon_error_rel",
    "num_tensors",
    "tree_depth",
    "expanded_from",
    "split_node",
    "elapsed_s",
]

CANDIDATE_FIELDS = [
    "round",
    "tree_id",
    "node_id",
    "candidate_id",
    "generator",
    "A_leaves",
    "B_leaves",
    "proxy_max_live_child_logdim",
    "proxy_live_balance",
    "actual_worst_peak_log2",
    "actual_worst_total_log2",
    "actual_max_error",
    "accepted",
    "reject_reason",
    "elapsed_s",
]


@dataclass
class Snapshot:
    step: int
    tensor: np.ndarray
    legs: list[LegMeta]

    @property
    def live(self) -> set[str]:
        return {leg.name for leg in self.legs}

    @property
    def leg_map(self) -> dict[str, LegMeta]:
        return {leg.name: leg for leg in self.legs}


@dataclass
class MaterializedNode:
    node_id: str
    tensor: np.ndarray
    legs: list[LegMeta]
    children: list["MaterializedNode"]
    split_rank: int | None = None


@dataclass
class StepEvalResult:
    step: int
    status: str
    old_numel: int
    old_bytes: int
    old_log2_numel: float
    peak_numel: int
    peak_bytes: int
    peak_log2: float
    total_numel: int
    total_bytes: int
    total_log2: float
    recon_error_abs: float
    recon_error_rel: float
    max_internal_rank: int
    max_internal_rank_log2: float
    num_tensors: int
    tree_depth_live: int
    node_stats: list[dict]
    notes: str = ""


@dataclass
class GlobalEvalResult:
    tree_id: str
    status: str
    worst_peak_log2: float
    worst_peak_step: int | None
    worst_total_log2: float
    worst_total_step: int | None
    max_recon_error_rel: float
    max_error_step: int | None
    median_peak_compression: float
    min_peak_compression: float
    max_internal_rank: int
    num_tensors: int
    tree_depth: int
    per_step: list[StepEvalResult]
    elapsed_s: float
    notes: str = ""

    def score(self):
        return (
            float(self.worst_peak_log2),
            float(self.worst_total_log2),
            float(self.max_recon_error_rel),
            int(self.num_tensors),
            int(self.tree_depth),
        )


def _prod(vals):
    out = 1
    for v in vals:
        out *= int(v)
    return int(out)


def _bytes_for_numel(numel):
    return int(numel) * 16


def _rank_from_singular_values(s, rank_rule, tol, abs_tol=0.0):
    if s.size == 0:
        return 1, 0.0, 0.0
    total = float(np.sum(s * s))
    if rank_rule == "rel":
        threshold = max(float(abs_tol), float(tol) * float(s[0]))
        rank = max(1, int(np.count_nonzero(s > threshold)))
    elif rank_rule == "energy":
        if total <= 0:
            rank = 1
        else:
            limit = (float(tol) ** 2) * total
            rank = len(s)
            tail = np.cumsum((s[::-1] * s[::-1]))[::-1]
            for cand in range(1, len(s) + 1):
                discarded = float(tail[cand]) if cand < len(s) else 0.0
                if discarded <= limit:
                    rank = cand
                    break
    else:
        raise ValueError(f"unknown rank rule: {rank_rule}")
    discarded = float(np.sum(s[rank:] * s[rank:]))
    rel = math.sqrt(discarded / total) if total > 0 else 0.0
    return rank, discarded, rel


def canonical_tree(leaves, node_id="root"):
    leaves = sorted(set(leaves))
    return dict(node_id=node_id, leaves=leaves, left=None, right=None)


def is_leaf(node):
    return node.get("left") is None and node.get("right") is None


def clone_tree(node):
    return json.loads(json.dumps(node))


def assign_node_ids(node, prefix="root"):
    node["node_id"] = prefix
    if not is_leaf(node):
        assign_node_ids(node["left"], prefix + "L")
        assign_node_ids(node["right"], prefix + "R")
        node["leaves"] = sorted(set(node["left"]["leaves"]) | set(node["right"]["leaves"]))
    else:
        node["leaves"] = sorted(set(node["leaves"]))
    return node


def split_leaf(node, target_id, A, B):
    if node["node_id"] == target_id:
        if not is_leaf(node):
            raise ValueError(f"target is not a leaf: {target_id}")
        node["left"] = canonical_tree(A, target_id + "L")
        node["right"] = canonical_tree(B, target_id + "R")
        node["leaves"] = sorted(set(A) | set(B))
        return True
    if is_leaf(node):
        return False
    return split_leaf(node["left"], target_id, A, B) or split_leaf(node["right"], target_id, A, B)


def iter_leaf_nodes(node):
    if is_leaf(node):
        yield node
        return
    yield from iter_leaf_nodes(node["left"])
    yield from iter_leaf_nodes(node["right"])


def tree_depth(node):
    if is_leaf(node):
        return 0
    return 1 + max(tree_depth(node["left"]), tree_depth(node["right"]))


def num_tree_leaves(node):
    return sum(1 for _ in iter_leaf_nodes(node))


def convert_static_tree(node):
    """Convert static compression tree JSON into the binary skeleton format."""
    if node.get("children"):
        left = convert_static_tree(node["children"][0])
        right = convert_static_tree(node["children"][1])
        out = dict(
            node_id=node.get("node_id", "n"),
            leaves=sorted(set(left["leaves"]) | set(right["leaves"])),
            left=left,
            right=right,
        )
    else:
        original_legs = [
            name for name in node.get("legs", node.get("leaves", []))
            if not str(name).startswith("internal:")
        ]
        out = dict(
            node_id=node.get("node_id", "n"),
            leaves=sorted(set(original_legs)),
            left=None,
            right=None,
        )
    return out


def restrict_tree_to_live(node, live):
    live_here = [x for x in node["leaves"] if x in live]
    if not live_here:
        return None
    if is_leaf(node):
        return dict(node_id=node["node_id"], leaves=live_here, left=None, right=None)
    left = restrict_tree_to_live(node["left"], live)
    right = restrict_tree_to_live(node["right"], live)
    if left is None:
        return right
    if right is None:
        return left
    return dict(
        node_id=node["node_id"],
        leaves=sorted(set(left["leaves"]) | set(right["leaves"])),
        left=left,
        right=right,
    )


def _leg(name, dim, kind="internal"):
    return LegMeta(name=name, kind=kind, dim=int(dim), original_axis=None, edge_id=name, log2_dim=_log2(dim))


def _materialized_leaves(node):
    if not node.children:
        return [node]
    out = []
    for child in node.children:
        out.extend(_materialized_leaves(child))
    return out


def _max_depth_materialized(node):
    if not node.children:
        return 0
    return 1 + max(_max_depth_materialized(c) for c in node.children)


def _reconstruct(node):
    if not node.children:
        return node.tensor, list(node.legs)
    left_t, left_legs = _reconstruct(node.children[0])
    right_t, right_legs = _reconstruct(node.children[1])
    left_names = [l.name for l in left_legs]
    right_names = [l.name for l in right_legs]
    shared = [name for name in left_names if name in set(right_names)]
    if len(shared) != 1:
        raise RuntimeError(f"expected one shared internal leg at {node.node_id}, got {shared}")
    sname = shared[0]
    li = left_names.index(sname)
    ri = right_names.index(sname)
    out = np.tensordot(left_t, right_t, axes=([li], [ri]))
    out_legs = [l for i, l in enumerate(left_legs) if i != li]
    out_legs += [l for i, l in enumerate(right_legs) if i != ri]
    target = [l.name for l in node.legs]
    cur = [l.name for l in out_legs]
    perm = [cur.index(name) for name in target]
    return np.transpose(out, perm), list(node.legs)


def evaluate_tree_on_step(tree, snapshot, rank_rule="rel", tol=1e-8, abs_tol=0.0):
    live = snapshot.live
    live_tree = restrict_tree_to_live(tree, live)
    if live_tree is None:
        raise ValueError(f"tree has no live leaves at step {snapshot.step}")

    leg_map = snapshot.leg_map
    name_to_axis = {leg.name: i for i, leg in enumerate(snapshot.legs)}
    root_order = [name for name in live_tree["leaves"] if name in live]
    perm = [name_to_axis[name] for name in root_order]
    tensor0 = np.transpose(snapshot.tensor, perm)
    legs0 = [leg_map[name] for name in root_order]
    ranks = []

    def rec(tensor, legs, node):
        if is_leaf(node):
            return MaterializedNode(node["node_id"], tensor, legs, [])
        cur_names = [leg.name for leg in legs]
        left_names = [name for name in node["left"]["leaves"] if name in cur_names]
        right_names = [name for name in node["right"]["leaves"] if name in cur_names]
        if not left_names:
            return rec(tensor, legs, node["right"])
        if not right_names:
            return rec(tensor, legs, node["left"])

        extra_names = [
            name for name in cur_names
            if name not in set(left_names) and name not in set(right_names)
        ]
        left_extra = []
        right_extra = []
        for name in extra_names:
            left_log = sum(legs[cur_names.index(x)].log2_dim for x in left_names + left_extra)
            right_log = sum(legs[cur_names.index(x)].log2_dim for x in right_names + right_extra)
            if left_log <= right_log:
                left_extra.append(name)
            else:
                right_extra.append(name)
        left_order = left_names + left_extra
        right_order = right_extra + right_names
        left_idx = [cur_names.index(name) for name in left_order]
        right_idx = [cur_names.index(name) for name in right_order]
        tensor2 = np.transpose(tensor, left_idx + right_idx)
        left_legs = [legs[i] for i in left_idx]
        right_legs = [legs[i] for i in right_idx]
        dim_left = _prod(leg.dim for leg in left_legs)
        dim_right = _prod(leg.dim for leg in right_legs)
        matrix = tensor2.reshape(dim_left, dim_right)
        U, s, Vh = linalg.svd(matrix, full_matrices=False, lapack_driver="gesdd")
        rank, _discarded, _rel = _rank_from_singular_values(s, rank_rule, tol, abs_tol)
        ranks.append(rank)
        sqrt_s = np.sqrt(s[:rank])
        left_m = U[:, :rank] * sqrt_s[None, :]
        right_m = sqrt_s[:, None] * Vh[:rank, :]
        bond_name = f"internal:{node['node_id']}"
        left_bond = _leg(bond_name, rank)
        right_bond = _leg(bond_name, rank)
        left_tensor = left_m.reshape(tuple(leg.dim for leg in left_legs) + (rank,))
        right_tensor = right_m.reshape((rank,) + tuple(leg.dim for leg in right_legs))
        left_node = rec(left_tensor, left_legs + [left_bond], node["left"])
        right_node = rec(right_tensor, [right_bond] + right_legs, node["right"])
        return MaterializedNode(
            node["node_id"],
            np.empty((0,), dtype=snapshot.tensor.dtype),
            legs,
            [left_node, right_node],
            split_rank=rank,
        )

    materialized = rec(tensor0, legs0, live_tree)
    recon, recon_legs = _reconstruct(materialized)
    recon_names = [leg.name for leg in recon_legs]
    perm_back = [recon_names.index(name) for name in root_order]
    recon = np.transpose(recon, perm_back)
    err_abs = float(np.linalg.norm((tensor0 - recon).ravel()))
    norm = float(np.linalg.norm(tensor0.ravel()))
    leaves = _materialized_leaves(materialized)
    node_stats = []
    for leaf in leaves:
        names = [leg.name for leg in leaf.legs]
        internal = [leg for leg in leaf.legs if leg.kind == "internal"]
        open_legs = [leg for leg in leaf.legs if leg.kind != "internal"]
        numel = int(leaf.tensor.size)
        node_stats.append(dict(
            node_id=leaf.node_id,
            shape=list(map(int, leaf.tensor.shape)),
            numel=numel,
            bytes=_bytes_for_numel(numel),
            log2_numel=_log2(numel),
            open_legs=[leg.name for leg in open_legs],
            internal_bonds=[leg.name for leg in internal],
            internal_bond_ranks=[int(leg.dim) for leg in internal],
            legs=names,
        ))
    numels = [int(leaf.tensor.size) for leaf in leaves]
    total_numel = int(sum(numels))
    peak_numel = int(max(numels, default=0))
    old_numel = int(snapshot.tensor.size)
    return StepEvalResult(
        step=snapshot.step,
        status="ok",
        old_numel=old_numel,
        old_bytes=int(snapshot.tensor.nbytes),
        old_log2_numel=_log2(old_numel),
        peak_numel=peak_numel,
        peak_bytes=_bytes_for_numel(peak_numel),
        peak_log2=_log2(peak_numel),
        total_numel=total_numel,
        total_bytes=_bytes_for_numel(total_numel),
        total_log2=_log2(total_numel),
        recon_error_abs=err_abs,
        recon_error_rel=err_abs / norm if norm else 0.0,
        max_internal_rank=max(ranks, default=1),
        max_internal_rank_log2=_log2(max(ranks, default=1)),
        num_tensors=len(leaves),
        tree_depth_live=_max_depth_materialized(materialized),
        node_stats=node_stats,
    )


def evaluate_tree_multisnapshot(tree, snapshots, rank_rule, tol, best_score=None, abs_tol=0.0):
    t0 = time.perf_counter()
    per_step = []
    notes = ""
    error_notes = []
    ordered = sorted(
        snapshots.values(),
        key=lambda s: (s.step != 944, -_log2(s.tensor.size)),
    )
    worst_peak = float("-inf")
    worst_peak_step = None
    for snap in ordered:
        try:
            result = evaluate_tree_on_step(tree, snap, rank_rule, tol, abs_tol)
        except Exception as exc:
            error_notes.append(f"step {snap.step}: {exc!r}")
            result = StepEvalResult(
                step=snap.step,
                status="error",
                old_numel=int(snap.tensor.size),
                old_bytes=int(snap.tensor.nbytes),
                old_log2_numel=_log2(snap.tensor.size),
                peak_numel=0,
                peak_bytes=0,
                peak_log2=float("inf"),
                total_numel=0,
                total_bytes=0,
                total_log2=float("inf"),
                recon_error_abs=float("inf"),
                recon_error_rel=float("inf"),
                max_internal_rank=0,
                max_internal_rank_log2=float("inf"),
                num_tensors=0,
                tree_depth_live=0,
                node_stats=[],
                notes=repr(exc),
            )
        per_step.append(result)
        if result.peak_log2 > worst_peak:
            worst_peak = result.peak_log2
            worst_peak_step = result.step
        if best_score is not None and worst_peak > float(best_score[0]) + 1e-12:
            notes = "early_rejected_by_peak"
            break

    ok = [r for r in per_step if r.status == "ok"]
    if not ok:
        return GlobalEvalResult(
            tree_id=tree.get("tree_id", tree.get("node_id", "tree")),
            status="error",
            worst_peak_log2=float("inf"),
            worst_peak_step=None,
            worst_total_log2=float("inf"),
            worst_total_step=None,
            max_recon_error_rel=float("inf"),
            max_error_step=None,
            median_peak_compression=0.0,
            min_peak_compression=0.0,
            max_internal_rank=0,
            num_tensors=num_tree_leaves(tree),
            tree_depth=tree_depth(tree),
            per_step=per_step,
            elapsed_s=time.perf_counter() - t0,
            notes="; ".join(error_notes[:3]) or notes,
        )
    worst_total_row = max(ok, key=lambda r: r.total_log2)
    max_err_row = max(ok, key=lambda r: r.recon_error_rel)
    ratios = sorted(r.old_numel / r.peak_numel for r in ok if r.peak_numel > 0)
    median_ratio = ratios[len(ratios) // 2] if ratios else 0.0
    return GlobalEvalResult(
        tree_id=tree.get("tree_id", tree.get("node_id", "tree")),
        status="ok" if len(ok) == len(snapshots) else "partial",
        worst_peak_log2=max(r.peak_log2 for r in ok),
        worst_peak_step=max(ok, key=lambda r: r.peak_log2).step,
        worst_total_log2=worst_total_row.total_log2,
        worst_total_step=worst_total_row.step,
        max_recon_error_rel=max_err_row.recon_error_rel,
        max_error_step=max_err_row.step,
        median_peak_compression=median_ratio,
        min_peak_compression=min(ratios) if ratios else 0.0,
        max_internal_rank=max(r.max_internal_rank for r in ok),
        num_tensors=max(r.num_tensors for r in ok),
        tree_depth=tree_depth(tree),
        per_step=per_step,
        elapsed_s=time.perf_counter() - t0,
        notes=notes,
    )


def _step_logdim(step_maps, step, leaves):
    m = step_maps[step]
    return float(sum(m[x].log2_dim for x in leaves if x in m))


def split_affects_any_step(A, B, live_sets):
    A = set(A); B = set(B)
    for live in live_sets.values():
        if A & live and B & live:
            return True
    return False


def split_proxy(A, B, step_maps):
    max_child = 0.0
    max_balance = 0.0
    for step in step_maps:
        la = _step_logdim(step_maps, step, A)
        lb = _step_logdim(step_maps, step, B)
        if la <= 0.0 or lb <= 0.0:
            continue
        max_child = max(max_child, la, lb)
        max_balance = max(max_balance, abs(la - lb))
    return max_child, max_balance


def _canonical_split(A, S):
    S = sorted(set(S))
    A = sorted(set(A) & set(S))
    B = sorted(set(S) - set(A))
    if not A or not B:
        return None
    # Avoid complement duplicates.
    if tuple(A) > tuple(B):
        A, B = B, A
    return tuple(A), tuple(B)


def generator_live_logdim_balance(S, step_maps):
    def load(leaf):
        return max((m[leaf].log2_dim for m in step_maps.values() if leaf in m), default=0.0)
    A, B = [], []
    la = lb = 0.0
    for leaf in sorted(S, key=lambda x: load(x), reverse=True):
        if la <= lb:
            A.append(leaf); la += load(leaf)
        else:
            B.append(leaf); lb += load(leaf)
    sp = _canonical_split(A, S)
    return [("live_logdim_balance", sp)] if sp else []


def generator_large_leg_separation(S, step_maps):
    def load(leaf):
        return max((m[leaf].log2_dim for m in step_maps.values() if leaf in m), default=0.0)
    order = sorted(S, key=lambda x: load(x), reverse=True)
    if len(order) < 2:
        return []
    A = [order[0]]
    B = [order[1]]
    la = load(order[0])
    lb = load(order[1])
    for leaf in order[2:]:
        if la <= lb:
            A.append(leaf); la += load(leaf)
        else:
            B.append(leaf); lb += load(leaf)
    sp = _canonical_split(A, S)
    return [("large_leg_separation", sp)] if sp else []


def generator_random_balanced(S, step_maps, rng, n_random, balance_tol):
    S = sorted(S)
    out = []
    for _ in range(int(n_random)):
        shuffled = S[:]
        rng.shuffle(shuffled)
        A = []
        cur = 0.0
        total = max(sum(m[x].log2_dim for x in S if x in m) for m in step_maps.values())
        target = total / 2.0
        for leaf in shuffled:
            if not A or cur < target:
                A.append(leaf)
                cur += max((m[leaf].log2_dim for m in step_maps.values() if leaf in m), default=0.0)
        sp = _canonical_split(A, S)
        if not sp:
            continue
        max_child, max_bal = split_proxy(sp[0], sp[1], step_maps)
        if max_bal <= float(balance_tol) and max_child > 0:
            out.append(("random_balanced", sp))
    return out


def generator_assignment_beam(S, step_maps, beam_width=16, max_outputs=4):
    """Generate split candidates by beam search over A/B labels.

    This is a parameter-bounded candidate generator, not the final objective.
    It searches the split label space with the same parameter-free proxy used
    for filtering: minimize max live child logdim, then live imbalance. Actual
    multi-snapshot SVD evaluation still decides whether the split is good.
    """
    S = sorted(S)
    if len(S) < 2:
        return []

    def load(leaf):
        return max((m[leaf].log2_dim for m in step_maps.values() if leaf in m), default=0.0)

    order = sorted(S, key=lambda x: load(x), reverse=True)
    states = [(tuple(), tuple())]
    for leaf in order:
        nxt = []
        for A, B in states:
            nxt.append((A + (leaf,), B))
            nxt.append((A, B + (leaf,)))
        scored = []
        for A, B in nxt:
            if not A or not B:
                score = (float("inf"), float("inf"), len(A) + len(B))
            else:
                score = (*split_proxy(A, B, step_maps), len(A) + len(B))
            scored.append((score, A, B))
        scored.sort(key=lambda x: x[0])
        states = [(A, B) for _score, A, B in scored[:int(beam_width)]]

    out = []
    seen = set()
    for A, B in states:
        sp = _canonical_split(A, S)
        if not sp:
            continue
        key = (sp[0], sp[1])
        if key in seen:
            continue
        seen.add(key)
        out.append(("assignment_beam", sp))
    return out[:int(max_outputs)]


def generator_colive_spectral(S, live_sets):
    S = sorted(S)
    n = len(S)
    if n < 3:
        return []
    idx = {x: i for i, x in enumerate(S)}
    W = np.zeros((n, n), dtype=float)
    for live in live_sets.values():
        local = [idx[x] for x in S if x in live]
        for i in local:
            W[i, local] += 1.0
    np.fill_diagonal(W, 0.0)
    deg = W.sum(axis=1)
    if np.count_nonzero(deg) <= 1:
        return []
    L = np.diag(deg) - W
    try:
        vals, vecs = np.linalg.eigh(L)
    except Exception:
        return []
    order = np.argsort(vecs[:, 1] if n > 1 else vecs[:, 0])
    outs = []
    mid = n // 2
    for cut in sorted(set([mid, max(1, mid - 2), min(n - 1, mid + 2), max(1, mid - 1), min(n - 1, mid + 1)])):
        A = [S[int(i)] for i in order[:cut]]
        sp = _canonical_split(A, S)
        if sp:
            outs.append(("colive_spectral", sp))
    return outs


def _collect_tree_splits(node, out=None):
    if out is None:
        out = []
    if not is_leaf(node):
        out.append((tuple(node["left"]["leaves"]), tuple(node["right"]["leaves"])))
        _collect_tree_splits(node["left"], out)
        _collect_tree_splits(node["right"], out)
    return out


def generator_previous_splits(S, previous_splits):
    out = []
    Sset = set(S)
    for A0, B0 in previous_splits:
        for side in (set(A0), set(B0)):
            A = sorted(side & Sset)
            sp = _canonical_split(A, S)
            if sp:
                out.append(("previous_topology", sp))
    return out


def generate_candidate_splits(S, step_maps, live_sets, rng, args, previous_splits):
    generated = []
    generated.extend(generator_live_logdim_balance(S, step_maps))
    generated.extend(generator_large_leg_separation(S, step_maps))
    generated.extend(generator_colive_spectral(S, live_sets))
    generated.extend(generator_assignment_beam(
        S,
        step_maps,
        beam_width=args.assignment_beam_width,
        max_outputs=args.assignment_beam_outputs,
    ))
    generated.extend(generator_previous_splits(S, previous_splits))
    generated.extend(generator_random_balanced(S, step_maps, rng, args.random_candidates, args.balance_tol))
    seen = {}
    for gen, split in generated:
        if not split:
            continue
        A, B = split
        if not split_affects_any_step(A, B, live_sets):
            continue
        key = (tuple(A), tuple(B))
        if key not in seen:
            max_child, balance = split_proxy(A, B, step_maps)
            seen[key] = dict(
                generator=gen,
                A=list(A),
                B=list(B),
                proxy_max_live_child_logdim=max_child,
                proxy_live_balance=balance,
            )
    rows = list(seen.values())
    rows.sort(key=lambda r: (r["proxy_max_live_child_logdim"], r["proxy_live_balance"], r["generator"]))
    return rows[:int(args.top_svd)]


def eligible_split_nodes(tree, step_maps, limit):
    nodes = list(iter_leaf_nodes(tree))
    scored = []
    for node in nodes:
        if len(node["leaves"]) < 2:
            continue
        max_log = max((_step_logdim(step_maps, step, node["leaves"]) for step in step_maps), default=0.0)
        if max_log <= 0.0:
            continue
        scored.append((max_log, len(node["leaves"]), node))
    scored.sort(key=lambda x: (-x[0], -x[1], x[2]["node_id"]))
    return [x[2] for x in scored[:int(limit)]]


def score_tuple(eval_result):
    return eval_result.score()


def summarize_method(method, eval_result, rank_rule, tol, steps, elapsed_s=None, notes=""):
    return dict(
        method=method,
        rank_rule=rank_rule,
        tol=tol,
        num_steps=len(steps),
        steps=" ".join(str(s) for s in steps),
        worst_peak_log2=eval_result.worst_peak_log2,
        worst_peak_step=eval_result.worst_peak_step,
        worst_total_log2=eval_result.worst_total_log2,
        worst_total_step=eval_result.worst_total_step,
        max_recon_error_rel=eval_result.max_recon_error_rel,
        max_error_step=eval_result.max_error_step,
        median_peak_compression_vs_old=eval_result.median_peak_compression,
        min_peak_compression_vs_old=eval_result.min_peak_compression,
        max_internal_rank=eval_result.max_internal_rank,
        max_internal_rank_log2=_log2(eval_result.max_internal_rank),
        num_tensors=eval_result.num_tensors,
        tree_depth=eval_result.tree_depth,
        elapsed_s=eval_result.elapsed_s if elapsed_s is None else elapsed_s,
        status=eval_result.status,
        notes=notes or eval_result.notes,
    )


def fixed_reuse_eval_from_csv(path, steps):
    if not path or not os.path.exists(path):
        return None
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            step = int(float(row["step"]))
            if step in set(steps) and row.get("status") == "ok":
                old_numel = int(float(row["old_B0_numel"]))
                peak_numel = int(float(row["fixed_topology_peak_numel"]))
                total_numel = int(float(row["fixed_topology_total_numel"]))
                rows.append(StepEvalResult(
                    step=step,
                    status="ok",
                    old_numel=old_numel,
                    old_bytes=int(float(row["old_B0_bytes"])),
                    old_log2_numel=float(row["old_B0_log2_numel"]),
                    peak_numel=peak_numel,
                    peak_bytes=int(float(row["fixed_topology_peak_bytes"])),
                    peak_log2=float(row["fixed_topology_peak_log2"]),
                    total_numel=total_numel,
                    total_bytes=int(float(row["fixed_topology_total_bytes"])),
                    total_log2=float(row["fixed_topology_total_log2"]),
                    recon_error_abs=float(row["reconstruction_error_abs"]),
                    recon_error_rel=float(row["reconstruction_error_rel"]),
                    max_internal_rank=int(float(row["max_internal_rank"])),
                    max_internal_rank_log2=float(row["max_internal_rank_log2"]),
                    num_tensors=int(float(row["num_tensors"])),
                    tree_depth_live=int(float(row["tree_depth"])),
                    node_stats=[],
                ))
    if not rows:
        return None
    ratios = sorted(r.old_numel / r.peak_numel for r in rows if r.peak_numel > 0)
    return GlobalEvalResult(
        tree_id="fixed_T977_csv",
        status="ok" if len(rows) == len(steps) else "partial",
        worst_peak_log2=max(r.peak_log2 for r in rows),
        worst_peak_step=max(rows, key=lambda r: r.peak_log2).step,
        worst_total_log2=max(r.total_log2 for r in rows),
        worst_total_step=max(rows, key=lambda r: r.total_log2).step,
        max_recon_error_rel=max(r.recon_error_rel for r in rows),
        max_error_step=max(rows, key=lambda r: r.recon_error_rel).step,
        median_peak_compression=ratios[len(ratios) // 2] if ratios else 0.0,
        min_peak_compression=min(ratios) if ratios else 0.0,
        max_internal_rank=max(r.max_internal_rank for r in rows),
        num_tensors=max(r.num_tensors for r in rows),
        tree_depth=max(r.tree_depth_live for r in rows),
        per_step=rows,
        elapsed_s=0.0,
        notes=f"loaded_from:{path}",
    )


def per_step_rows(method, eval_result):
    rows = []
    for r in eval_result.per_step:
        rows.append(dict(
            method=method,
            step=r.step,
            old_log2_numel=r.old_log2_numel,
            old_bytes=r.old_bytes,
            peak_log2=r.peak_log2,
            peak_bytes=r.peak_bytes,
            total_log2=r.total_log2,
            total_bytes=r.total_bytes,
            peak_compression_ratio_vs_old=(r.old_numel / r.peak_numel if r.peak_numel else ""),
            total_compression_ratio_vs_old=(r.old_numel / r.total_numel if r.total_numel else ""),
            recon_error_rel=r.recon_error_rel,
            max_internal_rank=r.max_internal_rank,
            max_internal_rank_log2=r.max_internal_rank_log2,
            num_tensors=r.num_tensors,
            tree_depth_live=r.tree_depth_live,
            status=r.status,
            notes=r.notes,
        ))
    return rows


def evaluate_current_hub(snapshots, rank_rule, tol):
    per_step = []
    for snap in snapshots.values():
        old_numel = int(snap.tensor.size)
        per_step.append(StepEvalResult(
            step=snap.step,
            status="ok",
            old_numel=old_numel,
            old_bytes=int(snap.tensor.nbytes),
            old_log2_numel=_log2(old_numel),
            peak_numel=old_numel,
            peak_bytes=int(snap.tensor.nbytes),
            peak_log2=_log2(old_numel),
            total_numel=old_numel,
            total_bytes=int(snap.tensor.nbytes),
            total_log2=_log2(old_numel),
            recon_error_abs=0.0,
            recon_error_rel=0.0,
            max_internal_rank=1,
            max_internal_rank_log2=0.0,
            num_tensors=1,
            tree_depth_live=0,
            node_stats=[],
        ))
    fake = canonical_tree([])
    return GlobalEvalResult(
        tree_id="current_hub",
        status="ok",
        worst_peak_log2=max(r.peak_log2 for r in per_step),
        worst_peak_step=max(per_step, key=lambda r: r.peak_log2).step,
        worst_total_log2=max(r.total_log2 for r in per_step),
        worst_total_step=max(per_step, key=lambda r: r.total_log2).step,
        max_recon_error_rel=0.0,
        max_error_step=None,
        median_peak_compression=1.0,
        min_peak_compression=1.0,
        max_internal_rank=1,
        num_tensors=1,
        tree_depth=tree_depth(fake),
        per_step=per_step,
        elapsed_s=0.0,
    )


def write_csv(path, fields, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def write_report(path, summary_rows, steps, previous_available):
    rows = {r["method"]: r for r in summary_rows}
    with open(path, "w") as f:
        f.write("# Multi-Snapshot Global TTN Skeleton Search\n\n")
        f.write("이 리포트는 full runtime patch가 아니라 offline/profile-time common-skeleton 탐색 결과입니다.\n\n")
        f.write(f"- critical steps: `{', '.join(str(s) for s in steps)}`\n")
        f.write("- final score: `(worst_peak_log2, worst_total_log2, max_error, num_tensors, tree_depth)` lexicographic\n")
        f.write("- lazy live allocation: snapshot에 없는 leaf는 materialize하지 않고, 한쪽 live side가 빈 split은 rank-1 inactive bond로 취급합니다.\n")
        f.write("- weighted alpha/beta/gamma objective: 사용하지 않음\n\n")

        f.write("## Summary\n\n")
        for method in ("current_hub", "fixed_T977", "common_global_tree"):
            if method not in rows:
                continue
            r = rows[method]
            f.write(
                f"- `{method}`: worst_peak_log2=`{float(r['worst_peak_log2']):.6f}` "
                f"(step `{r['worst_peak_step']}`), worst_total_log2=`{float(r['worst_total_log2']):.6f}`, "
                f"max_error=`{float(r['max_recon_error_rel']):.6g}`, "
                f"min_peak_compression=`{float(r['min_peak_compression_vs_old']):.6g}`\n"
            )
        f.write("\n")

        if "current_hub" in rows and "common_global_tree" in rows:
            old = float(rows["current_hub"]["worst_peak_log2"])
            new = float(rows["common_global_tree"]["worst_peak_log2"])
            f.write("## Interpretation\n\n")
            if new < old:
                f.write(
                    f"공통 global topology는 현재 B0 hub worst peak를 `{old:.3f}`에서 `{new:.3f}`로 낮췄습니다. "
                    "이는 선택된 critical snapshot set에서는 하나의 reusable B0-subtree 후보가 존재한다는 증거입니다.\n\n"
                )
            else:
                f.write(
                    "이번 search budget에서는 현재 B0 hub worst peak보다 낮은 공통 topology를 찾지 못했습니다. "
                    "이 경우 더 큰 beam budget 또는 multi-topology/runtime switching이 필요합니다.\n\n"
                )
        if previous_available and "fixed_T977" in rows and "common_global_tree" in rows:
            prev = float(rows["fixed_T977"]["worst_peak_log2"])
            new = float(rows["common_global_tree"]["worst_peak_log2"])
            if new < prev:
                f.write(
                    f"step-977 고정 topology의 hard counterexample worst peak `{prev:.3f}`보다 "
                    f"common topology가 `{new:.3f}`로 개선됐습니다.\n\n"
                )
            else:
                f.write(
                    f"common topology가 step-977 고정 topology worst peak `{prev:.3f}`를 넘어서지는 못했습니다. "
                    "이는 search budget 부족 또는 snapshot별 topology 충돌 가능성을 뜻합니다.\n\n"
                )

        f.write("## Algorithmic Status\n\n")
        f.write(
            "이 방법은 전역 최적해 보장이 있는 알고리즘은 아닙니다. 가능한 binary tree topology 수는 super-exponential이고, "
            "각 split의 numerical rank도 tensor 값에 의존하므로 exact global optimization은 작은 leaf 수를 제외하면 현실적으로 어렵습니다. "
            "대신 목적함수 자체는 수학적으로 명확한 min-max actual memory objective이며, beam search는 그 목적함수를 직접 평가하는 bounded anytime heuristic입니다. "
            "후보 생성기는 search space를 줄이는 역할만 하고 최종 순위에는 관여하지 않습니다.\n"
        )


def load_snapshots(args, steps):
    cached = capture_snapshots(
        args.circuit,
        steps,
        args.bag,
        args.snapshot_cache_dir,
        args.seed,
        args.runtime_timeout,
        args.force_refresh,
    )
    out = {}
    missing = []
    for step in steps:
        if step in cached:
            tensor, legs = cached[step]
            out[step] = Snapshot(step, tensor, legs)
        else:
            path = _snapshot_path(args.snapshot_cache_dir, args.circuit, step, args.bag)
            if path.exists():
                tensor, legs = _load_snapshot(path)
                out[step] = Snapshot(step, tensor, legs)
            else:
                missing.append(step)
    return out, missing


def load_previous_tree(path):
    if not path:
        return None
    if not os.path.exists(path):
        return None
    with open(path) as f:
        raw = json.load(f)
    tree = convert_static_tree(raw)
    assign_node_ids(tree)
    tree["tree_id"] = "fixed_T977"
    return tree


def prune_tree_to_universe(tree, U):
    pruned = restrict_tree_to_live(tree, set(U))
    if pruned is None:
        return None
    assign_node_ids(pruned)
    pruned["tree_id"] = tree.get("tree_id", "fixed_T977")
    return pruned


def search_global_tree(args, snapshots, U, previous_tree=None):
    rng = random.Random(args.seed)
    step_maps = {s: snapshots[s].leg_map for s in snapshots}
    live_sets = {s: snapshots[s].live for s in snapshots}
    previous_splits = _collect_tree_splits(previous_tree) if previous_tree is not None else []

    root = canonical_tree(U)
    root["tree_id"] = "root_unsplit"
    initial = [root]
    if args.seed_initial_previous and previous_tree is not None and set(previous_tree["leaves"]) == set(U):
        initial.append(previous_tree)

    beam = []
    seen = set()
    for tree in initial:
        key = json.dumps(tree, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        ev = evaluate_tree_multisnapshot(tree, snapshots, args.rank_rule, args.rel_tol)
        beam.append(dict(tree=tree, eval=ev, expanded_from="", split_node=""))
    beam.sort(key=lambda x: x["eval"].score())
    beam = beam[: int(args.beam_width)]

    beam_history = []
    candidate_rows = []
    debug = dict(
        generated_candidates=0,
        evaluated_candidates=0,
        reject_reasons={},
        rounds=[],
        final_best_score=None,
        final_best_num_tensors=None,
        final_best_tree_depth=None,
    )
    best_score = beam[0]["eval"].score() if beam else None
    t_start = time.perf_counter()
    for round_idx in range(int(args.beam_rounds)):
        print(f"beam_round={round_idx} beam_size={len(beam)}", flush=True)
        expanded = []
        round_info = dict(
            round=round_idx,
            input_beam_size=len(beam),
            eligible_nodes=0,
            candidate_splits=0,
            evaluated_candidates=0,
            expanded_trees=0,
            reject_reasons=Counter(),
            output_beam=[],
        )
        for bidx, state in enumerate(beam):
            expanded.append(state)
            nodes = eligible_split_nodes(state["tree"], step_maps, args.beam_node_splits)
            round_info["eligible_nodes"] += len(nodes)
            print(
                f"  expand tree={state['tree'].get('tree_id','')} rank={bidx} "
                f"eligible_nodes={len(nodes)} current_score={state['eval'].score()}",
                flush=True,
            )
            for node in nodes:
                splits = generate_candidate_splits(
                    node["leaves"],
                    step_maps,
                    live_sets,
                    rng,
                    args,
                    previous_splits,
                )
                print(
                    f"    node={node['node_id']} leaves={len(node['leaves'])} "
                    f"candidate_splits={len(splits)}",
                    flush=True,
                )
                round_info["candidate_splits"] += len(splits)
                debug["generated_candidates"] += len(splits)
                for cid, split in enumerate(splits):
                    t0 = time.perf_counter()
                    new_tree = clone_tree(state["tree"])
                    ok = split_leaf(new_tree, node["node_id"], split["A"], split["B"])
                    if not ok:
                        continue
                    assign_node_ids(new_tree)
                    new_tree["tree_id"] = f"tree_{round_idx}_{bidx}_{cid}_{uuid.uuid4().hex[:8]}"
                    ev = evaluate_tree_multisnapshot(
                        new_tree,
                        snapshots,
                        args.rank_rule,
                        args.rel_tol,
                        best_score=best_score if args.early_reject else None,
                    )
                    print(
                        f"      cand={cid} gen={split['generator']} "
                        f"score={ev.score()} status={ev.status} notes={ev.notes}",
                        flush=True,
                    )
                    debug["evaluated_candidates"] += 1
                    round_info["evaluated_candidates"] += 1
                    if ev.notes:
                        round_info["reject_reasons"][ev.notes] += 1
                    candidate_rows.append(dict(
                        round=round_idx,
                        tree_id=new_tree["tree_id"],
                        node_id=node["node_id"],
                        candidate_id=cid,
                        generator=split["generator"],
                        A_leaves=" ".join(split["A"]),
                        B_leaves=" ".join(split["B"]),
                        proxy_max_live_child_logdim=split["proxy_max_live_child_logdim"],
                        proxy_live_balance=split["proxy_live_balance"],
                        actual_worst_peak_log2=ev.worst_peak_log2,
                        actual_worst_total_log2=ev.worst_total_log2,
                        actual_max_error=ev.max_recon_error_rel,
                        accepted=False,
                        reject_reason=ev.notes,
                        elapsed_s=time.perf_counter() - t0,
                    ))
                    if ev.status in ("ok", "partial"):
                        expanded.append(dict(
                            tree=new_tree,
                            eval=ev,
                            expanded_from=state["tree"].get("tree_id", ""),
                            split_node=node["node_id"],
                        ))
        expanded.sort(key=lambda x: x["eval"].score())
        if args.depth_diverse_beam:
            selected = []
            seen_ids = set()
            for state in expanded:
                selected.append(state)
                seen_ids.add(id(state))
                if len(selected) >= int(args.beam_width):
                    break
            # Preserve the best state at each depth even when it is on a
            # lexicographic plateau with larger total memory. Without this,
            # a deeper tree cannot survive long enough for later splits to
            # reduce the peak node.
            best_by_depth = {}
            for state in expanded:
                depth = int(state["eval"].tree_depth)
                cur = best_by_depth.get(depth)
                if cur is None or state["eval"].score() < cur["eval"].score():
                    best_by_depth[depth] = state
            for depth in sorted(best_by_depth):
                state = best_by_depth[depth]
                if id(state) not in seen_ids:
                    selected.append(state)
                    seen_ids.add(id(state))
            selected.sort(key=lambda x: (
                x["eval"].score()[0],
                x["eval"].tree_depth,
                x["eval"].score()[1:],
            ))
            beam = selected[: int(args.beam_width)]
        else:
            beam = expanded[: int(args.beam_width)]
        if beam:
            best_score = beam[0]["eval"].score()
        for rank, state in enumerate(beam):
            ev = state["eval"]
            round_info["output_beam"].append(dict(
                beam_rank=rank,
                tree_id=state["tree"].get("tree_id", ""),
                score=list(ev.score()),
                num_tensors=ev.num_tensors,
                tree_depth=ev.tree_depth,
                worst_peak_step=ev.worst_peak_step,
                split_node=state.get("split_node", ""),
            ))
            beam_history.append(dict(
                round=round_idx,
                beam_rank=rank,
                tree_id=state["tree"].get("tree_id", ""),
                worst_peak_log2=ev.worst_peak_log2,
                worst_peak_step=ev.worst_peak_step,
                worst_total_log2=ev.worst_total_log2,
                worst_total_step=ev.worst_total_step,
                max_recon_error_rel=ev.max_recon_error_rel,
                num_tensors=ev.num_tensors,
                tree_depth=ev.tree_depth,
                expanded_from=state.get("expanded_from", ""),
                split_node=state.get("split_node", ""),
                elapsed_s=time.perf_counter() - t_start,
            ))
        round_info["expanded_trees"] = len(expanded)
        round_info["reject_reasons"] = dict(round_info["reject_reasons"])
        debug["rounds"].append(round_info)
    if beam:
        # Mark final accepted tree rows if their id appears in candidate log.
        best_id = beam[0]["tree"].get("tree_id")
        for row in candidate_rows:
            if row["tree_id"] == best_id:
                row["accepted"] = True
    if beam:
        debug["final_best_score"] = list(beam[0]["eval"].score())
        debug["final_best_num_tensors"] = beam[0]["eval"].num_tensors
        debug["final_best_tree_depth"] = beam[0]["eval"].tree_depth
    reject_counter = Counter()
    for r in debug["rounds"]:
        reject_counter.update(r["reject_reasons"])
    debug["reject_reasons"] = dict(reject_counter)
    return beam[0]["tree"], beam[0]["eval"], beam_history, candidate_rows, debug


def write_best_node_stats(path, eval_result):
    data = {}
    for r in eval_result.per_step:
        data[str(r.step)] = r.node_stats
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def write_search_debug_report(path, debug):
    with open(path, "w") as f:
        f.write("# Global Skeleton Search Debug\n\n")
        f.write(f"- generated candidate splits: `{debug.get('generated_candidates', 0)}`\n")
        f.write(f"- evaluated candidate trees: `{debug.get('evaluated_candidates', 0)}`\n")
        f.write(f"- final best score: `{debug.get('final_best_score')}`\n")
        f.write(f"- final best tensors: `{debug.get('final_best_num_tensors')}`\n")
        f.write(f"- final best depth: `{debug.get('final_best_tree_depth')}`\n\n")
        if debug.get("reject_reasons"):
            f.write("## Reject Reasons\n\n")
            for reason, count in sorted(debug["reject_reasons"].items(), key=lambda x: (-x[1], x[0])):
                f.write(f"- `{reason}`: `{count}`\n")
            f.write("\n")
        f.write("## Rounds\n\n")
        for r in debug.get("rounds", []):
            f.write(f"### Round {r['round']}\n\n")
            f.write(f"- input beam size: `{r['input_beam_size']}`\n")
            f.write(f"- eligible nodes: `{r['eligible_nodes']}`\n")
            f.write(f"- candidate splits: `{r['candidate_splits']}`\n")
            f.write(f"- evaluated candidates: `{r['evaluated_candidates']}`\n")
            f.write(f"- expanded trees before pruning: `{r['expanded_trees']}`\n\n")
            f.write("| beam_rank | tree_id | score | tensors | depth | worst_step | split_node |\n")
            f.write("|---:|---|---|---:|---:|---:|---|\n")
            for b in r.get("output_beam", []):
                f.write(
                    f"| {b['beam_rank']} | `{b['tree_id']}` | `{b['score']}` | "
                    f"{b['num_tensors']} | {b['tree_depth']} | {b['worst_peak_step']} | `{b['split_node']}` |\n"
                )
            f.write("\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--circuit", default="coherent_d5_r5")
    p.add_argument("--bag", default="B0")
    p.add_argument("--steps", nargs="*", type=int, default=None)
    p.add_argument("--time-steps-csv", default="reports/time_graph_steps.csv")
    p.add_argument("--time-critical-csv", default="reports/time_graph_critical.csv")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--delta", type=float, default=1.0)
    p.add_argument("--max-selected-steps", type=int, default=0)
    p.add_argument("--rank-rule", choices=["rel", "energy"], default="rel")
    p.add_argument("--rel-tol", type=float, default=1e-8)
    p.add_argument("--beam-width", type=int, default=4)
    p.add_argument("--beam-rounds", type=int, default=6)
    p.add_argument("--beam-node-splits", type=int, default=2)
    p.add_argument("--top-svd", type=int, default=8)
    p.add_argument("--random-candidates", type=int, default=300)
    p.add_argument("--balance-tol", type=float, default=6.0)
    p.add_argument("--assignment-beam-width", type=int, default=16)
    p.add_argument("--assignment-beam-outputs", type=int, default=4)
    p.add_argument("--previous-tree", default="reports/static_rel1e8_beam/static_ttn_b0_compression_tree_beam_rel_1em08.json")
    p.add_argument("--fixed-reuse-csv", default="reports/fixed_topology_reuse_rel1e8/reuse_summary.csv")
    p.add_argument("--evaluate-previous-tree", action="store_true",
                   help="expensive: directly re-evaluate previous tree instead of loading fixed-reuse CSV")
    p.add_argument("--seed-initial-previous", action="store_true",
                   help="expensive: include previous static tree as an initial beam state")
    p.add_argument("--depth-diverse-beam", action="store_true",
                   help="keep plateau deeper states so later splits can reduce peak memory")
    p.add_argument("--snapshot-cache-dir", default="reports/fixed_topology_reuse_rel1e8/snapshots")
    p.add_argument("--out-dir", default="reports/multisnapshot_global_rel1e8")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--runtime-timeout", type=float, default=80.0)
    p.add_argument("--force-refresh", action="store_true")
    p.add_argument("--early-reject", action="store_true")
    args = p.parse_args()

    if args.steps:
        steps = sorted(set(int(x) for x in args.steps))
    else:
        steps = select_critical_steps(args.circuit, args.time_steps_csv, args.time_critical_csv, args.top_k, args.delta)
        if int(args.max_selected_steps) > 0:
            steps = steps[: int(args.max_selected_steps)]
    if not steps:
        raise RuntimeError("no steps selected")
    print(f"selected_steps={steps}", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshots, missing = load_snapshots(args, steps)
    if missing:
        print(f"missing_snapshots={missing}", flush=True)
    steps = sorted(snapshots)
    if not steps:
        raise RuntimeError("no snapshots available")

    U = sorted(set().union(*(snap.live for snap in snapshots.values())))
    print(f"global_leaf_count={len(U)} available_steps={steps}", flush=True)
    previous_tree = load_previous_tree(args.previous_tree)
    previous_available = previous_tree is not None and bool(set(previous_tree["leaves"]) & set(U))
    if previous_tree is not None and set(previous_tree["leaves"]) != set(U):
        print(
            f"previous tree leg set mismatch: previous={len(previous_tree['leaves'])} global={len(U)}; "
            "pruning it to global U for lazy-live baseline",
            flush=True,
        )
        previous_tree = prune_tree_to_universe(previous_tree, U)
        previous_available = previous_tree is not None

    summary_rows = []
    per_rows = []

    current_eval = evaluate_current_hub(snapshots, args.rank_rule, args.rel_tol)
    summary_rows.append(summarize_method("current_hub", current_eval, args.rank_rule, args.rel_tol, steps))
    per_rows.extend(per_step_rows("current_hub", current_eval))

    fixed_eval = fixed_reuse_eval_from_csv(args.fixed_reuse_csv, steps)
    if fixed_eval is not None:
        summary_rows.append(summarize_method("fixed_T977", fixed_eval, args.rank_rule, args.rel_tol, steps))
        per_rows.extend(per_step_rows("fixed_T977", fixed_eval))
    elif previous_available and args.evaluate_previous_tree:
        print("evaluating fixed_T977 baseline directly", flush=True)
        fixed_eval = evaluate_tree_multisnapshot(previous_tree, snapshots, args.rank_rule, args.rel_tol)
        summary_rows.append(summarize_method("fixed_T977", fixed_eval, args.rank_rule, args.rel_tol, steps))
        per_rows.extend(per_step_rows("fixed_T977", fixed_eval))

    print("starting common global tree beam search", flush=True)
    best_tree, best_eval, beam_history, candidate_rows, debug = search_global_tree(args, snapshots, U, previous_tree)
    summary_rows.append(summarize_method("common_global_tree", best_eval, args.rank_rule, args.rel_tol, steps))
    per_rows.extend(per_step_rows("common_global_tree", best_eval))

    with open(out_dir / "best_tree.json", "w") as f:
        json.dump(best_tree, f, indent=2)
    write_best_node_stats(out_dir / "best_tree_node_stats.json", best_eval)
    write_csv(out_dir / "summary.csv", SUMMARY_FIELDS, summary_rows)
    write_csv(out_dir / "per_step.csv", PER_STEP_FIELDS, per_rows)
    write_csv(out_dir / "beam_history.csv", BEAM_HISTORY_FIELDS, beam_history)
    write_csv(out_dir / "candidate_splits.csv", CANDIDATE_FIELDS, candidate_rows)
    with open(out_dir / "rank_cache_stats.json", "w") as f:
        json.dump(dict(
            rank_cache_implemented=False,
            reason="v1 materializes full factors during each actual tree evaluation; no spectrum-only cache is used yet",
            snapshots=len(snapshots),
            global_leaves=len(U),
            candidate_evaluations=len(candidate_rows),
        ), f, indent=2)
    write_report(out_dir / "report.md", summary_rows, steps, previous_available)
    with open(out_dir / "search_debug.json", "w") as f:
        json.dump(debug, f, indent=2)
    write_search_debug_report(out_dir / "search_debug.md", debug)

    print(f"wrote {out_dir / 'summary.csv'}")
    print(
        f"best_common worst_peak_log2={best_eval.worst_peak_log2:.3f} "
        f"worst_step={best_eval.worst_peak_step} max_error={best_eval.max_recon_error_rel:.3g}",
        flush=True,
    )


if __name__ == "__main__":
    main()
