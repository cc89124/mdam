"""Exhaustive small-leg bag fission tree search.

This is an offline/profile-time tool. It captures a peak bag tensor and
evaluates every full binary tree over its original tensor legs. It is intended
for small offender tensors such as B72 with four open legs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np

from ttn_backend.scripts.bag_fission_offline import capture_bag_tensor, _prod


SUMMARY_FIELDS = [
    "circuit", "bag", "step", "mode", "tol", "old_shape", "old_bytes",
    "old_log2_numel", "best_peak_bytes", "best_total_bytes",
    "best_peak_log2_numel", "best_total_log2_numel",
    "peak_ratio", "total_ratio", "num_leaves", "tree_depth",
    "max_rank", "tree", "status", "notes",
]


def _log2(x):
    return float(math.log2(float(x))) if x else float("-inf")


def _rank_from_s(s, mode, tol):
    if s.size == 0:
        return 1
    if mode == "exact":
        threshold = max(1e-14, 1e-12 * float(s[0]))
    else:
        threshold = max(1e-14, float(tol) * float(s[0]))
    return max(1, int(np.count_nonzero(s > threshold)))


def _canonical_split_indices(n):
    rest = list(range(1, n))
    for mask in range(1 << len(rest)):
        left = [0]
        right = []
        for bit, ax in enumerate(rest):
            (left if (mask >> bit) & 1 else right).append(ax)
        if right:
            yield tuple(left), tuple(right)


@lru_cache(None)
def _tree_shapes(keys):
    keys = tuple(keys)
    if len(keys) == 1:
        return (keys[0],)
    out = []
    n = len(keys)
    for left_idx, right_idx in _canonical_split_indices(n):
        left_keys = tuple(keys[i] for i in left_idx)
        right_keys = tuple(keys[i] for i in right_idx)
        for lt in _tree_shapes(left_keys):
            for rt in _tree_shapes(right_keys):
                out.append((lt, rt))
    return tuple(out)


def _leaf_set(tree):
    if isinstance(tree, int):
        return frozenset([tree])
    return _leaf_set(tree[0]) | _leaf_set(tree[1])


def _depth(tree):
    if isinstance(tree, int):
        return 0
    return 1 + max(_depth(tree[0]), _depth(tree[1]))


def _factor_by_tree(tensor, axis_keys, tree, mode, tol):
    """Return (peak_numel, total_numel, max_rank, stats_tree).

    The function recursively factors the tensor according to `tree`. For
    internal split A|B, it applies SVD, splits sqrt(S) symmetrically, and then
    recursively factors the child tensors, keeping the new internal bond as an
    extra open leg on each child.
    """
    if isinstance(tree, int):
        return int(tensor.size), int(tensor.size), 1, {
            "leaf": int(tree),
            "shape": list(map(int, tensor.shape)),
            "numel": int(tensor.size),
        }

    left_tree, right_tree = tree
    left_keys = _leaf_set(left_tree)
    right_keys = _leaf_set(right_tree)
    left_axes = [i for i, k in enumerate(axis_keys) if k in left_keys]
    right_axes = [i for i, k in enumerate(axis_keys) if k in right_keys]
    passthrough_axes = [i for i, k in enumerate(axis_keys) if k not in left_keys and k not in right_keys]
    # Existing parent/internal bonds are not leaves of the remaining original
    # tree. They must remain attached to one side of the next split. For this
    # small exhaustive oracle, attach them to the left side. The mirrored tree
    # orientation covers the complementary case for the original first split.
    left_axes = passthrough_axes + left_axes

    order = left_axes + right_axes
    T = np.transpose(tensor, order)
    left_shape = T.shape[:len(left_axes)]
    right_shape = T.shape[len(left_axes):]
    d_left = _prod(left_shape)
    d_right = _prod(right_shape)
    M = T.reshape(d_left, d_right)
    U, s, Vh = np.linalg.svd(M, full_matrices=False)
    rank = _rank_from_s(s, mode, tol)
    sqrt_s = np.sqrt(s[:rank])
    left_tensor = (U[:, :rank] * sqrt_s[None, :]).reshape(left_shape + (rank,))
    right_tensor = (sqrt_s[:, None] * Vh[:rank, :]).reshape((rank,) + right_shape)

    left_axis_keys = [axis_keys[i] for i in left_axes] + [("bond", id(tree), rank)]
    right_axis_keys = [("bond", id(tree), rank)] + [axis_keys[i] for i in right_axes]
    left_peak, left_total, left_max_rank, left_stats = _factor_by_tree(
        left_tensor, tuple(left_axis_keys), left_tree, mode, tol)
    right_peak, right_total, right_max_rank, right_stats = _factor_by_tree(
        right_tensor, tuple(right_axis_keys), right_tree, mode, tol)
    peak = max(int(left_tensor.size), int(right_tensor.size), left_peak, right_peak)
    total = left_total + right_total
    return peak, total, max(rank, left_max_rank, right_max_rank), {
        "split": [repr(left_tree), repr(right_tree)],
        "rank": int(rank),
        "matrix_shape": [int(d_left), int(d_right)],
        "left_shape": list(map(int, left_tensor.shape)),
        "right_shape": list(map(int, right_tensor.shape)),
        "left": left_stats,
        "right": right_stats,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--circuit", default="coherent_d5_r5")
    p.add_argument("--bag", default="72")
    p.add_argument("--max-steps", type=int, default=1200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--runtime-timeout", type=float, default=300.0)
    p.add_argument("--seeder", default="recursive_balanced_mincut")
    p.add_argument("--refine", default="none")
    p.add_argument("--approx-tols", nargs="*", default=["1e-4", "1e-3", "1e-2"])
    p.add_argument("--out-dir", default="reports/bag_fission_fulltree")
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tensor, axes, metrics = capture_bag_tensor(
        args.circuit,
        int(str(args.bag).lstrip("Bb")),
        args.max_steps,
        args.seed,
        args.runtime_timeout,
        args.seeder,
        args.refine,
    )
    old_numel = int(tensor.size)
    old_bytes = int(tensor.nbytes)
    axis_keys = tuple(range(tensor.ndim))
    trees = _tree_shapes(axis_keys)

    rows = []
    best_trees = {}
    modes = [("exact", "0")] + [("approx", str(t)) for t in args.approx_tols]
    for mode, tol_s in modes:
        tol = float(tol_s)
        best = None
        for tree in trees:
            peak, total, max_rank, stats = _factor_by_tree(tensor, axis_keys, tree, mode, tol)
            key = (peak, total, max_rank, _depth(tree), repr(tree))
            if best is None or key < best[0]:
                best = (key, tree, peak, total, max_rank, stats)
        _, tree, peak, total, max_rank, stats = best
        rows.append({
            "circuit": args.circuit,
            "bag": f"B{int(str(args.bag).lstrip('Bb'))}",
            "step": metrics.get("actual_total_peak_step"),
            "mode": mode,
            "tol": tol_s,
            "old_shape": json.dumps(list(map(int, tensor.shape))),
            "old_bytes": old_bytes,
            "old_log2_numel": _log2(old_numel),
            "best_peak_bytes": int(peak * tensor.itemsize),
            "best_total_bytes": int(total * tensor.itemsize),
            "best_peak_log2_numel": _log2(peak),
            "best_total_log2_numel": _log2(total),
            "peak_ratio": old_numel / peak if peak else "",
            "total_ratio": old_numel / total if total else "",
            "num_leaves": tensor.ndim,
            "tree_depth": _depth(tree),
            "max_rank": int(max_rank),
            "tree": repr(tree),
            "status": "ok",
            "notes": f"evaluated_full_binary_trees={len(trees)}",
        })
        best_trees[f"{mode}_{tol_s}"] = {
            "tree": repr(tree),
            "stats": stats,
        }

    with open(out / "bag_fission_fulltree_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        w.writerows(rows)
    with open(out / "bag_fission_fulltree_best_trees.json", "w") as f:
        json.dump(best_trees, f, indent=2)
    print(f"wrote {out / 'bag_fission_fulltree_summary.csv'}")


if __name__ == "__main__":
    main()
