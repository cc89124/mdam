"""Exact subset-DP oracle for small carving instances."""

from __future__ import annotations

from functools import lru_cache

from .tree import TreeNode


def exact_dp(cost_model, max_n=20):
    axes = tuple(sorted(cost_model.trace.axes))
    n = len(axes)
    if n > max_n:
        raise ValueError(f"exact subset DP refuses n={n} > max_n={max_n}")
    idx = {i: axes[i] for i in range(n)}

    def members(mask):
        return frozenset(idx[i] for i in range(n) if mask & (1 << i))

    @lru_cache(None)
    def opt(mask):
        if mask & (mask - 1) == 0:
            i = (mask.bit_length() - 1)
            return cost_model.leaf_cost(idx[i]), TreeNode.leaf(idx[i])
        best = (float("inf"), None)
        sub = (mask - 1) & mask
        while sub:
            other = mask ^ sub
            if other and sub < other:  # complement duplicate removal
                A = members(sub)
                B = members(other)
                ca, ta = opt(sub)
                cb, tb = opt(other)
                val = max(ca, cb, cost_model.nodecost(A, B))
                if val < best[0]:
                    best = (val, TreeNode.join(ta, tb))
            sub = (sub - 1) & mask
        return best

    return opt((1 << n) - 1)


def fixed_tree_dp_decomposition_peak(cost_model, tree: TreeNode) -> float:
    if tree.is_leaf:
        return cost_model.leaf_cost(tree.axis)
    return max(
        fixed_tree_dp_decomposition_peak(cost_model, tree.left),
        fixed_tree_dp_decomposition_peak(cost_model, tree.right),
        cost_model.nodecost(tree.left.leaves(), tree.right.leaves()),
    )
