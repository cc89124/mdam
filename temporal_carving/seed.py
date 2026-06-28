"""Established seeding algorithms for carving trees."""

from __future__ import annotations

import random

import networkx as nx
from networkx.algorithms.community import kernighan_lin_bisection

from .surrogate import affinity, build_graph
from .tree import TreeNode, balanced_tree, caterpillar_tree, random_tree


def _balanced_axis_split(xs):
    xs = sorted(xs)
    mid = len(xs) // 2
    return set(xs[:mid]), set(xs[mid:])


def recursive_balanced_mincut(trace, seed=0, partitioner="networkx") -> TreeNode:
    A = affinity(trace)
    rng = random.Random(seed)

    def rec(xs):
        xs = sorted(xs)
        if len(xs) == 1:
            return TreeNode.leaf(xs[0])
        G = build_graph(trace, xs, A)
        try:
            if partitioner != "networkx":
                raise RuntimeError(f"partitioner {partitioner!r} unavailable; using networkx")
            left, right = kernighan_lin_bisection(G, weight="weight", seed=seed + len(xs))
            if not left or not right:
                left, right = _balanced_axis_split(xs)
        except Exception:
            left, right = _balanced_axis_split(xs)
        return TreeNode.join(rec(left), rec(right))

    return rec(trace.axes)


def louvain(trace, seed=0) -> TreeNode:
    G = build_graph(trace)
    try:
        comms = nx.algorithms.community.louvain_communities(G, weight="weight", seed=seed)
    except Exception:
        return balanced_tree(trace.axes)
    blocks = [balanced_tree(sorted(c)) for c in sorted(comms, key=lambda c: min(c))]
    if not blocks:
        return balanced_tree(trace.axes)
    while len(blocks) > 1:
        a = blocks.pop(0)
        b = blocks.pop(0)
        blocks.append(TreeNode.join(a, b))
    return blocks[0]


def linear_chain(trace, seed=0) -> TreeNode:
    return caterpillar_tree(sorted(trace.axes))


def star(trace, seed=0) -> TreeNode:
    return balanced_tree(sorted(trace.axes))


def random_seed(trace, seed=0) -> TreeNode:
    return random_tree(sorted(trace.axes), seed)


def build_seed(trace, name="recursive_balanced_mincut", seed=0, partitioner="networkx") -> TreeNode:
    if name == "recursive_balanced_mincut":
        return recursive_balanced_mincut(trace, seed=seed, partitioner=partitioner)
    if name == "louvain":
        return louvain(trace, seed=seed)
    if name in ("linear", "linear_chain", "mps"):
        return linear_chain(trace, seed=seed)
    if name == "star":
        return star(trace, seed=seed)
    if name == "random":
        return random_seed(trace, seed=seed)
    raise ValueError(f"unknown seeder: {name}")
