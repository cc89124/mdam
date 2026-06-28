"""Single source of truth for temporal-live carving costs."""

from __future__ import annotations

from dataclasses import dataclass
import functools
import math
from typing import Iterable

import numpy as np

from .tree import TreeNode


@dataclass(frozen=True)
class Trace:
    axes: tuple[int, ...]
    dims: dict[int, int]
    timeline: tuple[int, ...]
    live_sets: dict[int, frozenset[int]]
    events: dict[int, tuple[tuple[int, int], ...]]

    def __post_init__(self):
        object.__setattr__(self, "axes", tuple(map(int, self.axes)))
        object.__setattr__(self, "timeline", tuple(map(int, self.timeline)))

    @property
    def U(self) -> frozenset[int]:
        return frozenset(self.axes)

    def logdim_axis(self, u: int) -> float:
        return math.log2(int(self.dims.get(int(u), 2)))

    def ell(self, xs: Iterable[int]) -> float:
        return float(sum(self.logdim_axis(int(u)) for u in xs))


class CostModel:
    """Cost model defined in the build spec.

    All exact DP, refinement, and final evaluation call this object. No other
    module should reimplement C_S, rhat, nodecost, or tree peak.
    """

    def __init__(self, trace: Trace):
        self.trace = trace
        self.U = trace.U
        self.timeline = trace.timeline
        self._c_cache: dict[frozenset[int], np.ndarray] = {}
        self._r_cache: dict[frozenset[int], np.ndarray] = {}

    def canon(self, S: Iterable[int]) -> frozenset[int]:
        return frozenset(int(x) for x in S)

    def live_cumulative_cut_pressure(self, S: Iterable[int]) -> np.ndarray:
        S = self.canon(S)
        if S in self._c_cache:
            return self._c_cache[S]
        Sc = self.U - S
        acc = 0
        out = []
        for t in self.timeline:
            live = self.trace.live_sets.get(t, frozenset())
            Slive = S & live
            Sclive = Sc & live
            if not Slive or not Sclive:
                acc = 0
                out.append(0.0)
                continue
            cross = 0
            for i, j in self.trace.events.get(t, ()):
                if i in live and j in live and ((i in S and j in Sc) or (j in S and i in Sc)):
                    cross += 1
            acc += cross
            out.append(float(acc))
        arr = np.asarray(out, dtype=float)
        self._c_cache[S] = arr
        return arr

    def rhat(self, S: Iterable[int]) -> np.ndarray:
        S = self.canon(S)
        if S in self._r_cache:
            return self._r_cache[S]
        if not S or S == self.U:
            arr = np.zeros(len(self.timeline), dtype=float)
            self._r_cache[S] = arr
            return arr
        Sc = self.U - S
        C = self.live_cumulative_cut_pressure(S)
        vals = []
        for idx, t in enumerate(self.timeline):
            live = self.trace.live_sets.get(t, frozenset())
            Slive = S & live
            Sclive = Sc & live
            if not Slive or not Sclive:
                vals.append(0.0)
            else:
                vals.append(min(float(C[idx]), self.trace.ell(Slive), self.trace.ell(Sclive)))
        arr = np.asarray(vals, dtype=float)
        self._r_cache[S] = arr
        return arr

    def p_leaf_vector(self, u: int) -> np.ndarray:
        u = int(u)
        bit = self.trace.logdim_axis(u)
        return np.asarray([
            bit if u in self.trace.live_sets.get(t, frozenset()) else 0.0
            for t in self.timeline
        ], dtype=float)

    def leaf_cost(self, u: int) -> float:
        return float(np.max(self.p_leaf_vector(u) + self.rhat({u})))

    def nodecost(self, A: Iterable[int], B: Iterable[int]) -> float:
        A = self.canon(A)
        B = self.canon(B)
        if not A or not B or A & B:
            raise ValueError("nodecost requires a nonempty disjoint bipartition")
        S = A | B
        # Critical: sum pointwise before max.
        return float(np.max(self.rhat(A) + self.rhat(B) + self.rhat(S)))

    def tree_peak(self, tree: TreeNode) -> float:
        if tree.leaves() != self.U:
            raise ValueError(f"tree leaves {sorted(tree.leaves())} != trace axes {sorted(self.U)}")
        best = 0.0

        def rec(node: TreeNode, parent_cut: frozenset[int]):
            nonlocal best
            if node.is_leaf:
                vals = self.p_leaf_vector(node.axis) + self.rhat(parent_cut)
                best = max(best, float(np.max(vals)))
                return
            A = node.left.leaves()
            B = node.right.leaves()
            vals = self.rhat(A) + self.rhat(B) + self.rhat(parent_cut)
            best = max(best, float(np.max(vals)))
            rec(node.left, A)
            rec(node.right, B)

        rec(tree, self.U)
        return best

    def tree_profile(self, tree: TreeNode) -> list[float]:
        vals = np.zeros(len(self.timeline), dtype=float)

        def update(arr):
            nonlocal vals
            vals = np.maximum(vals, arr)

        def rec(node: TreeNode, parent_cut: frozenset[int]):
            if node.is_leaf:
                update(self.p_leaf_vector(node.axis) + self.rhat(parent_cut))
                return
            A = node.left.leaves()
            B = node.right.leaves()
            update(self.rhat(A) + self.rhat(B) + self.rhat(parent_cut))
            rec(node.left, A)
            rec(node.right, B)

        rec(tree, self.U)
        return [float(x) for x in vals]

    def union_graph_objective(self, tree: TreeNode) -> float:
        """Comparison-only union graph carving objective.

        This is not the true objective. It intentionally ignores temporal reset.
        """
        all_edges = set()
        for edges in self.trace.events.values():
            for i, j in edges:
                all_edges.add((min(i, j), max(i, j)))

        def cut_count(S):
            S = set(S)
            Sc = set(self.U) - S
            return sum(1 for i, j in all_edges if (i in S and j in Sc) or (j in S and i in Sc))

        best = 0.0

        def rec(node: TreeNode, parent_cut):
            nonlocal best
            if node.is_leaf:
                best = max(best, self.trace.logdim_axis(node.axis) + cut_count(parent_cut))
                return
            A = node.left.leaves()
            B = node.right.leaves()
            best = max(best, cut_count(A) + cut_count(B) + cut_count(parent_cut))
            rec(node.left, A)
            rec(node.right, B)

        rec(tree, self.U)
        return float(best)
