"""Submodular seeding surrogate Φ."""

from __future__ import annotations

import networkx as nx


def affinity(trace):
    A = {u: {v: 0.0 for v in trace.axes if v != u} for u in trace.axes}
    for t in trace.timeline:
        live = trace.live_sets.get(t, frozenset())
        for i, j in trace.events.get(t, ()):
            if i in live and j in live:
                A[i][j] = A[i].get(j, 0.0) + 1.0
                A[j][i] = A[j].get(i, 0.0) + 1.0
    return A


def phi(S, trace, A=None):
    if A is None:
        A = affinity(trace)
    S = set(S)
    Sc = set(trace.axes) - S
    return sum(A.get(i, {}).get(j, 0.0) for i in S for j in Sc)


def build_graph(trace, subset=None, A=None):
    if A is None:
        A = affinity(trace)
    subset = set(trace.axes if subset is None else subset)
    G = nx.Graph()
    G.add_nodes_from(sorted(subset))
    for i in sorted(subset):
        for j, w in A.get(i, {}).items():
            if j in subset and i < j and w:
                G.add_edge(i, j, weight=float(w))
    return G
