"""Synthetic traces for temporal-live carving tests."""

from __future__ import annotations

import random

from .cost import Trace


def _trace(n, timeline, live_sets, events):
    return Trace(
        axes=tuple(range(n)),
        dims={i: 2 for i in range(n)},
        timeline=tuple(timeline),
        live_sets={int(t): frozenset(v) for t, v in live_sets.items()},
        events={int(t): tuple((min(i, j), max(i, j)) for i, j in es) for t, es in events.items()},
    )


def random_brickwork(n=8, depth=12, p_live=0.85, seed=0):
    rng = random.Random(seed)
    live_sets = {}
    events = {}
    live = set(range(n))
    for t in range(depth):
        # Random measurement/reset-like live churn.
        live = {i for i in range(n) if rng.random() < p_live}
        if not live:
            live.add(rng.randrange(n))
        live_sets[t] = set(live)
        parity = t % 2
        es = []
        for i in range(parity, n - 1, 2):
            if i in live and i + 1 in live and rng.random() < 0.9:
                es.append((i, i + 1))
        events[t] = es
    return _trace(n, range(depth), live_sets, events)


def planted_temporal_masking(n=12, seed=0):
    """Two blocks with union crossings but no co-live X|Y window."""
    if n < 4:
        raise ValueError("n must be >= 4")
    mid = n // 2
    X = list(range(mid))
    Y = list(range(mid, n))
    timeline = list(range(8))
    live_sets = {}
    events = {}
    for t in timeline:
        if t < 4:
            live_sets[t] = set(X)
            block = X
            other = Y
        else:
            live_sets[t] = set(Y)
            block = Y
            other = X
        es = []
        for a in block:
            for b in block:
                if a < b:
                    es.append((a, b))
        # Cross events exist in the union graph but are never co-live.
        for k in range(min(len(X), len(Y))):
            es.append((X[k], Y[k]))
        events[t] = es
    return _trace(n, timeline, live_sets, events)


def two_block_qec(n=12, rounds=5, seed=0):
    rng = random.Random(seed)
    data = list(range(n // 2))
    anc = list(range(n // 2, n))
    live_sets = {}
    events = {}
    t = 0
    for _r in range(rounds):
        live = set(data) | set(anc)
        for a in anc:
            live_sets[t] = set(live)
            es = []
            for d in rng.sample(data, k=min(3, len(data))):
                es.append((a, d))
            events[t] = es
            t += 1
        # Ancilla measured/reset out.
        live_sets[t] = set(data)
        events[t] = []
        t += 1
    return _trace(n, range(t), live_sets, events)
