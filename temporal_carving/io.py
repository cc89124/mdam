"""CSV trace I/O and tree serialization."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .cost import Trace
from .tree import TreeNode


def load_trace(path) -> Trace:
    path = Path(path)
    dims = {}
    axes = set()
    axes_csv = path / "axes.csv"
    if axes_csv.exists():
        with open(axes_csv, newline="") as f:
            for r in csv.DictReader(f):
                axis = int(r["axis_id"])
                axes.add(axis)
                dims[axis] = int(r.get("dim") or 2)

    live_sets = {}
    live_csv = path / "live_sets.csv"
    with open(live_csv, newline="") as f:
        for r in csv.DictReader(f):
            t = int(r["t"])
            axis = int(r["axis_id"])
            axes.add(axis)
            live_sets.setdefault(t, set()).add(axis)

    events = {}
    events_csv = path / "events.csv"
    if events_csv.exists():
        with open(events_csv, newline="") as f:
            for r in csv.DictReader(f):
                t = int(r["t"])
                i = int(r["i"])
                j = int(r["j"])
                axes.update([i, j])
                if i > j:
                    i, j = j, i
                events.setdefault(t, []).append((i, j))

    for a in axes:
        dims.setdefault(a, 2)
    timeline = sorted(set(live_sets) | set(events))
    return Trace(
        axes=tuple(sorted(axes)),
        dims=dims,
        timeline=tuple(timeline),
        live_sets={t: frozenset(v) for t, v in live_sets.items()},
        events={t: tuple(v) for t, v in events.items()},
    )


def write_trace(trace: Trace, path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    with open(path / "axes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["axis_id", "dim"])
        w.writeheader()
        for a in trace.axes:
            w.writerow({"axis_id": a, "dim": trace.dims.get(a, 2)})
    with open(path / "live_sets.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["t", "axis_id"])
        w.writeheader()
        for t in trace.timeline:
            for a in sorted(trace.live_sets.get(t, ())):
                w.writerow({"t": t, "axis_id": a})
    with open(path / "events.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["t", "i", "j"])
        w.writeheader()
        for t in trace.timeline:
            for i, j in trace.events.get(t, ()):
                if i > j:
                    i, j = j, i
                w.writerow({"t": t, "i": i, "j": j})


def save_tree(tree: TreeNode, path):
    with open(path, "w") as f:
        json.dump(tree.to_dict(), f, indent=2, sort_keys=True)


def load_tree(path) -> TreeNode:
    with open(path) as f:
        return TreeNode.from_dict(json.load(f))
