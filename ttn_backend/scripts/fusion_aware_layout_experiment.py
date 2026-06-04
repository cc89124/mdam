"""Build a fusion-aware carving tree from MULTI_CNOT window demands.

This experiment answers whether finding fusion windows before choosing the tree
can make fused regions smaller than the current temporal-carving layout.
It does not execute the TTN backend.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import sys
from collections import defaultdict, deque
from pathlib import Path

sys.path.insert(0, ".")

import clifft

from temporal_carving.cost import Trace
from temporal_carving.io import save_tree
from temporal_carving.pipeline import run as run_pipeline
from ttn_backend.backend_spec import export_backend_spec
from ttn_backend.scripts.qec_temporal_carving_report import trace_from_program
from ttn_backend.scripts.qec_temporal_carving_runtime import _carving_bag_tree, build_carving_executable_spec


SUMMARY_FIELDS = [
    "method",
    "num_windows",
    "avg_region_size",
    "max_region_size",
    "avg_old_path_len_sum",
    "max_old_path_len_sum",
    "avg_window_openclose",
    "max_window_openclose",
    "total_old_transport_qr",
    "total_window_openclose",
    "reduction_vs_old",
    "tree_path",
]

WINDOW_FIELDS = [
    "method",
    "window_id",
    "first_step",
    "last_step",
    "num_multicnot_steps",
    "support_idents",
    "support_size",
    "region_bags",
    "region_size",
    "old_path_len_sum",
    "old_transport_qr",
    "window_openclose",
]


def _load_prog(circuit):
    with open(Path("qec_bench/circuits") / f"{circuit}.stim") as f:
        return clifft.compile(f.read())


def _bag_adj_from_edges(n_bags, edges):
    adj = {i: set() for i in range(n_bags)}
    for i, j, _ in edges:
        adj[int(i)].add(int(j))
        adj[int(j)].add(int(i))
    return adj


def _tree_path(adj, src, dst):
    if src == dst:
        return [src]
    parent = {src: None}
    q = deque([src])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in parent:
                parent[v] = u
                q.append(v)
    out = []
    cur = dst
    while cur is not None:
        out.append(cur)
        cur = parent[cur]
    return list(reversed(out))


def _minimal_subtree(adj, homes, support):
    support = sorted(set(support))
    if len(support) <= 1:
        return {homes[support[0]]} if support else set()
    region = set()
    root = support[0]
    for x in support[1:]:
        region.update(_tree_path(adj, homes[root], homes[x]))
    return region


def _current_tree_and_homing(base_spec, prog, args):
    trace = trace_from_program(prog, strict=False)
    result = run_pipeline(
        trace,
        seeder=args.seeder,
        refine_moves=tuple(x for x in args.refine.split(",") if x and x != "none"),
        seed=args.seed,
        partitioner=args.partitioner,
        exact=False,
    )
    spec, homing = build_carving_executable_spec(base_spec, result["tree"])
    return result["tree"], spec, homing


def _multicnot_step_ops(homing):
    by_step = defaultdict(list)
    for r in homing["op_classes"]:
        if r["kind"] == "two" and r["op"] == "OP_ARRAY_MULTI_CNOT":
            by_step[int(r["step"])].append(dict(
                axes=tuple(map(int, r["axes"])),
                path=list(r["path_bags"]),
                path_len=int(r["path_len"]),
            ))
    return dict(by_step)


def _make_windows_from_step_ops(step_ops, args):
    windows = []
    cur = None

    def finish():
        nonlocal cur
        if cur is not None:
            windows.append(cur)
            cur = None

    for step in sorted(step_ops):
        ops = step_ops[step]
        support = set()
        for op in ops:
            support.update(op["axes"])
        if not support:
            continue
        if cur is None:
            cur = dict(
                first_step=step,
                last_step=step,
                steps=[step],
                support=set(support),
                num_multicnot_steps=1,
                old_path_len_sum=sum(op["path_len"] for op in ops),
            )
            continue
        gap = step - cur["last_step"]
        merged = cur["support"] | support
        if (
            gap <= args.max_step_gap
            and bool(cur["support"] & support)
            and len(merged) <= args.max_support_size
        ):
            cur["last_step"] = step
            cur["steps"].append(step)
            cur["support"] = merged
            cur["num_multicnot_steps"] += 1
            cur["old_path_len_sum"] += sum(op["path_len"] for op in ops)
        else:
            finish()
            cur = dict(
                first_step=step,
                last_step=step,
                steps=[step],
                support=set(support),
                num_multicnot_steps=1,
                old_path_len_sum=sum(op["path_len"] for op in ops),
            )
    finish()
    for i, w in enumerate(windows):
        w["window_id"] = i
    return windows


def _fusion_trace(base_spec, windows, weight_mode="path"):
    axes = tuple(sorted(map(int, base_spec["lifecycle"])))
    dims = {a: 2 for a in axes}
    live_sets = {}
    events = {}
    for idx, w in enumerate(windows):
        support = sorted(w["support"])
        live_sets[idx] = frozenset(support)
        pairs = list(itertools.combinations(support, 2))
        if not pairs:
            events[idx] = tuple()
            continue
        if weight_mode == "path":
            repeat = max(1, int(w["old_path_len_sum"] / max(1, len(pairs))))
        else:
            repeat = 1
        ev = []
        for p in pairs:
            ev.extend([p] * repeat)
        events[idx] = tuple(ev)
    return Trace(axes=axes, dims=dims, timeline=tuple(range(len(windows))),
                 live_sets=live_sets, events=events)


def _evaluate_windows(method, tree, windows):
    bags, edges, homes = _carving_bag_tree(tree)
    adj = _bag_adj_from_edges(len(bags), edges)
    rows = []
    for w in windows:
        support = sorted(w["support"])
        region = _minimal_subtree(adj, homes, support)
        region_size = len(region)
        old_path = int(w["old_path_len_sum"])
        rows.append(dict(
            method=method,
            window_id=w["window_id"],
            first_step=w["first_step"],
            last_step=w["last_step"],
            num_multicnot_steps=w["num_multicnot_steps"],
            support_idents=support,
            support_size=len(support),
            region_bags=sorted(region),
            region_size=region_size,
            old_path_len_sum=old_path,
            old_transport_qr=2 * old_path,
            window_openclose=2 * max(0, region_size - 1),
        ))
    return rows


def _summary(method, rows, tree_path):
    n = len(rows)
    old = sum(r["old_transport_qr"] for r in rows)
    win = sum(r["window_openclose"] for r in rows)
    return dict(
        method=method,
        num_windows=n,
        avg_region_size=(sum(r["region_size"] for r in rows) / n) if n else 0,
        max_region_size=max((r["region_size"] for r in rows), default=0),
        avg_old_path_len_sum=(sum(r["old_path_len_sum"] for r in rows) / n) if n else 0,
        max_old_path_len_sum=max((r["old_path_len_sum"] for r in rows), default=0),
        avg_window_openclose=(sum(r["window_openclose"] for r in rows) / n) if n else 0,
        max_window_openclose=max((r["window_openclose"] for r in rows), default=0),
        total_old_transport_qr=old,
        total_window_openclose=win,
        reduction_vs_old=(old / win) if win else None,
        tree_path=str(tree_path),
    )


def write_outputs(out_dir, circuit, current_tree, fusion_tree, current_rows, fusion_rows):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    current_tree_path = out_dir / f"{circuit}_current_temporal_tree.json"
    fusion_tree_path = out_dir / f"{circuit}_fusion_aware_tree.json"
    save_tree(current_tree, current_tree_path)
    save_tree(fusion_tree, fusion_tree_path)
    summaries = [
        _summary("current_temporal", current_rows, current_tree_path),
        _summary("fusion_aware", fusion_rows, fusion_tree_path),
    ]
    with open(out_dir / f"{circuit}_fusion_aware_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        w.writerows(summaries)
    with open(out_dir / f"{circuit}_fusion_aware_windows.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=WINDOW_FIELDS)
        w.writeheader()
        for row in current_rows + fusion_rows:
            out = dict(row)
            for k in ("support_idents", "region_bags"):
                out[k] = json.dumps(out[k])
            w.writerow(out)
    with open(out_dir / f"{circuit}_fusion_aware_report.md", "w") as f:
        f.write(f"# Fusion-Aware Layout Experiment: {circuit}\n\n")
        f.write("This compares the current temporal-carving tree against a tree built from MULTI_CNOT fusion-window hyperedge demand. It does not execute the backend.\n\n")
        f.write("| method | windows | avg region | max region | total old QR | window open+close | reduction |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for s in summaries:
            f.write(
                f"| {s['method']} | {s['num_windows']} | {s['avg_region_size']:.3f} | "
                f"{s['max_region_size']} | {s['total_old_transport_qr']} | "
                f"{s['total_window_openclose']} | {s['reduction_vs_old']:.3g} |\n"
            )
    return summaries


def main():
    p = argparse.ArgumentParser()
    p.add_argument("circuits", nargs="*", default=["coherent_d5_r5"])
    p.add_argument("--seeder", default="recursive_balanced_mincut")
    p.add_argument("--refine", default="nni")
    p.add_argument("--fusion-refine", default="nni")
    p.add_argument("--partitioner", default="networkx")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-step-gap", type=int, default=64)
    p.add_argument("--max-support-size", type=int, default=16)
    p.add_argument("--out-dir", default="reports/fusion_aware_layout")
    args = p.parse_args()

    for circuit in args.circuits:
        print(f"running {circuit}", flush=True)
        prog = _load_prog(circuit)
        base_spec = export_backend_spec(prog, strict=False)
        current_tree, _spec, homing = _current_tree_and_homing(base_spec, prog, args)
        step_ops = _multicnot_step_ops(homing)
        windows = _make_windows_from_step_ops(step_ops, args)
        ftrace = _fusion_trace(base_spec, windows)
        fres = run_pipeline(
            ftrace,
            seeder=args.seeder,
            refine_moves=tuple(x for x in args.fusion_refine.split(",") if x and x != "none"),
            seed=args.seed,
            partitioner=args.partitioner,
            exact=False,
        )
        current_rows = _evaluate_windows("current_temporal", current_tree, windows)
        fusion_rows = _evaluate_windows("fusion_aware", fres["tree"], windows)
        summaries = write_outputs(args.out_dir, circuit, current_tree, fres["tree"], current_rows, fusion_rows)
        for s in summaries:
            print(
                f"  {s['method']}: windows={s['num_windows']} avg_region={s['avg_region_size']:.2f} "
                f"max_region={s['max_region_size']} openclose={s['total_window_openclose']} "
                f"reduction={s['reduction_vs_old']:.2f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
