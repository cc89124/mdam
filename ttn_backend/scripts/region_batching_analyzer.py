"""Analyze batched-region opportunities for TTN path refactors.

This is a scheduling analyzer, not an executor.  It estimates how many
per-operation transport QR calls could be replaced by opening one connected
region, applying several two-axis operations, and closing/compressing the region
once.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, ".")

import clifft

from temporal_carving.pipeline import run as run_pipeline
from ttn_backend.backend_spec import export_backend_spec
from ttn_backend.scripts.qec_temporal_carving_report import trace_from_program
from ttn_backend.scripts.qec_temporal_carving_runtime import build_carving_executable_spec


GROUP_FIELDS = [
    "group_id",
    "first_step",
    "last_step",
    "num_steps",
    "num_two_axis_ops",
    "op_kinds",
    "region_bags",
    "region_size",
    "region_tree_edges",
    "boundary_bags",
    "old_sum_path_length",
    "old_transport_qr",
    "batched_close_svd",
    "batched_open_close_refactor_upper",
    "qr_reduction_close_only",
    "qr_reduction_open_close",
    "max_path_len",
    "contains_peak_bag",
]


def _load_prog(circuit):
    with open(Path("qec_bench/circuits") / f"{circuit}.stim") as f:
        return clifft.compile(f.read())


def _bag_adj(spec):
    adj = {i: set() for i in range(int(spec["union"]["n_bags"]))}
    for i, j, _ in spec["union"]["bag_edges"]:
        adj[int(i)].add(int(j))
        adj[int(j)].add(int(i))
    return adj


def _connected_union(region, path):
    return set(region) | set(path)


def _region_edges(region, adj):
    region = set(region)
    n = 0
    boundary = set()
    for u in region:
        for v in adj[u]:
            if v in region:
                if u < v:
                    n += 1
            else:
                boundary.add(u)
    return n, sorted(boundary)


def _run_carving_homing(circuit, args):
    prog = _load_prog(circuit)
    base = export_backend_spec(prog, strict=False)
    trace = trace_from_program(prog, strict=False)
    result = run_pipeline(
        trace,
        seeder=args.seeder,
        refine_moves=tuple(x for x in args.refine.split(",") if x and x != "none"),
        seed=args.seed,
        partitioner=args.partitioner,
        exact=False,
    )
    spec, homing = build_carving_executable_spec(base, result["tree"])
    return prog, spec, homing


def _two_axis_path_ops(homing):
    ops = []
    for r in homing["op_classes"]:
        if r["kind"] != "two":
            continue
        path = list(r.get("path_bags") or [])
        if len(path) < 2:
            continue
        ops.append(dict(
            step=int(r["step"]),
            op=r["op"],
            axes=tuple(map(int, r["axes"])),
            path=path,
            path_len=len(path) - 1,
        ))
    ops.sort(key=lambda x: (x["step"], x["axes"]))
    return ops


def make_groups(ops, adj, args):
    groups = []
    current = None

    def flush():
        nonlocal current
        if current is None:
            return
        region = set(current["region"])
        region_edges, boundary = _region_edges(region, adj)
        old_sum = sum(op["path_len"] for op in current["ops"])
        old_transport_qr = 2 * old_sum
        close_svd = region_edges
        open_close = 2 * region_edges
        groups.append(dict(
            first_step=min(op["step"] for op in current["ops"]),
            last_step=max(op["step"] for op in current["ops"]),
            num_steps=len({op["step"] for op in current["ops"]}),
            num_two_axis_ops=len(current["ops"]),
            op_kinds=sorted({op["op"] for op in current["ops"]}),
            region_bags=sorted(region),
            region_size=len(region),
            region_tree_edges=region_edges,
            boundary_bags=boundary,
            old_sum_path_length=old_sum,
            old_transport_qr=old_transport_qr,
            batched_close_svd=close_svd,
            batched_open_close_refactor_upper=open_close,
            qr_reduction_close_only=(old_transport_qr / close_svd) if close_svd else None,
            qr_reduction_open_close=(old_transport_qr / open_close) if open_close else None,
            max_path_len=max(op["path_len"] for op in current["ops"]),
            contains_peak_bag=args.peak_bag in region if args.peak_bag is not None else False,
        ))
        current = None

    for op in ops:
        path = set(op["path"])
        if args.only_peak_bag is not None and args.only_peak_bag not in path:
            continue
        if op["path_len"] < args.min_path_len:
            continue
        if current is None:
            current = {"ops": [op], "region": set(path), "last_step": op["step"]}
            continue
        step_gap = op["step"] - current["last_step"]
        overlap = bool(path & current["region"])
        merged = _connected_union(current["region"], path)
        if (
            overlap
            and step_gap <= args.max_step_gap
            and len(merged) <= args.max_region_bags
            and len(current["ops"]) < args.max_ops_per_group
        ):
            current["ops"].append(op)
            current["region"] = merged
            current["last_step"] = op["step"]
        else:
            flush()
            current = {"ops": [op], "region": set(path), "last_step": op["step"]}
    flush()
    for i, g in enumerate(groups):
        g["group_id"] = i
    groups.sort(key=lambda g: (g["old_transport_qr"] - g["batched_close_svd"]), reverse=True)
    for i, g in enumerate(groups):
        g["rank_by_saving"] = i
    return groups


def _write(out_dir, circuit, groups, ops, args):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    group_path = out_dir / f"{circuit}_region_groups.csv"
    fields = GROUP_FIELDS + ["rank_by_saving"]
    with open(group_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for g in groups:
            row = dict(g)
            for key in ("op_kinds", "region_bags", "boundary_bags"):
                row[key] = json.dumps(row[key])
            w.writerow(row)

    total_old = sum(2 * op["path_len"] for op in ops)
    selected = groups[:args.top_groups]
    selected_old = sum(g["old_transport_qr"] for g in selected)
    selected_close = sum(g["batched_close_svd"] for g in selected)
    selected_open_close = sum(g["batched_open_close_refactor_upper"] for g in selected)
    summary = dict(
        circuit=circuit,
        total_two_axis_path_ops=len(ops),
        total_old_transport_qr=total_old,
        num_groups=len(groups),
        top_groups=args.top_groups,
        selected_old_transport_qr=selected_old,
        selected_batched_close_svd=selected_close,
        selected_batched_open_close_upper=selected_open_close,
        selected_close_only_reduction=(selected_old / selected_close) if selected_close else None,
        selected_open_close_reduction=(selected_old / selected_open_close) if selected_open_close else None,
        args=vars(args),
    )
    with open(out_dir / f"{circuit}_region_batching_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / f"{circuit}_region_batching_report.md", "w") as f:
        f.write(f"# Region Batching Analyzer: {circuit}\n\n")
        f.write("This is a scheduling estimate. It does not execute the batched region yet.\n\n")
        f.write("## Summary\n\n")
        for k, v in summary.items():
            if k != "args":
                f.write(f"- {k}: `{v}`\n")
        f.write("\n## Top Groups By QR Saving\n\n")
        f.write("| rank | steps | ops | region | old transport QR | close SVD | open+close upper | reduction close | reduction upper |\n")
        f.write("|---:|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for g in selected:
            f.write(
                f"| {g['rank_by_saving']} | {g['first_step']}-{g['last_step']} | "
                f"{g['num_two_axis_ops']} | {g['region_size']} | "
                f"{g['old_transport_qr']} | {g['batched_close_svd']} | "
                f"{g['batched_open_close_refactor_upper']} | "
                f"{g['qr_reduction_close_only']:.3g} | {g['qr_reduction_open_close']:.3g} |\n"
            )
    return group_path, summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("circuits", nargs="*", default=["coherent_d5_r5"])
    p.add_argument("--seeder", default="recursive_balanced_mincut")
    p.add_argument("--refine", default="nni")
    p.add_argument("--partitioner", default="networkx")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-path-len", type=int, default=2)
    p.add_argument("--max-step-gap", type=int, default=32)
    p.add_argument("--max-region-bags", type=int, default=16)
    p.add_argument("--max-ops-per-group", type=int, default=64)
    p.add_argument("--only-peak-bag", type=int, default=None)
    p.add_argument("--peak-bag", type=int, default=72)
    p.add_argument("--top-groups", type=int, default=20)
    p.add_argument("--out-dir", default="reports/region_batching")
    args = p.parse_args()

    for circuit in args.circuits:
        print(f"analyzing {circuit}", flush=True)
        _prog, spec, homing = _run_carving_homing(circuit, args)
        ops = _two_axis_path_ops(homing)
        adj = _bag_adj(spec)
        groups = make_groups(ops, adj, args)
        path, summary = _write(args.out_dir, circuit, groups, ops, args)
        print(f"  path_ops={len(ops)} old_transport_qr={summary['total_old_transport_qr']} groups={len(groups)}")
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
