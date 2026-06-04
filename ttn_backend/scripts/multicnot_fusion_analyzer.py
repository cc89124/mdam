"""Analyze OP_ARRAY_MULTI_CNOT fusion opportunities.

The current TTN backend expands one MULTI_CNOT into many per-control CNOT path
transports.  This analyzer groups all per-control paths from the same bytecode
step and estimates the structural savings of treating the entire MULTI_CNOT as
one fused region operation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, ".")

import clifft

from temporal_carving.pipeline import run as run_pipeline
from ttn_backend.backend_spec import export_backend_spec
from ttn_backend.scripts.qec_temporal_carving_report import trace_from_program
from ttn_backend.scripts.qec_temporal_carving_runtime import build_carving_executable_spec


FIELDS = [
    "step",
    "num_controls",
    "target_ident",
    "control_idents",
    "region_bags",
    "region_size",
    "region_tree_edges",
    "boundary_bags",
    "old_sum_path_length",
    "old_transport_qr",
    "fused_close_svd",
    "fused_open_close_upper",
    "qr_reduction_close_only",
    "qr_reduction_open_close",
    "max_path_len",
    "contains_peak_bag",
    "workspace_proxy_bag_count",
    "workspace_proxy_with_observed_chi_log2",
    "workspace_proxy_with_observed_chi_bytes",
    "memory_cap_pass",
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


def _region_edges(region, adj):
    region = set(region)
    internal = 0
    boundary = set()
    boundary_edges = []
    for u in region:
        for v in adj[u]:
            if v in region:
                if u < v:
                    internal += 1
            else:
                boundary.add(u)
                boundary_edges.append((min(u, v), max(u, v)))
    return internal, sorted(boundary), sorted(set(boundary_edges))


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
    return spec, homing


def _load_edge_chi(path):
    if not path:
        return {}
    with open(path) as f:
        m = json.load(f)
    out = {}
    for k, v in (m.get("edge_max_bond_dim") or {}).items():
        out[str(k)] = int(v)
    return out


def _boundary_workspace_proxy(region, boundary_edges, edge_chi, n_phys=0):
    """Very rough live-region upper proxy from observed max boundary chi.

    This deliberately ignores internal bond dimensions because a fused region
    contracts them away.  It is not an exact runtime workspace; it is a filter
    for whether the fused region is obviously too large.
    """
    log2_numel = float(n_phys)
    for a, b in boundary_edges:
        key = f"{a}-{b}"
        chi = int(edge_chi.get(key, 1))
        if chi > 1:
            log2_numel += math.log2(chi)
    bytes_ = int((2 ** log2_numel) * 16) if log2_numel < 60 else None
    return log2_numel, bytes_


def analyze(spec, homing, args, edge_chi):
    adj = _bag_adj(spec)
    groups = {}
    for r in homing["op_classes"]:
        if r["kind"] != "two" or r["op"] != "OP_ARRAY_MULTI_CNOT":
            continue
        path = list(r.get("path_bags") or [])
        if len(path) < 2:
            continue
        step = int(r["step"])
        groups.setdefault(step, []).append(dict(
            axes=tuple(map(int, r["axes"])),
            path=path,
            path_len=len(path) - 1,
        ))

    rows = []
    for step, ops in sorted(groups.items()):
        region = set()
        controls = []
        target_counts = {}
        for op in ops:
            region.update(op["path"])
            u, v = op["axes"]
            controls.append(u)
            target_counts[v] = target_counts.get(v, 0) + 1
        target = max(target_counts, key=target_counts.get) if target_counts else None
        internal, boundary, boundary_edges = _region_edges(region, adj)
        old_sum = sum(op["path_len"] for op in ops)
        old_qr = 2 * old_sum
        close = internal
        openclose = 2 * internal
        # The active physical axes inside a fully contracted fused region are at
        # most one target plus the controls.  This is a conservative local count.
        n_phys = len(set([target] + controls)) if target is not None else len(set(controls))
        ws_log2, ws_bytes = _boundary_workspace_proxy(region, boundary_edges, edge_chi, n_phys=n_phys)
        cap = args.memory_cap_bytes
        rows.append(dict(
            step=step,
            num_controls=len(ops),
            target_ident=target,
            control_idents=sorted(set(controls)),
            region_bags=sorted(region),
            region_size=len(region),
            region_tree_edges=internal,
            boundary_bags=boundary,
            old_sum_path_length=old_sum,
            old_transport_qr=old_qr,
            fused_close_svd=close,
            fused_open_close_upper=openclose,
            qr_reduction_close_only=(old_qr / close) if close else None,
            qr_reduction_open_close=(old_qr / openclose) if openclose else None,
            max_path_len=max(op["path_len"] for op in ops),
            contains_peak_bag=args.peak_bag in region if args.peak_bag is not None else False,
            workspace_proxy_bag_count=len(region),
            workspace_proxy_with_observed_chi_log2=ws_log2,
            workspace_proxy_with_observed_chi_bytes=ws_bytes,
            memory_cap_pass=(ws_bytes is not None and ws_bytes <= cap) if cap else None,
        ))
    rows.sort(key=lambda r: (r["old_transport_qr"] - r["fused_close_svd"]), reverse=True)
    return rows


def write_outputs(out_dir, circuit, rows, args):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{circuit}_multicnot_fusion.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            row = dict(r)
            for k in ("control_idents", "region_bags", "boundary_bags"):
                row[k] = json.dumps(row[k])
            w.writerow(row)
    selected = rows[:args.top]
    summary = dict(
        circuit=circuit,
        num_multicnot_steps=len(rows),
        total_old_transport_qr=sum(r["old_transport_qr"] for r in rows),
        total_fused_close_svd=sum(r["fused_close_svd"] for r in rows),
        total_fused_open_close_upper=sum(r["fused_open_close_upper"] for r in rows),
        top=args.top,
        top_old_transport_qr=sum(r["old_transport_qr"] for r in selected),
        top_fused_close_svd=sum(r["fused_close_svd"] for r in selected),
        top_fused_open_close_upper=sum(r["fused_open_close_upper"] for r in selected),
        top_memory_cap_pass=sum(1 for r in selected if r["memory_cap_pass"] is True),
    )
    with open(out_dir / f"{circuit}_multicnot_fusion_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / f"{circuit}_multicnot_fusion_report.md", "w") as f:
        f.write(f"# MULTI_CNOT Fusion Analyzer: {circuit}\n\n")
        f.write("This is a scheduling estimate. It does not execute fused MULTI_CNOT yet.\n\n")
        f.write("## Summary\n\n")
        for k, v in summary.items():
            f.write(f"- {k}: `{v}`\n")
        f.write("\n## Top MULTI_CNOT Steps\n\n")
        f.write("| step | controls | region | old QR | close SVD | open+close | reduction upper | ws proxy log2 | ws bytes | cap pass |\n")
        f.write("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in selected:
            red = r["qr_reduction_open_close"]
            f.write(
                f"| {r['step']} | {r['num_controls']} | {r['region_size']} | "
                f"{r['old_transport_qr']} | {r['fused_close_svd']} | "
                f"{r['fused_open_close_upper']} | {red:.3g} | "
                f"{r['workspace_proxy_with_observed_chi_log2']:.3f} | "
                f"{r['workspace_proxy_with_observed_chi_bytes']} | {r['memory_cap_pass']} |\n"
            )
    return csv_path, summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("circuits", nargs="*", default=["coherent_d5_r5"])
    p.add_argument("--seeder", default="recursive_balanced_mincut")
    p.add_argument("--refine", default="nni")
    p.add_argument("--partitioner", default="networkx")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--edge-metrics-json", default="")
    p.add_argument("--memory-cap-bytes", type=int, default=134217728)
    p.add_argument("--peak-bag", type=int, default=72)
    p.add_argument("--top", type=int, default=30)
    p.add_argument("--out-dir", default="reports/multicnot_fusion")
    args = p.parse_args()

    edge_chi = _load_edge_chi(args.edge_metrics_json)
    for circuit in args.circuits:
        print(f"analyzing {circuit}", flush=True)
        spec, homing = _run_carving_homing(circuit, args)
        rows = analyze(spec, homing, args, edge_chi)
        path, summary = write_outputs(args.out_dir, circuit, rows, args)
        print(
            f"  multicnot_steps={summary['num_multicnot_steps']} "
            f"old_qr={summary['total_old_transport_qr']} "
            f"close_svd={summary['total_fused_close_svd']} "
            f"openclose={summary['total_fused_open_close_upper']}",
            flush=True,
        )
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
