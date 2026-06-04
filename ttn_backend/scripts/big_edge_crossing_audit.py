"""Audit high-rank TTN edges and the operations that cross them.

This is a diagnostic for the "large bond, fewer touches" strategy.  It does
not change the runtime.  Given an executable temporal-carving layout and an
actual runtime metrics JSON, it reports which TTN edges have high observed bond
dimension and/or rank-weighted path hits, then maps those edges back to the
bytecode operations and MULTI_CNOT windows whose static paths cross them.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path

sys.path.insert(0, ".")

import clifft

from temporal_carving.pipeline import run as run_pipeline
from ttn_backend.backend_spec import export_backend_spec
from ttn_backend.scripts.qec_temporal_carving_report import trace_from_program
from ttn_backend.scripts.qec_temporal_carving_runtime import build_carving_executable_spec


EDGE_FIELDS = [
    "edge_id",
    "a",
    "b",
    "max_chi",
    "max_log2_chi",
    "runtime_hit_count",
    "runtime_rank_weighted_hits",
    "static_cross_ops",
    "static_cross_rank_weight",
    "opcode_counts",
    "multicnot_steps",
    "multicnot_cross_controls",
    "top_idents",
    "left_axes_count",
    "right_axes_count",
    "recommendation",
]

OP_FIELDS = [
    "edge_id",
    "step",
    "op",
    "axes",
    "path_len",
    "path_bags",
]

WINDOW_FIELDS = [
    "edge_id",
    "step",
    "target_ident",
    "cross_controls",
    "same_side_controls",
    "unknown_controls",
    "support_size",
    "crossing_fraction",
    "recommendation",
]


def _load_prog(circuit: str):
    with open(Path("qec_bench/circuits") / f"{circuit}.stim") as f:
        return clifft.compile(f.read())


def _edge_id(a: int, b: int) -> str:
    return f"{min(int(a), int(b))}-{max(int(a), int(b))}"


def _edge_tuple(edge_id: str) -> tuple[int, int]:
    a, b = edge_id.split("-")
    return int(a), int(b)


def _bag_adj(spec):
    adj = {i: set() for i in range(int(spec["union"]["n_bags"]))}
    for a, b, _ in spec["union"]["bag_edges"]:
        adj[int(a)].add(int(b))
        adj[int(b)].add(int(a))
    return adj


def _component_without_edge(adj, start: int, blocked: tuple[int, int]) -> set[int]:
    blocked = {tuple(blocked), tuple(reversed(blocked))}
    seen = {int(start)}
    q = deque([int(start)])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if (u, v) in blocked or v in seen:
                continue
            seen.add(v)
            q.append(v)
    return seen


def _edge_sides(spec, homing, edge_id: str):
    a, b = _edge_tuple(edge_id)
    adj = _bag_adj(spec)
    left_bags = _component_without_edge(adj, a, (a, b))
    right_bags = set(adj) - left_bags
    left_axes = {int(x) for x, h in homing["home"].items() if int(h) in left_bags}
    right_axes = {int(x) for x, h in homing["home"].items() if int(h) in right_bags}
    return left_bags, right_bags, left_axes, right_axes


def _path_crosses(path, edge_id: str) -> bool:
    if not path:
        return False
    return any(_edge_id(a, b) == edge_id for a, b in zip(path, path[1:]))


def _load_metrics(path: str | None):
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    with open(p) as f:
        return json.load(f)


def _rebuild_layout(circuit: str, args):
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


def _candidate_edges(metrics: dict, top: int):
    max_chi = metrics.get("edge_max_bond_dim") or {}
    hits = metrics.get("edge_hit_count") or {}
    weighted = metrics.get("edge_rank_weighted_hits") or {}
    keys = set(max_chi) | set(hits) | set(weighted)
    rows = []
    for k in keys:
        chi = int(max_chi.get(k, 1) or 1)
        hit = int(hits.get(k, 0) or 0)
        w = float(weighted.get(k, 0.0) or 0.0)
        logchi = math.log2(chi) if chi > 0 else 0.0
        # Prioritize actual rank-weighted work, then high chi, then hit count.
        score = (w, logchi * max(hit, 1), logchi, hit)
        rows.append((score, k))
    rows.sort(reverse=True)
    return [k for _, k in rows[:top]]


def _json(v):
    return json.dumps(v, sort_keys=True)


def _recommend(edge_row):
    if edge_row["multicnot_cross_controls"] >= 4:
        return "cluster_or_persistent_multicnot_window"
    if edge_row["runtime_hit_count"] >= 10 and edge_row["runtime_rank_weighted_hits"] >= 50:
        return "parking_or_lifetime_scheduling"
    if edge_row["max_chi"] >= 1024 and edge_row["runtime_hit_count"] <= 3:
        return "resident_streaming_more_relevant_than_crossing_reduction"
    return "layout_local_search_or_monitor"


def analyze(circuit: str, args):
    metrics = _load_metrics(args.metrics_json)
    prog, spec, homing = _rebuild_layout(circuit, args)
    candidate_edges = _candidate_edges(metrics, args.top_edges)
    max_chi = metrics.get("edge_max_bond_dim") or {}
    hits = metrics.get("edge_hit_count") or {}
    weighted = metrics.get("edge_rank_weighted_hits") or {}

    edge_rows = []
    op_rows = []
    window_acc = defaultdict(lambda: dict(target=None, cross=0, same=0, unknown=0, support=set()))
    ident_cross = defaultdict(Counter)

    for edge_id in candidate_edges:
        left_bags, right_bags, left_axes, right_axes = _edge_sides(spec, homing, edge_id)
        opcode_counts = Counter()
        static_cross_ops = 0
        static_cross_rank = 0.0

        for r in homing["op_classes"]:
            if r.get("kind") != "two":
                continue
            path = list(r.get("path_bags") or [])
            if not _path_crosses(path, edge_id):
                continue
            static_cross_ops += 1
            static_cross_rank += len(path) - 1
            op = str(r.get("op"))
            opcode_counts[op] += 1
            axes = tuple(int(x) for x in r.get("axes", ()))
            for x in axes:
                ident_cross[edge_id][int(x)] += 1
            op_rows.append(dict(
                edge_id=edge_id,
                step=int(r["step"]),
                op=op,
                axes=_json(list(axes)),
                path_len=len(path) - 1,
                path_bags=_json(path),
            ))
            if op == "OP_ARRAY_MULTI_CNOT" and len(axes) == 2:
                # The homing entry is one target-control pair.
                target, ctrl = axes
                key = (edge_id, int(r["step"]))
                win = window_acc[key]
                win["target"] = target
                win["support"].update([ctrl, target])
                if (ctrl in left_axes and target in right_axes) or (
                    ctrl in right_axes and target in left_axes
                ):
                    win["cross"] += 1
                elif (ctrl in left_axes and target in left_axes) or (
                    ctrl in right_axes and target in right_axes
                ):
                    win["same"] += 1
                else:
                    win["unknown"] += 1

        chi = int(max_chi.get(edge_id, 1) or 1)
        row = dict(
            edge_id=edge_id,
            a=_edge_tuple(edge_id)[0],
            b=_edge_tuple(edge_id)[1],
            max_chi=chi,
            max_log2_chi=math.log2(chi) if chi > 0 else 0.0,
            runtime_hit_count=int(hits.get(edge_id, 0) or 0),
            runtime_rank_weighted_hits=float(weighted.get(edge_id, 0.0) or 0.0),
            static_cross_ops=static_cross_ops,
            static_cross_rank_weight=static_cross_rank,
            opcode_counts=_json(dict(opcode_counts)),
            multicnot_steps=0,
            multicnot_cross_controls=0,
            top_idents=_json(ident_cross[edge_id].most_common(8)),
            left_axes_count=len(left_axes),
            right_axes_count=len(right_axes),
            recommendation="",
        )
        edge_rows.append(row)

    window_rows = []
    by_edge_multisteps = Counter()
    by_edge_multicross = Counter()
    for (edge_id, step), win in sorted(window_acc.items()):
        total = int(win["cross"] + win["same"] + win["unknown"])
        frac = (float(win["cross"]) / float(total)) if total else 0.0
        rec = "cluster_target_controls_across_edge" if win["cross"] >= 2 else "monitor"
        window_rows.append(dict(
            edge_id=edge_id,
            step=int(step),
            target_ident=win["target"],
            cross_controls=int(win["cross"]),
            same_side_controls=int(win["same"]),
            unknown_controls=int(win["unknown"]),
            support_size=len(win["support"]),
            crossing_fraction=frac,
            recommendation=rec,
        ))
        if total:
            by_edge_multisteps[edge_id] += 1
            by_edge_multicross[edge_id] += int(win["cross"])

    for row in edge_rows:
        edge_id = row["edge_id"]
        row["multicnot_steps"] = int(by_edge_multisteps.get(edge_id, 0))
        row["multicnot_cross_controls"] = int(by_edge_multicross.get(edge_id, 0))
        row["recommendation"] = _recommend(row)

    edge_rows.sort(
        key=lambda r: (
            float(r["runtime_rank_weighted_hits"]),
            float(r["max_log2_chi"]) * max(1, int(r["runtime_hit_count"])),
            int(r["runtime_hit_count"]),
        ),
        reverse=True,
    )
    op_rows.sort(key=lambda r: (r["edge_id"], int(r["step"]), str(r["op"])))
    window_rows.sort(key=lambda r: (r["edge_id"], -int(r["cross_controls"]), int(r["step"])))
    return edge_rows, op_rows, window_rows, metrics


def write_outputs(out_dir: Path, circuit: str, edge_rows, op_rows, window_rows, metrics):
    out_dir.mkdir(parents=True, exist_ok=True)
    edge_csv = out_dir / f"{circuit}_big_edge_crossing_edges.csv"
    op_csv = out_dir / f"{circuit}_big_edge_crossing_ops.csv"
    win_csv = out_dir / f"{circuit}_big_edge_crossing_windows.csv"
    md = out_dir / f"{circuit}_big_edge_crossing_report.md"
    js = out_dir / f"{circuit}_big_edge_crossing_summary.json"

    with open(edge_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=EDGE_FIELDS)
        w.writeheader()
        w.writerows(edge_rows)
    with open(op_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OP_FIELDS)
        w.writeheader()
        w.writerows(op_rows)
    with open(win_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=WINDOW_FIELDS)
        w.writeheader()
        w.writerows(window_rows)

    summary = dict(
        circuit=circuit,
        metrics_actual_total_peak_bytes=metrics.get("actual_total_peak_bytes"),
        metrics_peak_stored_bytes=metrics.get("peak_stored_bytes"),
        metrics_workspace_actual_peak_bytes=metrics.get("workspace_actual_peak_bytes"),
        metrics_max_bond_dim_observed=metrics.get("max_bond_dim_observed"),
        top_edges=edge_rows[:10],
        n_crossing_ops=len(op_rows),
        n_multicnot_windows=len(window_rows),
    )
    with open(js, "w") as f:
        json.dump(summary, f, indent=2)

    with open(md, "w") as f:
        f.write(f"# Big-Edge Crossing Audit: {circuit}\n\n")
        f.write("목표는 큰 bond dimension 자체를 줄이는 것이 아니라, 큰 edge를 지나는 "
                "transport/refactor 호출 수와 rank-weighted work를 줄일 수 있는지 "
                "진단하는 것이다.\n\n")
        f.write("## Runtime Context\n\n")
        f.write(f"- actual_total_peak_bytes: `{metrics.get('actual_total_peak_bytes')}`\n")
        f.write(f"- peak_stored_bytes: `{metrics.get('peak_stored_bytes')}`\n")
        f.write(f"- workspace_actual_peak_bytes: `{metrics.get('workspace_actual_peak_bytes')}`\n")
        f.write(f"- max_bond_dim_observed: `{metrics.get('max_bond_dim_observed')}`\n\n")
        f.write("## Top Edges\n\n")
        f.write("| edge | max chi | hit | rank-hit | static ops | MULTI cross controls | recommendation |\n")
        f.write("|---|---:|---:|---:|---:|---:|---|\n")
        for r in edge_rows[:15]:
            f.write(
                f"| {r['edge_id']} | {r['max_chi']} | {r['runtime_hit_count']} | "
                f"{float(r['runtime_rank_weighted_hits']):.1f} | {r['static_cross_ops']} | "
                f"{r['multicnot_cross_controls']} | {r['recommendation']} |\n"
            )
        f.write("\n## Interpretation\n\n")
        f.write("- `rank-hit = sum log2(chi_e)` over path crossings. 이 값이 큰 edge는 "
                "큰 bond를 반복해서 여는 실제 refactor work offender다.\n")
        f.write("- `cluster_or_persistent_multicnot_window`는 target/control이 같은 큰 edge를 "
                "반복해서 가르는 경우다. layout clustering 또는 persistent region 후보다.\n")
        f.write("- `parking_or_lifetime_scheduling`은 특정 ident가 같은 edge를 반복 왕복하는 "
                "경우다. home 즉시 복귀 대신 temporary parking을 검토할 후보다.\n")
        f.write("- `resident_streaming_more_relevant_than_crossing_reduction`은 큰 chi는 있지만 "
                "hit가 적은 경우다. 이때는 crossing 감소보다 out-of-core/block resident가 더 직접적이다.\n\n")
        f.write("## Output Files\n\n")
        f.write(f"- edges: `{edge_csv}`\n")
        f.write(f"- ops: `{op_csv}`\n")
        f.write(f"- MULTI_CNOT windows: `{win_csv}`\n")
        f.write(f"- summary: `{js}`\n")

    return edge_csv, op_csv, win_csv, md, js


def main():
    p = argparse.ArgumentParser()
    p.add_argument("circuit")
    p.add_argument("--metrics-json", required=True)
    p.add_argument("--seeder", default="recursive_balanced_mincut")
    p.add_argument("--refine", default="nni")
    p.add_argument("--partitioner", default="networkx")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--top-edges", type=int, default=20)
    p.add_argument("--out-dir", default="reports/big_edge_crossing_audit")
    args = p.parse_args()

    rows, ops, wins, metrics = analyze(args.circuit, args)
    paths = write_outputs(Path(args.out_dir), args.circuit, rows, ops, wins, metrics)
    print(f"wrote {paths[0]}")
    print(f"wrote {paths[3]}")


if __name__ == "__main__":
    main()
