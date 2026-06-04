"""Time-varying TTN live-graph evolution report.

This script records actual TTN tensor/bond states during execution and then
aggregates them into per-step live graphs. It is intentionally diagnostic: it
does not implement a new layout optimizer.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, ".")

import clifft
import numpy as np

from ttn_backend import TTNBackend
from ttn_backend.backend_spec import assign_homes_and_classify, export_backend_spec
from ttn_backend.layout_transform import reduce_hub_degree


DEFAULT_CIRCUITS = [
    "distillation",
    "cultivation_d3",
    "coherent_d3_r1",
    "coherent_d5_r1",
    "coherent_d5_r5",
    "coherent_d7_r1",
    "coherent_d7_r7",
]

SUMMARY_FIELDS = [
    "circuit",
    "layout_variant",
    "status",
    "timeout",
    "steps_completed",
    "total_steps",
    "elapsed_s",
    "n_trace_events",
    "n_step_snapshots",
    "global_peak_step",
    "global_peak_bag",
    "global_peak_E_B",
    "global_peak_stored_bytes",
    "global_peak_workspace_bytes",
    "b0_degree",
    "b0_incident_edges",
    "b0_union_sum_log2_bonds",
    "b0_max_live_sum_log2_bonds",
    "b0_avg_live_sum_log2_bonds_all_steps",
    "b0_avg_live_sum_log2_bonds_active_steps",
    "b0_current_allocated_sum_log2_bonds_at_peak",
    "b0_inactive_but_allocated_contribution",
    "lazy_allocation_feasibility",
    "critical_skeleton_n_steps",
    "critical_skeleton_live_bags",
    "critical_skeleton_live_edges",
    "critical_skeleton_live_axes",
    "critical_live_objective_E",
    "error",
]

STEP_FIELDS = [
    "circuit",
    "layout_variant",
    "step_id",
    "op_kind",
    "event_count",
    "stored_peak_bytes",
    "workspace_peak_bytes",
    "live_axis_count",
    "live_axes",
    "live_bag_count",
    "live_bags",
    "live_edge_count",
    "live_edges",
    "peak_bag",
    "peak_bag_E",
    "peak_bag_bytes",
    "b0_sum_log2_bonds",
    "b0_incident_bond_dims",
    "b0_live_incident_edges",
]

CRITICAL_FIELDS = [
    "circuit",
    "layout_variant",
    "step_id",
    "reason",
    "op_kind",
    "stored_peak_bytes",
    "workspace_peak_bytes",
    "peak_bag",
    "peak_bag_E",
    "b0_sum_log2_bonds",
    "live_edges",
]

B0_EDGE_FIELDS = [
    "circuit",
    "layout_variant",
    "edge_id",
    "neighbor",
    "max_chi",
    "max_log2_chi",
    "active_step_count",
    "live_intervals",
    "hit_count",
    "rank_weighted_hits",
    "static_separator_bits",
]


def _load_prog(name):
    with open(os.path.join("qec_bench/circuits", name + ".stim")) as f:
        return clifft.compile(f.read())


def _variant_spec(spec, variant, threshold):
    if variant == "baseline":
        return spec
    if variant.startswith("hub"):
        return reduce_hub_degree(spec, threshold)
    raise ValueError(f"unsupported layout variant: {variant}")


def _log2(x):
    x = float(x)
    return float(math.log2(x)) if x > 0 else 0.0


def _join(xs):
    return " ".join(str(x) for x in xs)


def _intervals(active_steps):
    steps = sorted(int(s) for s in active_steps)
    if not steps:
        return ""
    out = []
    start = prev = steps[0]
    for s in steps[1:]:
        if s == prev + 1:
            prev = s
            continue
        out.append(f"{start}-{prev}" if start != prev else str(start))
        start = prev = s
    out.append(f"{start}-{prev}" if start != prev else str(start))
    return ";".join(out)


def _edge_id(a, b):
    return f"{min(int(a), int(b))}-{max(int(a), int(b))}"


def _static_separator_bits(spec):
    out = {}
    for a, b, sep in spec["union"]["bag_edges"]:
        out[_edge_id(a, b)] = len(sep)
    return out


class TraceCollector:
    def __init__(self):
        self.events = []

    def __call__(self, row):
        self.events.append(row)


def _aggregate_steps(events, b0):
    steps = {}
    for ev in events:
        sid = -1 if ev.get("step_id") is None else int(ev["step_id"])
        cur = steps.setdefault(sid, dict(
            step_id=sid,
            op_kind=ev.get("op_kind") or "INIT",
            event_count=0,
            stored_peak_bytes=0,
            workspace_peak_bytes=0,
            live_axes=[],
            live_bags=[],
            live_edges=[],
            edge_dims={},
            peak_bag=None,
            peak_bag_E=-1.0,
            peak_bag_bytes=0,
            b0_sum_log2_bonds=0.0,
            b0_incident_bond_dims=[],
            b0_live_incident_edges=[],
        ))
        cur["event_count"] += 1
        if ev.get("op_kind"):
            cur["op_kind"] = ev["op_kind"]
        cur["stored_peak_bytes"] = max(cur["stored_peak_bytes"], int(ev["stored_bytes"]))
        cur["workspace_peak_bytes"] = max(
            cur["workspace_peak_bytes"], int(ev.get("pair_workspace_bytes", 0)))
        cur["live_axes"] = list(ev.get("live_axes", []))
        cur["live_bags"] = list(ev.get("live_bags", []))
        edge_dims = {r["edge_id"]: int(r["chi"]) for r in ev.get("edges", [])}
        cur["edge_dims"] = edge_dims
        cur["live_edges"] = sorted(eid for eid, dim in edge_dims.items() if dim > 1)
        peak = ev.get("peak_bag") or {}
        if float(peak.get("log2_numel", -1.0)) >= float(cur["peak_bag_E"]):
            cur["peak_bag"] = int(peak.get("bag", -1))
            cur["peak_bag_E"] = float(peak.get("log2_numel", -1.0))
            cur["peak_bag_bytes"] = int(peak.get("bytes", 0))
        b0_row = None
        for brow in ev.get("bags", []):
            if int(brow["bag"]) == int(b0):
                b0_row = brow
                break
        if b0_row is not None:
            dims = [int(x) for x in b0_row["incident_bond_dims"]]
            edges = list(b0_row["incident_edge_ids"])
            load = sum(_log2(x) for x in dims)
            if load >= float(cur["b0_sum_log2_bonds"]):
                cur["b0_sum_log2_bonds"] = load
                cur["b0_incident_bond_dims"] = dims
                cur["b0_live_incident_edges"] = [
                    eid for eid, dim in zip(edges, dims) if int(dim) > 1
                ]
    return steps


def _critical_steps(step_rows, top_k, delta):
    if not step_rows:
        return []
    rows = [r for r in step_rows if int(r["step_id"]) >= 0]
    by_step = {int(r["step_id"]): r for r in rows}
    reasons = defaultdict(list)
    for r in sorted(rows, key=lambda x: int(x["stored_peak_bytes"]), reverse=True)[:top_k]:
        reasons[int(r["step_id"])].append("top_stored")
    for r in sorted(rows, key=lambda x: int(x["workspace_peak_bytes"]), reverse=True)[:top_k]:
        if int(r["workspace_peak_bytes"]) > 0:
            reasons[int(r["step_id"])].append("top_workspace")
    for r in rows:
        if int(r["peak_bag"]) == 0:
            reasons[int(r["step_id"])].append("B0_offender")
    peak_E = max(float(r["peak_bag_E"]) for r in rows)
    for r in rows:
        if float(r["peak_bag_E"]) >= peak_E - float(delta):
            reasons[int(r["step_id"])].append(f"E>=peak-{delta:g}")
    out = []
    for sid in sorted(reasons):
        r = by_step[sid]
        out.append((r, sorted(set(reasons[sid]))))
    return out


def _b0_analysis(step_rows, spec, metrics, b0):
    sep_bits = _static_separator_bits(spec)
    b0_edges = sorted(
        _edge_id(a, b)
        for a, b, _ in spec["union"]["bag_edges"]
        if int(a) == int(b0) or int(b) == int(b0)
    )
    edge_active_steps = {eid: [] for eid in b0_edges}
    edge_max_chi = {eid: 1 for eid in b0_edges}
    loads_all = []
    loads_active = []
    peak_alloc_load = 0.0
    global_peak_stored = max((int(r["stored_peak_bytes"]) for r in step_rows), default=0)

    for r in step_rows:
        edge_dims = r.get("_edge_dims", {})
        load = 0.0
        for eid in b0_edges:
            dim = int(edge_dims.get(eid, 1))
            edge_max_chi[eid] = max(edge_max_chi[eid], dim)
            if dim > 1:
                edge_active_steps[eid].append(int(r["step_id"]))
            load += _log2(dim)
        loads_all.append(load)
        if load > 0:
            loads_active.append(load)
        if int(r["stored_peak_bytes"]) == global_peak_stored:
            peak_alloc_load = max(peak_alloc_load, load)

    union_load = sum(_log2(edge_max_chi[eid]) for eid in b0_edges)
    max_live_load = max(loads_all, default=0.0)
    avg_all = sum(loads_all) / len(loads_all) if loads_all else 0.0
    avg_active = sum(loads_active) / len(loads_active) if loads_active else 0.0
    inactive = max(0.0, union_load - max_live_load)
    if union_load <= 0:
        feasibility = "no_observed_b0_bond_load"
    elif inactive >= 2.0:
        feasibility = "lazy_allocation_can_help_b0_incident_load"
    elif inactive > 0.0:
        feasibility = "lazy_allocation_may_help_slightly"
    else:
        feasibility = "lazy_allocation_cannot_help_b0_union_equals_live"

    hit = metrics.get("edge_hit_count", {}) or {}
    weighted = metrics.get("edge_rank_weighted_hits", {}) or {}
    edge_rows = []
    for eid in b0_edges:
        a, b = (int(x) for x in eid.split("-"))
        nb = b if a == int(b0) else a
        edge_rows.append(dict(
            edge_id=eid,
            neighbor=nb,
            max_chi=int(edge_max_chi[eid]),
            max_log2_chi=_log2(edge_max_chi[eid]),
            active_step_count=len(set(edge_active_steps[eid])),
            live_intervals=_intervals(edge_active_steps[eid]),
            hit_count=int(hit.get(eid, 0)),
            rank_weighted_hits=float(weighted.get(eid, 0.0)),
            static_separator_bits=int(sep_bits.get(eid, 0)),
        ))

    overlap_rows = []
    active_sets = {eid: set(edge_active_steps[eid]) for eid in b0_edges}
    for eid in b0_edges:
        row = {"edge_id": eid}
        for other in b0_edges:
            row[other] = len(active_sets[eid] & active_sets[other])
        overlap_rows.append(row)

    return dict(
        b0_edges=b0_edges,
        b0_degree=len(b0_edges),
        union_load=union_load,
        max_live_load=max_live_load,
        avg_all=avg_all,
        avg_active=avg_active,
        peak_alloc_load=peak_alloc_load,
        inactive_contribution=inactive,
        feasibility=feasibility,
        edge_rows=edge_rows,
        overlap_rows=overlap_rows,
    )


def _step_csv_rows(circuit, variant, steps):
    rows = []
    for sid in sorted(steps):
        r = steps[sid]
        row = dict(
            circuit=circuit,
            layout_variant=variant,
            step_id=sid,
            op_kind=r["op_kind"],
            event_count=r["event_count"],
            stored_peak_bytes=r["stored_peak_bytes"],
            workspace_peak_bytes=r["workspace_peak_bytes"],
            live_axis_count=len(r["live_axes"]),
            live_axes=_join(r["live_axes"]),
            live_bag_count=len(r["live_bags"]),
            live_bags=_join(r["live_bags"]),
            live_edge_count=len(r["live_edges"]),
            live_edges=_join(r["live_edges"]),
            peak_bag=r["peak_bag"],
            peak_bag_E=r["peak_bag_E"],
            peak_bag_bytes=r["peak_bag_bytes"],
            b0_sum_log2_bonds=r["b0_sum_log2_bonds"],
            b0_incident_bond_dims=_join(r["b0_incident_bond_dims"]),
            b0_live_incident_edges=_join(r["b0_live_incident_edges"]),
        )
        row["_edge_dims"] = dict(r["edge_dims"])
        rows.append(row)
    return rows


def run_one(circuit, variant, threshold, seed, runtime_timeout, check_interval,
            top_k, delta, b0):
    prog = _load_prog(circuit)
    base_spec = export_backend_spec(prog, strict=False)
    spec = _variant_spec(base_spec, variant, threshold)
    homing = assign_homes_and_classify(spec)
    collector = TraceCollector()
    backend = TTNBackend(spec, homing, trace_recorder=collector)
    t0 = time.perf_counter()
    status = "complete"
    error = ""
    try:
        backend.run_shot(
            prog,
            seed,
            runtime_timeout=runtime_timeout,
            check_interval=check_interval,
        )
    except Exception as exc:
        status = "error"
        error = repr(exc)
    elapsed = time.perf_counter() - t0
    if backend.last_metrics is not None:
        metrics = backend.last_metrics
    elif hasattr(backend, "state"):
        metrics = backend._finish_metrics(0, len(prog), True, elapsed)
    else:
        metrics = dict(
            timeout=False,
            steps_completed=0,
            total_steps=len(prog),
            elapsed_time_seconds=elapsed,
            peak_stored_bytes=0,
            peak_pair_workspace_bytes=0,
            edge_hit_count={},
            edge_rank_weighted_hits={},
        )
    if metrics.get("timeout"):
        status = "timeout"

    steps = _aggregate_steps(collector.events, b0=b0)
    step_rows = _step_csv_rows(circuit, variant, steps)
    b0_info = _b0_analysis(step_rows, spec, metrics, b0=b0)
    critical = _critical_steps(step_rows, top_k=top_k, delta=delta)

    critical_live_bags = set()
    critical_live_edges = set()
    critical_live_axes = set()
    critical_objective = 0.0
    for r, _reasons in critical:
        critical_live_bags.update(int(x) for x in str(r["live_bags"]).split() if x != "")
        critical_live_edges.update(str(x) for x in str(r["live_edges"]).split() if x != "")
        critical_live_axes.update(int(x) for x in str(r["live_axes"]).split() if x != "")
        critical_objective = max(critical_objective, float(r["peak_bag_E"]))

    peak_row = max(
        (r for r in step_rows if int(r["step_id"]) >= 0),
        key=lambda r: (int(r["stored_peak_bytes"]), float(r["peak_bag_E"])),
        default=None,
    )
    summary = dict(
        circuit=circuit,
        layout_variant=variant,
        status=status,
        timeout=bool(metrics.get("timeout", False)),
        steps_completed=int(metrics.get("steps_completed", 0)),
        total_steps=int(metrics.get("total_steps", len(prog))),
        elapsed_s=float(metrics.get("elapsed_time_seconds", elapsed)),
        n_trace_events=len(collector.events),
        n_step_snapshots=len(step_rows),
        global_peak_step="" if peak_row is None else int(peak_row["step_id"]),
        global_peak_bag="" if peak_row is None else int(peak_row["peak_bag"]),
        global_peak_E_B="" if peak_row is None else float(peak_row["peak_bag_E"]),
        global_peak_stored_bytes=int(metrics.get("peak_stored_bytes", 0)),
        global_peak_workspace_bytes=int(metrics.get("peak_pair_workspace_bytes", 0)),
        b0_degree=int(b0_info["b0_degree"]),
        b0_incident_edges=_join(b0_info["b0_edges"]),
        b0_union_sum_log2_bonds=float(b0_info["union_load"]),
        b0_max_live_sum_log2_bonds=float(b0_info["max_live_load"]),
        b0_avg_live_sum_log2_bonds_all_steps=float(b0_info["avg_all"]),
        b0_avg_live_sum_log2_bonds_active_steps=float(b0_info["avg_active"]),
        b0_current_allocated_sum_log2_bonds_at_peak=float(b0_info["peak_alloc_load"]),
        b0_inactive_but_allocated_contribution=float(b0_info["inactive_contribution"]),
        lazy_allocation_feasibility=b0_info["feasibility"],
        critical_skeleton_n_steps=len(critical),
        critical_skeleton_live_bags=_join(sorted(critical_live_bags)),
        critical_skeleton_live_edges=_join(sorted(critical_live_edges)),
        critical_skeleton_live_axes=_join(sorted(critical_live_axes)),
        critical_live_objective_E=float(critical_objective),
        error=error,
    )

    critical_rows = []
    for r, reasons in critical:
        critical_rows.append(dict(
            circuit=circuit,
            layout_variant=variant,
            step_id=int(r["step_id"]),
            reason=";".join(reasons),
            op_kind=r["op_kind"],
            stored_peak_bytes=int(r["stored_peak_bytes"]),
            workspace_peak_bytes=int(r["workspace_peak_bytes"]),
            peak_bag=int(r["peak_bag"]),
            peak_bag_E=float(r["peak_bag_E"]),
            b0_sum_log2_bonds=float(r["b0_sum_log2_bonds"]),
            live_edges=r["live_edges"],
        ))

    b0_edge_rows = [
        dict(circuit=circuit, layout_variant=variant, **r)
        for r in b0_info["edge_rows"]
    ]
    overlap_rows = [
        dict(circuit=circuit, layout_variant=variant, **r)
        for r in b0_info["overlap_rows"]
    ]
    for row in step_rows:
        row.pop("_edge_dims", None)
    return summary, step_rows, critical_rows, b0_edge_rows, overlap_rows


def _write_csv(path, rows, fields):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _write_md(path, summaries, critical_rows):
    def fmt_num(x, digits=3):
        if x in ("", None):
            return "n/a"
        try:
            return f"{float(x):.{digits}f}"
        except (TypeError, ValueError):
            return "n/a"

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    crit_by_key = defaultdict(list)
    for row in critical_rows:
        crit_by_key[(row["circuit"], row["layout_variant"])].append(row)
    with open(path, "w") as f:
        f.write("# Time-Varying TTN Graph Evolution Report\n\n")
        f.write("이 리포트는 actual TTN 실행 trace에서 step별 live graph, B0 incident bond simultaneity, union-vs-live gap을 집계한다.\n\n")
        for s in summaries:
            f.write(f"## {s['circuit']} / {s['layout_variant']}\n\n")
            f.write(f"- status: `{s['status']}`, timeout: `{s['timeout']}`, steps: `{s['steps_completed']}/{s['total_steps']}`\n")
            f.write(f"- global peak: step `{s['global_peak_step']}`, bag `B{s['global_peak_bag']}`, E_B=`{s['global_peak_E_B']}`\n")
            f.write(f"- peak stored bytes: `{s['global_peak_stored_bytes']}`, peak workspace bytes: `{s['global_peak_workspace_bytes']}`\n")
            f.write(f"- B0 degree: `{s['b0_degree']}`\n")
            f.write(f"- B0 union sum log2 bonds: `{fmt_num(s['b0_union_sum_log2_bonds'])}`\n")
            f.write(f"- B0 max live sum log2 bonds: `{fmt_num(s['b0_max_live_sum_log2_bonds'])}`\n")
            f.write(f"- B0 inactive-but-allocated contribution: `{fmt_num(s['b0_inactive_but_allocated_contribution'])}`\n")
            f.write(f"- lazy allocation feasibility: `{s['lazy_allocation_feasibility']}`\n")
            f.write(f"- critical skeleton: steps=`{s['critical_skeleton_n_steps']}`, objective E=`{fmt_num(s['critical_live_objective_E'])}`\n\n")
            f.write("Top critical steps:\n\n")
            f.write("| step | reason | op | peak bag | E_B | stored bytes | workspace bytes | B0 load |\n")
            f.write("|---:|---|---|---:|---:|---:|---:|---:|\n")
            for r in crit_by_key[(s["circuit"], s["layout_variant"])][:20]:
                f.write(
                    f"| {r['step_id']} | {r['reason']} | {r['op_kind']} | "
                    f"{r['peak_bag']} | {float(r['peak_bag_E']):.3f} | "
                    f"{r['stored_peak_bytes']} | {r['workspace_peak_bytes']} | "
                    f"{float(r['b0_sum_log2_bonds']):.3f} |\n"
                )
            f.write("\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("circuits", nargs="*", default=DEFAULT_CIRCUITS)
    p.add_argument("--variants", default="baseline")
    p.add_argument("--hub-degree-threshold", type=int, default=3)
    p.add_argument("--runtime-timeout", type=float, default=60.0)
    p.add_argument("--check-interval", type=int, default=1)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--b0", type=int, default=0)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--critical-delta", type=float, default=1.0)
    p.add_argument("--out-summary-csv", default="reports/time_graph_summary.csv")
    p.add_argument("--out-steps-csv", default="reports/time_graph_steps.csv")
    p.add_argument("--out-critical-csv", default="reports/time_graph_critical.csv")
    p.add_argument("--out-b0-edges-csv", default="reports/time_graph_b0_edges.csv")
    p.add_argument("--out-overlap-csv", default="reports/time_graph_b0_overlap.csv")
    p.add_argument("--out-json", default="reports/time_graph_report.json")
    p.add_argument("--out-md", default="reports/time_graph_report.md")
    args = p.parse_args()

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    summaries = []
    step_rows = []
    critical_rows = []
    b0_edge_rows = []
    overlap_rows = []

    for circuit in args.circuits:
        for variant in variants:
            print(f"[time-graph] {circuit} / {variant}", flush=True)
            try:
                s, st, cr, er, ov = run_one(
                    circuit,
                    variant,
                    args.hub_degree_threshold,
                    args.seed,
                    args.runtime_timeout,
                    args.check_interval,
                    args.top_k,
                    args.critical_delta,
                    args.b0,
                )
            except Exception as exc:
                s = {k: "" for k in SUMMARY_FIELDS}
                s.update(dict(
                    circuit=circuit,
                    layout_variant=variant,
                    status="error",
                    timeout=False,
                    error=repr(exc),
                ))
                st, cr, er, ov = [], [], [], []
            summaries.append(s)
            step_rows.extend(st)
            critical_rows.extend(cr)
            b0_edge_rows.extend(er)
            overlap_rows.extend(ov)
            print(
                f"  status={s['status']} peak_step={s['global_peak_step']} "
                f"B0 union={s['b0_union_sum_log2_bonds']} "
                f"max_live={s['b0_max_live_sum_log2_bonds']} "
                f"lazy={s['lazy_allocation_feasibility']}",
                flush=True,
            )

    _write_csv(args.out_summary_csv, summaries, SUMMARY_FIELDS)
    _write_csv(args.out_steps_csv, step_rows, STEP_FIELDS)
    _write_csv(args.out_critical_csv, critical_rows, CRITICAL_FIELDS)
    _write_csv(args.out_b0_edges_csv, b0_edge_rows, B0_EDGE_FIELDS)
    overlap_fields = ["circuit", "layout_variant", "edge_id"]
    extra = sorted({k for r in overlap_rows for k in r if k not in overlap_fields})
    _write_csv(args.out_overlap_csv, overlap_rows, overlap_fields + extra)
    with open(args.out_json, "w") as f:
        json.dump(
            dict(
                summaries=summaries,
                critical=critical_rows,
                b0_edges=b0_edge_rows,
                b0_overlap=overlap_rows,
            ),
            f,
            indent=2,
        )
    _write_md(args.out_md, summaries, critical_rows)

    print("\nsummary:")
    for s in summaries:
        print(
            f"{s['circuit']:18s} {s['layout_variant']:8s} "
            f"status={s['status']:8s} peak_step={s['global_peak_step']} "
            f"B0 union={s['b0_union_sum_log2_bonds']} "
            f"max_live={s['b0_max_live_sum_log2_bonds']} "
            f"inactive={s['b0_inactive_but_allocated_contribution']}"
        )


if __name__ == "__main__":
    main()
