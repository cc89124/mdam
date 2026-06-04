"""Actual TTN peak-memory decomposition and local bond-rank diagnosis.

This report answers a narrower question than memory_risk_report.py:

  * Which actual bag tensor caused the resident peak?
  * Is the offender dominated by physical axes p_B or incident bond product?
  * For observed bonds, is allocated chi larger than a local two-bag SVD rank?

The SVD-rank diagnostic is local to the adjacent two-bag tensor at the end of
the run (or timeout partial state). It is not a full global canonical-rank
certificate, but it is the first check for obvious exact-compression slack.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time

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
]

SUMMARY_FIELDS = [
    "circuit",
    "layout_variant",
    "status",
    "timeout",
    "steps_completed",
    "total_steps",
    "elapsed_s",
    "peak_step",
    "peak_bag_id",
    "peak_bag_tensor_shape",
    "peak_bag_numel",
    "peak_bag_log2_numel",
    "peak_bag_bytes",
    "p_B",
    "incident_edges",
    "incident_bond_dims",
    "sum_log2_bonds",
    "E_B",
    "total_stored_peak_bytes",
    "workspace_actual_peak_bytes",
    "dense_bytes",
    "dense_over_total_stored",
    "dense_over_peak_bag",
    "diagnosis",
    "error",
]

EDGE_FIELDS = [
    "circuit",
    "layout_variant",
    "edge_id",
    "allocated_chi_current",
    "allocated_chi_max_observed",
    "allocated_log2_current",
    "allocated_log2_max_observed",
    "local_svd_rank",
    "local_svd_log2_rank",
    "allocated_over_local_rank",
    "matrix_shape",
    "matrix_numel",
    "svd_status",
    "is_peak_offender_edge",
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
    return float(math.log2(x)) if x > 0 else None


def _rank_from_singular_values(s, rtol=1e-12, atol=1e-14):
    if s.size == 0:
        return 0
    scale = float(np.max(s))
    return int(np.count_nonzero(s > max(atol, rtol * scale)))


def _edge_matrix_for_local_rank(state, a, b):
    ba = state.bags[a]
    bb = state.bags[b]
    a_inner = ba.bond_axis_pos(b)
    b_inner = bb.bond_axis_pos(a)

    a_t = np.moveaxis(ba.tensor, a_inner, -1)
    b_t = np.moveaxis(bb.tensor, b_inner, 0)
    theta = np.tensordot(a_t, b_t, axes=([a_t.ndim - 1], [0]))

    left_ndim = a_t.ndim - 1
    left_shape = theta.shape[:left_ndim]
    right_shape = theta.shape[left_ndim:]
    left_dim = int(np.prod(left_shape)) if left_shape else 1
    right_dim = int(np.prod(right_shape)) if right_shape else 1
    return theta.reshape(left_dim, right_dim)


class SnapshotBag:
    def __init__(self, row):
        self.bag_id = int(row["bag_id"])
        self.neighbors = list(map(int, row["neighbors"]))
        self.own_idents = list(map(int, row["own_idents"]))
        self.tensor = row["tensor"]

    def n_own(self):
        return len(self.own_idents)

    def bond_axis_pos(self, neighbor_id):
        return self.n_own() + self.neighbors.index(neighbor_id)


class SnapshotState:
    def __init__(self, snapshot):
        self.bags = [SnapshotBag(row) for row in snapshot["bags"]]


def local_edge_rank_rows(state, metrics, circuit, variant, peak_edges, max_svd_numel):
    rows = []
    max_observed = metrics.get("edge_max_bond_dim", {}) or {}
    for a, ba in enumerate(state.bags):
        for b in ba.neighbors:
            if b < a:
                continue
            edge_id = f"{a}-{b}"
            chi_current = int(ba.tensor.shape[ba.bond_axis_pos(b)])
            row = dict(
                circuit=circuit,
                layout_variant=variant,
                edge_id=edge_id,
                allocated_chi_current=chi_current,
                allocated_chi_max_observed=int(max_observed.get(edge_id, chi_current)),
                allocated_log2_current=_log2(chi_current),
                allocated_log2_max_observed=_log2(max_observed.get(edge_id, chi_current)),
                local_svd_rank="",
                local_svd_log2_rank="",
                allocated_over_local_rank="",
                matrix_shape="",
                matrix_numel="",
                svd_status="not_run",
                is_peak_offender_edge=edge_id in set(peak_edges or []),
            )
            try:
                M = _edge_matrix_for_local_rank(state, a, b)
                row["matrix_shape"] = f"{M.shape[0]}x{M.shape[1]}"
                row["matrix_numel"] = int(M.size)
                if int(M.size) > int(max_svd_numel):
                    row["svd_status"] = f"skipped_matrix_numel>{max_svd_numel}"
                else:
                    s = np.linalg.svd(M, compute_uv=False)
                    rank = max(1, _rank_from_singular_values(s))
                    row["local_svd_rank"] = rank
                    row["local_svd_log2_rank"] = _log2(rank)
                    row["allocated_over_local_rank"] = (
                        float(chi_current) / float(rank) if rank else ""
                    )
                    row["svd_status"] = "ok"
            except Exception as exc:
                row["svd_status"] = f"error:{exc!r}"
            rows.append(row)
    rows.sort(key=lambda r: (
        r["is_peak_offender_edge"],
        int(r["allocated_chi_max_observed"]),
        int(r["allocated_chi_current"]),
    ), reverse=True)
    return rows


def _diagnose(p_b, sum_log2_bonds):
    if p_b >= sum_log2_bonds + 4:
        return "physical_axis_dominated_layout_split_candidate"
    if sum_log2_bonds >= p_b + 4:
        return "bond_product_dominated_hub_or_separator_candidate"
    return "mixed_physical_and_bond_cost"


def run_one(circuit, variant, timeout_s, seed, threshold, max_svd_numel):
    prog = _load_prog(circuit)
    dense_bytes = int(16 * (1 << int(prog.peak_rank)))
    base_spec = export_backend_spec(prog, strict=False)
    spec = _variant_spec(base_spec, variant, threshold)
    homing = assign_homes_and_classify(spec)
    backend = TTNBackend(spec, homing, capture_peak_snapshot=True)
    t0 = time.perf_counter()
    error = ""
    try:
        backend.run_shot(prog, seed, runtime_timeout=timeout_s, check_interval=1)
        metrics = backend.last_metrics or {}
        status = "timeout" if metrics.get("timeout") else "complete"
    except Exception as exc:
        metrics = backend.last_metrics or {}
        status = "error"
        error = repr(exc)

    elapsed = float(metrics.get("elapsed_time_seconds", time.perf_counter() - t0))
    peak_edges = metrics.get("actual_peak_offender_incident_edge_ids") or []
    bond_dims = [int(x) for x in metrics.get("actual_peak_offender_incident_bond_dims", [])]
    sum_log2_bonds = sum(_log2(x) or 0.0 for x in bond_dims)
    p_b = int(metrics.get("actual_peak_offender_p_B") or 0)
    e_b = p_b + sum_log2_bonds
    peak_bytes = int(metrics.get("resident_actual_peak_bytes") or 0)
    stored = int(metrics.get("peak_stored_bytes") or 0)

    summary = dict(
        circuit=circuit,
        layout_variant=variant,
        status=status,
        timeout=bool(metrics.get("timeout", False)),
        steps_completed=int(metrics.get("steps_completed", 0)),
        total_steps=int(metrics.get("total_steps", len(prog))),
        elapsed_s=elapsed,
        peak_step=metrics.get("actual_peak_offender_step"),
        peak_bag_id=metrics.get("actual_peak_offender_bag"),
        peak_bag_tensor_shape=" ".join(map(str, metrics.get("actual_peak_offender_shape", []))),
        peak_bag_numel=int(metrics.get("resident_actual_peak_numel") or 0),
        peak_bag_log2_numel=metrics.get("resident_actual_peak_log2_numel"),
        peak_bag_bytes=peak_bytes,
        p_B=p_b,
        incident_edges=" ".join(map(str, peak_edges)),
        incident_bond_dims=" ".join(map(str, bond_dims)),
        sum_log2_bonds=sum_log2_bonds,
        E_B=e_b,
        total_stored_peak_bytes=stored,
        workspace_actual_peak_bytes=int(metrics.get("workspace_actual_peak_bytes") or 0),
        dense_bytes=dense_bytes,
        dense_over_total_stored=(float(dense_bytes) / stored if stored else ""),
        dense_over_peak_bag=(float(dense_bytes) / peak_bytes if peak_bytes else ""),
        diagnosis=_diagnose(p_b, sum_log2_bonds),
        error=error,
    )
    edge_rows = []
    snapshot = metrics.get("peak_snapshot")
    rank_state = SnapshotState(snapshot) if snapshot else getattr(backend, "state", None)
    if status != "error" and rank_state is not None:
        edge_rows = local_edge_rank_rows(
            rank_state, metrics, circuit, variant, peak_edges, max_svd_numel)
        for row in edge_rows:
            if snapshot:
                row["svd_status"] = (
                    row["svd_status"] if row["svd_status"] != "not_run"
                    else "not_run_peak_snapshot"
                )
    return summary, edge_rows


def _write_csv(path, fields, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def _write_md(path, summaries, edge_rows_by_key, skipped_layouts):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("# TTN Actual Memory Diagnosis\n\n")
        f.write("이 리포트는 proxy가 아니라 실행 중 관측된 TTN tensor shape와 bond dimension을 기준으로 peak memory를 분해한다.\n\n")
        f.write("Bag tensor size formula:\n\n")
        f.write("```text\n")
        f.write("N_B(t) = 2^p_B(t) * prod_{e~B} chi_e(t)\n")
        f.write("log2 N_B(t) = p_B(t) + sum_{e~B} log2 chi_e(t)\n")
        f.write("M_store(t) = 16 * sum_B N_B(t)\n")
        f.write("```\n\n")
        if skipped_layouts:
            f.write("## Layout Candidates Not Yet Measured\n\n")
            for name, reason in skipped_layouts:
                f.write(f"- `{name}`: {reason}\n")
            f.write("\n")
        f.write("## Summary\n\n")
        f.write("| circuit | layout | status | peak bag | p_B | sum log2 bonds | E_B | peak bag MB | stored MB | dense/stored | diagnosis |\n")
        f.write("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|\n")
        for r in summaries:
            f.write(
                f"| {r['circuit']} | {r['layout_variant']} | {r['status']} | "
                f"{r['peak_bag_id']} | {r['p_B']} | {float(r['sum_log2_bonds']):.3f} | "
                f"{float(r['E_B']):.3f} | {int(r['peak_bag_bytes'])/1e6:.3f} | "
                f"{int(r['total_stored_peak_bytes'])/1e6:.3f} | "
                f"{r['dense_over_total_stored']} | {r['diagnosis']} |\n"
            )
        f.write("\n## Per-Circuit Notes\n\n")
        for r in summaries:
            key = (r["circuit"], r["layout_variant"])
            f.write(f"### {r['circuit']} / {r['layout_variant']}\n\n")
            f.write(f"- status: `{r['status']}`, steps: `{r['steps_completed']}/{r['total_steps']}`\n")
            f.write(f"- peak step: `{r['peak_step']}`, peak bag: `B{r['peak_bag_id']}`\n")
            f.write(f"- shape: `{r['peak_bag_tensor_shape']}`\n")
            f.write(f"- p_B: `{r['p_B']}`\n")
            f.write(f"- incident edges: `{r['incident_edges']}`\n")
            f.write(f"- incident bond dims: `{r['incident_bond_dims']}`\n")
            f.write(f"- sum_log2_bonds: `{float(r['sum_log2_bonds']):.3f}`\n")
            f.write(f"- E_B: `{float(r['E_B']):.3f}`\n")
            f.write(f"- diagnosis: `{r['diagnosis']}`\n\n")
            rows = edge_rows_by_key.get(key, [])
            if rows:
                f.write("Top local bond-rank rows:\n\n")
                f.write("| edge | chi current | chi max | local rank | chi/rank | matrix | status | peak edge |\n")
                f.write("|---|---:|---:|---:|---:|---|---|---|\n")
                for e in rows[:10]:
                    f.write(
                        f"| {e['edge_id']} | {e['allocated_chi_current']} | "
                        f"{e['allocated_chi_max_observed']} | {e['local_svd_rank']} | "
                        f"{e['allocated_over_local_rank']} | {e['matrix_shape']} | "
                        f"{e['svd_status']} | {e['is_peak_offender_edge']} |\n"
                    )
                f.write("\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("circuits", nargs="*", default=DEFAULT_CIRCUITS)
    p.add_argument("--variants", default="baseline")
    p.add_argument("--hub-degree-threshold", type=int, default=3)
    p.add_argument("--runtime-timeout", type=float, default=60.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-svd-numel", type=int, default=8_000_000)
    p.add_argument("--out-summary-csv", default="reports/ttn_memory_diagnosis_summary.csv")
    p.add_argument("--out-edges-csv", default="reports/ttn_memory_diagnosis_edges.csv")
    p.add_argument("--out-json", default="reports/ttn_memory_diagnosis.json")
    p.add_argument("--out-md", default="reports/ttn_memory_diagnosis.md")
    args = p.parse_args()

    variants = [x.strip() for x in args.variants.split(",") if x.strip()]
    runnable = []
    skipped = []
    for v in variants:
        if v == "baseline" or v.startswith("hub"):
            runnable.append(v)
        elif v in {"balanced", "pair-demand", "random"}:
            skipped.append((v, "layout generator is not implemented yet; diagnosis framework is ready"))
        else:
            raise ValueError(f"unknown layout variant: {v}")

    summaries = []
    all_edges = []
    edges_by_key = {}
    for circuit in args.circuits:
        for variant in runnable:
            print(f"[diagnosis] {circuit} / {variant}", flush=True)
            summary, edges = run_one(
                circuit, variant, args.runtime_timeout, args.seed,
                args.hub_degree_threshold, args.max_svd_numel)
            summaries.append(summary)
            all_edges.extend(edges)
            edges_by_key[(circuit, variant)] = edges
            print(
                f"  status={summary['status']} peak=B{summary['peak_bag_id']} "
                f"p_B={summary['p_B']} sum_log2_bonds={summary['sum_log2_bonds']:.3f} "
                f"E={summary['E_B']:.3f}",
                flush=True,
            )

    _write_csv(args.out_summary_csv, SUMMARY_FIELDS, summaries)
    _write_csv(args.out_edges_csv, EDGE_FIELDS, all_edges)
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(dict(summaries=summaries, edges=all_edges, skipped_layouts=skipped), f, indent=2)
    _write_md(args.out_md, summaries, edges_by_key, skipped)
    print(f"wrote {args.out_summary_csv}")
    print(f"wrote {args.out_edges_csv}")
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_md}")


if __name__ == "__main__":
    main()
