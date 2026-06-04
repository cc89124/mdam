"""Actual TTN metric experiment for default vs conservative RASL execution.

This script deliberately separates proxy RASL metrics from actual TTN runtime
metrics. The RASL execution mode is conservative: it only substitutes accepted
active_z_route CNOT sequences recorded by rasl_report.py at OP_ARRAY_MULTI_CNOT
steps. All other choices fall back to default bytecode execution.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time

sys.path.insert(0, ".")

import clifft
import numpy as np

from ttn_backend.backend_spec import assign_homes_and_classify, export_backend_spec
from ttn_backend import TTNBackend
from ttn_backend.rasl.candidate import CliffordOp


DEFAULT_CIRCUITS = [
    "distillation",
    "cultivation_d3",
    "coherent_d3_r1",
    "coherent_d5_r1",
    "coherent_d5_r5",
]

COMPARISON_FIELDS = [
    "circuit",
    "mode_default_available",
    "mode_rasl_available",
    "num_rasl_executable_changes",
    "num_rasl_analysis_changes",
    "default_resident_actual_peak_log2_numel",
    "rasl_resident_actual_peak_log2_numel",
    "delta_resident_actual_log2",
    "default_resident_actual_peak_bytes",
    "rasl_resident_actual_peak_bytes",
    "delta_resident_actual_bytes",
    "default_workspace_actual_peak_bytes",
    "rasl_workspace_actual_peak_bytes",
    "default_num_qr",
    "rasl_num_qr",
    "default_num_svd",
    "rasl_num_svd",
    "default_num_refactor",
    "rasl_num_refactor",
    "default_sum_path_length",
    "rasl_sum_path_length",
    "default_sum_rank_weighted_path_length",
    "rasl_sum_rank_weighted_path_length",
    "default_total_elapsed_s",
    "rasl_total_elapsed_s",
    "correctness_passed",
    "notes",
]

STEP_FIELDS = [
    "step_id",
    "op_kind",
    "resident_actual_peak_log2_numel",
    "resident_actual_peak_numel",
    "resident_actual_peak_bytes",
    "actual_peak_offender_bag",
    "actual_peak_offender_shape",
    "actual_peak_offender_p_B",
    "actual_peak_offender_incident_bond_dims",
    "actual_peak_offender_incident_edge_ids",
]


_OP_RE = re.compile(r"([A-Z_]+)\((\d+)(?:,(\d+))?\)")


def _load_prog(name):
    with open(os.path.join("qec_bench/circuits", name + ".stim")) as f:
        return clifft.compile(f.read())


def _parse_ops(text):
    ops = []
    for name, a, b in _OP_RE.findall(text or ""):
        ops.append(CliffordOp(name, int(a), None if b == "" else int(b)))
    return ops


def load_rasl_decisions(path, circuit):
    decisions = {}
    analysis_changes = 0
    if not path or not os.path.exists(path):
        return decisions, analysis_changes
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("circuit") != circuit:
                continue
            if str(row.get("accepted", "")).lower() != "true":
                continue
            analysis_changes += 1
            kind = row.get("builder_kind", "")
            if not kind.startswith("active_z_route"):
                continue
            if str(row.get("active_only", "")).lower() != "true":
                continue
            if str(row.get("has_dormant", "")).lower() == "true":
                continue
            ops = _parse_ops(row.get("chosen_v_sequence", ""))
            if ops and all(op.name == "CNOT" and op.b is not None for op in ops):
                decisions[int(row["step_id"])] = ops
    return decisions, analysis_changes


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    return x


def _write_json(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(_jsonable(obj), f, indent=2)


def _write_steps(path, metrics):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    rows = []
    for _, row in sorted(
        metrics.get("actual_step_peaks", {}).items(),
        key=lambda kv: (-1 if kv[0] == "init" else int(kv[0])),
    ):
        rows.append(dict(
            step_id="" if row.get("step_id") is None else row.get("step_id"),
            op_kind=row.get("op_kind"),
            resident_actual_peak_log2_numel=row.get("resident_actual_peak_log2_numel"),
            resident_actual_peak_numel=row.get("resident_actual_peak_numel"),
            resident_actual_peak_bytes=row.get("resident_actual_peak_bytes"),
            actual_peak_offender_bag=row.get("peak_offender_bag"),
            actual_peak_offender_shape=" ".join(map(str, row.get("peak_offender_shape", []))),
            actual_peak_offender_p_B=row.get("peak_offender_p_B"),
            actual_peak_offender_incident_bond_dims=" ".join(
                map(str, row.get("peak_offender_incident_bond_dims", []))),
            actual_peak_offender_incident_edge_ids=" ".join(
                map(str, row.get("peak_offender_incident_edge_ids", []))),
        ))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=STEP_FIELDS)
        w.writeheader()
        w.writerows(rows)


def _summary(circuit, mode, metrics, status, error="", correctness_passed=None):
    return dict(
        circuit=circuit,
        mode=mode,
        status=status,
        error=error,
        resident_actual_peak_log2_numel=metrics.get("resident_actual_peak_log2_numel"),
        resident_actual_peak_numel=metrics.get("resident_actual_peak_numel"),
        resident_actual_peak_bytes=metrics.get("resident_actual_peak_bytes"),
        peak_step=metrics.get("actual_peak_offender_step"),
        peak_bag=metrics.get("actual_peak_offender_bag"),
        peak_bag_shape=metrics.get("actual_peak_offender_shape"),
        peak_p_B=metrics.get("actual_peak_offender_p_B"),
        peak_incident_bond_dims=metrics.get("actual_peak_offender_incident_bond_dims"),
        peak_incident_edges=metrics.get("actual_peak_offender_incident_edge_ids"),
        workspace_actual_peak_log2_numel=metrics.get("workspace_actual_peak_log2_numel"),
        workspace_actual_peak_bytes=metrics.get("workspace_actual_peak_bytes"),
        max_bond_dim_observed=metrics.get("max_bond_dim_observed"),
        edge_max_bond_dim=metrics.get("edge_max_bond_dim", {}),
        edge_hit_count=metrics.get("edge_hit_count", {}),
        edge_rank_weighted_hits=metrics.get("edge_rank_weighted_hits", {}),
        num_qr=metrics.get("n_qr"),
        num_svd=metrics.get("n_svd"),
        num_refactor=metrics.get("num_refactor"),
        num_path_contract=metrics.get("num_path_contract"),
        num_center_move=metrics.get("num_center_move"),
        sum_path_length=metrics.get("sum_path_length"),
        sum_rank_weighted_path_length=metrics.get("sum_rank_weighted_path_length"),
        sum_refactor_input_numel=metrics.get("sum_refactor_input_numel"),
        max_refactor_input_numel=metrics.get("max_refactor_input_numel"),
        elapsed_s=metrics.get("elapsed_time_seconds"),
        timeout=metrics.get("timeout"),
        steps_completed=metrics.get("steps_completed"),
        total_steps=metrics.get("total_steps"),
        rasl_exec_changes_used=metrics.get("rasl_exec_changes_used", 0),
        rasl_exec_changes_skipped=metrics.get("rasl_exec_changes_skipped", 0),
        correctness_passed=correctness_passed,
    )


def run_mode(circuit, prog, spec, homing, seed, timeout, mode, decisions=None, max_changes=None):
    backend = TTNBackend(spec, homing)
    t0 = time.perf_counter()
    try:
        rec = backend.run_shot(
            prog,
            seed,
            runtime_timeout=timeout,
            check_interval=1,
            rasl_exec_decisions=decisions if mode.startswith("rasl") else None,
            rasl_exec_max_changes=max_changes,
        )
        metrics = dict(backend.last_metrics or {})
        status = "timeout" if metrics.get("timeout") else "complete"
        return rec, metrics, _summary(circuit, mode, metrics, status), ""
    except Exception as exc:
        metrics = dict(backend.last_metrics or {})
        metrics.setdefault("elapsed_time_seconds", time.perf_counter() - t0)
        return None, metrics, _summary(circuit, mode, metrics, "error", repr(exc)), repr(exc)


def _edge_diff_rows(circuit, default_summary, rasl_summary):
    d_edges = default_summary.get("edge_max_bond_dim") or {}
    r_edges = rasl_summary.get("edge_max_bond_dim") or {}
    d_hits = default_summary.get("edge_hit_count") or {}
    r_hits = rasl_summary.get("edge_hit_count") or {}
    d_weight = default_summary.get("edge_rank_weighted_hits") or {}
    r_weight = rasl_summary.get("edge_rank_weighted_hits") or {}
    peak_d = set(default_summary.get("peak_incident_edges") or [])
    peak_r = set(rasl_summary.get("peak_incident_edges") or [])
    rows = []
    for edge in sorted(set(d_edges) | set(r_edges) | set(d_hits) | set(r_hits)):
        d_chi = int(d_edges.get(edge, 1))
        r_chi = int(r_edges.get(edge, 1))
        d_log = math.log2(d_chi) if d_chi > 0 else 0.0
        r_log = math.log2(r_chi) if r_chi > 0 else 0.0
        rows.append(dict(
            circuit=circuit,
            edge_id=edge,
            default_max_chi=d_chi,
            rasl_max_chi=r_chi,
            delta_max_chi=r_chi - d_chi,
            default_max_log2_chi=d_log,
            rasl_max_log2_chi=r_log,
            delta_max_log2_chi=r_log - d_log,
            default_hit_count=int(d_hits.get(edge, 0)),
            rasl_hit_count=int(r_hits.get(edge, 0)),
            delta_hit_count=int(r_hits.get(edge, 0)) - int(d_hits.get(edge, 0)),
            default_rank_weighted_hits=float(d_weight.get(edge, 0.0)),
            rasl_rank_weighted_hits=float(r_weight.get(edge, 0.0)),
            delta_rank_weighted_hits=float(r_weight.get(edge, 0.0)) - float(d_weight.get(edge, 0.0)),
            is_peak_offender_default=edge in peak_d,
            is_peak_offender_rasl=edge in peak_r,
        ))
    return rows


def _write_edge_diff(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fields = [
        "circuit", "edge_id", "default_max_chi", "rasl_max_chi", "delta_max_chi",
        "default_max_log2_chi", "rasl_max_log2_chi", "delta_max_log2_chi",
        "default_hit_count", "rasl_hit_count", "delta_hit_count",
        "default_rank_weighted_hits", "rasl_rank_weighted_hits",
        "delta_rank_weighted_hits", "is_peak_offender_default", "is_peak_offender_rasl",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _comparison_row(circuit, default_summary, rasl_summary, analysis_changes, notes):
    d_ok = default_summary.get("status") in ("complete", "timeout")
    rasl_used = (rasl_summary or {}).get("rasl_exec_changes_used", 0)
    r_ok = (
        rasl_summary is not None and
        rasl_summary.get("status") in ("complete", "timeout") and
        int(rasl_used or 0) > 0
    )
    d_log = default_summary.get("resident_actual_peak_log2_numel")
    r_log = rasl_summary.get("resident_actual_peak_log2_numel") if rasl_summary else None
    d_bytes = default_summary.get("resident_actual_peak_bytes")
    r_bytes = rasl_summary.get("resident_actual_peak_bytes") if rasl_summary else None
    return dict(
        circuit=circuit,
        mode_default_available=d_ok,
        mode_rasl_available=r_ok,
        num_rasl_executable_changes=rasl_used,
        num_rasl_analysis_changes=analysis_changes,
        default_resident_actual_peak_log2_numel=d_log,
        rasl_resident_actual_peak_log2_numel=r_log,
        delta_resident_actual_log2=None if d_log is None or r_log is None else r_log - d_log,
        default_resident_actual_peak_bytes=d_bytes,
        rasl_resident_actual_peak_bytes=r_bytes,
        delta_resident_actual_bytes=None if d_bytes is None or r_bytes is None else r_bytes - d_bytes,
        default_workspace_actual_peak_bytes=default_summary.get("workspace_actual_peak_bytes"),
        rasl_workspace_actual_peak_bytes=(rasl_summary or {}).get("workspace_actual_peak_bytes"),
        default_num_qr=default_summary.get("num_qr"),
        rasl_num_qr=(rasl_summary or {}).get("num_qr"),
        default_num_svd=default_summary.get("num_svd"),
        rasl_num_svd=(rasl_summary or {}).get("num_svd"),
        default_num_refactor=default_summary.get("num_refactor"),
        rasl_num_refactor=(rasl_summary or {}).get("num_refactor"),
        default_sum_path_length=default_summary.get("sum_path_length"),
        rasl_sum_path_length=(rasl_summary or {}).get("sum_path_length"),
        default_sum_rank_weighted_path_length=default_summary.get("sum_rank_weighted_path_length"),
        rasl_sum_rank_weighted_path_length=(rasl_summary or {}).get("sum_rank_weighted_path_length"),
        default_total_elapsed_s=default_summary.get("elapsed_s"),
        rasl_total_elapsed_s=(rasl_summary or {}).get("elapsed_s"),
        correctness_passed=(rasl_summary or {}).get("correctness_passed"),
        notes=notes,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("circuits", nargs="*", default=DEFAULT_CIRCUITS)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--runtime-timeout", type=float, default=60.0)
    p.add_argument("--rasl-steps-csv", default="reports/rasl_steps_full.csv")
    p.add_argument("--enable-rasl-exec-active-only", action="store_true")
    p.add_argument("--rasl-exec-max-changes", type=int, default=None)
    p.add_argument("--out-csv", default="reports/actual_rasl_comparison.csv")
    p.add_argument("--out-json", default="reports/actual_rasl_comparison.json")
    p.add_argument("--out-md", default="reports/actual_rasl_report.md")
    args = p.parse_args()

    os.makedirs("reports", exist_ok=True)
    comparison = []
    json_rows = []

    for circuit in args.circuits:
        print(f"[actual] {circuit}", flush=True)
        prog = _load_prog(circuit)
        spec = export_backend_spec(prog, strict=False)
        homing = assign_homes_and_classify(spec)

        default_rec, default_metrics, default_summary, default_error = run_mode(
            circuit, prog, spec, homing, args.seed, args.runtime_timeout, "default_actual")
        _write_steps(f"reports/actual_default_steps_{circuit}.csv", default_metrics)
        _write_json(f"reports/actual_default_summary_{circuit}.json", default_summary)

        rasl_summary = None
        rasl_metrics = {}
        notes = ""
        decisions, analysis_changes = load_rasl_decisions(args.rasl_steps_csv, circuit)
        if not args.enable_rasl_exec_active_only:
            notes = "rasl exec disabled; actual RASL fields unavailable"
        elif not decisions:
            notes = "no executable active-only RASL decisions; RASL actual equals default not rerun"
        else:
            rasl_rec, rasl_metrics, rasl_summary, rasl_error = run_mode(
                circuit, prog, spec, homing, args.seed, args.runtime_timeout,
                "rasl_exec_active_only", decisions, args.rasl_exec_max_changes)
            correctness = (
                rasl_rec is not None and default_rec is not None and
                dict(rasl_rec) == dict(default_rec)
            )
            rasl_summary["correctness_passed"] = bool(correctness)
            if not correctness:
                notes = "RASL executable record mismatch or error; do not use for correctness claims"
            elif int(rasl_summary.get("rasl_exec_changes_used", 0) or 0) == 0:
                notes = "timeout/step window ended before any executable RASL change; actual RASL effect not measured"
            else:
                notes = "RASL executable record matched default for this seed"
            _write_steps(f"reports/actual_rasl_steps_{circuit}.csv", rasl_metrics)
            _write_json(f"reports/actual_rasl_summary_{circuit}.json", rasl_summary)
            _write_edge_diff(
                f"reports/actual_edge_rank_diff_{circuit}.csv",
                _edge_diff_rows(circuit, default_summary, rasl_summary),
            )

        row = _comparison_row(circuit, default_summary, rasl_summary, analysis_changes, notes)
        comparison.append(row)
        json_rows.append(dict(default=default_summary, rasl=rasl_summary, comparison=row))
        print(
            f"  default peak log2={row['default_resident_actual_peak_log2_numel']} "
            f"rasl peak log2={row['rasl_resident_actual_peak_log2_numel']} "
            f"correct={row['correctness_passed']} notes={notes}",
            flush=True,
        )

    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COMPARISON_FIELDS)
        w.writeheader()
        for row in comparison:
            w.writerow(row)
    _write_json(args.out_json, json_rows)

    with open(args.out_md, "w") as f:
        f.write("# Actual RASL Experiment\n\n")
        f.write("Proxy metrics are not used as actual memory claims in this report.\n\n")
        f.write("| circuit | actual resident default | actual resident RASL | delta log2 | QR default/RASL | path default/RASL | correctness | notes |\n")
        f.write("|---|---:|---:|---:|---:|---:|---|---|\n")
        for row in comparison:
            f.write(
                f"| {row['circuit']} | {row['default_resident_actual_peak_log2_numel']} | "
                f"{row['rasl_resident_actual_peak_log2_numel']} | "
                f"{row['delta_resident_actual_log2']} | "
                f"{row['default_num_qr']}/{row['rasl_num_qr']} | "
                f"{row['default_sum_rank_weighted_path_length']}/{row['rasl_sum_rank_weighted_path_length']} | "
                f"{row['correctness_passed']} | {row['notes']} |\n"
            )
    print(f"wrote {args.out_csv}")
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_md}")


if __name__ == "__main__":
    main()
