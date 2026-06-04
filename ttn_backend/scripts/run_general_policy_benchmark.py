"""Run generalized TTN executor policies across QEC benchmark circuits."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_CIRCUITS = [
    "distillation",
    "cultivation_d3",
    "coherent_d3_r1",
    "coherent_d5_r1",
    "coherent_d5_r5",
    "cultivation_d5",
    "coherent_d7_r1",
]


POLICIES = {
    "carving_base": {},
    "fuse_only": {
        "TTN_FUSE_MULTICNOT": "1",
        "TTN_DESTRUCTIVE_OPEN": "1",
        "TTN_FUSE_MULTICNOT_BATCH": "1",
        "TTN_FUSE_MULTICNOT_CAP_BYTES": str(64 * 1024 * 1024),
        "TTN_FUSE_MULTICNOT_TOTAL_CAP_BYTES": str(64 * 1024 * 1024),
    },
    "general_policy": {
        "TTN_FUSE_MULTICNOT": "1",
        "TTN_PERSISTENT_MULTICNOT": "1",
        "TTN_PERSISTENT_MULTICNOT_MIN_MULTIS": "2",
        "TTN_DESTRUCTIVE_OPEN": "1",
        "TTN_FUSE_MULTICNOT_BATCH": "1",
        "TTN_FUSE_MULTICNOT_CAP_BYTES": str(64 * 1024 * 1024),
        "TTN_FUSE_MULTICNOT_TOTAL_CAP_BYTES": str(64 * 1024 * 1024),
        "TTN_PREFISSION_TRANSPORT_CAP_BYTES": str(64 * 1024 * 1024),
        "TTN_PREFISSION_MIN_GAIN": "1.01",
    },
    # Block-streamed (staged) transport: above the exact cap, the big theta
    # workspace is never materialized; the GEMM + QR is done over row-blocks.
    "staged_transport": {
        "TTN_FUSE_MULTICNOT": "1",
        "TTN_PERSISTENT_MULTICNOT": "1",
        "TTN_PERSISTENT_MULTICNOT_MIN_MULTIS": "2",
        "TTN_DESTRUCTIVE_OPEN": "1",
        "TTN_FUSE_MULTICNOT_BATCH": "1",
        "TTN_FUSE_MULTICNOT_CAP_BYTES": str(64 * 1024 * 1024),
        "TTN_FUSE_MULTICNOT_TOTAL_CAP_BYTES": str(64 * 1024 * 1024),
        "TTN_EXACT_TOTAL_CAP_BYTES": str(64 * 1024 * 1024),
        "TTN_STAGED_TRANSPORT": "1",
        "TTN_STAGED_BLOCK_BYTES": str(8 * 1024 * 1024),
    },
    # Staged transport + output fission: if a Q/R output bag still exceeds the
    # cap, store it as an exact local microtree (separates workspace reduction
    # from resident reduction).
    "staged_transport_fission": {
        "TTN_FUSE_MULTICNOT": "1",
        "TTN_PERSISTENT_MULTICNOT": "1",
        "TTN_PERSISTENT_MULTICNOT_MIN_MULTIS": "2",
        "TTN_DESTRUCTIVE_OPEN": "1",
        "TTN_FUSE_MULTICNOT_BATCH": "1",
        "TTN_FUSE_MULTICNOT_CAP_BYTES": str(64 * 1024 * 1024),
        "TTN_FUSE_MULTICNOT_TOTAL_CAP_BYTES": str(64 * 1024 * 1024),
        "TTN_EXACT_TOTAL_CAP_BYTES": str(64 * 1024 * 1024),
        "TTN_STAGED_TRANSPORT": "1",
        "TTN_STAGED_BLOCK_BYTES": str(8 * 1024 * 1024),
        "TTN_STAGED_OUTPUT_FISSION": "1",
        "TTN_BAG_FISSION_CAP_BYTES": str(64 * 1024 * 1024),
        "TTN_BAG_FISSION_MIN_GAIN": "1.05",
    },
}


OUT_FIELDS = [
    "circuit", "policy", "status", "timeout", "steps_completed", "total_steps",
    "elapsed_s", "flat_peak_k", "dense_peak_bytes",
    "actual_total_peak_bytes", "dense_over_actual",
    "peak_stored_bytes", "workspace_actual_peak_bytes",
    "resident_actual_peak_bytes", "max_bond_dim_observed",
    "n_qr", "n_transports", "num_refactor", "sum_path_length",
    "sum_rank_weighted_path_length", "qr_work_proxy",
    "persistent_multicnot_windows", "persistent_multicnot_controls",
    "multicnot_region_fused", "multicnot_region_fallback",
    "prefission_transport_attempts", "prefission_transport_success",
    "prefission_transport_failed", "num_bag_fissions",
    "staged_transport_count", "staged_max_theta_block_bytes",
    "staged_qr_temp_peak_bytes", "staged_max_q_bytes",
    "staged_max_full_theta_bytes_avoided", "staged_fallback_count",
    "staged_reorth_count",
    "frame_lifted_windows", "num_frame_updates",
    "num_frame_materializations", "num_avoided_tensor_applies",
    "peak_offender_bag", "peak_offender_step", "peak_offender_shape",
    "notes",
]


def _dense_bytes(flat_peak_k):
    if flat_peak_k in ("", None):
        return None
    return int(16 * (2 ** int(float(flat_peak_k))))


def _read_row(path: Path):
    with open(path) as f:
        data = json.load(f)
    return data["rows"][0]


def _read_metrics(path: Path):
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("circuits", nargs="*", default=DEFAULT_CIRCUITS)
    p.add_argument("--policies", default="carving_base,fuse_only,general_policy")
    p.add_argument("--runtime-timeout", type=float, default=120.0)
    p.add_argument("--max-steps", type=int, default=1200)
    p.add_argument("--out-dir", default="reports/general_policy_benchmark")
    p.add_argument("--python", default=sys.executable)
    args = p.parse_args()

    root = Path(args.out_dir)
    root.mkdir(parents=True, exist_ok=True)
    policies = [x.strip() for x in args.policies.split(",") if x.strip()]

    rows = []
    for circuit in args.circuits:
        for policy in policies:
            if policy not in POLICIES:
                raise ValueError(f"unknown policy {policy}")
            out = root / policy / circuit
            out.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            # Clear knobs this benchmark controls to avoid accidental leakage.
            for k in {
                "TTN_FUSE_MULTICNOT",
                "TTN_PERSISTENT_MULTICNOT",
                "TTN_PERSISTENT_MULTICNOT_MIN_MULTIS",
                "TTN_DESTRUCTIVE_OPEN",
                "TTN_FUSE_MULTICNOT_BATCH",
                "TTN_FUSE_MULTICNOT_CAP_BYTES",
                "TTN_FUSE_MULTICNOT_TOTAL_CAP_BYTES",
                "TTN_PREFISSION_TRANSPORT_CAP_BYTES",
                "TTN_PREFISSION_MIN_GAIN",
                "TTN_BAG_FISSION_CAP_BYTES",
                "TTN_BAG_FISSION_MIN_GAIN",
                "TTN_CLIFFORD_FRAME_LIFT",
                "TTN_PERSISTENT_INCLUDE_ARRAY_CLIFFORD",
                "TTN_EXACT_TOTAL_CAP_BYTES",
                "TTN_STAGED_TRANSPORT",
                "TTN_STAGED_TRANSPORT_FORCE",
                "TTN_STAGED_BLOCK_BYTES",
                "TTN_STAGED_MIN_BYTES",
                "TTN_STAGED_OUTPUT_FISSION",
                "TTN_STAGED_FORCE_REORTH",
                "TTN_STAGED_COND_MAX",
            }:
                env.pop(k, None)
            env.update(POLICIES[policy])

            cmd = [
                args.python,
                "ttn_backend/scripts/qec_temporal_carving_runtime.py",
                circuit,
                "--runtime-timeout", str(args.runtime_timeout),
                "--modes", "carving_leaf",
                "--max-steps", str(args.max_steps),
                "--out-dir", str(out),
            ]
            print(f"running {circuit} {policy}", flush=True)
            proc = subprocess.run(
                cmd,
                cwd=Path.cwd(),
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            (out / "runner.log").write_text(proc.stdout)
            summary_path = out / "summary.json"
            if summary_path.exists():
                r = _read_row(summary_path)
            else:
                r = {
                    "circuit": circuit,
                    "status": "runner_error",
                    "timeout": False,
                    "steps_completed": 0,
                    "total_steps": 0,
                    "elapsed_s": 0.0,
                    "flat_peak_k": "",
                    "notes": proc.stdout[-1000:],
                }
            dense = _dense_bytes(r.get("flat_peak_k"))
            metrics_path = out / circuit / "carving_leaf_metrics.json"
            m = _read_metrics(metrics_path)
            actual = r.get("actual_total_peak_bytes")
            dense_over = (float(dense) / float(actual)) if dense and actual else ""
            row = {field: "" for field in OUT_FIELDS}
            row.update({
                "circuit": circuit,
                "policy": policy,
                "status": r.get("status"),
                "timeout": r.get("timeout"),
                "steps_completed": r.get("steps_completed"),
                "total_steps": r.get("total_steps"),
                "elapsed_s": r.get("elapsed_s"),
                "flat_peak_k": r.get("flat_peak_k"),
                "dense_peak_bytes": dense,
                "actual_total_peak_bytes": actual,
                "dense_over_actual": dense_over,
                "peak_stored_bytes": r.get("peak_stored_bytes"),
                "workspace_actual_peak_bytes": r.get("workspace_actual_peak_bytes"),
                "resident_actual_peak_bytes": r.get("resident_actual_peak_bytes"),
                "max_bond_dim_observed": r.get("max_bond_dim_observed"),
                "n_qr": r.get("n_qr"),
                "n_transports": r.get("n_transports"),
                "num_refactor": r.get("num_refactor"),
                "sum_path_length": r.get("sum_path_length"),
                "sum_rank_weighted_path_length": r.get("sum_rank_weighted_path_length"),
                "qr_work_proxy": r.get("qr_work_proxy"),
                "persistent_multicnot_windows": r.get("persistent_multicnot_windows"),
                "persistent_multicnot_controls": r.get("persistent_multicnot_controls"),
                "multicnot_region_fused": r.get("multicnot_region_fused"),
                "multicnot_region_fallback": r.get("multicnot_region_fallback"),
                "prefission_transport_attempts": m.get("prefission_transport_attempts"),
                "prefission_transport_success": m.get("prefission_transport_success"),
                "prefission_transport_failed": m.get("prefission_transport_failed"),
                "num_bag_fissions": m.get("num_bag_fissions"),
                "staged_transport_count": m.get("staged_transport_count"),
                "staged_max_theta_block_bytes": m.get("staged_max_theta_block_bytes"),
                "staged_qr_temp_peak_bytes": m.get("staged_qr_temp_peak_bytes"),
                "staged_max_q_bytes": m.get("staged_max_q_bytes"),
                "staged_max_full_theta_bytes_avoided": m.get("staged_max_full_theta_bytes_avoided"),
                "staged_fallback_count": m.get("staged_fallback_count"),
                "staged_reorth_count": m.get("staged_reorth_count"),
                "frame_lifted_windows": m.get("frame_lifted_windows"),
                "num_frame_updates": m.get("num_frame_updates"),
                "num_frame_materializations": m.get("num_frame_materializations"),
                "num_avoided_tensor_applies": m.get("num_avoided_tensor_applies"),
                "peak_offender_bag": r.get("peak_offender_bag"),
                "peak_offender_step": r.get("peak_offender_step"),
                "peak_offender_shape": r.get("peak_offender_shape"),
                "notes": r.get("notes"),
            })
            rows.append(row)

    csv_path = root / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUT_FIELDS)
        w.writeheader()
        w.writerows(rows)
    json_path = root / "summary.json"
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
