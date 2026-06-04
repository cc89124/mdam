"""Paper-ready static/runtime memory metrics for the Clifft TTN backend."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

sys.path.insert(0, ".")

import clifft
import numpy as np

from ttn_backend.backend_spec import (
    assign_homes_and_classify,
    compute_memory_estimates,
    export_backend_spec,
)
from ttn_backend.layout_transform import reduce_hub_degree
from ttn_backend import TTNBackend


DEFAULT_CIRCUITS = [
    "distillation",
    "cultivation_d3",
    "coherent_d3_r1",
    "coherent_d5_r1",
    "coherent_d5_r5",
    "coherent_d7_r1",
    "coherent_d7_r7",
]

CSV_FIELDS = [
    "circuit",
    "layout_variant",
    "shots_requested",
    "shots_completed",
    "M_static_bytes",
    "M_sep_worst_bytes",
    "M_sep_max_bag_bytes",
    "D_max",
    "S_max",
    "n_bags",
    "max_bag_k",
    "runtime_peak_stored_bytes",
    "runtime_peak_workspace_bytes",
    "max_bond_dim_observed",
    "max_separator_size_observed",
    "max_bag_degree_observed",
    "largest_bag_tensor_bytes",
    "largest_pair_workspace_bytes",
    "n_qr",
    "n_transport",
    "steps_completed",
    "total_steps",
    "timeout",
    "elapsed_time_seconds",
    "runtime_status",
    "error",
]


def _load_prog(name):
    path = os.path.join("qec_bench/circuits", name + ".stim")
    with open(path) as f:
        return clifft.compile(f.read())


def _variant_spec(spec, variant, threshold):
    if variant == "baseline":
        return spec
    if variant.startswith("hub"):
        return reduce_hub_degree(spec, threshold)
    raise ValueError(f"unknown layout variant: {variant}")


def _fmt_mb(n):
    if n in ("", None):
        return ""
    mb = float(n) / 1e6
    if abs(mb) >= 1e6:
        return f"{mb:.3e}"
    return f"{mb:.3f}"


def _merge_top(rows, key="bytes", k=5):
    rows = [dict(r) for r in rows if r]
    rows.sort(key=lambda x: int(x.get(key, 0)), reverse=True)
    return rows[:k]


def _runtime_metrics(prog, spec, homing, shots, seed, runtime_timeout):
    backend = TTNBackend(spec, homing)
    master = np.random.default_rng(seed)
    aggregate = dict(
        runtime_peak_stored_bytes=0,
        runtime_peak_workspace_bytes=0,
        max_bond_dim_observed=0,
        max_separator_size_observed=0,
        max_bag_degree_observed=0,
        largest_bag_tensor_bytes=0,
        largest_pair_workspace_bytes=0,
        n_qr=0,
        n_transport=0,
        steps_completed=0,
        total_steps=0,
        timeout=False,
        elapsed_time_seconds=0.0,
        shots_completed=0,
        runtime_status="complete",
        error="",
        top5_bag_sizes=[],
        top5_pair_workspace=[],
    )

    for _ in range(shots):
        shot_seed = int(master.integers(0, 2**63 - 1))
        t0 = time.perf_counter()
        try:
            backend.run_shot(
                prog,
                shot_seed,
                runtime_timeout=runtime_timeout,
                check_interval=1,
            )
        except Exception as exc:
            aggregate["runtime_status"] = "error"
            aggregate["error"] = repr(exc)
            aggregate["elapsed_time_seconds"] += time.perf_counter() - t0
            break

        m = backend.last_metrics or {}
        aggregate["shots_completed"] += 1
        aggregate["runtime_peak_stored_bytes"] = max(
            aggregate["runtime_peak_stored_bytes"],
            int(m.get("peak_stored_bytes", 0)),
        )
        aggregate["runtime_peak_workspace_bytes"] = max(
            aggregate["runtime_peak_workspace_bytes"],
            int(m.get("peak_pair_workspace_bytes", 0)),
        )
        aggregate["max_bond_dim_observed"] = max(
            aggregate["max_bond_dim_observed"],
            int(m.get("max_bond_dim_observed", 0)),
        )
        aggregate["max_separator_size_observed"] = max(
            aggregate["max_separator_size_observed"],
            int(m.get("max_separator_size_observed", 0)),
        )
        aggregate["max_bag_degree_observed"] = max(
            aggregate["max_bag_degree_observed"],
            int(m.get("max_bag_degree_observed", 0)),
        )
        top_bags = m.get("top5_bag_sizes", [])
        top_pairs = m.get("top5_pair_workspace", [])
        aggregate["top5_bag_sizes"] = _merge_top(
            aggregate["top5_bag_sizes"] + top_bags)
        aggregate["top5_pair_workspace"] = _merge_top(
            aggregate["top5_pair_workspace"] + top_pairs)
        aggregate["largest_bag_tensor_bytes"] = max(
            aggregate["largest_bag_tensor_bytes"],
            int(top_bags[0]["bytes"]) if top_bags else 0,
        )
        aggregate["largest_pair_workspace_bytes"] = max(
            aggregate["largest_pair_workspace_bytes"],
            int(top_pairs[0]["bytes"]) if top_pairs else 0,
        )
        aggregate["n_qr"] += int(m.get("n_qr", 0))
        aggregate["n_transport"] += int(m.get("n_transports", 0))
        aggregate["steps_completed"] += int(m.get("steps_completed", 0))
        aggregate["total_steps"] += int(m.get("total_steps", len(prog)))
        aggregate["elapsed_time_seconds"] += float(
            m.get("elapsed_time_seconds", time.perf_counter() - t0))

        if bool(m.get("timeout", False)):
            aggregate["timeout"] = True
            aggregate["runtime_status"] = "timeout"
            break

    if aggregate["shots_completed"] == 0 and aggregate["runtime_status"] == "complete":
        aggregate["runtime_status"] = "not-run"
    return aggregate


def build_row(name, variant, threshold, shots, seed, runtime_timeout):
    prog = _load_prog(name)
    base_spec = export_backend_spec(prog, strict=False)
    spec = _variant_spec(base_spec, variant, threshold)
    homing = assign_homes_and_classify(spec)
    mem = compute_memory_estimates(spec, homing)
    runtime = _runtime_metrics(prog, spec, homing, shots, seed, runtime_timeout)

    row = dict(
        circuit=name,
        layout_variant=variant,
        shots_requested=shots,
        shots_completed=runtime["shots_completed"],
        M_static_bytes=int(mem["M_static"]),
        M_sep_worst_bytes=int(mem["M_separator_worst"]),
        M_sep_max_bag_bytes=int(mem["M_separator_max_bag"]),
        D_max=int(mem["D_max"]),
        S_max=int(mem["S_max"]),
        n_bags=int(spec["union"]["n_bags"]),
        max_bag_k=int(spec["union"]["max_bag"]),
        runtime_peak_stored_bytes=int(runtime["runtime_peak_stored_bytes"]),
        runtime_peak_workspace_bytes=int(runtime["runtime_peak_workspace_bytes"]),
        max_bond_dim_observed=int(runtime["max_bond_dim_observed"]),
        max_separator_size_observed=int(runtime["max_separator_size_observed"]),
        max_bag_degree_observed=int(runtime["max_bag_degree_observed"]),
        largest_bag_tensor_bytes=int(runtime["largest_bag_tensor_bytes"]),
        largest_pair_workspace_bytes=int(runtime["largest_pair_workspace_bytes"]),
        n_qr=int(runtime["n_qr"]),
        n_transport=int(runtime["n_transport"]),
        steps_completed=int(runtime["steps_completed"]),
        total_steps=int(runtime["total_steps"]),
        timeout=bool(runtime["timeout"]),
        elapsed_time_seconds=float(runtime["elapsed_time_seconds"]),
        runtime_status=runtime["runtime_status"],
        error=runtime["error"],
    )
    extra = dict(
        top5_bag_sizes=runtime["top5_bag_sizes"],
        top5_pair_workspace=runtime["top5_pair_workspace"],
        static_top5_separator_bags=sorted(
            mem["per_bag"], key=lambda x: x["bytes"], reverse=True)[:5],
    )
    return row, extra


def _parse_variants(s, threshold):
    variants = [x.strip() for x in s.split(",") if x.strip()]
    out = []
    for v in variants:
        if v == "hub3":
            out.append("hub3")
        elif v == "baseline":
            out.append("baseline")
        elif v.startswith("hub"):
            out.append(v)
        else:
            raise ValueError(f"unknown variant: {v}")
    return out


def _write_outputs(rows, records, csv_path, json_path):
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})
    with open(json_path, "w") as f:
        json.dump(records, f, indent=2)


def _print_table(rows):
    print("circuit            variant | M_static  M_sep_worst  M_runtime | "
          "D S | max_bond steps/total timeout")
    print("-" * 112)
    for r in rows:
        print(
            f"{r['circuit']:18s} {r['layout_variant']:7s} | "
            f"{_fmt_mb(r['M_static_bytes']):>8s}  "
            f"{_fmt_mb(r['M_sep_worst_bytes']):>11s}  "
            f"{_fmt_mb(r['runtime_peak_stored_bytes']):>9s} | "
            f"{r['D_max']:2d} {r['S_max']:2d} | "
            f"{r['max_bond_dim_observed']:8d} "
            f"{r['steps_completed']:5d}/{r['total_steps']:<5d} "
            f"{str(r['timeout']):>7s} "
            f"{r['elapsed_time_seconds']:.2f}s"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("circuits", nargs="*", default=DEFAULT_CIRCUITS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runtime-timeout", type=float, default=60.0)
    parser.add_argument("--hub-degree-threshold", type=int, default=3)
    parser.add_argument("--no-hub-reduce", action="store_true")
    parser.add_argument("--shots", type=int, default=1)
    parser.add_argument("--out-csv", default="metrics_report.csv")
    parser.add_argument("--out-json", default="metrics_report.json")
    parser.add_argument("--variants", default="baseline")
    args = parser.parse_args()

    variants = _parse_variants(args.variants, args.hub_degree_threshold)
    if args.no_hub_reduce:
        variants = [v for v in variants if not v.startswith("hub")]
    if not variants:
        raise ValueError("no layout variants selected")

    rows = []
    records = []
    for variant in variants:
        for name in args.circuits:
            print(f"[run] circuit={name} variant={variant}", flush=True)
            row, extra = build_row(
                name,
                variant,
                args.hub_degree_threshold,
                args.shots,
                args.seed,
                args.runtime_timeout,
            )
            rows.append(row)
            records.append(dict(row=row, **extra))

    _write_outputs(rows, records, args.out_csv, args.out_json)
    _print_table(rows)
    print(f"\nwrote CSV:  {args.out_csv}")
    print(f"wrote JSON: {args.out_json}")


if __name__ == "__main__":
    main()
