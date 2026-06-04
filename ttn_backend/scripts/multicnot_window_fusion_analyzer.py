"""Analyze window-level fusion of OP_ARRAY_MULTI_CNOT regions.

This consumes the per-step MULTI_CNOT fusion CSV and greedily merges nearby
steps whose fused regions overlap, subject to region-size and workspace caps.
It estimates replacing repeated per-step open/close refactors by one persistent
region per window.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = [
    "window_id",
    "first_step",
    "last_step",
    "num_multicnot_steps",
    "total_controls",
    "region_bags",
    "region_size",
    "old_transport_qr",
    "step_fused_open_close_upper",
    "window_close_svd",
    "window_open_close_upper",
    "reduction_vs_old_close",
    "reduction_vs_old_open_close",
    "reduction_vs_step_fused_open_close",
    "max_workspace_proxy_bytes",
    "max_workspace_proxy_log2",
    "all_steps_cap_pass",
    "contains_peak_bag",
]


def _loads_rows(path):
    rows = []
    for r in csv.DictReader(open(path)):
        row = dict(r)
        row["step"] = int(row["step"])
        row["num_controls"] = int(row["num_controls"])
        row["region_bags"] = set(json.loads(row["region_bags"]))
        row["region_size"] = int(row["region_size"])
        row["region_tree_edges"] = int(row["region_tree_edges"])
        row["old_transport_qr"] = int(row["old_transport_qr"])
        row["fused_open_close_upper"] = int(row["fused_open_close_upper"])
        row["fused_close_svd"] = int(row["fused_close_svd"])
        row["workspace_proxy_with_observed_chi_log2"] = float(row["workspace_proxy_with_observed_chi_log2"])
        ws = row["workspace_proxy_with_observed_chi_bytes"]
        row["workspace_proxy_with_observed_chi_bytes"] = None if ws in ("", "None") else int(float(ws))
        row["memory_cap_pass"] = row["memory_cap_pass"] == "True"
        row["contains_peak_bag"] = row["contains_peak_bag"] == "True"
        rows.append(row)
    rows.sort(key=lambda r: r["step"])
    return rows


def _tree_edges_for_region_size(region_size):
    # The executable carving layout is a tree; a connected region with n bags has
    # n-1 internal edges.  This analyzer only merges overlapping connected
    # regions from previous analyzer output, so use n-1 as an optimistic close
    # cost estimate.
    return max(0, int(region_size) - 1)


def _can_merge(window, row, args):
    if row["step"] - window["last_step"] > args.max_step_gap:
        return False
    overlap = bool(window["region_bags"] & row["region_bags"])
    if not overlap:
        return False
    merged_region = window["region_bags"] | row["region_bags"]
    if len(merged_region) > args.max_region_bags:
        return False
    if args.require_cap_pass and not row["memory_cap_pass"]:
        return False
    return True


def _finish(window):
    region_size = len(window["region_bags"])
    close = _tree_edges_for_region_size(region_size)
    openclose = 2 * close
    old = window["old_transport_qr"]
    step_fused = window["step_fused_open_close_upper"]
    return dict(
        first_step=window["first_step"],
        last_step=window["last_step"],
        num_multicnot_steps=len(window["steps"]),
        total_controls=window["total_controls"],
        region_bags=sorted(window["region_bags"]),
        region_size=region_size,
        old_transport_qr=old,
        step_fused_open_close_upper=step_fused,
        window_close_svd=close,
        window_open_close_upper=openclose,
        reduction_vs_old_close=(old / close) if close else None,
        reduction_vs_old_open_close=(old / openclose) if openclose else None,
        reduction_vs_step_fused_open_close=(step_fused / openclose) if openclose else None,
        max_workspace_proxy_bytes=window["max_workspace_proxy_bytes"],
        max_workspace_proxy_log2=window["max_workspace_proxy_log2"],
        all_steps_cap_pass=window["all_steps_cap_pass"],
        contains_peak_bag=window["contains_peak_bag"],
    )


def make_windows(rows, args):
    windows = []
    cur = None
    for row in rows:
        if args.require_cap_pass and not row["memory_cap_pass"]:
            if cur is not None:
                windows.append(_finish(cur))
                cur = None
            continue
        if cur is None:
            cur = dict(
                first_step=row["step"],
                last_step=row["step"],
                steps=[row["step"]],
                total_controls=row["num_controls"],
                region_bags=set(row["region_bags"]),
                old_transport_qr=row["old_transport_qr"],
                step_fused_open_close_upper=row["fused_open_close_upper"],
                max_workspace_proxy_bytes=row["workspace_proxy_with_observed_chi_bytes"] or 0,
                max_workspace_proxy_log2=row["workspace_proxy_with_observed_chi_log2"],
                all_steps_cap_pass=row["memory_cap_pass"],
                contains_peak_bag=row["contains_peak_bag"],
            )
            continue
        if _can_merge(cur, row, args):
            cur["last_step"] = row["step"]
            cur["steps"].append(row["step"])
            cur["total_controls"] += row["num_controls"]
            cur["region_bags"] |= row["region_bags"]
            cur["old_transport_qr"] += row["old_transport_qr"]
            cur["step_fused_open_close_upper"] += row["fused_open_close_upper"]
            cur["max_workspace_proxy_bytes"] = max(
                cur["max_workspace_proxy_bytes"],
                row["workspace_proxy_with_observed_chi_bytes"] or 0,
            )
            cur["max_workspace_proxy_log2"] = max(
                cur["max_workspace_proxy_log2"],
                row["workspace_proxy_with_observed_chi_log2"],
            )
            cur["all_steps_cap_pass"] = cur["all_steps_cap_pass"] and row["memory_cap_pass"]
            cur["contains_peak_bag"] = cur["contains_peak_bag"] or row["contains_peak_bag"]
        else:
            windows.append(_finish(cur))
            cur = dict(
                first_step=row["step"],
                last_step=row["step"],
                steps=[row["step"]],
                total_controls=row["num_controls"],
                region_bags=set(row["region_bags"]),
                old_transport_qr=row["old_transport_qr"],
                step_fused_open_close_upper=row["fused_open_close_upper"],
                max_workspace_proxy_bytes=row["workspace_proxy_with_observed_chi_bytes"] or 0,
                max_workspace_proxy_log2=row["workspace_proxy_with_observed_chi_log2"],
                all_steps_cap_pass=row["memory_cap_pass"],
                contains_peak_bag=row["contains_peak_bag"],
            )
    if cur is not None:
        windows.append(_finish(cur))
    windows.sort(key=lambda w: (w["old_transport_qr"] - w["window_open_close_upper"]), reverse=True)
    for i, w in enumerate(windows):
        w["window_id"] = i
    return windows


def write_outputs(out_dir, circuit, windows, args):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{circuit}_multicnot_windows.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for row in windows:
            out = dict(row)
            out["region_bags"] = json.dumps(out["region_bags"])
            w.writerow(out)
    top = windows[:args.top]
    summary = dict(
        circuit=circuit,
        num_windows=len(windows),
        top=args.top,
        total_old_transport_qr=sum(w["old_transport_qr"] for w in windows),
        total_step_fused_open_close_upper=sum(w["step_fused_open_close_upper"] for w in windows),
        total_window_open_close_upper=sum(w["window_open_close_upper"] for w in windows),
        top_old_transport_qr=sum(w["old_transport_qr"] for w in top),
        top_step_fused_open_close_upper=sum(w["step_fused_open_close_upper"] for w in top),
        top_window_open_close_upper=sum(w["window_open_close_upper"] for w in top),
    )
    with open(out_dir / f"{circuit}_multicnot_windows_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / f"{circuit}_multicnot_windows_report.md", "w") as f:
        f.write(f"# MULTI_CNOT Window Fusion Analyzer: {circuit}\n\n")
        f.write("This is a scheduling estimate. It does not execute persistent windows yet.\n\n")
        f.write("## Summary\n\n")
        for k, v in summary.items():
            f.write(f"- {k}: `{v}`\n")
        f.write("\n## Top Windows\n\n")
        f.write("| id | steps | mcnots | controls | region | old QR | step fused | window open+close | red vs old | red vs step fused | ws bytes |\n")
        f.write("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for w in top:
            f.write(
                f"| {w['window_id']} | {w['first_step']}-{w['last_step']} | "
                f"{w['num_multicnot_steps']} | {w['total_controls']} | {w['region_size']} | "
                f"{w['old_transport_qr']} | {w['step_fused_open_close_upper']} | "
                f"{w['window_open_close_upper']} | {w['reduction_vs_old_open_close']:.3g} | "
                f"{w['reduction_vs_step_fused_open_close']:.3g} | {w['max_workspace_proxy_bytes']} |\n"
            )
    return csv_path, summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-csv", required=True)
    p.add_argument("--circuit", default="coherent_d5_r5")
    p.add_argument("--max-step-gap", type=int, default=64)
    p.add_argument("--max-region-bags", type=int, default=64)
    p.add_argument("--require-cap-pass", action="store_true")
    p.add_argument("--top", type=int, default=30)
    p.add_argument("--out-dir", default="reports/multicnot_window_fusion")
    args = p.parse_args()
    rows = _loads_rows(args.input_csv)
    windows = make_windows(rows, args)
    path, summary = write_outputs(args.out_dir, args.circuit, windows, args)
    print(
        f"windows={summary['num_windows']} old_qr={summary['total_old_transport_qr']} "
        f"step_fused={summary['total_step_fused_open_close_upper']} "
        f"window_openclose={summary['total_window_open_close_upper']}"
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
