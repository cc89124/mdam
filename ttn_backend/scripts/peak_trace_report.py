"""Write a compact report from TTN actual-total-peak events.

Input is a `*_metrics.json` file produced by the TTN runtime. The runtime stores
`actual_total_peak_events` whenever the concurrent total peak increases.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = [
    "event_index",
    "step",
    "opcode",
    "kind",
    "selected_executor",
    "actual_total_peak_bytes",
    "actual_total_peak_mib",
    "actual_total_peak_log2_numel",
    "stored_bytes",
    "open_region_bytes",
    "temporary_bytes",
    "pair_workspace_bytes",
    "pair_src",
    "pair_dst",
    "offender_bag",
    "offender_shape",
    "offender_p_B",
    "offender_incident_bond_dims",
    "path",
    "path_length",
    "transport_ident",
    "transport_src",
    "transport_dst",
    "region",
]


def _get(d, key, default=None):
    cur = d
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _json(v):
    if isinstance(v, (list, dict)):
        return json.dumps(v, sort_keys=True)
    return v


def event_row(i: int, e: dict) -> dict:
    dbg = e.get("debug") or {}
    ctx = e.get("executor_context") or {}
    total = int(e.get("actual_total_peak_bytes") or 0)
    return {
        "event_index": i,
        "step": e.get("step"),
        "opcode": e.get("opcode"),
        "kind": e.get("kind"),
        "selected_executor": e.get("selected_executor") or ctx.get("selected_executor"),
        "actual_total_peak_bytes": total,
        "actual_total_peak_mib": total / (1024 * 1024),
        "actual_total_peak_log2_numel": e.get("actual_total_peak_log2_numel"),
        "stored_bytes": e.get("stored_bytes"),
        "open_region_bytes": dbg.get("open_region_bytes"),
        "temporary_bytes": dbg.get("temporary_bytes"),
        "pair_workspace_bytes": dbg.get("pair_workspace_bytes"),
        "pair_src": dbg.get("pair_src"),
        "pair_dst": dbg.get("pair_dst"),
        "offender_bag": e.get("offender_bag"),
        "offender_shape": _json(e.get("offender_shape")),
        "offender_p_B": e.get("offender_p_B"),
        "offender_incident_bond_dims": _json(e.get("offender_incident_bond_dims")),
        "path": _json(ctx.get("path")),
        "path_length": ctx.get("path_length"),
        "transport_ident": ctx.get("transport_ident"),
        "transport_src": ctx.get("transport_src"),
        "transport_dst": ctx.get("transport_dst"),
        "region": _json(ctx.get("region") or dbg.get("region")),
    }


def write_md(path: Path, metrics_path: Path, metrics: dict, rows: list[dict]):
    with open(path, "w") as f:
        f.write("# TTN Peak Trace Report\n\n")
        f.write(f"- metrics: `{metrics_path}`\n")
        f.write(f"- total peak bytes: `{metrics.get('actual_total_peak_bytes')}`\n")
        f.write(f"- total peak step: `{metrics.get('actual_total_peak_step')}`\n")
        f.write(f"- total peak kind: `{metrics.get('actual_total_peak_kind')}`\n")
        f.write(f"- offender bag: `{metrics.get('actual_peak_offender_bag')}`\n")
        f.write(f"- offender shape: `{metrics.get('actual_peak_offender_shape')}`\n")
        f.write(f"- incident bonds: `{metrics.get('actual_peak_offender_incident_bond_dims')}`\n\n")

        f.write("## Peak-Increase Events\n\n")
        f.write("| idx | step | opcode | kind | executor | total MiB | offender | shape | path/region |\n")
        f.write("|---:|---:|---|---|---|---:|---:|---|---|\n")
        for r in rows:
            path_region = r.get("path") if r.get("path") not in (None, "null") else r.get("region")
            f.write(
                f"| {r['event_index']} | {r['step']} | {r['opcode']} | {r['kind']} | "
                f"{r['selected_executor']} | {float(r['actual_total_peak_mib']):.2f} | "
                f"{r['offender_bag']} | `{r['offender_shape']}` | `{path_region}` |\n"
            )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metrics", required=True)
    p.add_argument("--out-csv", required=True)
    p.add_argument("--out-md", required=True)
    args = p.parse_args()

    metrics_path = Path(args.metrics)
    metrics = json.load(open(metrics_path))
    events = metrics.get("actual_total_peak_events") or []
    rows = [event_row(i, e) for i, e in enumerate(events)]

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    write_md(out_md, metrics_path, metrics, rows)
    print(f"wrote {out_csv}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
