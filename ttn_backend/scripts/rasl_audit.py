"""Print an audit table for accepted RASL target changes."""

from __future__ import annotations

import argparse
import csv
import json
import os


FIELDS = [
    "step_id",
    "circuit",
    "support_axes",
    "support_size",
    "active_only",
    "default_target",
    "chosen_target",
    "builder_kind",
    "default_v_sequence",
    "chosen_v_sequence",
    "default_edge_hits",
    "chosen_edge_hits",
    "reduced_edges",
    "resident_proxy_reason",
    "refactor_proxy_delta",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="reports/rasl_steps_full.csv")
    p.add_argument("--json", default="reports/rasl_changed_audit.json")
    p.add_argument("--circuit", default="")
    args = p.parse_args()

    rows = []
    with open(args.csv) as f:
        for row in csv.DictReader(f):
            if row.get("accepted") != "True":
                continue
            if row.get("default_target") == row.get("chosen_target"):
                continue
            if args.circuit and row.get("circuit") != args.circuit:
                continue
            rows.append({k: row.get(k, "") for k in FIELDS})

    os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
    with open(args.json, "w") as f:
        json.dump(rows, f, indent=2)

    for row in rows:
        print(
            f"{row['circuit']} step={row['step_id']} support=[{row['support_axes']}] "
            f"target {row['default_target']}->{row['chosen_target']} "
            f"builder={row['builder_kind']} delta_refactor={row['refactor_proxy_delta']}"
        )
        print(f"  default V: {row['default_v_sequence']}")
        print(f"  chosen  V: {row['chosen_v_sequence']}")
        print(f"  default edges: {row['default_edge_hits']}")
        print(f"  chosen  edges: {row['chosen_edge_hits']}")
        print(f"  reduced edges: {row['reduced_edges'] or '(none; same edge count, lower weighted path)'}")
        print(f"  resident proxy: {row['resident_proxy_reason']}")
    print(f"\nwrote JSON: {args.json}")


if __name__ == "__main__":
    main()
