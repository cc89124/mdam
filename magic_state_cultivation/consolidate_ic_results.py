#!/usr/bin/env python3
"""Consolidate inject+cultivate chunk files into one file per stratum.

Merges all {key}_k{k}_chunk*.json files into a single {key}_k{k}.json
by summing total_shots, passed_shots, logical_errors, and seconds.
Existing unchunked files are included in the aggregation.

After consolidation, chunk files are deleted. The resulting files are
compatible with both plot_results.py (load_strata) and run_ic_tiered.py
(_existing_shots_and_max_chunk).

Usage:
    uv run python consolidate_ic_results.py
    uv run python consolidate_ic_results.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib

RESULTS_DIR = pathlib.Path(__file__).parent / "results" / "inject_cultivate"


def consolidate(dry_run: bool = False):
    # Collect all files grouped by (key, k)
    groups: dict[tuple[str, int], list[pathlib.Path]] = {}
    for path in sorted(RESULTS_DIR.glob("*_d*_k*.json")):
        if path.name == "analysis.json":
            continue
        data = json.load(open(path))
        key = f"{data['circuit']}_d{data['dcolor']}"
        k = data["k"]
        groups.setdefault((key, k), []).append(path)

    consolidated = 0
    deleted = 0

    for (key, k), paths in sorted(groups.items()):
        if len(paths) == 1 and "chunk" not in paths[0].stem:
            # Already a single unchunked file
            continue

        # Aggregate all chunks + unchunked file
        agg_shots = 0
        agg_passed = 0
        agg_errors = 0
        agg_seconds = 0.0
        circuit = None
        dcolor = None

        for path in paths:
            data = json.load(open(path))
            circuit = data["circuit"]
            dcolor = data["dcolor"]
            agg_shots += data["total_shots"]
            agg_passed += data["passed_shots"]
            agg_errors += data.get("logical_errors", 0)
            agg_seconds += data.get("seconds", 0)

        result = {
            "circuit": circuit,
            "dcolor": dcolor,
            "k": k,
            "total_shots": agg_shots,
            "passed_shots": agg_passed,
            "logical_errors": agg_errors,
            "seconds": round(agg_seconds, 1),
        }

        out_path = RESULTS_DIR / f"{key}_k{k}.json"
        print(
            f"  {key} k={k}: {len(paths)} files -> {out_path.name} "
            f"({agg_shots:,} shots, {agg_seconds/3600:.1f} CPU-hrs)"
        )

        if not dry_run:
            # Write consolidated file
            tmp = out_path.with_suffix(".json.tmp")
            with open(tmp, "w") as f:
                json.dump(result, f, indent=2)
            os.replace(tmp, out_path)

            # Delete chunk files (but not the file we just wrote)
            for path in paths:
                if path != out_path and path.exists():
                    path.unlink()
                    deleted += 1

        consolidated += 1

    print(f"\nConsolidated {consolidated} strata, deleted {deleted} chunk files")


def main():
    parser = argparse.ArgumentParser(
        description="Consolidate IC chunk files into one file per stratum",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show what would happen")
    args = parser.parse_args()

    if args.dry_run:
        print("DRY RUN — no files will be modified\n")

    consolidate(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
