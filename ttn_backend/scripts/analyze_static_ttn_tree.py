"""Analyze a static TTN compression tree and explain its bottleneck nodes."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from collections import Counter
from pathlib import Path


NODE_FIELDS = [
    "node_id",
    "parent_id",
    "side",
    "depth",
    "kind",
    "shape",
    "numel",
    "bytes",
    "log2_numel",
    "open_legs",
    "internal_bonds",
    "internal_bond_ranks",
    "open_leg_logsum",
    "internal_bond_logsum",
    "log2_numel_decomposition",
    "created_by_rank",
    "created_by_candidate_id",
    "created_by_discarded_energy",
    "created_by_discarded_relative",
]

HIST_FIELDS = ["rank", "rank_log2", "count"]


def _log2(x):
    x = float(x)
    return math.log2(x) if x > 0 else float("-inf")


def _prod(xs):
    out = 1
    for x in xs:
        out *= int(x)
    return int(out)


def _read_candidates(path):
    if not path or not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def _candidate_key(parent_id, split):
    def norm(xs):
        return " ".join(xs)
    return (
        str(parent_id),
        norm(split.get("A_legs", [])),
        norm(split.get("B_legs", [])),
        str(int(split.get("rank", 0))),
    )


def _candidate_index(rows):
    idx = {}
    for r in rows:
        try:
            key = (
                r["node_id"],
                " ".join(r["A_legs"].split()),
                " ".join(r["B_legs"].split()),
                str(int(float(r["rank"]))),
            )
        except Exception:
            continue
        if r.get("accepted") in ("True", "true", "1", True):
            idx[key] = r
    return idx


def _leaf_rows(tree, cand_idx):
    rows = []
    ranks = []

    def walk(node, parent_id="", side="", depth=0, creator=None):
        kind = node.get("kind")
        split = node.get("split") or {}
        if kind == "internal":
            rank = int(split.get("rank", 1))
            ranks.append(rank)
            children = node.get("children", [])
            if len(children) >= 1:
                walk(children[0], node["node_id"], "L", depth + 1, split)
            if len(children) >= 2:
                walk(children[1], node["node_id"], "R", depth + 1, split)
            return

        legs = list(node.get("legs", []))
        shape = [int(x) for x in node.get("shape", [])]
        if len(legs) != len(shape):
            raise ValueError(f"node {node.get('node_id')} legs/shape mismatch")
        open_legs = []
        internal = []
        internal_ranks = []
        open_log = 0.0
        internal_log = 0.0
        for name, dim in zip(legs, shape):
            if str(name).startswith("internal:"):
                internal.append(name)
                internal_ranks.append(dim)
                internal_log += _log2(dim)
            else:
                open_legs.append(f"{name}:{dim}")
                open_log += _log2(dim)
        numel = int(node.get("numel") or _prod(shape))
        created = {}
        if creator:
            key = _candidate_key(parent_id, creator)
            created = cand_idx.get(key, {})
        rows.append(dict(
            node_id=node.get("node_id", ""),
            parent_id=parent_id,
            side=side,
            depth=depth,
            kind=kind,
            shape="x".join(str(x) for x in shape),
            numel=numel,
            bytes=int(node.get("bytes") or numel * 16),
            log2_numel=_log2(numel),
            open_legs=" ".join(open_legs),
            internal_bonds=" ".join(internal),
            internal_bond_ranks=" ".join(str(x) for x in internal_ranks),
            open_leg_logsum=open_log,
            internal_bond_logsum=internal_log,
            log2_numel_decomposition=f"{open_log:.6g}+{internal_log:.6g}",
            created_by_rank=creator.get("rank", "") if creator else "",
            created_by_candidate_id=created.get("candidate_id", ""),
            created_by_discarded_energy=created.get("discarded_energy", creator.get("discarded_energy", "") if creator else ""),
            created_by_discarded_relative=created.get("discarded_relative", ""),
        ))

    walk(tree, depth=0)
    return rows, ranks


def _write_csv(path, fields, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _write_report(path, rows, ranks, summary_rows):
    peak = max(rows, key=lambda r: (float(r["log2_numel"]), int(r["bytes"])))
    rank_logs = [_log2(r) for r in ranks]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("# Beam Tree Bottleneck Report\n\n")
        if summary_rows:
            f.write("## Source Summary\n\n")
            for r in summary_rows:
                f.write(
                    f"- mode={r.get('mode')} rule={r.get('rank_rule')} tol={r.get('tol')} "
                    f"peak_log2={r.get('new_peak_log2_numel')} total_log2={r.get('new_total_log2_numel')} "
                    f"error={r.get('recon_error_rel')}\n"
                )
            f.write("\n")
        f.write("## Peak Node\n\n")
        f.write(f"- peak_node_id: `{peak['node_id']}`\n")
        f.write(f"- parent: `{peak['parent_id']}` side `{peak['side']}`\n")
        f.write(f"- depth: `{peak['depth']}`\n")
        f.write(f"- shape: `{peak['shape']}`\n")
        f.write(f"- log2_numel: `{float(peak['log2_numel']):.6f}`\n")
        f.write(f"- bytes: `{peak['bytes']}`\n")
        f.write(f"- open_leg_logsum: `{float(peak['open_leg_logsum']):.6f}`\n")
        f.write(f"- internal_bond_logsum: `{float(peak['internal_bond_logsum']):.6f}`\n")
        f.write(f"- decomposition: `{peak['log2_numel_decomposition']}`\n")
        f.write(f"- internal bonds: `{peak['internal_bonds']}`\n")
        f.write(f"- internal bond ranks: `{peak['internal_bond_ranks']}`\n")
        f.write(f"- open legs: `{peak['open_legs']}`\n")
        f.write(f"- created_by_rank: `{peak['created_by_rank']}`\n")
        f.write(f"- created_by_candidate_id: `{peak['created_by_candidate_id']}`\n")
        f.write(f"- created_by_discarded_energy: `{peak['created_by_discarded_energy']}`\n\n")

        f.write("## Rank Statistics\n\n")
        if ranks:
            sorted_logs = sorted(rank_logs)
            mean_log = sum(rank_logs) / len(rank_logs)
            median_log = sorted_logs[len(sorted_logs) // 2]
            f.write(f"- internal bond count: `{len(ranks)}`\n")
            f.write(f"- max_internal_rank: `{max(ranks)}`\n")
            f.write(f"- max_internal_rank_log2: `{_log2(max(ranks)):.6f}`\n")
            f.write(f"- mean_internal_rank_log2: `{mean_log:.6f}`\n")
            f.write(f"- median_internal_rank_log2: `{median_log:.6f}`\n\n")
        else:
            f.write("- no internal bonds\n\n")

        f.write("## Interpretation\n\n")
        open_log = float(peak["open_leg_logsum"])
        int_log = float(peak["internal_bond_logsum"])
        if open_log >= int_log:
            f.write(
                "Q1/Q2: peak node is mostly open-leg-product limited. "
                "To reduce below this peak, the next search should split or regroup "
                "the peak node's open original legs.\n\n"
            )
        else:
            f.write(
                "Q1/Q2: peak node is mostly internal-rank limited. "
                "To reduce below this peak, the next search should target the split "
                "that created the large internal rank or use a multi-snapshot/rank-aware "
                "objective.\n\n"
            )
        f.write(
            "Singular spectrum data is not stored in the current candidate CSV. "
            "Only rank, discarded energy, discarded relative error, and SVD matrix "
            "shape are available. To inspect top singular values, extend the SVD "
            "evaluator to persist spectra for accepted splits.\n"
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tree-json", default=None)
    p.add_argument("--summary-csv", default=None)
    p.add_argument("--candidates-csv", default=None)
    p.add_argument("--out-dir", default="reports/static_rel1e8_beam")
    args = p.parse_args()

    tree_json = args.tree_json
    if tree_json is None:
        matches = glob.glob(os.path.join(args.out_dir, "static_ttn_b0_compression_tree_*.json"))
        if not matches:
            raise FileNotFoundError(f"no tree json in {args.out_dir}")
        tree_json = sorted(matches)[0]
    summary_csv = args.summary_csv or os.path.join(args.out_dir, "static_ttn_b0_compression_summary.csv")
    candidates_csv = args.candidates_csv or os.path.join(args.out_dir, "static_ttn_b0_compression_candidates.csv")

    tree = json.load(open(tree_json))
    candidates = _read_candidates(candidates_csv)
    cand_idx = _candidate_index(candidates)
    rows, ranks = _leaf_rows(tree, cand_idx)
    hist = [dict(rank=r, rank_log2=_log2(r), count=c) for r, c in sorted(Counter(ranks).items())]
    summary_rows = _read_candidates(summary_csv)

    _write_csv(os.path.join(args.out_dir, "beam_tree_node_decomposition.csv"), NODE_FIELDS, rows)
    _write_csv(os.path.join(args.out_dir, "beam_tree_rank_histogram.csv"), HIST_FIELDS, hist)
    _write_report(os.path.join(args.out_dir, "beam_tree_bottleneck_report.md"), rows, ranks, summary_rows)

    peak = max(rows, key=lambda r: (float(r["log2_numel"]), int(r["bytes"])))
    print(
        f"peak_node={peak['node_id']} log2={float(peak['log2_numel']):.3f} "
        f"open={float(peak['open_leg_logsum']):.3f} internal={float(peak['internal_bond_logsum']):.3f}"
    )


if __name__ == "__main__":
    main()
