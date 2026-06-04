"""Local SVD refinement experiment for the actual TTN peak offender node.

This is an offline/profile-time experiment.  It does not modify the runtime
layout.  It reruns one TTN layout with peak snapshot capture, extracts the peak
offender bag tensor, enumerates all bipartitions of that node's open legs, and
reports the best local two-tensor replacement under a numerical-rank threshold.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

import clifft
import numpy as np

from temporal_carving.pipeline import run as run_pipeline
from ttn_backend import TTNBackend
from ttn_backend.backend_spec import assign_homes_and_classify, export_backend_spec
from ttn_backend.scripts.qec_temporal_carving_report import trace_from_program
from ttn_backend.scripts.qec_temporal_carving_runtime import build_carving_executable_spec


CANDIDATE_FIELDS = [
    "candidate_id",
    "left_axes",
    "right_axes",
    "left_labels",
    "right_labels",
    "matrix_shape",
    "rank",
    "rank_log2",
    "discarded_rel_error",
    "recon_error_rel",
    "old_numel",
    "old_log2",
    "old_bytes",
    "new_peak_numel",
    "new_peak_log2",
    "new_peak_bytes",
    "new_total_numel",
    "new_total_log2",
    "new_total_bytes",
    "peak_reduction",
    "total_reduction",
    "elapsed_s_svd",
]


def _load_prog(circuit):
    with open(Path("qec_bench/circuits") / f"{circuit}.stim") as f:
        return clifft.compile(f.read())


def _run_capture(circuit, mode, args):
    prog = _load_prog(circuit)
    base = export_backend_spec(prog, strict=False)
    if mode == "baseline_jt":
        spec = base
        homing = assign_homes_and_classify(base)
    elif mode == "carving_leaf":
        trace = trace_from_program(prog, strict=False)
        result = run_pipeline(
            trace,
            seeder=args.seeder,
            refine_moves=tuple(x for x in args.refine.split(",") if x and x != "none"),
            seed=args.seed,
            partitioner=args.partitioner,
            exact=False,
        )
        spec, homing = build_carving_executable_spec(base, result["tree"])
    else:
        raise ValueError(f"unknown mode {mode}")
    backend = TTNBackend(spec, homing, capture_peak_snapshot=True)
    backend.run_shot(
        prog,
        seed=args.shot_seed,
        runtime_timeout=args.runtime_timeout,
        check_interval=1,
    )
    return backend.last_metrics or {}


def _axis_metadata(peak_bag):
    tensor = peak_bag["tensor"]
    own = list(map(int, peak_bag["own_idents"]))
    neighbors = list(map(int, peak_bag["neighbors"]))
    nown = len(own)
    axes = []
    for ax, ident in enumerate(own):
        axes.append(dict(
            axis=ax,
            kind="physical",
            label=f"phys:{ident}",
            dim=int(tensor.shape[ax]),
        ))
    bid = int(peak_bag["bag_id"])
    for i, nb in enumerate(neighbors):
        axes.append(dict(
            axis=nown + i,
            kind="bond",
            label=f"bond:{min(bid, nb)}-{max(bid, nb)}",
            dim=int(tensor.shape[nown + i]),
        ))
    return axes


def _rank_from_singular_values(s, rel_tol, abs_tol):
    if s.size == 0:
        return 1
    threshold = max(float(abs_tol), float(rel_tol) * float(s[0]))
    rank = int(np.count_nonzero(s > threshold))
    return max(rank, 1)


def _evaluate_split(tensor, axes, left, rel_tol, abs_tol):
    n = tensor.ndim
    right = [i for i in range(n) if i not in left]
    perm = list(left) + right
    T = np.transpose(tensor, perm)
    left_dim = int(np.prod([tensor.shape[i] for i in left]))
    right_dim = int(np.prod([tensor.shape[i] for i in right]))
    M = T.reshape(left_dim, right_dim)
    t0 = time.perf_counter()
    U, s, Vh = np.linalg.svd(M, full_matrices=False)
    elapsed = time.perf_counter() - t0
    rank = _rank_from_singular_values(s, rel_tol, abs_tol)
    discarded = float(np.sum(s[rank:] ** 2))
    total = float(np.sum(s ** 2))
    discarded_rel = math.sqrt(discarded / total) if total else 0.0

    Ur = U[:, :rank]
    sr = s[:rank]
    Vhr = Vh[:rank, :]
    recon_M = (Ur * sr[None, :]) @ Vhr
    recon_rel = float(np.linalg.norm(M - recon_M) / np.linalg.norm(M)) if np.linalg.norm(M) else 0.0

    old_numel = int(tensor.size)
    old_bytes = int(tensor.nbytes)
    left_numel = left_dim * rank
    right_numel = rank * right_dim
    peak = max(left_numel, right_numel)
    total_numel = left_numel + right_numel
    itemsize = int(tensor.dtype.itemsize)
    left_labels = [axes[i]["label"] for i in left]
    right_labels = [axes[i]["label"] for i in right]
    return dict(
        left_axes=list(left),
        right_axes=right,
        left_labels=left_labels,
        right_labels=right_labels,
        matrix_shape=[left_dim, right_dim],
        rank=rank,
        rank_log2=float(math.log2(rank)) if rank > 0 else None,
        discarded_rel_error=discarded_rel,
        recon_error_rel=recon_rel,
        old_numel=old_numel,
        old_log2=float(math.log2(old_numel)),
        old_bytes=old_bytes,
        new_peak_numel=int(peak),
        new_peak_log2=float(math.log2(peak)),
        new_peak_bytes=int(peak * itemsize),
        new_total_numel=int(total_numel),
        new_total_log2=float(math.log2(total_numel)),
        new_total_bytes=int(total_numel * itemsize),
        peak_reduction=float(old_numel / peak) if peak else None,
        total_reduction=float(old_numel / total_numel) if total_numel else None,
        elapsed_s_svd=elapsed,
    )


def enumerate_splits(tensor, axes, rel_tol, abs_tol):
    n = tensor.ndim
    rows = []
    seen = set()
    cid = 0
    for r in range(1, n):
        for left in itertools.combinations(range(n), r):
            left = tuple(left)
            right = tuple(i for i in range(n) if i not in left)
            key = frozenset([left, right])
            if key in seen:
                continue
            seen.add(key)
            row = _evaluate_split(tensor, axes, left, rel_tol, abs_tol)
            row["candidate_id"] = cid
            rows.append(row)
            cid += 1
    rows.sort(key=lambda x: (
        x["new_peak_log2"],
        x["new_total_log2"],
        x["recon_error_rel"],
    ))
    return rows


def write_outputs(out_dir, circuit, mode, metrics, axes, rows, args):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cand_path = out_dir / "peak_node_refinement_candidates.csv"
    with open(cand_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CANDIDATE_FIELDS)
        w.writeheader()
        for row in rows:
            out = dict(row)
            for key in ("left_axes", "right_axes", "left_labels", "right_labels", "matrix_shape"):
                out[key] = json.dumps(out[key])
            w.writerow(out)
    best = rows[0] if rows else None
    summary = dict(
        circuit=circuit,
        mode=mode,
        rel_tol=args.rel_tol,
        abs_tol=args.abs_tol,
        runtime_timeout=args.runtime_timeout,
        metrics={k: metrics.get(k) for k in [
            "steps_completed",
            "total_steps",
            "timeout",
            "resident_actual_peak_log2_numel",
            "resident_actual_peak_bytes",
            "actual_peak_offender_bag",
            "actual_peak_offender_step",
            "actual_peak_offender_shape",
            "actual_peak_offender_p_B",
            "actual_peak_offender_incident_bond_dims",
            "actual_peak_offender_incident_edge_ids",
            "workspace_actual_peak_bytes",
            "peak_stored_bytes",
            "max_bond_dim_observed",
            "n_qr",
        ]},
        axes=axes,
        best=best,
    )
    with open(out_dir / "peak_node_refinement_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    with open(out_dir / "peak_node_refinement_report.md", "w") as f:
        f.write(f"# Peak Node Refinement Experiment: {circuit} {mode}\n\n")
        f.write("This is an offline local SVD split experiment on the actual runtime peak offender tensor. It does not modify the full backend layout.\n\n")
        f.write(f"- rel_tol: `{args.rel_tol}`\n")
        f.write(f"- abs_tol: `{args.abs_tol}`\n")
        f.write(f"- steps: `{metrics.get('steps_completed')}/{metrics.get('total_steps')}` timeout={metrics.get('timeout')}\n")
        f.write(f"- peak bag: `{metrics.get('actual_peak_offender_bag')}` at step `{metrics.get('actual_peak_offender_step')}`\n")
        f.write(f"- old shape: `{metrics.get('actual_peak_offender_shape')}`\n")
        f.write(f"- old bytes: `{metrics.get('resident_actual_peak_bytes')}`\n\n")
        if best:
            f.write("## Best Local Split\n\n")
            f.write(f"- left: `{best['left_labels']}`\n")
            f.write(f"- right: `{best['right_labels']}`\n")
            f.write(f"- matrix shape: `{best['matrix_shape']}`\n")
            f.write(f"- numerical rank: `{best['rank']}`\n")
            f.write(f"- new peak log2: `{best['new_peak_log2']:.6f}`\n")
            f.write(f"- new peak bytes: `{best['new_peak_bytes']}`\n")
            f.write(f"- peak reduction: `{best['peak_reduction']:.6g}x`\n")
            f.write(f"- total reduction: `{best['total_reduction']:.6g}x`\n")
            f.write(f"- reconstruction error rel: `{best['recon_error_rel']:.6g}`\n")
            f.write(f"- discarded rel error: `{best['discarded_rel_error']:.6g}`\n")
        f.write("\n## Top Candidates\n\n")
        f.write("| left | right | rank | new peak log2 | peak reduction | recon err |\n")
        f.write("|---|---|---:|---:|---:|---:|\n")
        for row in rows[:10]:
            f.write(
                f"| {row['left_labels']} | {row['right_labels']} | {row['rank']} | "
                f"{row['new_peak_log2']:.3f} | {row['peak_reduction']:.3g} | "
                f"{row['recon_error_rel']:.3g} |\n"
            )
    return cand_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--circuit", default="coherent_d5_r5")
    p.add_argument("--mode", choices=["baseline_jt", "carving_leaf"], default="carving_leaf")
    p.add_argument("--rel-tol", type=float, default=1e-4)
    p.add_argument("--abs-tol", type=float, default=1e-14)
    p.add_argument("--runtime-timeout", type=float, default=60.0)
    p.add_argument("--shot-seed", type=int, default=42)
    p.add_argument("--seeder", default="recursive_balanced_mincut")
    p.add_argument("--refine", default="nni")
    p.add_argument("--partitioner", default="networkx")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="reports/peak_node_refinement")
    args = p.parse_args()

    print(f"capturing peak: circuit={args.circuit} mode={args.mode}", flush=True)
    metrics = _run_capture(args.circuit, args.mode, args)
    snap = metrics.get("peak_snapshot")
    if not snap:
        raise RuntimeError("peak snapshot was not captured")
    peak_bid = int(metrics["actual_peak_offender_bag"])
    peak_bag = next(b for b in snap["bags"] if int(b["bag_id"]) == peak_bid)
    tensor = peak_bag["tensor"]
    axes = _axis_metadata(peak_bag)
    print(
        f"peak bag={peak_bid} step={metrics.get('actual_peak_offender_step')} "
        f"shape={tensor.shape} bytes={tensor.nbytes}",
        flush=True,
    )
    rows = enumerate_splits(tensor, axes, args.rel_tol, args.abs_tol)
    out_dir = Path(args.out_dir) / args.circuit / args.mode
    write_outputs(out_dir, args.circuit, args.mode, metrics, axes, rows, args)
    best = rows[0]
    print(
        f"best split rank={best['rank']} new_peak_log2={best['new_peak_log2']:.3f} "
        f"new_peak_bytes={best['new_peak_bytes']} "
        f"peak_reduction={best['peak_reduction']:.3g} "
        f"recon_error={best['recon_error_rel']:.3g}",
        flush=True,
    )
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
