"""Offline cap-triggered bag fission feasibility experiment.

This script captures the runtime peak snapshot for an executable temporal-carving
TTN run, extracts one offending bag tensor, and evaluates local binary fission
of that tensor. It does not modify runtime execution.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, ".")

import clifft
import numpy as np

from temporal_carving.cost import CostModel
from temporal_carving.pipeline import run as run_pipeline
from ttn_backend import TTNBackend
from ttn_backend.backend_spec import export_backend_spec
from ttn_backend.scripts.qec_temporal_carving_report import trace_from_program
from ttn_backend.scripts.qec_temporal_carving_runtime import build_carving_executable_spec


SUMMARY_FIELDS = [
    "circuit", "step", "bag", "mode", "tol",
    "old_shape", "old_numel", "old_bytes", "old_log2_numel",
    "best_peak_numel", "best_total_numel", "best_peak_bytes", "best_total_bytes",
    "best_peak_ratio", "best_total_ratio", "best_rank", "best_split",
    "best_error", "num_tensors", "tree_depth", "status", "notes",
]

CANDIDATE_FIELDS = [
    "mode", "tol", "split_id", "node_id", "depth",
    "left_axes", "right_axes", "left_dims", "right_dims",
    "matrix_shape", "rank", "old_numel", "new_peak_numel",
    "new_total_numel", "peak_ratio", "total_ratio", "error",
    "future_cross_penalty", "accepted", "elapsed_s",
]


@dataclass
class AxisMeta:
    index: int
    name: str
    kind: str
    dim: int


class Node:
    def __init__(self, node_id, depth, tensor, axes):
        self.node_id = str(node_id)
        self.depth = int(depth)
        self.tensor = tensor
        self.axes = list(axes)
        self.children = []
        self.split = None

    @property
    def is_leaf(self):
        return not self.children


def _load_prog(name):
    with open(Path("qec_bench/circuits") / f"{name}.stim") as f:
        return clifft.compile(f.read())


def _prod(xs):
    out = 1
    for x in xs:
        out *= int(x)
    return int(out)


def _log2(x):
    return float(math.log2(float(x))) if x else float("-inf")


def _all_bipartitions(n):
    # Unique unordered non-empty splits; side containing axis 0 is canonical.
    if n < 2:
        return []
    out = []
    rest = list(range(1, n))
    for mask in range(1 << len(rest)):
        left = [0]
        right = []
        for bit, ax in enumerate(rest):
            (left if (mask >> bit) & 1 else right).append(ax)
        if not right:
            continue
        out.append((tuple(left), tuple(right)))
    return out


def _rank_from_s(s, mode, tol):
    if mode == "exact":
        if s.size == 0:
            return 1
        threshold = max(1e-14, 1e-12 * float(s[0]))
        return max(1, int(np.count_nonzero(s > threshold)))
    threshold = max(1e-14, float(tol) * float(s[0])) if s.size else 0.0
    return max(1, int(np.count_nonzero(s > threshold)))


def _split_tensor(tensor, axes, left_local, right_local, mode, tol):
    order = list(left_local) + list(right_local)
    T = np.transpose(tensor, order)
    left_shape = T.shape[:len(left_local)]
    right_shape = T.shape[len(left_local):]
    d_left = _prod(left_shape)
    d_right = _prod(right_shape)
    M = T.reshape(d_left, d_right)
    t0 = time.perf_counter()
    U, s, Vh = np.linalg.svd(M, full_matrices=False)
    elapsed = time.perf_counter() - t0
    rank = _rank_from_s(s, mode, tol)
    Ur = U[:, :rank]
    sr = s[:rank]
    Vhr = Vh[:rank, :]
    if mode == "exact":
        err = 0.0
    else:
        tail = float(np.sum(s[rank:] * s[rank:]))
        total = float(np.sum(s * s))
        err = math.sqrt(tail / total) if total > 0 else 0.0
    sqrt_s = np.sqrt(sr)
    left_tensor = (Ur * sqrt_s[None, :]).reshape(left_shape + (rank,))
    right_tensor = (sqrt_s[:, None] * Vhr).reshape((rank,) + right_shape)
    left_axes = [axes[i] for i in left_local] + [f"bond:{rank}"]
    right_axes = [f"bond:{rank}"] + [axes[i] for i in right_local]
    return dict(
        matrix_shape=(d_left, d_right),
        rank=int(rank),
        left_tensor=left_tensor,
        right_tensor=right_tensor,
        left_axes=left_axes,
        right_axes=right_axes,
        peak_numel=max(int(left_tensor.size), int(right_tensor.size)),
        total_numel=int(left_tensor.size + right_tensor.size),
        error=float(err),
        elapsed_s=float(elapsed),
    )


def _best_split(node, mode, tol, candidates, rows):
    n = node.tensor.ndim
    old_numel = int(node.tensor.size)
    best = None
    best_key = None
    for sid, (left, right) in enumerate(_all_bipartitions(n)):
        res = _split_tensor(node.tensor, node.axes, left, right, mode, tol)
        key = (res["peak_numel"], res["total_numel"], res["error"])
        accepted = best_key is None or key < best_key
        row = dict(
            mode=mode,
            tol=tol,
            split_id=sid,
            node_id=node.node_id,
            depth=node.depth,
            left_axes=json.dumps([node.axes[i] for i in left]),
            right_axes=json.dumps([node.axes[i] for i in right]),
            left_dims=json.dumps([int(node.tensor.shape[i]) for i in left]),
            right_dims=json.dumps([int(node.tensor.shape[i]) for i in right]),
            matrix_shape=json.dumps(list(res["matrix_shape"])),
            rank=res["rank"],
            old_numel=old_numel,
            new_peak_numel=res["peak_numel"],
            new_total_numel=res["total_numel"],
            peak_ratio=old_numel / res["peak_numel"] if res["peak_numel"] else "",
            total_ratio=old_numel / res["total_numel"] if res["total_numel"] else "",
            error=res["error"],
            future_cross_penalty=0,
            accepted=accepted,
            elapsed_s=res["elapsed_s"],
        )
        rows.append(row)
        if accepted:
            best = (left, right, res, sid)
            best_key = key
    return best


def _collect_leaves(node):
    if node.is_leaf:
        return [node]
    out = []
    for c in node.children:
        out.extend(_collect_leaves(c))
    return out


def _recursive_fission(root, mode, tol, max_depth, rows):
    for _ in range(max_depth):
        leaves = [n for n in _collect_leaves(root) if n.tensor.ndim >= 2]
        if not leaves:
            break
        node = max(leaves, key=lambda n: n.tensor.size)
        best = _best_split(node, mode, tol, [], rows)
        if best is None:
            break
        left, right, res, sid = best
        if res["peak_numel"] >= int(node.tensor.size):
            break
        node.split = dict(
            split_id=sid,
            left_axes=[node.axes[i] for i in left],
            right_axes=[node.axes[i] for i in right],
            rank=res["rank"],
            mode=mode,
            tol=tol,
            error=res["error"],
        )
        node.children = [
            Node(f"{node.node_id}L", node.depth + 1, res["left_tensor"], res["left_axes"]),
            Node(f"{node.node_id}R", node.depth + 1, res["right_tensor"], res["right_axes"]),
        ]
    return root


def _tree_stats(root):
    leaves = _collect_leaves(root)
    peak = max((int(n.tensor.size) for n in leaves), default=0)
    total = sum(int(n.tensor.size) for n in leaves)
    depth = max((int(n.depth) for n in leaves), default=0)
    return peak, total, len(leaves), depth


def _tree_json(node):
    return dict(
        node_id=node.node_id,
        depth=node.depth,
        axes=list(map(str, node.axes)),
        shape=list(map(int, node.tensor.shape)),
        numel=int(node.tensor.size),
        split=node.split,
        children=[_tree_json(c) for c in node.children],
    )


def capture_bag_tensor(circuit, bag_id, max_steps, seed, timeout, seeder, refine):
    prog = _load_prog(circuit)
    base_spec = export_backend_spec(prog, strict=False)
    trace = trace_from_program(prog, strict=False)
    carving_result = run_pipeline(
        trace,
        seeder=seeder,
        refine_moves=tuple(x for x in refine.split(",") if x and x != "none"),
        exact=False,
    )
    spec, homing = build_carving_executable_spec(base_spec, carving_result["tree"])
    backend = TTNBackend(spec, homing, capture_peak_snapshot=True)
    backend.run_shot(prog, seed=seed, runtime_timeout=timeout, check_interval=1, max_steps=max_steps)
    metrics = backend.last_metrics or {}
    snapshot = metrics.get("peak_snapshot")
    if not snapshot:
        raise RuntimeError("no peak snapshot captured")
    row = next((r for r in snapshot["bags"] if int(r["bag_id"]) == int(bag_id)), None)
    if row is None:
        raise RuntimeError(f"B{bag_id} not found in peak snapshot")
    tensor = np.asarray(row["tensor"])
    axes = []
    for ident in row["own_idents"]:
        axes.append(f"phys:{int(ident)}")
    for nb in row["neighbors"]:
        axes.append(f"bond:{min(int(bag_id), int(nb))}-{max(int(bag_id), int(nb))}")
    return tensor, axes, metrics


def write_report(path, summaries):
    with open(path, "w") as f:
        f.write("# Bag Fission Offline Report\n\n")
        f.write("| mode | tol | old log2 | best peak log2 | peak ratio | total ratio | error | status |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---|\n")
        for r in summaries:
            f.write(
                f"| {r['mode']} | {r['tol']} | {float(r['old_log2_numel']):.3f} | "
                f"{_log2(int(r['best_peak_numel'])):.3f} | {float(r['best_peak_ratio']):.3f} | "
                f"{float(r['best_total_ratio']):.3f} | {float(r['best_error']):.3e} | {r['status']} |\n"
            )
        f.write("\nExact rows use numerical zero cleanup only. Approx rows use truncated SVD and must be reported separately.\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--circuit", default="coherent_d5_r5")
    p.add_argument("--bag", default="72")
    p.add_argument("--max-steps", type=int, default=1200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--runtime-timeout", type=float, default=300.0)
    p.add_argument("--seeder", default="recursive_balanced_mincut")
    p.add_argument("--refine", default="none")
    p.add_argument("--approx-tols", nargs="*", default=["1e-4", "1e-3", "1e-2"])
    p.add_argument("--max-depth", type=int, default=3)
    p.add_argument("--out-dir", default="reports/bag_fission_offline")
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tensor, axes, metrics = capture_bag_tensor(
        args.circuit,
        int(str(args.bag).lstrip("Bb")),
        args.max_steps,
        args.seed,
        args.runtime_timeout,
        args.seeder,
        args.refine,
    )
    old_numel = int(tensor.size)
    old_bytes = int(tensor.nbytes)

    summaries = []
    rows = []
    tree_outputs = {}
    modes = [("exact", "0")] + [("approx", str(t)) for t in args.approx_tols]
    for mode, tol_s in modes:
        tol = float(tol_s)
        root = Node("root", 0, tensor.copy(), list(axes))
        _recursive_fission(root, mode, tol, args.max_depth, rows)
        peak, total, ntensors, depth = _tree_stats(root)
        leaves = _collect_leaves(root)
        best_rank = ""
        best_split = ""
        best_error = max((float((n.split or {}).get("error", 0.0)) for n in [root] + leaves), default=0.0)
        if root.split:
            best_rank = root.split.get("rank")
            best_split = json.dumps(dict(left=root.split.get("left_axes"), right=root.split.get("right_axes")))
        summaries.append(dict(
            circuit=args.circuit,
            step=metrics.get("actual_total_peak_step"),
            bag=f"B{int(str(args.bag).lstrip('Bb'))}",
            mode=mode,
            tol=tol_s,
            old_shape=json.dumps(list(map(int, tensor.shape))),
            old_numel=old_numel,
            old_bytes=old_bytes,
            old_log2_numel=_log2(old_numel),
            best_peak_numel=peak,
            best_total_numel=total,
            best_peak_bytes=peak * tensor.itemsize,
            best_total_bytes=total * tensor.itemsize,
            best_peak_ratio=old_numel / peak if peak else "",
            best_total_ratio=old_numel / total if total else "",
            best_rank=best_rank,
            best_split=best_split,
            best_error=best_error,
            num_tensors=ntensors,
            tree_depth=depth,
            status="ok",
            notes="offline_only",
        ))
        tree_outputs[f"{mode}_{tol_s}"] = _tree_json(root)

    with open(out / "bag_fission_offline_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        w.writerows(summaries)
    with open(out / "bag_fission_offline_candidates.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CANDIDATE_FIELDS)
        w.writeheader()
        w.writerows(rows)
    with open(out / "bag_fission_offline_tree.json", "w") as f:
        json.dump(tree_outputs, f, indent=2)
    write_report(out / "bag_fission_offline_report.md", summaries)
    with open(out / "source_metrics.json", "w") as f:
        json.dump({
            k: v for k, v in metrics.items()
            if k != "peak_snapshot"
        }, f, indent=2)
    print(f"wrote {out / 'bag_fission_offline_summary.csv'}")
    print(f"wrote {out / 'bag_fission_offline_report.md'}")


if __name__ == "__main__":
    main()
