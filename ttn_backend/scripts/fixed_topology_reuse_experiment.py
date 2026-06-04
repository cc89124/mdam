"""Apply one static TTN tree topology to multiple B0 critical snapshots."""

from __future__ import annotations

import argparse
import csv
import glob
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
from scipy import linalg

from ttn_backend import TTNBackend
from ttn_backend.backend_spec import assign_homes_and_classify, export_backend_spec
from ttn_backend.core import ds_mod


SUMMARY_FIELDS = [
    "step",
    "old_B0_numel",
    "old_B0_bytes",
    "old_B0_log2_numel",
    "fixed_topology_peak_numel",
    "fixed_topology_peak_bytes",
    "fixed_topology_peak_log2",
    "fixed_topology_total_numel",
    "fixed_topology_total_bytes",
    "fixed_topology_total_log2",
    "peak_compression_ratio",
    "total_compression_ratio",
    "reconstruction_error_abs",
    "reconstruction_error_rel",
    "max_internal_rank",
    "max_internal_rank_log2",
    "num_tensors",
    "tree_depth",
    "status",
    "notes",
]


@dataclass(frozen=True)
class LegMeta:
    name: str
    kind: str
    dim: int
    original_axis: int | None
    edge_id: str | None
    log2_dim: float


class SnapshotRecorder:
    def __init__(self, target_steps, bag_id):
        self.target_steps = {int(x) for x in target_steps}
        self.bag_id = int(bag_id)
        self.state = None
        self.snapshots = {}

    def __call__(self, row):
        step = row.get("step_id")
        if step is None:
            return
        step = int(step)
        if step not in self.target_steps or self.state is None:
            return
        bag = self.state.bags[self.bag_id]
        self.snapshots[step] = dict(
            step_id=step,
            bag_id=self.bag_id,
            neighbors=list(map(int, bag.neighbors)),
            own_idents=list(map(int, bag.own_idents)),
            tensor=bag.tensor.copy(),
        )


class SnapshotBackend(TTNBackend):
    def _reset(self):
        super()._reset()
        if hasattr(self.trace_recorder, "state"):
            self.trace_recorder.state = self.state


def _log2(x):
    x = float(x)
    return math.log2(x) if x > 0 else float("-inf")


def _prod(xs):
    out = 1
    for x in xs:
        out *= int(x)
    return int(out)


def _load_prog(name):
    with open(os.path.join("qec_bench/circuits", name + ".stim")) as f:
        return clifft.compile(f.read())


def _snapshot_path(cache_dir, circuit, step, bag_name):
    return Path(cache_dir) / f"fixed_reuse_snapshot_{circuit}_step{int(step)}_{bag_name}.npz"


def _save_snapshot(path, tensor, legs):
    os.makedirs(path.parent, exist_ok=True)
    meta = [l.__dict__ for l in legs]
    np.savez_compressed(path, tensor=tensor, legs_json=json.dumps(meta))


def _load_snapshot(path):
    data = np.load(path, allow_pickle=False)
    tensor = data["tensor"]
    legs = [LegMeta(**row) for row in json.loads(str(data["legs_json"]))]
    return tensor, legs


def _legs_from_bag_row(row):
    tensor = np.asarray(row["tensor"])
    bag_id = int(row["bag_id"])
    legs = []
    axis = 0
    for ident in row["own_idents"]:
        dim = int(tensor.shape[axis])
        legs.append(LegMeta(
            name=f"phys:{int(ident)}",
            kind="physical",
            dim=dim,
            original_axis=axis,
            edge_id=None,
            log2_dim=_log2(dim),
        ))
        axis += 1
    for nb in row["neighbors"]:
        dim = int(tensor.shape[axis])
        eid = f"{min(bag_id, int(nb))}-{max(bag_id, int(nb))}"
        legs.append(LegMeta(
            name=f"bond:{eid}",
            kind="bond",
            dim=dim,
            original_axis=axis,
            edge_id=eid,
            log2_dim=_log2(dim),
        ))
        axis += 1
    return legs


def capture_snapshots(circuit, steps, bag_name, cache_dir, seed, timeout_s, force=False):
    bag_id = int(str(bag_name).lstrip("Bb"))
    cached = {}
    missing = []
    for step in steps:
        path = _snapshot_path(cache_dir, circuit, step, bag_name)
        if path.exists() and not force:
            cached[int(step)] = _load_snapshot(path)
        else:
            missing.append(int(step))
    if not missing:
        return cached

    prog = _load_prog(circuit)
    spec = export_backend_spec(prog, strict=False)
    homing = assign_homes_and_classify(spec)
    recorder = SnapshotRecorder(missing, bag_id)
    backend = SnapshotBackend(spec, homing, trace_recorder=recorder)
    backend.run_shot(prog, seed, runtime_timeout=timeout_s, check_interval=1)
    for step, row in recorder.snapshots.items():
        tensor = np.asarray(row["tensor"])
        legs = _legs_from_bag_row(row)
        path = _snapshot_path(cache_dir, circuit, step, bag_name)
        _save_snapshot(path, tensor, legs)
        cached[int(step)] = (tensor, legs)
    return cached


def select_critical_steps(circuit, steps_csv, critical_csv, top_k=10, delta=1.0):
    rows = []
    with open(steps_csv) as f:
        for r in csv.DictReader(f):
            if r.get("circuit") == circuit and r.get("layout_variant", "baseline") == "baseline":
                rows.append(r)
    crit_rows = []
    if os.path.exists(critical_csv):
        with open(critical_csv) as f:
            for r in csv.DictReader(f):
                if r.get("circuit") == circuit and r.get("layout_variant", "baseline") == "baseline":
                    crit_rows.append(r)
    steps = set()
    for key in ("stored_peak_bytes", "peak_bag_bytes"):
        for r in sorted(rows, key=lambda x: float(x.get(key) or 0.0), reverse=True)[:top_k]:
            steps.add(int(float(r["step_id"])))
    b0_rows = [r for r in crit_rows if "B0" in r.get("reason", "") or r.get("peak_bag") == "0"]
    for r in sorted(b0_rows, key=lambda x: float(x.get("stored_peak_bytes") or 0.0), reverse=True)[:top_k]:
        steps.add(int(float(r["step_id"])))
    peak_E = max((float(r.get("peak_bag_E") or 0.0) for r in rows if r.get("peak_bag") == "0"), default=None)
    if peak_E is not None:
        for r in rows:
            if r.get("peak_bag") == "0" and float(r.get("peak_bag_E") or 0.0) >= peak_E - float(delta):
                steps.add(int(float(r["step_id"])))
    return sorted(x for x in steps if x >= 0)


def _rank_from_singular_values(s, tol, abs_tol=0.0):
    if s.size == 0:
        return 1, 0.0
    threshold = max(float(abs_tol), float(tol) * float(s[0]))
    r = max(1, int(np.count_nonzero(s > threshold)))
    discarded = float(np.sum(s[r:] * s[r:]))
    return r, discarded


def apply_topology(tensor, legs, tree, tol):
    leg_map = {l.name: l for l in legs}
    if set(tree["legs"]) != set(leg_map):
        missing = sorted(set(tree["legs"]) - set(leg_map))
        extra = sorted(set(leg_map) - set(tree["legs"]))
        raise ValueError(f"leg_mismatch missing={missing[:8]} extra={extra[:8]}")
    name_to_axis = {l.name: i for i, l in enumerate(legs)}
    ordered = [leg_map[name] for name in tree["legs"]]
    perm = [name_to_axis[name] for name in tree["legs"]]
    tensor = np.transpose(tensor, perm)
    leaves = []
    ranks = []
    discarded = 0.0

    def rec(T, cur_legs, node):
        nonlocal discarded
        if node.get("kind") == "leaf":
            leaves.append((T, cur_legs))
            return T, cur_legs
        split = node["split"]
        A_names = list(split["A_legs"])
        B_names = list(split["B_legs"])
        cur_names = [l.name for l in cur_legs]
        A_idx = [cur_names.index(x) for x in A_names]
        B_idx = [cur_names.index(x) for x in B_names]
        perm2 = A_idx + B_idx
        T2 = np.transpose(T, perm2)
        A_legs = [cur_legs[i] for i in A_idx]
        B_legs = [cur_legs[i] for i in B_idx]
        dim_A = _prod(l.dim for l in A_legs)
        dim_B = _prod(l.dim for l in B_legs)
        M = T2.reshape(dim_A, dim_B)
        U, s, Vh = linalg.svd(M, full_matrices=False, lapack_driver="gesdd")
        r, disc = _rank_from_singular_values(s, tol)
        discarded += disc
        ranks.append(r)
        sqrt_s = np.sqrt(s[:r])
        left_m = U[:, :r] * sqrt_s[None, :]
        right_m = sqrt_s[:, None] * Vh[:r, :]
        bond_name = f"internal:{node['node_id']}"
        left_bond = LegMeta(bond_name, "internal", r, None, bond_name, _log2(r))
        right_bond = LegMeta(bond_name, "internal", r, None, bond_name, _log2(r))
        left = left_m.reshape(tuple(l.dim for l in A_legs) + (r,))
        right = right_m.reshape((r,) + tuple(l.dim for l in B_legs))
        rec(left, A_legs + [left_bond], node["children"][0])
        rec(right, [right_bond] + B_legs, node["children"][1])

    rec(tensor, ordered, tree)

    def reconstruct_leaf_tree(node, leaf_iter):
        if node.get("kind") == "leaf":
            return next(leaf_iter)
        left_t, left_legs = reconstruct_leaf_tree(node["children"][0], leaf_iter)
        right_t, right_legs = reconstruct_leaf_tree(node["children"][1], leaf_iter)
        left_names = [l.name for l in left_legs]
        right_names = [l.name for l in right_legs]
        shared = [x for x in left_names if x in set(right_names)]
        if len(shared) != 1:
            raise RuntimeError(f"expected one shared internal leg at {node['node_id']}, got {shared}")
        sname = shared[0]
        li = left_names.index(sname)
        ri = right_names.index(sname)
        out = np.tensordot(left_t, right_t, axes=([li], [ri]))
        out_legs = [l for i, l in enumerate(left_legs) if i != li]
        out_legs += [l for i, l in enumerate(right_legs) if i != ri]
        target = node["legs"]
        cur = [l.name for l in out_legs]
        return np.transpose(out, [cur.index(x) for x in target]), [leg_map.get(x, l) for x, l in zip(target, out_legs)]

    recon, _ = reconstruct_leaf_tree(tree, iter(leaves))
    err_abs = float(np.linalg.norm((tensor - recon).ravel()))
    norm = float(np.linalg.norm(tensor.ravel()))
    numels = [int(T.size) for T, _ in leaves]
    return dict(
        total_numel=int(sum(numels)),
        peak_numel=int(max(numels, default=0)),
        num_tensors=len(leaves),
        max_rank=max(ranks, default=1),
        tree_depth=max((len(str(node_id)) for node_id in []), default=0),
        recon_error_abs=err_abs,
        recon_error_rel=err_abs / norm if norm else 0.0,
        ranks=ranks,
    )


def _tree_depth_json(node):
    if node.get("kind") == "leaf":
        return 0
    return 1 + max(_tree_depth_json(c) for c in node.get("children", []))


def _write_csv(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _write_report(path, rows):
    ok = [r for r in rows if r["status"] == "ok"]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("# Fixed Topology Reuse Report\n\n")
        f.write(f"- steps tested: `{len(rows)}`\n")
        f.write(f"- successful: `{len(ok)}`\n")
        f.write(f"- failed: `{len(rows) - len(ok)}`\n")
        if ok:
            peaks = [float(r["fixed_topology_peak_log2"]) for r in ok]
            totals = [int(r["fixed_topology_total_bytes"]) for r in ok]
            ratios = [float(r["peak_compression_ratio"]) for r in ok]
            errs = [float(r["reconstruction_error_rel"]) for r in ok]
            f.write(f"- worst fixed peak log2: `{max(peaks):.6f}`\n")
            f.write(f"- worst fixed total bytes: `{max(totals)}`\n")
            f.write(f"- median peak compression ratio: `{sorted(ratios)[len(ratios)//2]:.6g}`\n")
            f.write(f"- min peak compression ratio: `{min(ratios):.6g}`\n")
            f.write(f"- max reconstruction error: `{max(errs):.6g}`\n\n")
        f.write("## Interpretation\n\n")
        if not ok:
            f.write("No selected critical snapshot matched the step-977 topology. Multi-snapshot common-skeleton search is required.\n")
        else:
            failures = len(rows) - len(ok)
            min_ratio = min(float(r["peak_compression_ratio"]) for r in ok)
            if failures:
                f.write(
                    "The topology works only on a subset of selected snapshots. "
                    "This points toward multi-snapshot common-skeleton search or a topology family keyed by active leg set.\n"
                )
            elif min_ratio < 8.0:
                f.write(
                    "The topology is structurally reusable across the selected snapshots, "
                    "but it is not uniformly memory-good. At least one critical step has "
                    "low peak compression, so the next step should be multi-snapshot "
                    "common-skeleton search before any full backend patch.\n"
                )
            else:
                f.write(
                    "The step-977 topology was reusable across the selected critical snapshots. "
                    "It is a plausible B0-subtree layout candidate for the next offline-to-runtime integration step.\n"
                )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--circuit", default="coherent_d5_r5")
    p.add_argument("--bag", default="B0")
    p.add_argument("--tree-json", default=None)
    p.add_argument("--time-steps-csv", default="reports/time_graph_steps.csv")
    p.add_argument("--time-critical-csv", default="reports/time_graph_critical.csv")
    p.add_argument("--snapshot-cache-dir", default="reports/fixed_topology_reuse_rel1e8/snapshots")
    p.add_argument("--out-dir", default="reports/fixed_topology_reuse_rel1e8")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--max-selected-steps", type=int, default=0,
                   help="0 means use all selected critical steps")
    p.add_argument("--steps", nargs="*", type=int, default=None,
                   help="explicit step list; overrides CSV critical-step selection")
    p.add_argument("--delta", type=float, default=1.0)
    p.add_argument("--rel-tol", type=float, default=1e-8)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--runtime-timeout", type=float, default=80.0)
    p.add_argument("--force-refresh", action="store_true")
    args = p.parse_args()

    tree_json = args.tree_json
    if tree_json is None:
        matches = glob.glob("reports/static_rel1e8_beam/static_ttn_b0_compression_tree_*.json")
        if not matches:
            raise FileNotFoundError("no beam tree json found")
        tree_json = sorted(matches)[0]
    tree = json.load(open(tree_json))
    if args.steps:
        steps = sorted(set(int(x) for x in args.steps))
    else:
        steps = select_critical_steps(args.circuit, args.time_steps_csv, args.time_critical_csv, args.top_k, args.delta)
        if int(args.max_selected_steps) > 0:
            steps = steps[:int(args.max_selected_steps)]
    print(f"selected_steps={steps}", flush=True)
    snapshots = capture_snapshots(
        args.circuit,
        steps,
        args.bag,
        args.snapshot_cache_dir,
        args.seed,
        args.runtime_timeout,
        args.force_refresh,
    )
    rows = []
    stats_json = {}
    depth = _tree_depth_json(tree)
    for step in steps:
        t0 = time.perf_counter()
        status = "ok"
        notes = ""
        try:
            if step not in snapshots:
                raise RuntimeError("snapshot_not_captured")
            tensor, legs = snapshots[step]
            result = apply_topology(tensor, legs, tree, args.rel_tol)
            old_numel = int(tensor.size)
            total = int(result["total_numel"])
            peak = int(result["peak_numel"])
            row = dict(
                step=step,
                old_B0_numel=old_numel,
                old_B0_bytes=int(tensor.nbytes),
                old_B0_log2_numel=_log2(old_numel),
                fixed_topology_peak_numel=peak,
                fixed_topology_peak_bytes=peak * 16,
                fixed_topology_peak_log2=_log2(peak),
                fixed_topology_total_numel=total,
                fixed_topology_total_bytes=total * 16,
                fixed_topology_total_log2=_log2(total),
                peak_compression_ratio=old_numel / peak if peak else "",
                total_compression_ratio=old_numel / total if total else "",
                reconstruction_error_abs=result["recon_error_abs"],
                reconstruction_error_rel=result["recon_error_rel"],
                max_internal_rank=int(result["max_rank"]),
                max_internal_rank_log2=_log2(result["max_rank"]),
                num_tensors=int(result["num_tensors"]),
                tree_depth=depth,
                status=status,
                notes=notes,
            )
            stats_json[str(step)] = dict(row, ranks=result["ranks"])
        except Exception as exc:
            row = dict(
                step=step,
                old_B0_numel="",
                old_B0_bytes="",
                old_B0_log2_numel="",
                fixed_topology_peak_numel="",
                fixed_topology_peak_bytes="",
                fixed_topology_peak_log2="",
                fixed_topology_total_numel="",
                fixed_topology_total_bytes="",
                fixed_topology_total_log2="",
                peak_compression_ratio="",
                total_compression_ratio="",
                reconstruction_error_abs="",
                reconstruction_error_rel="",
                max_internal_rank="",
                max_internal_rank_log2="",
                num_tensors="",
                tree_depth=depth,
                status="error",
                notes=repr(exc),
            )
            stats_json[str(step)] = dict(row)
        row["elapsed_s"] = time.perf_counter() - t0
        rows.append(row)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "reuse_summary.csv", rows)
    with open(out / "reuse_per_step_tree_stats.json", "w") as f:
        json.dump(stats_json, f, indent=2)
    _write_report(out / "reuse_report.md", rows)
    print(f"wrote {out / 'reuse_summary.csv'}")
    ok = [r for r in rows if r["status"] == "ok"]
    print(f"steps={len(rows)} ok={len(ok)} failed={len(rows)-len(ok)}")
    if ok:
        print(f"worst_peak_log2={max(float(r['fixed_topology_peak_log2']) for r in ok):.3f}")


if __name__ == "__main__":
    main()
