"""Execute Clifft bytecode on a temporal-carving TTN layout.

This script is intentionally different from qec_temporal_carving_report.py.
The report script evaluates the temporal-carving objective as a compile-time
proxy.  This script converts the carving tree into an executable TTN bag tree
and runs the existing TTNBackend bytecode dispatcher, collecting actual tensor
and bond metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, ".")

import clifft

from temporal_carving.cost import CostModel
from temporal_carving.io import save_tree
from temporal_carving.pipeline import run as run_pipeline
from temporal_carving.tree import TreeNode
from ttn_backend import TTNBackend
from ttn_backend.backend_spec import assign_homes_and_classify, export_backend_spec
from ttn_backend.scripts.qec_temporal_carving_report import trace_from_program


SUMMARY_FIELDS = [
    "circuit",
    "mode",
    "status",
    "timeout",
    "steps_completed",
    "total_steps",
    "elapsed_s",
    "n_axes",
    "n_bags",
    "tree_depth",
    "flat_peak_k",
    "temporal_proxy_peak_E",
    "resident_actual_peak_log2_numel",
    "resident_actual_peak_bytes",
    "peak_stored_bytes",
    "workspace_actual_peak_bytes",
    "workspace_actual_peak_log2_numel",
    "actual_total_peak_bytes",
    "actual_total_peak_log2_numel",
    "actual_total_peak_step",
    "actual_total_peak_kind",
    "destructive_total_peak_bytes",
    "destructive_total_peak_log2_numel",
    "destructive_total_peak_step",
    "destructive_total_peak_kind",
    "destructive_total_peak_debug",
    "max_bond_dim_observed",
    "n_qr",
    "n_transports",
    "num_refactor",
    "sum_path_length",
    "sum_rank_weighted_path_length",
    "qr_work_proxy",
    "multicnot_region_fused",
    "multicnot_region_controls",
    "multicnot_region_fallback",
    "multicnot_region_workspace_peak_bytes",
    "persistent_multicnot_windows",
    "persistent_multicnot_steps",
    "persistent_multicnot_controls",
    "multicnot_region_batches",
    "peak_offender_bag",
    "peak_offender_step",
    "peak_offender_shape",
    "notes",
]


PROFILE_FIELDS = [
    "circuit",
    "mode",
    "step_id",
    "op_kind",
    "resident_actual_peak_log2_numel",
    "resident_actual_peak_bytes",
    "peak_offender_bag",
    "peak_offender_shape",
    "peak_offender_p_B",
    "peak_offender_incident_bond_dims",
]


def _load_prog(name: str):
    path = Path("qec_bench/circuits") / f"{name}.stim"
    with open(path) as f:
        return clifft.compile(f.read())


def _tree_depth(node: TreeNode) -> int:
    if node.is_leaf:
        return 0
    return 1 + max(_tree_depth(node.left), _tree_depth(node.right))


def _carving_bag_tree(tree: TreeNode, couse=None, group_min: int = 0):
    """Return (bags, bag_edges, leaf_home) for a pure binary carving tree.

    Each carving leaf becomes a TTN bag owning exactly one active ident.
    Internal carving nodes become TTN bags owning no idents.

    Grouping (Task 4): when `group_min > 0`, a sibling pair of leaves whose two
    idents are co-used in >= group_min two-axis ops is merged into one bag owning
    both idents (p_B = 2). Their gates then classify as Class A (local, no
    transport). Only leaf pairs are merged, so p_B stays <= 2 (cap-safe); this
    reduces transport/QR, it does not shrink internal hub bonds.
    """
    bags: list[list[int]] = []
    edges: list[tuple[int, int, list[int]]] = []
    leaf_home: dict[int, int] = {}

    def visit(node: TreeNode, parent: int | None = None) -> int:
        bid = len(bags)
        if (group_min and not node.is_leaf
                and node.left.is_leaf and node.right.is_leaf):
            a = int(node.left.axis); b = int(node.right.axis)
            key = (min(a, b), max(a, b))
            if couse and int(couse.get(key, 0)) >= int(group_min):
                bags.append([a, b])
                leaf_home[a] = bid
                leaf_home[b] = bid
                if parent is not None:
                    edges.append((parent, bid, []))
                return bid
        if node.is_leaf:
            axis = int(node.axis)
            bags.append([axis])
            leaf_home[axis] = bid
        else:
            bags.append([])
        if parent is not None:
            edges.append((parent, bid, []))
        if not node.is_leaf:
            visit(node.left, bid)
            visit(node.right, bid)
        return bid

    visit(tree)
    return bags, edges, leaf_home


def _bag_adj(n_bags: int, bag_edges):
    adj = {i: [] for i in range(n_bags)}
    for i, j, _ in bag_edges:
        adj[int(i)].append(int(j))
        adj[int(j)].append(int(i))
    return adj


def _tree_path(adj, src: int, dst: int) -> list[int]:
    if src == dst:
        return [src]
    parent = {src: None}
    q = deque([src])
    while q:
        u = q.popleft()
        if u == dst:
            break
        for v in adj[u]:
            if v not in parent:
                parent[v] = u
                q.append(v)
    if dst not in parent:
        raise ValueError(f"no path between bags {src} and {dst}")
    out = []
    cur = dst
    while cur is not None:
        out.append(cur)
        cur = parent[cur]
    return list(reversed(out))


def build_carving_executable_spec(base_spec, tree: TreeNode):
    group_min = int(os.environ.get("TTN_GROUP_COUSE_MIN", "0"))
    couse = {}
    if group_min:
        for r in base_spec["op_to_bag"]:
            if r.get("kind") == "two":
                u, v = (int(x) for x in r["axes"])
                key = (min(u, v), max(u, v))
                couse[key] = couse.get(key, 0) + 1
    bags, bag_edges, leaf_home = _carving_bag_tree(tree, couse, group_min)

    # Optional static clustering for small MULTI_CNOT supports.  This is a
    # general layout repair knob: it clusters target/control axes of high-value
    # MULTI_CNOT windows into the target's home bag to reduce high-rank edge
    # crossings.  It is deliberately disabled by default because it trades lower
    # transport work against larger local p_B.
    cluster_top = int(os.environ.get("TTN_CLUSTER_MULTICNOT_TOP", "0"))
    if cluster_top > 0:
        min_controls = int(os.environ.get("TTN_CLUSTER_MULTICNOT_MIN_CONTROLS", "3"))
        max_support = int(os.environ.get("TTN_CLUSTER_MULTICNOT_MAX_SUPPORT", "5"))
        max_bag_own = int(os.environ.get("TTN_CLUSTER_MULTICNOT_MAX_BAG_OWN", "6"))
        groups = {}
        for r in base_spec["op_to_bag"]:
            if r.get("kind") != "two" or r.get("op") != "OP_ARRAY_MULTI_CNOT":
                continue
            axes = tuple(map(int, r["axes"]))
            if len(axes) != 2:
                continue
            # backend_spec records MULTI_CNOT pairs as (target, control), matching
            # Clifft's axis_1 target convention.
            target, ctrl = axes
            g = groups.setdefault(int(r["step"]), dict(target=target, controls=set()))
            g["target"] = target
            g["controls"].add(ctrl)
        candidates = []
        for step, g in groups.items():
            support = set(g["controls"]) | {int(g["target"])}
            if len(g["controls"]) < min_controls:
                continue
            if len(support) > max_support:
                continue
            candidates.append((len(g["controls"]), step, int(g["target"]), support))
        candidates.sort(reverse=True)
        moved = set()
        for _, step, target, support in candidates[:cluster_top]:
            if target not in leaf_home:
                continue
            dst = int(leaf_home[target])
            new_axes = [int(x) for x in sorted(support) if int(leaf_home.get(int(x), -1)) != dst]
            if not new_axes:
                continue
            if len(set(bags[dst]) | set(new_axes)) > max_bag_own:
                continue
            # Avoid repeatedly moving the same ident through conflicting windows.
            if any(x in moved for x in new_axes):
                continue
            for ident in new_axes:
                old = int(leaf_home[ident])
                if ident in bags[old]:
                    bags[old].remove(ident)
                if ident not in bags[dst]:
                    bags[dst].append(ident)
                    bags[dst].sort()
                leaf_home[ident] = dst
                moved.add(ident)

    n_bags = len(bags)
    adj = _bag_adj(n_bags, bag_edges)

    missing = sorted(set(map(int, base_spec["lifecycle"])) - set(leaf_home))
    if missing:
        raise ValueError(f"carving tree is missing {len(missing)} identities: {missing[:10]}")

    spec = dict(base_spec)
    spec["union"] = dict(base_spec["union"])
    spec["union"].update(
        n_ids=len(base_spec["lifecycle"]),
        n_bags=n_bags,
        bags=[list(b) for b in bags],
        bag_edges=[(int(i), int(j), list(s)) for i, j, s in bag_edges],
        tau=1,
        max_bag=max((len(b) for b in bags), default=0),
        max_sep=0,
        sum2=sum(2 ** len(b) for b in bags),
        shape="binary_carving_leaf_home",
    )
    spec["invariants"] = dict(base_spec.get("invariants", {}))
    spec["invariants"]["two_axis_coverage"] = False
    spec["invariants"]["layout_kind"] = "binary_carving_leaf_home"

    home = {int(ident): int(leaf_home[int(ident)]) for ident in base_spec["lifecycle"]}
    owned_phys = {bid: [] for bid in range(n_bags)}
    for ident, bid in home.items():
        owned_phys[bid].append(ident)
    for bid in owned_phys:
        owned_phys[bid].sort()

    op_classes = []
    path_lens = []
    for r in base_spec["op_to_bag"]:
        axes = tuple(map(int, r["axes"]))
        if r["kind"] == "single":
            ident = axes[0]
            h = home[ident]
            op_classes.append(dict(
                step=int(r["step"]),
                op=r["op"],
                kind="single",
                cls="-",
                axes=axes,
                home=h,
                compute_bag=h,
                path_bags=None,
                path_len=None,
                refactor_cost=None,
            ))
            continue

        u, v = axes
        hu = home[u]
        hv = home[v]
        if hu == hv:
            cls = "A"
            path = [hu]
        else:
            cls = "C"
            path = _tree_path(adj, hu, hv)
            path_lens.append(len(path) - 1)
        op_classes.append(dict(
            step=int(r["step"]),
            op=r["op"],
            kind="two",
            axes=(u, v),
            home_u=hu,
            home_v=hv,
            compute_bag=hv,
            cls=cls,
            path_bags=path,
            path_len=(len(path) - 1),
            refactor_cost=len(path),
        ))

    n_two = sum(1 for r in op_classes if r["kind"] == "two")
    homing = dict(
        home=home,
        owned_phys=owned_phys,
        op_classes=op_classes,
        stats=dict(
            n_two_axis=n_two,
            n_A=sum(1 for r in op_classes if r.get("cls") == "A"),
            n_B=0,
            n_C=sum(1 for r in op_classes if r.get("cls") == "C"),
            pctA=0.0,
            pctB=0.0,
            pctC=100.0 if n_two else 0.0,
            avg_path_len=(sum(path_lens) / len(path_lens)) if path_lens else 0.0,
            max_path_len=max(path_lens) if path_lens else 0,
            max_refactor_cost=max(path_lens) if path_lens else 0,
            sum_refactor_cost=sum(path_lens),
        ),
    )
    return spec, homing


def build_leaf_home_homing(spec, home):
    """Build path-based homing/classification for an executable TTN tree.

    Unlike assign_homes_and_classify(), this does not require a junction-tree
    compute bag containing both operands.  It classifies every two-axis op by
    the path between the static homes of the two identities.
    """
    n_bags = int(spec["union"]["n_bags"])
    adj = _bag_adj(n_bags, spec["union"]["bag_edges"])
    owned_phys = {bid: [] for bid in range(n_bags)}
    for ident, bid in home.items():
        owned_phys[int(bid)].append(int(ident))
    for bid in owned_phys:
        owned_phys[bid].sort()

    op_classes = []
    path_lens = []
    for r in spec["op_to_bag"]:
        axes = tuple(map(int, r["axes"]))
        if r["kind"] == "single":
            ident = axes[0]
            h = int(home[ident])
            op_classes.append(dict(
                step=int(r["step"]),
                op=r["op"],
                kind="single",
                cls="-",
                axes=axes,
                home=h,
                compute_bag=h,
                path_bags=None,
                path_len=None,
                refactor_cost=None,
            ))
            continue

        u, v = axes
        hu = int(home[u])
        hv = int(home[v])
        if hu == hv:
            cls = "A"
            path = [hu]
        else:
            cls = "C"
            path = _tree_path(adj, hu, hv)
            path_lens.append(len(path) - 1)
        op_classes.append(dict(
            step=int(r["step"]),
            op=r["op"],
            kind="two",
            axes=(u, v),
            home_u=hu,
            home_v=hv,
            compute_bag=hv,
            cls=cls,
            path_bags=path,
            path_len=(len(path) - 1),
            refactor_cost=len(path),
        ))

    n_two = sum(1 for r in op_classes if r["kind"] == "two")
    return dict(
        home=dict(home),
        owned_phys=owned_phys,
        op_classes=op_classes,
        stats=dict(
            n_two_axis=n_two,
            n_A=sum(1 for r in op_classes if r.get("cls") == "A"),
            n_B=0,
            n_C=sum(1 for r in op_classes if r.get("cls") == "C"),
            pctA=0.0,
            pctB=0.0,
            pctC=100.0 if n_two else 0.0,
            avg_path_len=(sum(path_lens) / len(path_lens)) if path_lens else 0.0,
            max_path_len=max(path_lens) if path_lens else 0,
            max_refactor_cost=max(path_lens) if path_lens else 0,
            sum_refactor_cost=sum(path_lens),
        ),
    )


def split_executable_bag(spec, homing, bag_id, left_neighbors, right_neighbors):
    """Split one executable TTN bag by distributing incident edges.

    The original bag remains the left bag.  A new right bag is appended and
    connected to the original.  Edges to right_neighbors are moved to the new
    bag.  Homes remain unchanged; if an ident was originally homed at bag_id it
    stays on the left side.
    """
    bag_id = int(bag_id)
    left_neighbors = {int(x) for x in left_neighbors}
    right_neighbors = {int(x) for x in right_neighbors}
    n_old = int(spec["union"]["n_bags"])
    new_id = n_old
    old_edges = [(int(i), int(j), list(s)) for i, j, s in spec["union"]["bag_edges"]]
    incident = {
        j if i == bag_id else i
        for i, j, _ in old_edges
        if i == bag_id or j == bag_id
    }
    if right_neighbors - incident:
        raise ValueError(f"right_neighbors not incident to B{bag_id}: {sorted(right_neighbors - incident)}")
    if left_neighbors - incident:
        raise ValueError(f"left_neighbors not incident to B{bag_id}: {sorted(left_neighbors - incident)}")
    if not right_neighbors:
        raise ValueError("right_neighbors must be non-empty")

    new_edges = []
    for i, j, sep in old_edges:
        if i == bag_id and j in right_neighbors:
            new_edges.append((new_id, j, sep))
        elif j == bag_id and i in right_neighbors:
            new_edges.append((i, new_id, sep))
        else:
            new_edges.append((i, j, sep))
    new_edges.append((bag_id, new_id, []))

    new_spec = dict(spec)
    new_spec["union"] = dict(spec["union"])
    new_bags = [list(b) for b in spec["union"]["bags"]]
    new_bags.append([])
    new_spec["union"].update(
        n_bags=n_old + 1,
        bags=new_bags,
        bag_edges=new_edges,
        max_bag=max((len(b) for b in new_bags), default=0),
        max_sep=max((len(s) for _, _, s in new_edges), default=0),
        sum2=sum(2 ** len(b) for b in new_bags),
        shape=str(spec["union"].get("shape", "")) + f"+split_B{bag_id}",
    )
    new_homing = build_leaf_home_homing(new_spec, homing["home"])
    return new_spec, new_homing


def _summarize_metrics(circuit, mode, status, spec, tree_depth, flat_peak, proxy_peak, metrics, notes=""):
    shape = metrics.get("actual_peak_offender_shape")
    return dict(
        circuit=circuit,
        mode=mode,
        status=status,
        timeout=bool(metrics.get("timeout", False)),
        steps_completed=int(metrics.get("steps_completed", 0)),
        total_steps=int(metrics.get("total_steps", 0)),
        elapsed_s=float(metrics.get("elapsed_time_seconds", 0.0)),
        n_axes=int(spec["union"].get("n_ids", len(spec.get("lifecycle", {})))),
        n_bags=int(spec["union"]["n_bags"]),
        tree_depth=tree_depth,
        flat_peak_k=float(flat_peak),
        temporal_proxy_peak_E="" if proxy_peak is None else float(proxy_peak),
        resident_actual_peak_log2_numel=metrics.get("resident_actual_peak_log2_numel"),
        resident_actual_peak_bytes=metrics.get("resident_actual_peak_bytes"),
        peak_stored_bytes=metrics.get("peak_stored_bytes"),
        workspace_actual_peak_bytes=metrics.get("workspace_actual_peak_bytes"),
        workspace_actual_peak_log2_numel=metrics.get("workspace_actual_peak_log2_numel"),
        actual_total_peak_bytes=metrics.get("actual_total_peak_bytes"),
        actual_total_peak_log2_numel=metrics.get("actual_total_peak_log2_numel"),
        actual_total_peak_step=metrics.get("actual_total_peak_step"),
        actual_total_peak_kind=metrics.get("actual_total_peak_kind"),
        destructive_total_peak_bytes=metrics.get("destructive_total_peak_bytes"),
        destructive_total_peak_log2_numel=metrics.get("destructive_total_peak_log2_numel"),
        destructive_total_peak_step=metrics.get("destructive_total_peak_step"),
        destructive_total_peak_kind=metrics.get("destructive_total_peak_kind"),
        destructive_total_peak_debug=json.dumps(metrics.get("destructive_total_peak_debug", {})),
        max_bond_dim_observed=metrics.get("max_bond_dim_observed"),
        n_qr=metrics.get("n_qr"),
        n_transports=metrics.get("n_transports"),
        num_refactor=metrics.get("num_refactor"),
        sum_path_length=metrics.get("sum_path_length"),
        sum_rank_weighted_path_length=metrics.get("sum_rank_weighted_path_length"),
        qr_work_proxy=metrics.get("qr_work_proxy"),
        multicnot_region_fused=metrics.get("multicnot_region_fused"),
        multicnot_region_controls=metrics.get("multicnot_region_controls"),
        multicnot_region_fallback=metrics.get("multicnot_region_fallback"),
        multicnot_region_workspace_peak_bytes=metrics.get("multicnot_region_workspace_peak_bytes"),
        persistent_multicnot_windows=metrics.get("persistent_multicnot_windows"),
        persistent_multicnot_steps=metrics.get("persistent_multicnot_steps"),
        persistent_multicnot_controls=metrics.get("persistent_multicnot_controls"),
        multicnot_region_batches=metrics.get("multicnot_region_batches"),
        peak_offender_bag=metrics.get("actual_peak_offender_bag"),
        peak_offender_step=metrics.get("actual_peak_offender_step"),
        peak_offender_shape=json.dumps(shape),
        notes=notes,
    )


def _write_step_profile(path, circuit, mode, metrics):
    rows = []
    for key, r in sorted(
        metrics.get("actual_step_peaks", {}).items(),
        key=lambda kv: -1 if kv[0] == "init" else int(kv[0]),
    ):
        rows.append(dict(
            circuit=circuit,
            mode=mode,
            step_id=r.get("step_id"),
            op_kind=r.get("op_kind"),
            resident_actual_peak_log2_numel=r.get("resident_actual_peak_log2_numel"),
            resident_actual_peak_bytes=r.get("resident_actual_peak_bytes"),
            peak_offender_bag=r.get("peak_offender_bag"),
            peak_offender_shape=json.dumps(r.get("peak_offender_shape")),
            peak_offender_p_B=r.get("peak_offender_p_B"),
            peak_offender_incident_bond_dims=json.dumps(r.get("peak_offender_incident_bond_dims")),
        ))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PROFILE_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return rows


def _run_backend(prog, spec, homing, seed, timeout, max_steps=None):
    backend = TTNBackend(spec, homing)
    backend.run_shot(
        prog,
        seed=seed,
        runtime_timeout=timeout,
        check_interval=1,
        max_steps=max_steps,
    )
    return backend.last_metrics or {}


def run_one(circuit: str, args):
    prog = _load_prog(circuit)
    t0 = time.perf_counter()
    base_spec = export_backend_spec(prog, strict=False)
    trace = trace_from_program(prog, strict=False)
    flat_peak = max((len(trace.live_sets.get(t, ())) for t in trace.timeline), default=0)
    carving_result = run_pipeline(
        trace,
        seeder=args.seeder,
        refine_moves=tuple(x for x in args.refine.split(",") if x and x != "none"),
        seed=args.seed,
        partitioner=args.partitioner,
        exact=False,
    )
    tree = carving_result["tree"]
    proxy_peak = float(CostModel(trace).tree_peak(tree))
    carve_spec, carve_homing = build_carving_executable_spec(base_spec, tree)
    split_spec = split_homing = None
    if args.enable_peak_split:
        left = [int(x) for x in args.peak_split_left.split(",") if x != ""]
        right = [int(x) for x in args.peak_split_right.split(",") if x != ""]
        split_spec, split_homing = split_executable_bag(
            carve_spec,
            carve_homing,
            args.peak_split_bag,
            left,
            right,
        )
    base_homing = assign_homes_and_classify(base_spec)

    out_dir = Path(args.out_dir) / circuit
    out_dir.mkdir(parents=True, exist_ok=True)
    save_tree(tree, out_dir / "carving_tree.json")

    rows = []
    summaries = {}
    run_specs = [
        ("baseline_jt", base_spec, base_homing, "", None),
        ("carving_leaf", carve_spec, carve_homing, _tree_depth(tree), proxy_peak),
    ]
    if split_spec is not None:
        run_specs.append((
            "carving_peak_split",
            split_spec,
            split_homing,
            _tree_depth(tree) + 1,
            proxy_peak,
        ))

    requested_modes = None
    if args.modes:
        requested_modes = {x.strip() for x in args.modes.split(",") if x.strip()}

    for mode, spec, homing, depth, proxy in run_specs:
        if requested_modes is not None and mode not in requested_modes:
            continue
        print(f"  executing {circuit} {mode}", flush=True)
        try:
            metrics = _run_backend(
                prog,
                spec,
                homing,
                args.shot_seed,
                args.runtime_timeout,
                max_steps=args.max_steps,
            )
            status = "ok"
            notes = ""
        except Exception as exc:
            metrics = dict(
                timeout=False,
                steps_completed=0,
                total_steps=len(prog),
                elapsed_time_seconds=0.0,
            )
            status = "error"
            notes = repr(exc)
        summaries[mode] = metrics
        rows.append(_summarize_metrics(
            circuit, mode, status, spec, depth, flat_peak, proxy, metrics, notes=notes))
        _write_step_profile(out_dir / f"{mode}_actual_step_profile.csv", circuit, mode, metrics)
        with open(out_dir / f"{mode}_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2, default=str)

    elapsed = time.perf_counter() - t0
    return rows, dict(
        circuit=circuit,
        elapsed_s=elapsed,
        carving_tree_path=str(out_dir / "carving_tree.json"),
        baseline_metrics_path=str(out_dir / "baseline_jt_metrics.json"),
        carving_metrics_path=str(out_dir / "carving_leaf_metrics.json"),
    )


def write_report(path, rows):
    with open(path, "w") as f:
        f.write("# QEC Temporal Carving Runtime Actual Report\n\n")
        f.write("이 리포트는 temporal carving tree를 기존 `TTNBackend`의 실행 가능한 binary bag tree로 변환한 뒤, Clifft bytecode를 직접 실행해서 actual tensor/bond metric을 측정한다. `temporal_proxy_peak_E`는 layout 생성용 참고값이며, memory 결론은 actual fields만 사용한다.\n\n")
        f.write("| circuit | mode | status | timeout | steps | actual peak log2 numel | peak stored bytes | workspace bytes | max bond | QR | transports |\n")
        f.write("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(
                f"| {r['circuit']} | {r['mode']} | {r['status']} | {r['timeout']} | "
                f"{r['steps_completed']}/{r['total_steps']} | "
                f"{r['resident_actual_peak_log2_numel']} | {r['peak_stored_bytes']} | "
                f"{r['workspace_actual_peak_bytes']} | {r['max_bond_dim_observed']} | "
                f"{r['n_qr']} | {r['n_transports']} |\n"
            )
        f.write("\n## 해석 기준\n\n")
        f.write("- `baseline_jt`: 기존 junction-tree layout + 기존 homing/classification.\n")
        f.write("- `carving_leaf`: temporal carving tree의 leaf에 active ident를 배치하고, 모든 2축 active op를 leaf-home path transport로 실행.\n")
        f.write("- `resident_actual_peak_*`, `peak_stored_bytes`, `workspace_actual_peak_bytes`, `max_bond_dim_observed`, `n_qr`는 실제 TTN tensor 실행에서 나온 값이다.\n")
        f.write("- `temporal_proxy_peak_E`는 실행 결과가 아니라 layout seeding objective 값이므로 actual memory와 섞어 해석하면 안 된다.\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("circuits", nargs="*", default=["coherent_d5_r1"])
    p.add_argument("--seeder", default="recursive_balanced_mincut")
    p.add_argument("--refine", default="nni")
    p.add_argument("--partitioner", default="networkx")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shot-seed", type=int, default=42)
    p.add_argument("--runtime-timeout", type=float, default=60.0)
    p.add_argument("--max-steps", type=int, default=None,
                   help="stop after this many bytecode steps for fixed-prefix comparisons")
    p.add_argument("--modes", default="",
                   help="comma-separated subset: baseline_jt,carving_leaf,carving_peak_split")
    p.add_argument("--enable-peak-split", action="store_true",
                   help="also run a local static split of one carving bag")
    p.add_argument("--peak-split-bag", type=int, default=72)
    p.add_argument("--peak-split-left", default="0",
                   help="comma-separated old neighbors kept on the original bag")
    p.add_argument("--peak-split-right", default="73,108",
                   help="comma-separated old neighbors moved to the new bag")
    p.add_argument("--out-dir", default="reports/qec_temporal_carving_runtime")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    meta = []
    for circuit in args.circuits:
        print(f"running {circuit}", flush=True)
        r, m = run_one(circuit, args)
        rows.extend(r)
        meta.append(m)
        for row in r:
            print(
                f"  {row['mode']}: status={row['status']} timeout={row['timeout']} "
                f"steps={row['steps_completed']}/{row['total_steps']} "
                f"actual_peak_log2={row['resident_actual_peak_log2_numel']} "
                f"stored={row['peak_stored_bytes']} ws={row['workspace_actual_peak_bytes']} "
                f"qr={row['n_qr']}",
                flush=True,
            )

    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        w.writerows(rows)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(dict(rows=rows, meta=meta), f, indent=2, default=str)
    write_report(out_dir / "report.md", rows)
    print(f"wrote {out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
