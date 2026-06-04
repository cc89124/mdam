"""RASL v1 analysis/profiling pass.

This does not rewrite Clifft bytecode. It identifies active-only localization
windows already present in Clifft bytecode, generates bounded alternative
symplectic routing candidates, scores them on the fixed TTN layout, and writes
CSV/JSON reports. Phase tracking is intentionally absent in v1, so choices are
not emitted as executable bytecode.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

sys.path.insert(0, ".")

import clifft

from ttn_backend import treewidth as T
from ttn_backend.backend_spec import assign_homes_and_classify, export_backend_spec
from ttn_backend.rasl.builders import active_z_route_candidates, default_candidate, symplectic_greedy_candidates
from ttn_backend.rasl.candidate import CliffordOp
from ttn_backend.rasl.cost import LayoutCost
from ttn_backend.rasl.select import choose_candidate
from ttn_backend.rasl.symplectic import PauliVec


DEFAULT_CIRCUITS = [
    "distillation",
    "cultivation_d3",
    "coherent_d3_r1",
    "coherent_d5_r1",
    "coherent_d5_r5",
]

STEP_FIELDS = [
    "circuit",
    "step_id",
    "op_kind",
    "support_size",
    "support_axes",
    "active_only",
    "has_dormant",
    "default_target",
    "chosen_target",
    "builder_kind",
    "candidate_count",
    "valid_candidate_count",
    "default_path_cost",
    "chosen_path_cost",
    "default_workspace_proxy_score",
    "chosen_workspace_proxy_score",
    "default_workspace_actual_peak_bytes",
    "chosen_workspace_actual_peak_bytes",
    "default_resident_bound_proxy",
    "chosen_resident_bound_proxy",
    "default_resident_actual_peak_numel",
    "chosen_resident_actual_peak_numel",
    "default_resident_actual_peak_log2_numel",
    "chosen_resident_actual_peak_log2_numel",
    "default_resident_actual_peak_bytes",
    "chosen_resident_actual_peak_bytes",
    "default_peak_offender_bag",
    "chosen_peak_offender_bag",
    "default_peak_offender_p_B",
    "chosen_peak_offender_p_B",
    "default_peak_offender_incident_bonds",
    "chosen_peak_offender_incident_bonds",
    "default_peak_offender_bond_dims",
    "chosen_peak_offender_bond_dims",
    "accepted",
    "reject_reason",
    "num_2q_ops_default",
    "num_2q_ops_chosen",
    "path_length_default",
    "path_length_chosen",
    "default_v_sequence",
    "chosen_v_sequence",
    "default_edge_hits",
    "chosen_edge_hits",
    "reduced_edges",
    "resident_proxy_reason",
    "refactor_proxy_delta",
]


def _load_prog(name):
    with open(os.path.join("qec_bench/circuits", name + ".stim")) as f:
        return clifft.compile(f.read())


def _bits(mask):
    while mask:
        low = mask & -mask
        yield low.bit_length() - 1
        mask ^= low


def _mapped_z_pauli(support):
    n = max(support) + 1 if support else 1
    p = PauliVec.zeros(n)
    for q in support:
        p.z[q] = True
    return p


def _path_len(layout: LayoutCost, axis_to_ident, cand) -> int:
    total = 0
    for op in cand.ops:
        if not op.is_2q() or op.a not in axis_to_ident or op.b not in axis_to_ident:
            continue
        hu = layout.home[axis_to_ident[op.a]]
        hv = layout.home[axis_to_ident[op.b]]
        path = layout.tree_path_bags(hu, hv)
        total += max(0, len(path) - 1)
    return total


def _op_sequence(cand) -> str:
    parts = []
    for op in cand.ops:
        if op.is_2q():
            parts.append(f"{op.name}({op.a},{op.b})")
        else:
            parts.append(f"{op.name}({op.a})")
    return " ".join(parts)


def _edge_hit_counts(layout: LayoutCost, axis_to_ident, cand):
    counts = {}
    for op in cand.ops:
        if not op.is_2q() or op.a not in axis_to_ident or op.b not in axis_to_ident:
            continue
        hu = layout.home[axis_to_ident[op.a]]
        hv = layout.home[axis_to_ident[op.b]]
        path = layout.tree_path_bags(hu, hv)
        for a, b in zip(path, path[1:]):
            key = tuple(sorted((a, b)))
            counts[key] = counts.get(key, 0) + 1
    return counts


def _edge_count_string(counts):
    return " ".join(f"{a}-{b}:{n}" for (a, b), n in sorted(counts.items()))


def _reduced_edges(default_counts, chosen_counts):
    rows = []
    for edge, n0 in sorted(default_counts.items()):
        n1 = chosen_counts.get(edge, 0)
        if n1 < n0:
            rows.append(f"{edge[0]}-{edge[1]}:{n0}->{n1}")
    return " ".join(rows)


def _score_all(layout, axis_to_ident, default, candidates):
    layout.score_candidate(default, axis_to_ident)
    for c in candidates:
        if c.valid:
            layout.score_candidate(c, axis_to_ident, default.proxy_resident_bound)


def _candidate_step_from_inst(step, name, inst, slot2id):
    a1 = int(inst.axis_1)
    a2 = int(inst.axis_2)
    if name == "OP_ARRAY_CNOT":
        if a1 not in slot2id or a2 not in slot2id:
            return None
        support = sorted({a1, a2})
        ops = [CliffordOp("CNOT", a1, a2)]
        return dict(step=step, op_kind=name, support=support, target=a2, ops=ops)
    if name == "OP_ARRAY_MULTI_CNOT":
        d = inst.as_dict()
        ctrls = [q for q in _bits(int(d.get("mask", 0))) if q in slot2id and q != a1]
        if a1 not in slot2id or not ctrls:
            return None
        support = sorted(set(ctrls + [a1]))
        ops = [CliffordOp("CNOT", q, a1) for q in ctrls]
        return dict(step=step, op_kind=name, support=support, target=a1, ops=ops)
    return None


def _update_active_mapping(name, inst, slot2id, next_ident):
    a1 = int(inst.axis_1)
    a2 = int(inst.axis_2)
    if name in T.NODE_ADD:
        if a1 not in slot2id:
            slot2id[a1] = next_ident
            next_ident += 1
    elif name in T.SWAP_PAIR:
        i1 = slot2id.get(a1)
        i2 = slot2id.get(a2)
        if i1 is not None:
            del slot2id[a1]
        if i2 is not None:
            del slot2id[a2]
        if i1 is not None:
            slot2id[a2] = i1
        if i2 is not None:
            slot2id[a1] = i2
    elif name in T.NODE_DEL:
        slot2id.pop(a1, None)
    elif name in T.SWAP_MEAS:
        i_from = slot2id.get(a1)
        i_to = slot2id.get(a2)
        if i_from is None:
            pass
        elif i_to is None:
            slot2id.pop(a1, None)
        else:
            slot2id[a1] = i_to
            slot2id.pop(a2, None)
    return next_ident


def analyze_program(name, prog, max_steps=200, max_support=10, top_k=32, builder="full"):
    t0 = time.perf_counter()
    spec = export_backend_spec(prog, strict=False)
    homing = assign_homes_and_classify(spec)
    layout = LayoutCost(spec, homing)
    compile_default_s = time.perf_counter() - t0

    rows = []
    decisions = []
    slot2id = {}
    next_ident = 0
    considered = 0
    rejected_invalid = 0
    rejected_resident = 0
    default_workspace_peak = 0.0
    chosen_workspace_peak = 0.0
    default_refactor = 0.0
    chosen_refactor = 0.0
    default_cross_ops = 0
    chosen_cross_ops = 0

    t1 = time.perf_counter()
    for step in range(len(prog)):
        inst = prog[step]
        opname = T._opname(inst.opcode)
        info = _candidate_step_from_inst(step, opname, inst, slot2id)
        if info is not None and considered < max_steps and len(info["support"]) <= max_support:
            axis_to_ident = {axis: slot2id[axis] for axis in info["support"] if axis in slot2id}
            mapped = _mapped_z_pauli(info["support"])
            default = default_candidate(step, mapped, info["ops"], info["target"])
            active_axes = set(slot2id)
            candidates = []
            if builder in ("active_z", "full"):
                candidates.extend(active_z_route_candidates(
                    step, mapped, active_axes, max_support=max_support))
            if builder in ("symplectic", "full"):
                candidates.extend(symplectic_greedy_candidates(
                    step, mapped, active_axes, max_support=max_support))
            candidates = candidates[:top_k]
            _score_all(layout, axis_to_ident, default, candidates)
            valid = [c for c in candidates if c.valid]
            rejected_invalid += len(candidates) - len(valid)
            chosen = choose_candidate(default, valid)
            if chosen.proxy_resident_bound > default.proxy_resident_bound:
                rejected_resident += 1
                chosen = default
            accepted = chosen is not default and (
                chosen.proxy_resident_bound <= default.proxy_resident_bound and
                chosen.refactor_cost <= default.refactor_cost and
                chosen.proxy_path_cost <= default.proxy_path_cost and
                (
                    chosen.proxy_workspace < default.proxy_workspace or
                    chosen.refactor_cost < default.refactor_cost or
                    chosen.proxy_path_cost < default.proxy_path_cost
                )
            )
            if not accepted:
                chosen = default
            considered += 1

            default_workspace_peak = max(default_workspace_peak, default.proxy_workspace)
            chosen_workspace_peak = max(chosen_workspace_peak, chosen.proxy_workspace)
            default_refactor += default.refactor_cost
            chosen_refactor += chosen.refactor_cost
            default_cross_ops += default.num_2q_ops()
            chosen_cross_ops += chosen.num_2q_ops()
            if accepted:
                decisions.append(dict(step=step, from_target=default.target_axis,
                                      to_target=chosen.target_axis,
                                      kind=chosen.kind,
                                      default_path=default.proxy_path_cost,
                                      chosen_path=chosen.proxy_path_cost))

            rows.append(dict(
                circuit=name,
                step_id=step,
                op_kind=opname,
                support_size=len(info["support"]),
                support_axes=" ".join(str(x) for x in info["support"]),
                active_only=True,
                has_dormant=False,
                default_target=default.target_axis,
                chosen_target=chosen.target_axis,
                builder_kind=chosen.kind,
                candidate_count=len(candidates) + 1,
                valid_candidate_count=len(valid) + (1 if default.valid else 0),
                default_path_cost=default.proxy_path_cost,
                chosen_path_cost=chosen.proxy_path_cost,
                default_workspace_proxy_score=default.proxy_workspace,
                chosen_workspace_proxy_score=chosen.proxy_workspace,
                default_workspace_actual_peak_bytes="",
                chosen_workspace_actual_peak_bytes="",
                default_resident_bound_proxy=default.proxy_resident_bound,
                chosen_resident_bound_proxy=chosen.proxy_resident_bound,
                default_resident_actual_peak_numel="",
                chosen_resident_actual_peak_numel="",
                default_resident_actual_peak_log2_numel="",
                chosen_resident_actual_peak_log2_numel="",
                default_resident_actual_peak_bytes="",
                chosen_resident_actual_peak_bytes="",
                default_peak_offender_bag="",
                chosen_peak_offender_bag="",
                default_peak_offender_p_B="",
                chosen_peak_offender_p_B="",
                default_peak_offender_incident_bonds="",
                chosen_peak_offender_incident_bonds="",
                default_peak_offender_bond_dims="",
                chosen_peak_offender_bond_dims="",
                accepted=accepted,
                reject_reason="" if accepted else "default_kept_or_no_improvement",
                num_2q_ops_default=default.num_2q_ops(),
                num_2q_ops_chosen=chosen.num_2q_ops(),
                path_length_default=_path_len(layout, axis_to_ident, default),
                path_length_chosen=_path_len(layout, axis_to_ident, chosen),
                default_v_sequence=_op_sequence(default),
                chosen_v_sequence=_op_sequence(chosen),
                default_edge_hits=_edge_count_string(_edge_hit_counts(layout, axis_to_ident, default)),
                chosen_edge_hits=_edge_count_string(_edge_hit_counts(layout, axis_to_ident, chosen)),
                reduced_edges=_reduced_edges(
                    _edge_hit_counts(layout, axis_to_ident, default),
                    _edge_hit_counts(layout, axis_to_ident, chosen),
                ),
                resident_proxy_reason="same fixed TTN layout and proxy r_e; no executable retrace in v1",
                refactor_proxy_delta=default.refactor_cost - chosen.refactor_cost,
            ))

        next_ident = _update_active_mapping(opname, inst, slot2id, next_ident)

    summary = dict(
        circuit=name,
        layout="baseline",
        executable_emit=False,
        emit_blocked_reason="phase_tracking_not_implemented_in_v1",
        num_localization_steps=len(rows),
        num_rasl_considered=considered,
        num_rasl_changed=len(decisions),
        num_rejected_invalid=rejected_invalid,
        num_rejected_resident=rejected_resident,
        report_mode="proxy_only",
        default_resident_bound_proxy=layout.r_resident,
        rasl_resident_bound_proxy=layout.r_resident,
        default_resident_actual_peak_numel=None,
        rasl_resident_actual_peak_numel=None,
        default_resident_actual_peak_log2_numel=None,
        rasl_resident_actual_peak_log2_numel=None,
        default_resident_actual_peak_bytes=None,
        rasl_resident_actual_peak_bytes=None,
        default_peak_offender_bag=None,
        rasl_peak_offender_bag=None,
        default_peak_offender_p_B=None,
        rasl_peak_offender_p_B=None,
        default_peak_offender_incident_bonds=None,
        rasl_peak_offender_incident_bonds=None,
        default_peak_offender_bond_dims=None,
        rasl_peak_offender_bond_dims=None,
        default_workspace_proxy_peak=default_workspace_peak,
        rasl_workspace_proxy_peak=chosen_workspace_peak,
        default_workspace_actual_peak_bytes=None,
        rasl_workspace_actual_peak_bytes=None,
        default_refactor_cost=default_refactor,
        rasl_refactor_cost=chosen_refactor,
        default_num_cross_bag_ops=default_cross_ops,
        rasl_num_cross_bag_ops=chosen_cross_ops,
        compile_time_default_s=compile_default_s,
        compile_time_rasl_s=time.perf_counter() - t1,
        decisions=decisions[:50],
    )
    return rows, summary


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=STEP_FIELDS)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in STEP_FIELDS})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("circuits", nargs="*", default=DEFAULT_CIRCUITS)
    p.add_argument("--enable-rasl", action="store_true")
    p.add_argument("--rasl-max-steps", type=int, default=200)
    p.add_argument("--rasl-max-support", type=int, default=10)
    p.add_argument("--rasl-top-k", type=int, default=32)
    p.add_argument("--rasl-builder", choices=["active_z", "symplectic", "full"], default="full")
    p.add_argument("--rasl-global-rollback", action="store_true")
    p.add_argument("--out-csv", default="reports/rasl_steps.csv")
    p.add_argument("--out-json", default="reports/rasl_summary.json")
    args = p.parse_args()

    if not args.enable_rasl:
        raise SystemExit("RASL is behind a feature flag. Re-run with --enable-rasl.")

    all_rows = []
    summaries = []
    for circuit in args.circuits:
        print(f"[rasl] circuit={circuit}", flush=True)
        prog = _load_prog(circuit)
        rows, summary = analyze_program(
            circuit, prog,
            max_steps=args.rasl_max_steps,
            max_support=args.rasl_max_support,
            top_k=args.rasl_top_k,
            builder=args.rasl_builder,
        )
        all_rows.extend(rows)
        summaries.append(summary)
        print(
            f"  considered={summary['num_rasl_considered']} "
            f"changed={summary['num_rasl_changed']} "
            f"resident_proxy={summary['default_resident_bound_proxy']}->{summary['rasl_resident_bound_proxy']} "
            f"workspace_proxy={summary['default_workspace_proxy_peak']:.1f}->{summary['rasl_workspace_proxy_peak']:.1f} "
            f"refactor={summary['default_refactor_cost']:.1f}->{summary['rasl_refactor_cost']:.1f}",
            flush=True,
        )

    write_csv(args.out_csv, all_rows)
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"wrote CSV:  {args.out_csv}")
    print(f"wrote JSON: {args.out_json}")


if __name__ == "__main__":
    main()
