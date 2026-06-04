"""Conservative structural SVD/QR reuse analyzer.

This script does not sample tensors and does not claim numerical similarity.
It classifies only cases that can be certified structurally as local/frame
equivalent. Everything else is refresh-required or unknown.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import clifft

sys.path.insert(0, ".")

from ttn_backend import treewidth as T_mod
from ttn_backend.backend_spec import assign_homes_and_classify, export_backend_spec


SUMMARY_FIELDS = [
    "circuit",
    "edge_count",
    "candidate_svd_sites",
    "reusable_sites",
    "refresh_sites",
    "reusable_ratio",
    "max_reusable_window",
    "avg_reusable_window",
    "dominant_refresh_reason",
]


EVENT_FIELDS = [
    "step",
    "edge_id",
    "operation",
    "left_support_size",
    "right_support_size",
    "is_frame_equivalent",
    "needs_refresh",
    "refresh_reason",
    "notes",
]


FRAME_REUSABLE_OPS = {
    "OP_FRAME_H",
    "OP_FRAME_S",
    "OP_FRAME_S_DAG",
    "OP_FRAME_CNOT",
    "OP_FRAME_CZ",
    "OP_FRAME_SWAP",
}

ACTIVE_CLIFFORD_OPS = {
    "OP_ARRAY_CNOT",
    "OP_ARRAY_CZ",
    "OP_ARRAY_MULTI_CNOT",
    "OP_ARRAY_MULTI_CZ",
    "OP_ARRAY_SWAP",
}

ACTIVE_NONCLIFFORD_OPS = {
    "OP_ARRAY_T",
    "OP_ARRAY_T_DAG",
    "OP_ARRAY_ROT",
    "OP_ARRAY_U2",
    "OP_ARRAY_U4",
    "OP_PHASE_T",
    "OP_PHASE_T_DAG",
    "OP_PHASE_ROT",
}

MEASUREMENT_OPS = {
    "OP_MEAS_ACTIVE_DIAGONAL",
    "OP_MEAS_ACTIVE_DIAGONAL_FORCED",
    "OP_MEAS_ACTIVE_INTERFERE",
    "OP_MEAS_ACTIVE_INTERFERE_FORCED",
}


def _load_prog(circuit: str):
    path = Path("qec_bench/circuits") / f"{circuit}.stim"
    with open(path) as f:
        return clifft.compile(f.read())


def _edge_id_for_path(path):
    if not path or len(path) < 2:
        return ""
    return ";".join(f"{min(a,b)}-{max(a,b)}" for a, b in zip(path, path[1:]))


def analyze_circuit(circuit: str):
    prog = _load_prog(circuit)
    spec = export_backend_spec(prog, strict=False)
    homing = assign_homes_and_classify(spec)
    op_classes = defaultdict(list)
    for r in homing["op_classes"]:
        if r.get("kind") == "two":
            op_classes[int(r["step"])].append(r)

    rows = []
    reusable = 0
    refresh = 0
    reason_counts = Counter()
    reusable_windows = []
    current_window = 0

    for step, inst in enumerate(prog):
        name = T_mod._opname(inst.opcode)
        step_rows = []
        if name in FRAME_REUSABLE_OPS:
            # Frame-only changes are local Clifford frame updates, so all
            # existing SVD factors can be reused for this operation.
            reusable += 1
            current_window += 1
            rows.append(dict(
                step=step,
                edge_id="all",
                operation=name,
                left_support_size=0,
                right_support_size=0,
                is_frame_equivalent=True,
                needs_refresh=False,
                refresh_reason="",
                notes="certified frame-only local Clifford update",
            ))
            continue

        for r in op_classes.get(step, []):
            path = list(r.get("path_bags", []))
            edge_id = _edge_id_for_path(path)
            axes = list(r.get("axes", []))
            if name in ACTIVE_CLIFFORD_OPS:
                # This would be reusable only if the backend keeps the entire
                # affected region as a symbolic local Clifford frame. The
                # current backend materializes and refactors, so classify as a
                # materialization boundary, not as proven reusable.
                reason = "materialization_boundary"
                is_reuse = False
                note = "active Clifford; reusable only with region-local Clifford frame executor"
            elif name in ACTIVE_NONCLIFFORD_OPS:
                reason = "cross_cut_nonclifford" if len(axes) >= 2 else "local_nonclifford_no_svd_site"
                is_reuse = False
                note = "non-Clifford active operation; cross-cut reuse not certified"
            elif name in MEASUREMENT_OPS:
                reason = "active_measurement"
                is_reuse = False
                note = "projection/collapse changes rank factors"
            else:
                reason = "unknown"
                is_reuse = False
                note = "no structural reuse certificate"

            if is_reuse:
                reusable += 1
                current_window += 1
            else:
                refresh += 1
                reason_counts[reason] += 1
                if current_window:
                    reusable_windows.append(current_window)
                    current_window = 0
            step_rows.append(dict(
                step=step,
                edge_id=edge_id,
                operation=name,
                left_support_size=1,
                right_support_size=max(1, len(axes) - 1),
                is_frame_equivalent=is_reuse,
                needs_refresh=not is_reuse,
                refresh_reason=reason,
                notes=note,
            ))
        rows.extend(step_rows)

    if current_window:
        reusable_windows.append(current_window)
    candidate = reusable + refresh
    summary = dict(
        circuit=circuit,
        edge_count=len(spec["union"].get("bag_edges", [])),
        candidate_svd_sites=candidate,
        reusable_sites=reusable,
        refresh_sites=refresh,
        reusable_ratio=(reusable / candidate) if candidate else 0.0,
        max_reusable_window=max(reusable_windows) if reusable_windows else 0,
        avg_reusable_window=(sum(reusable_windows) / len(reusable_windows)) if reusable_windows else 0.0,
        dominant_refresh_reason=reason_counts.most_common(1)[0][0] if reason_counts else "",
    )
    return summary, rows, reason_counts


def write_report(path, summaries, reason_by_circuit):
    with open(path, "w") as f:
        f.write("# SVD/QR Reuse Structural Analysis\n\n")
        f.write("판정 기준은 `M_omega = L M_0 R^dagger`를 구조적으로 증명할 수 있는 경우만 reusable로 세는 것이다. 증명 불가 또는 현재 backend가 materialize/refactor하는 active operation은 refresh-required로 분류했다.\n\n")
        f.write("| circuit | candidates | reusable | ratio | dominant refresh reason |\n")
        f.write("|---|---:|---:|---:|---|\n")
        for s in summaries:
            f.write(
                f"| {s['circuit']} | {s['candidate_svd_sites']} | {s['reusable_sites']} | "
                f"{float(s['reusable_ratio']):.3f} | {s['dominant_refresh_reason']} |\n"
            )
        f.write("\n## Refresh reason breakdown\n\n")
        for circuit, counts in reason_by_circuit.items():
            f.write(f"### {circuit}\n\n")
            for reason, count in counts.most_common():
                f.write(f"- `{reason}`: {count}\n")
            f.write("\n")
        f.write("## 해석\n\n")
        f.write("- frame-only Clifford update는 SVD basis 재사용이 구조적으로 가능하다.\n")
        f.write("- 현재 active Clifford CNOT/MULTI_CNOT은 region-local Clifford frame executor가 없으면 tensor를 materialize하고 닫아야 하므로 `materialization_boundary`로 둔다.\n")
        f.write("- 따라서 실제 reuse 비율을 올리는 다음 단계는 SVD 캐시 자체보다 region-local Clifford frame을 유지하는 executor다.\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("circuits", nargs="*", default=["coherent_d5_r1", "coherent_d5_r5"])
    p.add_argument("--out-dir", default="reports")
    args = p.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    summaries = []
    all_rows = []
    reason_by_circuit = {}
    for circuit in args.circuits:
        s, rows, counts = analyze_circuit(circuit)
        summaries.append(s)
        reason_by_circuit[circuit] = counts
        all_rows.extend(dict(circuit=circuit, **r) for r in rows)

    with open(out / "svd_reuse_analysis_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        w.writerows(summaries)
    with open(out / "svd_reuse_analysis_events.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["circuit"] + EVENT_FIELDS)
        w.writeheader()
        w.writerows(all_rows)
    write_report(out / "svd_reuse_analysis_report.md", summaries, reason_by_circuit)
    print(f"wrote {out / 'svd_reuse_analysis_summary.csv'}")
    print(f"wrote {out / 'svd_reuse_analysis_events.csv'}")
    print(f"wrote {out / 'svd_reuse_analysis_report.md'}")


if __name__ == "__main__":
    main()
