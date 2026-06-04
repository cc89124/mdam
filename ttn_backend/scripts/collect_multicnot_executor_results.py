"""Collect MULTI_CNOT fused executor experiment outputs into paper CSV/MD files."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path("reports")


SUMMARY_FIELDS = [
    "circuit",
    "layout",
    "cap_mib",
    "topk",
    "steps_done",
    "steps_total",
    "actual_peak_stored_bytes",
    "actual_peak_workspace_bytes",
    "actual_peak_temp_bytes",
    "actual_total_peak_bytes",
    "destructive_total_peak_bytes",
    "qr_count",
    "qr_work_proxy",
    "transport_count",
    "center_move_count",
    "path_refactor_count",
    "runtime_sec",
    "correctness_pass",
]


WINDOW_FIELDS = [
    "window_id",
    "step_begin",
    "step_end",
    "support_size",
    "region_size",
    "old_qr_proxy",
    "new_open_close_qr",
    "workspace_proxy_bytes",
    "actual_workspace_bytes",
    "selected",
    "fallback_reason",
]


def _read_one(path: Path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"empty CSV: {path}")
    return rows[-1]


def _summary_row(path, circuit, layout, cap_mib, topk, correctness="not_checked"):
    r = _read_one(Path(path))
    stored = int(float(r["peak_stored_bytes"]))
    ws = int(float(r["workspace_actual_peak_bytes"]))
    total = int(float(r.get("actual_total_peak_bytes") or max(stored, ws)))
    destructive_total = int(float(r.get("destructive_total_peak_bytes") or total))
    return dict(
        circuit=circuit,
        layout=layout,
        cap_mib=cap_mib,
        topk=topk,
        steps_done=int(float(r["steps_completed"])),
        steps_total=int(float(r["total_steps"])),
        actual_peak_stored_bytes=stored,
        actual_peak_workspace_bytes=ws,
        actual_peak_temp_bytes=ws,
        actual_total_peak_bytes=total,
        destructive_total_peak_bytes=destructive_total,
        qr_count=int(float(r["n_qr"])),
        qr_work_proxy=float(r.get("qr_work_proxy") or 0.0),
        transport_count=int(float(r["n_transports"])),
        center_move_count="",
        path_refactor_count=int(float(r["num_refactor"])),
        runtime_sec=float(r["elapsed_s"]),
        correctness_pass=correctness,
    )


def collect_summary(out_path: Path):
    rows = [
        _summary_row(
            ROOT / "multicnot_executor_d5r1_off/summary.csv",
            "coherent_d5_r1",
            "carving_leaf_exact_baseline",
            "",
            "none",
            "reference",
        ),
        _summary_row(
            ROOT / "multicnot_executor_d5r1_on_v2/summary.csv",
            "coherent_d5_r1",
            "carving_leaf_step_fused",
            128,
            "all_step_feasible",
            "passed_same_seed_record",
        ),
        _summary_row(
            ROOT / "multicnot_executor_d5r5_off/summary.csv",
            "coherent_d5_r5",
            "carving_leaf_exact_baseline",
            "",
            "none",
            "reference_partial",
        ),
        _summary_row(
            ROOT / "multicnot_executor_d5r5_cap32_v2/summary.csv",
            "coherent_d5_r5",
            "carving_leaf_step_fused",
            32,
            "all_step_feasible",
            "not_checked_partial",
        ),
        _summary_row(
            ROOT / "multicnot_executor_d5r5_cap64_v2/summary.csv",
            "coherent_d5_r5",
            "carving_leaf_step_fused",
            64,
            "all_step_feasible",
            "not_checked_partial",
        ),
        _summary_row(
            ROOT / "multicnot_executor_d5r5_cap96_v2/summary.csv",
            "coherent_d5_r5",
            "carving_leaf_step_fused",
            96,
            "all_step_feasible",
            "not_checked_partial",
        ),
        _summary_row(
            ROOT / "multicnot_executor_d5r5_cap128_v2/summary.csv",
            "coherent_d5_r5",
            "carving_leaf_step_fused",
            128,
            "all_step_feasible",
            "not_checked_partial",
        ),
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return rows


def collect_windows(out_path: Path):
    src = ROOT / "multicnot_window_fusion_d5r5_cap/coherent_d5_r5_multicnot_windows.csv"
    rows = []
    with open(src, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(dict(
                window_id=r["window_id"],
                step_begin=r["first_step"],
                step_end=r["last_step"],
                support_size=r["total_controls"],
                region_size=r["region_size"],
                old_qr_proxy=r["old_transport_qr"],
                new_open_close_qr=r["window_open_close_upper"],
                workspace_proxy_bytes=r["max_workspace_proxy_bytes"],
                actual_workspace_bytes="",
                selected=r["all_steps_cap_pass"],
                fallback_reason="" if r["all_steps_cap_pass"] == "True" else "cap_proxy",
            ))
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=WINDOW_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return rows


def write_report(path: Path, summary_rows):
    def mib(x):
        if x == "":
            return ""
        return f"{float(x) / (1024 * 1024):.2f}"

    d5_off = next(r for r in summary_rows if r["circuit"] == "coherent_d5_r5" and "baseline" in r["layout"])
    d5_on = next(r for r in summary_rows if r["circuit"] == "coherent_d5_r5" and r["cap_mib"] == 64)
    d1_off = next(r for r in summary_rows if r["circuit"] == "coherent_d5_r1" and "baseline" in r["layout"])
    d1_on = next(r for r in summary_rows if r["circuit"] == "coherent_d5_r1" and "fused" in r["layout"])
    with open(path, "w") as f:
        f.write("# MULTI_CNOT Region-Fused Executor Actual Report\n\n")
        f.write("이 리포트는 analyzer proxy가 아니라 `TTNBackend.run_shot()`의 실제 tensor 실행 metric을 모은 것이다. 현재 구현은 persistent multi-step window가 아니라 안전한 첫 단계인 **single-step MULTI_CNOT region fusion**이다. 즉 한 `OP_ARRAY_MULTI_CNOT` 안의 control CNOT들을 한 region open/apply/close로 묶는다.\n\n")
        f.write("| circuit | mode | steps | peak stored MiB | peak workspace MiB | concurrent total MiB | destructive-open total MiB | QR | transports | runtime s |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in summary_rows:
            f.write(
                f"| {r['circuit']} | {r['layout']} cap={r['cap_mib']} | "
                f"{r['steps_done']}/{r['steps_total']} | {mib(r['actual_peak_stored_bytes'])} | "
                f"{mib(r['actual_peak_workspace_bytes'])} | {mib(r['actual_total_peak_bytes'])} | "
                f"{mib(r['destructive_total_peak_bytes'])} | "
                f"{r['qr_count']} | {r['transport_count']} | {float(r['runtime_sec']):.2f} |\n"
            )
        f.write("\n## 핵심 비교\n\n")
        f.write(f"- `coherent_d5_r1`: QR `{d1_off['qr_count']} -> {d1_on['qr_count']}`, transports `{d1_off['transport_count']} -> {d1_on['transport_count']}`, total peak `{mib(d1_off['actual_total_peak_bytes'])} MiB -> {mib(d1_on['actual_total_peak_bytes'])} MiB`.\n")
        f.write(f"- `coherent_d5_r5` 60초 partial best cap64: steps `{d5_off['steps_done']} -> {d5_on['steps_done']}`, concurrent total peak `{mib(d5_off['actual_total_peak_bytes'])} MiB -> {mib(d5_on['actual_total_peak_bytes'])} MiB`, QR `{d5_off['qr_count']} -> {d5_on['qr_count']}`, transports `{d5_off['transport_count']} -> {d5_on['transport_count']}`.\n")
        f.write("- cap128은 더 큰 fused region을 허용하면서 concurrent total이 273MiB까지 올라가므로 dense 256MiB claim에는 cap64가 더 맞다. cap32는 fallback transport가 늘어 오히려 악화됐다.\n")
        f.write("\n## 해석\n\n")
        f.write("- 이 구현은 window-level persistent executor의 완성본은 아니다. 그러나 실제 runtime executor이며, per-control transport를 한 step 안에서 제거한다.\n")
        f.write("- 큰 회로에서 cap64 fused partial은 dense 256MiB보다 작게 유지됐다. concurrent actual total peak는 약 253.91MiB다. 이전 32MiB 값은 stored/workspace 동시합이 아니라 workspace peak 기준이라 memory claim으로 쓰면 안 된다.\n")
        f.write("- 다음 구현 단계는 여러 `OP_ARRAY_MULTI_CNOT`을 같은 open region 안에 유지하는 persistent multi-step window다.\n")


def main():
    summary = collect_summary(ROOT / "multicnot_window_executor_summary.csv")
    collect_windows(ROOT / "multicnot_window_executor_windows.csv")
    write_report(ROOT / "multicnot_window_executor_report.md", summary)
    print(f"wrote {ROOT / 'multicnot_window_executor_summary.csv'}")
    print(f"wrote {ROOT / 'multicnot_window_executor_windows.csv'}")
    print(f"wrote {ROOT / 'multicnot_window_executor_report.md'}")


if __name__ == "__main__":
    main()
