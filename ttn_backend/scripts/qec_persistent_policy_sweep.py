"""Run a circuit-independent persistent-region policy sweep on QEC benchmarks.

The policy is intentionally generic:

* Build the same temporal-carving executable layout for every circuit.
* Compute dense active-state peak bytes from `flat_peak_k`.
* Set the region/total cap from `dense_peak_bytes / target_ratio`.
* Enable persistent MULTI_CNOT windows with destructive-open liveness.

No circuit name is used to choose windows or caps, except optional per-circuit
max-step limits for making long benchmark runs finite.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


PY = Path("/home/jung/clifft_env/bin/python")
ROOT = Path(__file__).resolve().parents[2]


OUT_FIELDS = [
    "circuit",
    "policy",
    "target_ratio",
    "cap_bytes",
    "cap_mib",
    "max_steps",
    "status",
    "timeout",
    "steps_completed",
    "total_steps",
    "dense_peak_bytes",
    "actual_total_peak_bytes",
    "memory_ratio_dense_over_actual",
    "destructive_total_peak_bytes",
    "resident_actual_peak_bytes",
    "peak_stored_bytes",
    "workspace_actual_peak_bytes",
    "n_qr",
    "n_transports",
    "num_refactor",
    "persistent_multicnot_windows",
    "persistent_multicnot_steps",
    "persistent_multicnot_controls",
    "multicnot_region_fused",
    "multicnot_region_batches",
    "multicnot_region_fallback",
    "elapsed_s",
    "summary_path",
    "metrics_path",
]


def _read_summary(path: Path) -> dict:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if r.get("mode") == "carving_leaf"]
    if not rows:
        raise RuntimeError(f"no carving_leaf row in {path}")
    return rows[-1]


def _intish(v, default=0):
    if v in (None, ""):
        return default
    return int(float(v))


def _floatish(v, default=0.0):
    if v in (None, ""):
        return default
    return float(v)


def _run_runtime(circuit: str, out_dir: Path, env: dict, max_steps: int | None,
                 runtime_timeout: float) -> dict:
    cmd = [
        str(PY),
        str(ROOT / "ttn_backend/scripts/qec_temporal_carving_runtime.py"),
        circuit,
        "--runtime-timeout",
        str(runtime_timeout),
        "--modes",
        "carving_leaf",
        "--out-dir",
        str(out_dir),
    ]
    if max_steps is not None:
        cmd += ["--max-steps", str(max_steps)]
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)
    return _read_summary(out_dir / "summary.csv")


def _dense_peak_bytes(row: dict) -> int:
    k = int(float(row.get("flat_peak_k") or 0))
    return int((2 ** k) * 16)


def _make_env(policy: str, cap_bytes: int | None, svd_rtol: str = "1e-4",
              svd_min_matrix_elems: int = 1048576) -> dict:
    env = os.environ.copy()
    for key in [
        "TTN_FUSE_MULTICNOT",
        "TTN_PERSISTENT_MULTICNOT",
        "TTN_PERSISTENT_MULTICNOT_MIN_MULTIS",
        "TTN_DESTRUCTIVE_OPEN",
        "TTN_FUSE_MULTICNOT_BATCH",
        "TTN_FUSE_MULTICNOT_CAP_BYTES",
        "TTN_FUSE_MULTICNOT_TOTAL_CAP_BYTES",
        "TTN_SVD_TRUNC_RTOL",
        "TTN_SVD_TRUNC_MIN_MATRIX_ELEMS",
    ]:
        env.pop(key, None)
    if policy == "baseline":
        env["TTN_FUSE_MULTICNOT"] = "0"
    elif policy == "persistent":
        env["TTN_FUSE_MULTICNOT"] = "1"
        env["TTN_PERSISTENT_MULTICNOT"] = "1"
        env["TTN_PERSISTENT_MULTICNOT_MIN_MULTIS"] = "2"
        env["TTN_DESTRUCTIVE_OPEN"] = "1"
        env["TTN_FUSE_MULTICNOT_BATCH"] = "1"
        if cap_bytes is not None:
            env["TTN_FUSE_MULTICNOT_CAP_BYTES"] = str(int(cap_bytes))
            env["TTN_FUSE_MULTICNOT_TOTAL_CAP_BYTES"] = str(int(cap_bytes))
    elif policy == "persistent_svd":
        env = _make_env("persistent", cap_bytes, svd_rtol, svd_min_matrix_elems)
        env["TTN_SVD_TRUNC_RTOL"] = str(svd_rtol)
        env["TTN_SVD_TRUNC_MIN_MATRIX_ELEMS"] = str(int(svd_min_matrix_elems))
    else:
        raise ValueError(policy)
    return env


def _output_row(circuit: str, policy: str, target_ratio: float, cap_bytes: int | None,
                max_steps: int | None, row: dict, out_dir: Path) -> dict:
    dense = _dense_peak_bytes(row)
    actual = _intish(row.get("actual_total_peak_bytes"))
    return dict(
        circuit=circuit,
        policy=policy,
        target_ratio=target_ratio,
        cap_bytes="" if cap_bytes is None else int(cap_bytes),
        cap_mib="" if cap_bytes is None else float(cap_bytes) / (1024 * 1024),
        max_steps="" if max_steps is None else int(max_steps),
        status=row.get("status"),
        timeout=row.get("timeout"),
        steps_completed=_intish(row.get("steps_completed")),
        total_steps=_intish(row.get("total_steps")),
        dense_peak_bytes=dense,
        actual_total_peak_bytes=actual,
        memory_ratio_dense_over_actual=(dense / actual) if actual else "",
        destructive_total_peak_bytes=_intish(row.get("destructive_total_peak_bytes")),
        resident_actual_peak_bytes=_intish(row.get("resident_actual_peak_bytes")),
        peak_stored_bytes=_intish(row.get("peak_stored_bytes")),
        workspace_actual_peak_bytes=_intish(row.get("workspace_actual_peak_bytes")),
        n_qr=_intish(row.get("n_qr")),
        n_transports=_intish(row.get("n_transports")),
        num_refactor=_intish(row.get("num_refactor")),
        persistent_multicnot_windows=_intish(row.get("persistent_multicnot_windows")),
        persistent_multicnot_steps=_intish(row.get("persistent_multicnot_steps")),
        persistent_multicnot_controls=_intish(row.get("persistent_multicnot_controls")),
        multicnot_region_fused=_intish(row.get("multicnot_region_fused")),
        multicnot_region_batches=_intish(row.get("multicnot_region_batches")),
        multicnot_region_fallback=_intish(row.get("multicnot_region_fallback")),
        elapsed_s=_floatish(row.get("elapsed_s")),
        summary_path=str(out_dir / "summary.csv"),
        metrics_path=str(out_dir / circuit / "carving_leaf_metrics.json"),
    )


def write_report(path: Path, rows: list[dict]):
    with open(path, "w") as f:
        f.write("# QEC Persistent Region Policy Sweep\n\n")
        f.write("이 sweep은 회로별 손튜닝 없이 같은 policy를 적용한다. cap은 dense active-state peak를 target ratio로 나눈 값이다.\n\n")
        f.write("| circuit | policy | steps | dense/actual | actual MiB | QR | transport | persistent windows | fallback |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            actual_mib = int(r["actual_total_peak_bytes"]) / (1024 * 1024) if r["actual_total_peak_bytes"] else 0.0
            ratio = r["memory_ratio_dense_over_actual"]
            ratio_s = "" if ratio == "" else f"{float(ratio):.2f}"
            f.write(
                f"| {r['circuit']} | {r['policy']} | {r['steps_completed']}/{r['total_steps']} | "
                f"{ratio_s} | {actual_mib:.2f} | {r['n_qr']} | {r['n_transports']} | "
                f"{r['persistent_multicnot_windows']} | {r['multicnot_region_fallback']} |\n"
            )
        f.write("\n## 해석\n\n")
        f.write("- `baseline`은 temporal-carving leaf layout에서 기존 per-control execution이다.\n")
        f.write("- `persistent`는 destructive-open persistent MULTI_CNOT window + batch fallback이다.\n")
        f.write("- `persistent_svd`는 peak-risk 구간에서만 큰 matrix SVD compression을 추가한 approximation policy다.\n")
        f.write("- 이 파일의 목적은 특정 회로 튜닝이 아니라, 동일 bytecode-structured policy가 여러 QEC benchmark에서 얼마나 일반적으로 작동하는지 보는 것이다.\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("circuits", nargs="*", default=[
        "distillation",
        "cultivation_d3",
        "coherent_d3_r1",
        "coherent_d5_r1",
    ])
    p.add_argument("--include-d5r5", action="store_true")
    p.add_argument("--d5r5-max-steps", type=int, default=839)
    p.add_argument("--target-ratio", type=float, default=8.0)
    p.add_argument("--runtime-timeout", type=float, default=120.0)
    p.add_argument("--policies", default="baseline,persistent",
                   help="comma-separated: baseline,persistent,persistent_svd")
    p.add_argument("--svd-rtol", default="1e-4",
                   help="relative SVD threshold for persistent_svd")
    p.add_argument("--svd-min-matrix-elems", type=int, default=1048576,
                   help="only use SVD compression for matrices with at least this many elements")
    p.add_argument("--out-dir", default="reports/qec_persistent_policy_sweep")
    args = p.parse_args()

    circuits = list(args.circuits)
    if args.include_d5r5 and "coherent_d5_r5" not in circuits:
        circuits.append("coherent_d5_r5")
    policies = [x.strip() for x in args.policies.split(",") if x.strip()]
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for circuit in circuits:
        max_steps = args.d5r5_max_steps if circuit == "coherent_d5_r5" else None
        # Baseline first, also to compute dense peak bytes in the same script.
        baseline_out = out_root / circuit / "baseline"
        baseline_row = _run_runtime(
            circuit,
            baseline_out,
            _make_env("baseline", None),
            max_steps=max_steps,
            runtime_timeout=args.runtime_timeout,
        )
        dense = _dense_peak_bytes(baseline_row)
        cap = max(16, int(dense / float(args.target_ratio)))
        if "baseline" in policies:
            rows.append(_output_row(circuit, "baseline", args.target_ratio, None,
                                    max_steps, baseline_row, baseline_out))
        for policy in policies:
            if policy == "baseline":
                continue
            policy_out = out_root / circuit / f"{policy}_ratio{args.target_ratio:g}"
            row = _run_runtime(
                circuit,
                policy_out,
                _make_env(policy, cap, args.svd_rtol, args.svd_min_matrix_elems),
                max_steps=max_steps,
                runtime_timeout=args.runtime_timeout,
            )
            rows.append(_output_row(circuit, policy, args.target_ratio, cap,
                                    max_steps, row, policy_out))

    with open(out_root / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUT_FIELDS)
        w.writeheader()
        w.writerows(rows)
    with open(out_root / "summary.json", "w") as f:
        json.dump(rows, f, indent=2)
    write_report(out_root / "report.md", rows)
    print(f"wrote {out_root / 'summary.csv'}")
    print(f"wrote {out_root / 'summary.json'}")
    print(f"wrote {out_root / 'report.md'}")


if __name__ == "__main__":
    main()
