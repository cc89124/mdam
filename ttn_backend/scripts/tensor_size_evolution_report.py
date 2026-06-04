"""Build tensor-size evolution reports from TTN actual step profiles."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


OUT_FIELDS = [
    "mode",
    "step_id",
    "op_kind",
    "resident_actual_peak_log2_numel",
    "resident_actual_peak_bytes",
    "peak_offender_bag",
    "peak_offender_shape",
    "peak_offender_p_B",
    "peak_offender_incident_bond_dims",
    "dense_k",
    "dense_bytes",
    "dense_over_ttn_ratio",
]


def _read_summary(path):
    rows = list(csv.DictReader(open(path)))
    return {r["mode"]: r for r in rows}


def _read_profile(path):
    rows = []
    for r in csv.DictReader(open(path)):
        if r["step_id"] in ("", "None"):
            continue
        rows.append(r)
    rows.sort(key=lambda r: int(r["step_id"]))
    return rows


def _safe_float(x):
    if x in ("", None, "None"):
        return None
    return float(x)


def _safe_int(x):
    if x in ("", None, "None"):
        return None
    return int(float(x))


def build_rows(mode, profile_rows, dense_k):
    out = []
    dense_bytes = None if dense_k is None else int(16 * (2 ** int(dense_k)))
    for r in profile_rows:
        b = _safe_int(r["resident_actual_peak_bytes"])
        ratio = None
        if dense_bytes is not None and b:
            ratio = dense_bytes / b
        out.append(dict(
            mode=mode,
            step_id=int(r["step_id"]),
            op_kind=r["op_kind"],
            resident_actual_peak_log2_numel=_safe_float(r["resident_actual_peak_log2_numel"]),
            resident_actual_peak_bytes=b,
            peak_offender_bag=r["peak_offender_bag"],
            peak_offender_shape=r["peak_offender_shape"],
            peak_offender_p_B=r["peak_offender_p_B"],
            peak_offender_incident_bond_dims=r["peak_offender_incident_bond_dims"],
            dense_k=dense_k,
            dense_bytes=dense_bytes,
            dense_over_ttn_ratio=ratio,
        ))
    return out


def write_outputs(out_dir, all_rows, summary):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    profile_path = out_dir / "tensor_size_evolution.csv"
    with open(profile_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUT_FIELDS)
        w.writeheader()
        w.writerows(all_rows)

    by_mode = {}
    for row in all_rows:
        by_mode.setdefault(row["mode"], []).append(row)

    stats = {}
    for mode, rows in by_mode.items():
        vals = [float(r["resident_actual_peak_log2_numel"]) for r in rows]
        bytes_vals = [int(r["resident_actual_peak_bytes"]) for r in rows]
        stats[mode] = dict(
            n_steps=len(rows),
            max_log2=max(vals) if vals else None,
            p99_log2=percentile(vals, 0.99),
            p95_log2=percentile(vals, 0.95),
            median_log2=percentile(vals, 0.50),
            max_bytes=max(bytes_vals) if bytes_vals else None,
            p99_bytes=percentile(bytes_vals, 0.99),
            p95_bytes=percentile(bytes_vals, 0.95),
            median_bytes=percentile(bytes_vals, 0.50),
            summary=summary.get(mode, {}),
        )

    with open(out_dir / "tensor_size_evolution_summary.json", "w") as f:
        json.dump(stats, f, indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(11, 4))
        for mode, rows in by_mode.items():
            plt.plot(
                [int(r["step_id"]) for r in rows],
                [float(r["resident_actual_peak_log2_numel"]) for r in rows],
                label=mode,
                linewidth=1.0,
            )
        if all_rows and all_rows[0]["dense_k"] not in ("", None):
            plt.axhline(float(all_rows[0]["dense_k"]), color="black", linestyle="--", linewidth=0.8, label="dense k")
        plt.xlabel("bytecode step")
        plt.ylabel("peak tensor log2(numel)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "tensor_size_evolution.png", dpi=160)
        plt.close()
    except Exception as exc:
        stats["_plot_error"] = repr(exc)

    with open(out_dir / "tensor_size_evolution_report.md", "w") as f:
        f.write("# Tensor Size Evolution Report\n\n")
        f.write("This report uses actual TTN runtime step profiles. Values are per-step peak single tensor sizes, not proxy layout scores.\n\n")
        f.write("| mode | steps | median log2 | p95 log2 | p99 log2 | max log2 | max bytes |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for mode, s in stats.items():
            if mode.startswith("_"):
                continue
            f.write(
                f"| {mode} | {s['n_steps']} | {s['median_log2']:.3f} | "
                f"{s['p95_log2']:.3f} | {s['p99_log2']:.3f} | {s['max_log2']:.3f} | "
                f"{int(s['max_bytes'])} |\n"
            )
        f.write("\n## Top Events\n\n")
        for mode, rows in by_mode.items():
            f.write(f"### {mode}\n\n")
            f.write("| step | op | log2 | bytes | bag | shape | bonds |\n")
            f.write("|---:|---|---:|---:|---:|---|---|\n")
            top = sorted(rows, key=lambda r: float(r["resident_actual_peak_log2_numel"]), reverse=True)[:15]
            for r in top:
                f.write(
                    f"| {r['step_id']} | {r['op_kind']} | "
                    f"{float(r['resident_actual_peak_log2_numel']):.3f} | "
                    f"{r['resident_actual_peak_bytes']} | {r['peak_offender_bag']} | "
                    f"{r['peak_offender_shape']} | {r['peak_offender_incident_bond_dims']} |\n"
                )
            f.write("\n")
    return profile_path


def percentile(vals, q):
    vals = sorted(vals)
    if not vals:
        return None
    if len(vals) == 1:
        return float(vals[0])
    pos = q * (len(vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(vals[lo])
    w = pos - lo
    return float(vals[lo] * (1 - w) + vals[hi] * w)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True,
                   help="entries mode:summary_csv:profile_csv")
    p.add_argument("--dense-k", type=int, default=None)
    p.add_argument("--out-dir", default="reports/tensor_size_evolution")
    args = p.parse_args()

    all_rows = []
    summary = {}
    for entry in args.inputs:
        mode, summary_path, profile_path = entry.split(":", 2)
        summary.update(_read_summary(summary_path))
        rows = _read_profile(profile_path)
        all_rows.extend(build_rows(mode, rows, args.dense_k))
    path = write_outputs(args.out_dir, all_rows, summary)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
