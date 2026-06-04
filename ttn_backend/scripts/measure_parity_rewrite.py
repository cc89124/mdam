"""Measure the MULTI_CNOT parity-gather rewrite on the carving_leaf layout.

Runs qec_temporal_carving_runtime.py as a subprocess for a circuit/prefix under
chosen policy envs, with the rewrite OFF then ON, and prints the work/peak deltas.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

POLICIES = {
    # exact persistent + prefission (the doc's "general_policy")
    "general_policy": dict(
        TTN_FUSE_MULTICNOT="1",
        TTN_PERSISTENT_MULTICNOT="1",
        TTN_PERSISTENT_MULTICNOT_MIN_MULTIS="2",
        TTN_DESTRUCTIVE_OPEN="1",
        TTN_FUSE_MULTICNOT_BATCH="1",
        TTN_FUSE_MULTICNOT_CAP_BYTES=str(64 * 1024 * 1024),
        TTN_FUSE_MULTICNOT_TOTAL_CAP_BYTES=str(64 * 1024 * 1024),
        TTN_PREFISSION_TRANSPORT_CAP_BYTES=str(64 * 1024 * 1024),
        TTN_PREFISSION_MIN_GAIN="1.01",
    ),
    # block-streamed staged transport on top of general_policy (the doc's best)
    "staged_transport": dict(
        TTN_FUSE_MULTICNOT="1",
        TTN_PERSISTENT_MULTICNOT="1",
        TTN_PERSISTENT_MULTICNOT_MIN_MULTIS="2",
        TTN_DESTRUCTIVE_OPEN="1",
        TTN_FUSE_MULTICNOT_BATCH="1",
        TTN_FUSE_MULTICNOT_CAP_BYTES=str(64 * 1024 * 1024),
        TTN_FUSE_MULTICNOT_TOTAL_CAP_BYTES=str(64 * 1024 * 1024),
        TTN_EXACT_TOTAL_CAP_BYTES=str(64 * 1024 * 1024),
        TTN_STAGED_TRANSPORT="1",
        TTN_STAGED_BLOCK_BYTES=str(8 * 1024 * 1024),
    ),
    # no fuse/persistent: every MULTI_CNOT runs the per-control fallback the
    # rewrite replaces -> shows the raw rewrite effect.
    "pure_fallback": {},
}

ALL_KNOBS = sorted(set().union(*[set(v) for v in POLICIES.values()]) | {
    "TTN_MULTICNOT_PARITY_REWRITE",
    "TTN_CLUSTER_MULTICNOT_TOP",
})

KEYS = [
    "status", "steps_completed", "total_steps", "elapsed_s",
    "actual_total_peak_bytes", "peak_stored_bytes",
    "workspace_actual_peak_bytes", "resident_actual_peak_bytes",
    "max_bond_dim_observed", "n_qr", "n_transports",
    "sum_path_length", "sum_rank_weighted_path_length",
    "num_refactor", "peak_offender_bag", "peak_offender_step",
]


def run(circuit, policy, rewrite, max_steps, timeout, out_root):
    out = Path(out_root) / f"{policy}_{'rw' if rewrite else 'base'}"
    out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    for k in ALL_KNOBS:
        env.pop(k, None)
    env.update(POLICIES[policy])
    env["TTN_MULTICNOT_PARITY_REWRITE"] = "1" if rewrite else "0"
    cmd = [
        sys.executable, "ttn_backend/scripts/qec_temporal_carving_runtime.py",
        circuit, "--runtime-timeout", str(timeout), "--modes", "carving_leaf",
        "--max-steps", str(max_steps), "--out-dir", str(out),
    ]
    proc = subprocess.run(cmd, env=env, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out / "runner.log").write_text(proc.stdout)
    summary = out / "summary.json"
    metrics = out / circuit / "carving_leaf_metrics.json"
    row = {}
    if summary.exists():
        data = json.loads(summary.read_text())
        if isinstance(data, dict) and "rows" in data:
            rows = data["rows"]
            rec = rows[0] if rows else {}
        elif isinstance(data, list):
            rec = data[0] if data else {}
        else:
            rec = data
        for k in KEYS:
            row[k] = rec.get(k)
        if row.get("elapsed_s") is None:
            row["elapsed_s"] = rec.get("elapsed_time_seconds")
    if metrics.exists():
        m = json.loads(metrics.read_text())
        for k in ("multicnot_parity_rewrite_windows", "multicnot_parity_local_cnots",
                  "multicnot_parity_crossing_cnots", "multicnot_parity_groups_folded",
                  "multicnot_parity_groups_direct", "multicnot_parity_max_fold_depth",
                  "n_qr", "n_transports",
                  "sum_rank_weighted_path_length", "multicnot_region_fallback"):
            if m.get(k) is not None:
                row[k] = m.get(k)
    if not row:
        row["status"] = "runner_error"
        row["log_tail"] = proc.stdout[-500:]
    return row


def fmt(v):
    if isinstance(v, float):
        return f"{v:.3g}"
    return str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("circuit", default="coherent_d5_r5", nargs="?")
    ap.add_argument("--policy", default="general_policy", choices=list(POLICIES))
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--out-dir", default="reports/parity_rewrite_measure")
    args = ap.parse_args()

    print(f"circuit={args.circuit} policy={args.policy} "
          f"max_steps={args.max_steps} timeout={args.timeout}")
    base = run(args.circuit, args.policy, False, args.max_steps, args.timeout, args.out_dir)
    rw = run(args.circuit, args.policy, True, args.max_steps, args.timeout, args.out_dir)

    fields = ["status", "steps_completed", "elapsed_s",
              "actual_total_peak_bytes", "peak_stored_bytes",
              "workspace_actual_peak_bytes", "max_bond_dim_observed",
              "n_qr", "n_transports", "sum_path_length",
              "sum_rank_weighted_path_length", "multicnot_region_fallback",
              "multicnot_parity_rewrite_windows", "multicnot_parity_groups_folded",
              "multicnot_parity_groups_direct", "multicnot_parity_max_fold_depth",
              "multicnot_parity_local_cnots", "multicnot_parity_crossing_cnots"]
    print(f"\n{'metric':36s} {'baseline':>16s} {'rewrite':>16s} {'delta%':>9s}")
    for k in fields:
        b = base.get(k)
        r = rw.get(k)
        d = ""
        try:
            if b not in (None, "", 0) and isinstance(b, (int, float)) and isinstance(r, (int, float)):
                d = f"{100.0 * (r - b) / b:+.1f}"
        except Exception:
            d = ""
        print(f"{k:36s} {fmt(b):>16s} {fmt(r):>16s} {d:>9s}")


if __name__ == "__main__":
    main()
