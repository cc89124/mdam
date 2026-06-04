"""Run temporal-live carving layout optimization on Clifft QEC circuits.

This is a compile-time/profile-time adapter:

  Clifft bytecode -> active live trace L(t), E_t
  -> temporal_carving layout T*
  -> lazy-live peak profile max_v Ehat_v(t)

It does not modify the runtime TTN backend.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, ".")

import clifft

from temporal_carving.cost import CostModel, Trace
from temporal_carving.exact import exact_dp
from temporal_carving.io import save_tree, write_trace
from temporal_carving.pipeline import run as run_pipeline
from temporal_carving.seed import build_seed
from ttn_backend.backend_spec import _instrumented_replay


PROFILE_FIELDS = [
    "circuit",
    "t",
    "flat_k",
    "temporal_ttn_peak_E",
    "saving_log2",
    "saving_ratio",
]

SUMMARY_FIELDS = [
    "circuit",
    "n_axes",
    "n_steps",
    "n_two_axis_events",
    "seeder",
    "refine",
    "seed_peak",
    "refined_peak",
    "flat_peak_k",
    "peak_saving_log2",
    "peak_saving_ratio",
    "union_graph_peak",
    "exact_peak",
    "accepted_moves",
    "elapsed_s",
    "tree_path",
    "profile_path",
    "plot_path",
]


def _load_prog(name):
    with open(os.path.join("qec_bench/circuits", name + ".stim")) as f:
        return clifft.compile(f.read())


def trace_from_program(program, strict=False) -> Trace:
    rec = _instrumented_replay(program, strict=strict)
    axes = tuple(sorted(rec["lifecycle"]))
    dims = {i: 2 for i in axes}
    timeline = tuple(range(len(program)))
    live_sets = {}
    for t in timeline:
        live = []
        for ident, info in rec["lifecycle"].items():
            start = int(info["promote_step"])
            end = info["demote_step"]
            if start <= t and (end is None or t <= int(end)):
                live.append(int(ident))
        live_sets[t] = frozenset(live)
    events = defaultdict(list)
    for step, _name, u, v in rec["two_axis_ops"]:
        if int(u) == int(v):
            continue
        a, b = sorted((int(u), int(v)))
        events[int(step)].append((a, b))
    return Trace(
        axes=axes,
        dims=dims,
        timeline=timeline,
        live_sets=live_sets,
        events={t: tuple(es) for t, es in events.items()},
    )


def _safe_ratio(log_saving):
    if log_saving <= -100:
        return 0.0
    if log_saving >= 100:
        return float("inf")
    return float(2.0 ** log_saving)


def write_profile(path, circuit, trace, profile):
    rows = []
    for idx, t in enumerate(trace.timeline):
        flat_k = len(trace.live_sets.get(t, ()))
        E = float(profile[idx])
        saving = float(flat_k - E)
        rows.append(dict(
            circuit=circuit,
            t=t,
            flat_k=flat_k,
            temporal_ttn_peak_E=E,
            saving_log2=saving,
            saving_ratio=_safe_ratio(saving),
        ))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PROFILE_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return rows


def maybe_plot(path, title, profile_rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return ""
    xs = [int(r["t"]) for r in profile_rows]
    flat = [float(r["flat_k"]) for r in profile_rows]
    ttn = [float(r["temporal_ttn_peak_E"]) for r in profile_rows]
    plt.figure(figsize=(10, 4))
    plt.plot(xs, flat, label="Clifft flat k(t)", linewidth=1.0)
    plt.plot(xs, ttn, label="Temporal-live TTN max_v Ehat_v(t)", linewidth=1.0)
    plt.xlabel("bytecode step t")
    plt.ylabel("log2 tensor elements")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return str(path)


def write_report(path, summary_rows):
    with open(path, "w") as f:
        f.write("# QEC Temporal-Live Carving Report\n\n")
        f.write("이 리포트는 Clifft QEC bytecode에서 active live trace `L(t)`와 two-axis event `E_t`를 추출한 뒤, 하나의 static carving tree `T*`를 찾고 lazy-live allocation objective를 시간축에서 평가한다.\n\n")
        f.write("| circuit | axes | steps | seeder | refine | flat peak k | TTN peak E | saving log2 | saving ratio | union peak |\n")
        f.write("|---|---:|---:|---|---|---:|---:|---:|---:|---:|\n")
        for r in summary_rows:
            f.write(
                f"| {r['circuit']} | {r['n_axes']} | {r['n_steps']} | {r['seeder']} | {r['refine']} | "
                f"{float(r['flat_peak_k']):.3f} | {float(r['refined_peak']):.3f} | "
                f"{float(r['peak_saving_log2']):.3f} | {float(r['peak_saving_ratio']):.3g} | "
                f"{float(r['union_graph_peak']):.3f} |\n"
            )
        f.write("\n## Interpretation\n\n")
        f.write("- `flat_peak_k`는 dense active-state 기준 `|L(t)|` peak다.\n")
        f.write("- `refined_peak`는 lazy-live carving tree 위의 `max_t max_v Ehat_v(t)`다.\n")
        f.write("- inactive cut은 `rhat=0`으로 마스킹되므로, 이 값이 lazy allocation을 반영한 log-memory profile이다.\n")
        f.write("- `union_graph_peak`는 temporal reset을 무시한 비교용 값이며 final objective가 아니다.\n")


def run_one(circuit, args):
    t0 = time.perf_counter()
    prog = _load_prog(circuit)
    trace = trace_from_program(prog, strict=False)
    out_dir = Path(args.out_dir) / circuit
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.write_trace:
        write_trace(trace, out_dir / "trace")

    moves = tuple(x for x in args.refine.split(",") if x and x != "none")
    result = run_pipeline(
        trace,
        seeder=args.seeder,
        refine_moves=moves,
        seed=args.seed,
        partitioner=args.partitioner,
        exact=args.exact and len(trace.axes) <= args.max_exact_n,
        max_exact_n=args.max_exact_n,
    )
    cost = CostModel(trace)
    tree = result["tree"]
    profile = cost.tree_profile(tree)
    tree_path = out_dir / "tree.json"
    profile_path = out_dir / "profile.csv"
    plot_path = out_dir / "profile.png"
    save_tree(tree, tree_path)
    profile_rows = write_profile(profile_path, circuit, trace, profile)
    plot_written = maybe_plot(plot_path, circuit, profile_rows)
    flat_peak = max((len(trace.live_sets.get(t, ())) for t in trace.timeline), default=0)
    refined_peak = float(result["refined_peak"])
    union_peak = cost.union_graph_objective(tree)
    summary = dict(
        circuit=circuit,
        n_axes=len(trace.axes),
        n_steps=len(trace.timeline),
        n_two_axis_events=sum(len(v) for v in trace.events.values()),
        seeder=args.seeder,
        refine=args.refine,
        seed_peak=float(result["seed_peak"]),
        refined_peak=refined_peak,
        flat_peak_k=float(flat_peak),
        peak_saving_log2=float(flat_peak - refined_peak),
        peak_saving_ratio=_safe_ratio(float(flat_peak - refined_peak)),
        union_graph_peak=float(union_peak),
        exact_peak="" if result["exact_peak"] is None else float(result["exact_peak"]),
        accepted_moves=int(result["accepted_moves"]),
        elapsed_s=time.perf_counter() - t0,
        tree_path=str(tree_path),
        profile_path=str(profile_path),
        plot_path=plot_written,
    )
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("circuits", nargs="*", default=["coherent_d5_r1"])
    p.add_argument("--seeder", default="recursive_balanced_mincut")
    p.add_argument("--refine", default="nni")
    p.add_argument("--partitioner", default="networkx")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--exact", action="store_true")
    p.add_argument("--max-exact-n", type=int, default=20)
    p.add_argument("--write-trace", action="store_true")
    p.add_argument("--out-dir", default="reports/qec_temporal_carving")
    args = p.parse_args()

    rows = []
    for circuit in args.circuits:
        print(f"running {circuit}", flush=True)
        row = run_one(circuit, args)
        rows.append(row)
        print(
            f"  axes={row['n_axes']} flat_peak={row['flat_peak_k']} "
            f"ttn_peak={row['refined_peak']:.3f} saving_log2={row['peak_saving_log2']:.3f}",
            flush=True,
        )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        w.writerows(rows)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(rows, f, indent=2)
    write_report(out_dir / "report.md", rows)
    print(f"wrote {out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
