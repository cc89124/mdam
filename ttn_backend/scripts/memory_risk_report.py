"""Bond-aware memory-risk offender analysis for TTN layouts."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys

sys.path.insert(0, ".")

import clifft

from ttn_backend.backend_spec import assign_homes_and_classify, export_backend_spec
from ttn_backend.layout_transform import reduce_hub_degree
from ttn_backend import TTNBackend


DEFAULT_CIRCUITS = [
    "distillation",
    "cultivation_d3",
    "coherent_d3_r1",
    "coherent_d5_r1",
    "coherent_d5_r5",
    "coherent_d7_r1",
    "coherent_d7_r7",
]

CSV_FIELDS = [
    "circuit",
    "layout_variant",
    "n_bags",
    "n_edges",
    "M_static_bytes",
    "M_store_worst_exp",
    "M_store_worst_total_bytes",
    "M_ws_worst_exp",
    "M_ws_worst_max_bytes",
    "R_store",
    "R_workspace",
    "R_mem",
    "max_bag_degree",
    "max_separator_bits",
    "runtime_available",
    "runtime_status",
    "runtime_timeout",
    "runtime_steps_completed",
    "runtime_total_steps",
    "runtime_R_store_obs",
    "runtime_R_store_gap",
]


def _load_prog(name):
    with open(os.path.join("qec_bench/circuits", name + ".stim")) as f:
        return clifft.compile(f.read())


def _variant_spec(spec, variant, threshold):
    if variant == "baseline":
        return spec
    if variant.startswith("hub"):
        return reduce_hub_degree(spec, threshold)
    raise ValueError(f"unknown variant: {variant}")


def _bytes_from_exp(exp):
    return 16 * (1 << int(exp))


def _fmt_bytes(n):
    if n is None:
        return ""
    mb = float(n) / 1e6
    if abs(mb) >= 1e6:
        return f"{mb:.3e} MB"
    return f"{mb:.3f} MB"


def _exp_to_bytes_string(exp):
    return str(_bytes_from_exp(exp))


def _ceil_log2_dim(dim):
    dim = int(dim)
    if dim <= 1:
        return 0
    return int(math.ceil(math.log2(dim)))


def _build_layout(spec, homing):
    bags = [set(b) for b in spec["union"]["bags"]]
    owned = {int(k): list(v) for k, v in homing["owned_phys"].items()}
    adj = {i: [] for i in range(len(bags))}
    edge_sep = {}
    for i, j, sep in spec["union"]["bag_edges"]:
        s = len(sep)
        adj[i].append((j, s))
        adj[j].append((i, s))
        edge_sep[tuple(sorted((i, j)))] = s
    return bags, owned, adj, edge_sep


def compute_risk(spec, homing):
    bags, owned, adj, edge_sep = _build_layout(spec, homing)
    store_rows = []
    for bid in range(len(bags)):
        sep_bits = [s for _, s in adj[bid]]
        own_count = len(owned.get(bid, []))
        exp = own_count + sum(sep_bits)
        store_rows.append(dict(
            bag_id=bid,
            own_count=own_count,
            degree=len(adj[bid]),
            separator_bits=sorted(sep_bits, reverse=True),
            store_exp=exp,
            M_store_worst_bytes=_exp_to_bytes_string(exp),
        ))

    ws_rows = []
    for (a, b), sep in sorted(edge_sep.items()):
        own_a = len(owned.get(a, []))
        own_b = len(owned.get(b, []))
        ext_a = [s for nb, s in adj[a] if nb != b]
        ext_b = [s for nb, s in adj[b] if nb != a]
        exp = own_a + own_b + sum(ext_a) + sum(ext_b)
        ws_rows.append(dict(
            edge=[a, b],
            own_count_A=own_a,
            own_count_B=own_b,
            degree_A=len(adj[a]),
            degree_B=len(adj[b]),
            external_separator_bits_A=sorted(ext_a, reverse=True),
            external_separator_bits_B=sorted(ext_b, reverse=True),
            ws_exp=exp,
            M_ws_worst_bytes=_exp_to_bytes_string(exp),
        ))

    store_rows.sort(key=lambda x: x["store_exp"], reverse=True)
    ws_rows.sort(key=lambda x: x["ws_exp"], reverse=True)
    r_store = store_rows[0]["store_exp"] if store_rows else 0
    r_ws = ws_rows[0]["ws_exp"] if ws_rows else 0
    return dict(
        n_bags=len(bags),
        n_edges=len(edge_sep),
        M_static_bytes=16 * int(spec["union"]["sum2"]),
        M_store_worst_exp=r_store,
        M_store_worst_total_bytes=str(sum(_bytes_from_exp(r["store_exp"]) for r in store_rows)),
        M_ws_worst_exp=r_ws,
        M_ws_worst_max_bytes=_exp_to_bytes_string(r_ws),
        R_store=r_store,
        R_workspace=r_ws,
        R_mem=max(r_store, r_ws),
        max_bag_degree=max((len(v) for v in adj.values()), default=0),
        max_separator_bits=max(edge_sep.values(), default=0),
        top10_store=store_rows[:10],
        top10_workspace=ws_rows[:10],
        store_rows=store_rows,
        adj=adj,
    )


def _runtime_from_existing(circuit, variant):
    path = "reports/all_variants.json"
    if not os.path.exists(path):
        return None
    try:
        data = json.load(open(path))
    except Exception:
        return None
    for rec in data:
        row = rec.get("row", {})
        if row.get("circuit") == circuit and row.get("layout_variant") == variant:
            return rec
    return None


def _runtime_by_running(prog, spec, homing, timeout_s, seed=42):
    backend = TTNBackend(spec, homing)
    try:
        backend.run_shot(prog, seed, runtime_timeout=timeout_s, check_interval=1)
        metrics = backend.last_metrics or {}
        status = "timeout" if metrics.get("timeout") else "complete"
        return dict(row=dict(runtime_status=status), metrics=metrics)
    except Exception as exc:
        return dict(
            row=dict(runtime_status="error", error=repr(exc)),
            metrics=getattr(backend, "last_metrics", None) or {},
        )


def _runtime_metrics(rec):
    if rec is None:
        return None
    if "metrics" in rec:
        return rec.get("metrics") or {}
    row = rec.get("row", {})
    return dict(
        runtime_status=row.get("runtime_status"),
        timeout=row.get("timeout"),
        steps_completed=row.get("steps_completed"),
        total_steps=row.get("total_steps"),
    )


def observed_store_rows(risk, runtime):
    metrics = _runtime_metrics(runtime)
    if not metrics:
        return [], None
    edge_dims = metrics.get("edge_max_bond_dim")
    if not edge_dims:
        return [], None
    edge_exp = {}
    for key, dim in edge_dims.items():
        a, b = [int(x) for x in key.split("-")]
        edge_exp[tuple(sorted((a, b)))] = _ceil_log2_dim(dim)

    rows = []
    by_bag = {r["bag_id"]: r for r in risk["store_rows"]}
    for bid, pred in by_bag.items():
        obs = pred["own_count"]
        for nb, _ in risk["adj"][bid]:
            obs += edge_exp.get(tuple(sorted((bid, nb))), 0)
        rows.append(dict(
            bag_id=bid,
            store_exp_predicted_worst=pred["store_exp"],
            store_exp_observed=obs,
            gap=pred["store_exp"] - obs,
        ))
    rows.sort(key=lambda x: x["store_exp_predicted_worst"], reverse=True)
    r_obs = max((r["store_exp_observed"] for r in rows), default=None)
    return rows, r_obs


def build_record(circuit, variant, threshold, include_runtime, runtime_timeout):
    prog = _load_prog(circuit)
    base_spec = export_backend_spec(prog, strict=False)
    spec = _variant_spec(base_spec, variant, threshold)
    homing = assign_homes_and_classify(spec)
    risk = compute_risk(spec, homing)

    runtime = None
    if include_runtime:
        runtime = _runtime_from_existing(circuit, variant)
        metrics = _runtime_metrics(runtime)
        if runtime is None or not metrics or not metrics.get("edge_max_bond_dim"):
            runtime = _runtime_by_running(prog, spec, homing, runtime_timeout)
    obs_rows, r_obs = observed_store_rows(risk, runtime)
    rt_row = (runtime or {}).get("row", {})
    metrics = _runtime_metrics(runtime) or {}
    status = rt_row.get("runtime_status") or metrics.get("runtime_status") or ""
    timeout = rt_row.get("timeout", metrics.get("timeout", ""))
    steps = rt_row.get("steps_completed", metrics.get("steps_completed", ""))
    total = rt_row.get("total_steps", metrics.get("total_steps", ""))

    row = {k: risk[k] for k in CSV_FIELDS if k in risk}
    row.update(dict(
        circuit=circuit,
        layout_variant=variant,
        runtime_available=runtime is not None,
        runtime_status=status,
        runtime_timeout=timeout,
        runtime_steps_completed=steps,
        runtime_total_steps=total,
        runtime_R_store_obs="" if r_obs is None else r_obs,
        runtime_R_store_gap="" if r_obs is None else risk["R_store"] - r_obs,
    ))
    record = dict(
        row=row,
        top10_store_offenders=risk["top10_store"],
        top10_workspace_offenders=risk["top10_workspace"],
        observed_store_rows=obs_rows,
    )
    return row, record


def _parse_variants(s):
    return [x.strip() for x in s.split(",") if x.strip()]


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in CSV_FIELDS})


def write_json(path, records):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(records, f, indent=2)


def write_md(path, records):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("# Memory Risk Summary\n\n")
        for rec in records:
            row = rec["row"]
            f.write(f"## {row['circuit']} {row['layout_variant']}\n\n")
            f.write("Global:\n\n")
            f.write(f"- n_bags = {row['n_bags']}, n_edges = {row['n_edges']}\n")
            f.write(f"- R_store = {row['R_store']}\n")
            f.write(f"- R_workspace = {row['R_workspace']}\n")
            f.write(f"- R_mem = {row['R_mem']}\n")
            f.write(f"- D_max = {row['max_bag_degree']}, S_max = {row['max_separator_bits']}\n")
            f.write(f"- M_static = {_fmt_bytes(row['M_static_bytes'])}\n")
            f.write(f"- M_store_worst_total = {_fmt_bytes(int(row['M_store_worst_total_bytes']))}\n")
            f.write(f"- M_ws_worst_max = {_fmt_bytes(int(row['M_ws_worst_max_bytes']))}\n")
            if row.get("runtime_available"):
                f.write(
                    f"- runtime observed R_store = {row.get('runtime_R_store_obs')} "
                    f"(gap {row.get('runtime_R_store_gap')})\n"
                )

            f.write("\nTop-10 store offenders:\n\n")
            for r in rec["top10_store_offenders"]:
                f.write(
                    f"- B{r['bag_id']}: own={r['own_count']}, deg={r['degree']}, "
                    f"sep_bits={r['separator_bits']}, store_exp={r['store_exp']}, "
                    f"bytes={_fmt_bytes(int(r['M_store_worst_bytes']))}\n"
                )

            f.write("\nTop-10 workspace offenders:\n\n")
            for r in rec["top10_workspace_offenders"]:
                f.write(
                    f"- B{r['edge'][0]}-B{r['edge'][1]}: own=({r['own_count_A']},"
                    f"{r['own_count_B']}), deg=({r['degree_A']},{r['degree_B']}), "
                    f"ext_sep_A={r['external_separator_bits_A']}, "
                    f"ext_sep_B={r['external_separator_bits_B']}, "
                    f"ws_exp={r['ws_exp']}, bytes={_fmt_bytes(int(r['M_ws_worst_bytes']))}\n"
                )

            obs = rec.get("observed_store_rows", [])[:10]
            if obs:
                f.write("\nRuntime observed vs predicted store exponents, top predicted bags:\n\n")
                for r in obs:
                    f.write(
                        f"- B{r['bag_id']}: pred={r['store_exp_predicted_worst']}, "
                        f"obs={r['store_exp_observed']}, gap={r['gap']}\n"
                    )
            f.write("\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("circuits", nargs="*", default=DEFAULT_CIRCUITS)
    p.add_argument("--variants", default="baseline,hub3")
    p.add_argument("--hub-degree-threshold", type=int, default=3)
    p.add_argument("--include-runtime", action="store_true")
    p.add_argument("--runtime-timeout", type=float, default=60.0)
    p.add_argument("--out-csv", default="reports/memory_risk.csv")
    p.add_argument("--out-json", default="reports/memory_risk.json")
    p.add_argument("--out-md", default="reports/memory_risk_summary.md")
    args = p.parse_args()

    rows = []
    records = []
    for variant in _parse_variants(args.variants):
        for circuit in args.circuits:
            print(f"[risk] circuit={circuit} variant={variant}", flush=True)
            row, rec = build_record(
                circuit,
                variant,
                args.hub_degree_threshold,
                args.include_runtime,
                args.runtime_timeout,
            )
            rows.append(row)
            records.append(rec)

    write_csv(args.out_csv, rows)
    write_json(args.out_json, records)
    write_md(args.out_md, records)
    print(f"wrote CSV:  {args.out_csv}")
    print(f"wrote JSON: {args.out_json}")
    print(f"wrote MD:   {args.out_md}")


if __name__ == "__main__":
    main()
