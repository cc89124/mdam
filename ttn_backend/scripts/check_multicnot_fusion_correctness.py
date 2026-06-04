"""Compare default per-control MULTI_CNOT execution against fused execution."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import clifft

sys.path.insert(0, ".")

from ttn_backend import TTNBackend
from ttn_backend.backend_spec import assign_homes_and_classify, export_backend_spec
from ttn_backend.scripts.qec_temporal_carving_report import trace_from_program
from ttn_backend.scripts.qec_temporal_carving_runtime import build_carving_executable_spec
from temporal_carving.pipeline import run as run_pipeline


def _load_prog(circuit):
    with open(Path("qec_bench/circuits") / f"{circuit}.stim") as f:
        return clifft.compile(f.read())


def _specs(circuit, mode):
    prog = _load_prog(circuit)
    base_spec = export_backend_spec(prog, strict=False)
    if mode == "baseline_jt":
        return prog, base_spec, assign_homes_and_classify(base_spec)
    trace = trace_from_program(prog, strict=False)
    tree = run_pipeline(trace, seeder="recursive_balanced_mincut", refine_moves=(), seed=1, exact=False)["tree"]
    return (prog,) + build_carving_executable_spec(base_spec, tree)


def _run(prog, spec, homing, seed, fuse, cap):
    if fuse:
        os.environ["TTN_FUSE_MULTICNOT"] = "1"
        os.environ["TTN_FUSE_MULTICNOT_CAP_BYTES"] = str(int(cap))
    else:
        os.environ["TTN_FUSE_MULTICNOT"] = "0"
        os.environ.pop("TTN_FUSE_MULTICNOT_CAP_BYTES", None)
    b = TTNBackend(spec, homing)
    rec = b.run_shot(prog, seed=seed, runtime_timeout=None)
    return rec, b.last_metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("circuits", nargs="*", default=["cultivation_d3", "coherent_d5_r1"])
    p.add_argument("--mode", default="carving_leaf", choices=["baseline_jt", "carving_leaf"])
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--cap-bytes", type=int, default=134217728)
    args = p.parse_args()

    ok = True
    for circuit in args.circuits:
        prog, spec, homing = _specs(circuit, args.mode)
        rec0, m0 = _run(prog, spec, homing, args.seed, False, args.cap_bytes)
        rec1, m1 = _run(prog, spec, homing, args.seed, True, args.cap_bytes)
        same = rec0 == rec1
        ok = ok and same
        print(
            f"{circuit} {args.mode}: same_record={same} "
            f"qr {m0.get('n_qr')}->{m1.get('n_qr')} "
            f"transport {m0.get('n_transports')}->{m1.get('n_transports')} "
            f"fused={m1.get('multicnot_region_fused')}"
        )
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
