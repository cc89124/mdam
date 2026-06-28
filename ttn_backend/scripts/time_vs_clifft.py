"""Wall-clock: native Clifft sampling vs the TTN backend, same circuit.

Honest caveat: the TTN backend is an unoptimized Python/numpy prototype; Clifft
is a compiled simulator. So this measures the prototype's slowdown, and the TTN
value is memory (running circuits past Clifft's 2^k wall), not speed.
"""
import argparse, os, time
import numpy as np
import clifft
from ttn_backend.backend_spec import export_backend_spec, assign_homes_and_classify
from ttn_backend.scripts.qec_temporal_carving_runtime import build_carving_executable_spec
from ttn_backend.scripts.qec_temporal_carving_report import trace_from_program
from temporal_carving.pipeline import run as run_pipeline
from ttn_backend import TTNBackend

POLICY = dict(
    TTN_FUSE_MULTICNOT="1", TTN_PERSISTENT_MULTICNOT="1",
    TTN_PERSISTENT_MULTICNOT_MIN_MULTIS="2", TTN_DESTRUCTIVE_OPEN="1",
    TTN_FUSE_MULTICNOT_BATCH="1",
    TTN_FUSE_MULTICNOT_CAP_BYTES=str(64*1024*1024),
    TTN_FUSE_MULTICNOT_TOTAL_CAP_BYTES=str(64*1024*1024),
    TTN_PREFISSION_TRANSPORT_CAP_BYTES=str(64*1024*1024),
    TTN_PREFISSION_MIN_GAIN="1.01",
    TTN_MULTICNOT_PARITY_REWRITE="1",
)


def time_clifft(prog, shots, seed):
    t0 = time.perf_counter()
    clifft.sample(prog, shots, seed=seed)
    return (time.perf_counter() - t0) / shots


def time_ttn(prog, shots, seed, max_steps):
    for k, v in POLICY.items():
        os.environ[k] = v
    spec = export_backend_spec(prog, strict=False)
    if max_steps:  # big circuit: carving layout, prefix
        trace = trace_from_program(prog, strict=False)
        carving = run_pipeline(trace, seeder="recursive_balanced_mincut",
                               refine_moves=("nni",), seed=0,
                               partitioner="networkx", exact=False)
        spec, homing = build_carving_executable_spec(spec, carving["tree"])
    else:
        homing = assign_homes_and_classify(spec)
    backend = TTNBackend(spec, homing)
    t0 = time.perf_counter()
    for s in range(shots):
        backend.run_shot(prog, seed + s, max_steps=max_steps)
    return (time.perf_counter() - t0) / shots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("circuits", nargs="*",
                    default=["distillation", "cultivation_d3", "coherent_d5_r1"])
    ap.add_argument("--shots", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ttn-max-steps", type=int, default=0,
                    help="prefix cap for the TTN run (0 = full circuit)")
    args = ap.parse_args()

    print(f"{'circuit':16s} {'k_peak':>6s} {'steps':>7s} "
          f"{'clifft/shot':>12s} {'ttn/shot':>12s} {'ttn slower':>11s}")
    for c in args.circuits:
        prog = clifft.compile(open(f"qec_bench/circuits/{c}.stim").read())
        kpk = int(prog.peak_rank) if hasattr(prog, "peak_rank") else -1
        ms = args.ttn_max_steps or None
        ni = int(prog.num_instructions)
        tcl = time_clifft(prog, args.shots, args.seed)
        ttn = time_ttn(prog, args.shots, args.seed, ms)
        steps = ms or ni
        print(f"{c:16s} {kpk:6d} {steps:7d} "
              f"{tcl*1e3:10.2f}ms {ttn*1e3:10.2f}ms {ttn/tcl:9.1f}x")


if __name__ == "__main__":
    main()
