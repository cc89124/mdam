"""Where does the d5_r5 wall-clock actually go? Split the real TTN run by
primitive, to decide whether an (A,f) frame can help.

The (A,f) frame's premise is: defer CNOT/MULTI_CNOT (the transport-heavy ops)
and handle rotations as deferred diagonal phases. But in THIS backend a diagonal
2-qubit gate (CZ) is applied through the SAME transport path as a CNOT -- there
is no cheap rank-exploiting diagonal primitive. So the decisive question is:

  how much of the wall is transport/QR (CNOT/MULTI_CNOT -- what (A,f) defers)
  vs apply_diag (ROT/T/S -- already local & cheap in eager mode)?

If the diagonal work is already ~free and the wall is ~all transport/QR, then
deferring CNOTs can only help if the linear-map compression saving exceeds the
NEW cost of realizing the (now nonlocal) phases -- which, without a cheap
diagonal primitive, also goes through transport. We instrument the real run by
wrapping the state primitives and the QR kernel.
"""
from __future__ import annotations

import argparse
import os
import time

import clifft
import ttn_backend.core as core
from ttn_backend.backend_spec import export_backend_spec, assign_homes_and_classify

POLICY = dict(   # staged_transport_fission (the fast cached policy, ~43s)
    TTN_FUSE_MULTICNOT="1", TTN_PERSISTENT_MULTICNOT="1",
    TTN_PERSISTENT_MULTICNOT_MIN_MULTIS="2", TTN_DESTRUCTIVE_OPEN="1",
    TTN_FUSE_MULTICNOT_BATCH="1",
    TTN_FUSE_MULTICNOT_CAP_BYTES=str(64 * 1024 * 1024),
    TTN_FUSE_MULTICNOT_TOTAL_CAP_BYTES=str(64 * 1024 * 1024),
    TTN_EXACT_TOTAL_CAP_BYTES=str(64 * 1024 * 1024),
    TTN_STAGED_TRANSPORT="1",
    TTN_STAGED_BLOCK_BYTES=str(8 * 1024 * 1024),
    TTN_STAGED_OUTPUT_FISSION="1",
    TTN_BAG_FISSION_CAP_BYTES=str(64 * 1024 * 1024),
    TTN_BAG_FISSION_MIN_GAIN="1.05",
)

ACC = {}      # name -> [calls, seconds]


def tick(name, fn):
    def wrapped(*a, **k):
        t0 = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            e = ACC.setdefault(name, [0, 0.0])
            e[0] += 1; e[1] += time.perf_counter() - t0
    return wrapped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("circuit", nargs="?", default="coherent_d5_r5")
    ap.add_argument("--layout", choices=["carving", "union"], default="carving")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--timeout", type=float, default=1200.0)
    ap.add_argument("--max-steps", type=int, default=0, help="0=full")
    args = ap.parse_args()
    for k, v in POLICY.items():
        os.environ.setdefault(k, v)

    from ttn_backend import TTNBackend
    src = open(f"qec_bench/circuits/{args.circuit}.stim").read()
    prog = clifft.compile(src)
    base = export_backend_spec(prog, strict=False)
    t_build = time.perf_counter()
    if args.layout == "union":
        spec, homing = base, assign_homes_and_classify(base)
    else:
        from temporal_carving.pipeline import run as run_pipeline
        from ttn_backend.scripts.qec_temporal_carving_report import trace_from_program
        from ttn_backend.scripts.qec_temporal_carving_runtime import build_carving_executable_spec
        trace = trace_from_program(prog, strict=False)
        carving = run_pipeline(trace, seeder="recursive_balanced_mincut",
                               refine_moves=("nni",), seed=0,
                               partitioner="networkx", exact=False)
        spec, homing = build_carving_executable_spec(base, carving["tree"])
    build_s = time.perf_counter() - t_build
    print(f"[{args.circuit}] layout/carving build: {build_s:.2f}s", flush=True)

    # wrap the QR kernel (all transport/contraction/refactor linear algebra)
    core._thin_qr = tick("_thin_qr (all QR)", core._thin_qr)
    # wrap state primitives at the CLASS level (state is built inside run_shot)
    for meth, label in [
        ("apply_diag", "apply_diag  (ROT/T/S local)"),
        ("apply_1q", "apply_1q    (H/U2 boundary)"),
        ("apply_2q_class_A", "apply_2q_A  (local 2q)"),
        ("apply_2q_class_B_path", "apply_2q_Bpath (CNOT/CZ transport)"),
    ]:
        if hasattr(core.TTNState, meth):
            setattr(core.TTNState, meth, tick(label, getattr(core.TTNState, meth)))

    backend = TTNBackend(spec, homing)
    t0 = time.perf_counter()
    backend.run_shot(prog, args.seed, runtime_timeout=args.timeout,
                     max_steps=(args.max_steps or None))
    wall = time.perf_counter() - t0
    m = backend.state.metrics

    print(f"\n=== {args.circuit} primitive wall-clock split ===", flush=True)
    print(f"total run_shot wall: {wall:.2f}s   (carving build: {build_s:.2f}s)")
    print(f"metrics: n_qr={m.get('n_qr')} n_transports={m.get('n_transports')} "
          f"n_svd={m.get('n_svd')} peak={m.get('peak_stored_bytes',0)/2**20:.1f}MiB "
          f"maxχ={m.get('max_bond_dim_observed')}")
    print(f"\n{'primitive':40s} {'calls':>8s} {'seconds':>10s} {'%wall':>7s}")
    for name in sorted(ACC, key=lambda n: -ACC[n][1]):
        c, s = ACC[name]
        print(f"{name:40s} {c:8d} {s:10.2f} {100*s/wall:6.1f}%")
    # the decisive ratio
    diag = ACC.get("apply_diag  (ROT/T/S local)", [0, 0.0])[1]
    qr = ACC.get("_thin_qr (all QR)", [0, 0.0])[1]
    print(f"\nDECISIVE: diagonal(ROT) work = {diag:.2f}s ({100*diag/wall:.1f}%); "
          f"QR/transport work = {qr:.2f}s ({100*qr/wall:.1f}%)")
    print("If diagonal<<QR: rotations are already ~free; (A,f) can only win by")
    print("compressing the CNOT linear map MORE than the cost it adds realizing")
    print("the (now nonlocal) phases -- which this backend also transports.")


if __name__ == "__main__":
    main()
