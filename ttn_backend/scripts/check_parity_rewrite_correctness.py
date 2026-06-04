"""Correctness check: the MULTI_CNOT parity-gather rewrite must reproduce the
baseline measurement record bit-for-bit.

The rewrite changes only the CNOT execution order/grouping (an exact GF(2)
identity) and adds no RNG draws, so with an identical seed the sampled records
must match the no-rewrite path exactly.

Run:
    /home/jung/clifft_env/bin/python -m ttn_backend.scripts.check_parity_rewrite_correctness \
        distillation coherent_d3_r1 coherent_d5_r1 cultivation_d3
"""
import os
import sys
import numpy as np

import clifft
from ttn_backend.backend_spec import export_backend_spec, assign_homes_and_classify
from ttn_backend import TTNBackend

CIRCUITS = sys.argv[1:] or ["distillation", "coherent_d3_r1", "coherent_d5_r1"]
SHOTS = int(os.environ.get("SHOTS", "32"))
SEED = int(os.environ.get("SEED", "42"))

# Two policy contexts:
#   "pure"  : no fuse/persistent -> the per-control fallback path runs, so the
#             rewrite actually fires (this is the path it replaces).
#   "fused" : fuse + persistent windows on -> confirms the rewrite is compatible
#             and harmless when regions already absorb most controls.
POLICY_ENV = {
    "pure": {},
    "fused": dict(
        TTN_FUSE_MULTICNOT="1",
        TTN_PERSISTENT_MULTICNOT="1",
        TTN_PERSISTENT_MULTICNOT_MIN_MULTIS="2",
        TTN_DESTRUCTIVE_OPEN="1",
        TTN_FUSE_MULTICNOT_BATCH="1",
        TTN_FUSE_MULTICNOT_CAP_BYTES=str(64 * 1024 * 1024),
        TTN_FUSE_MULTICNOT_TOTAL_CAP_BYTES=str(64 * 1024 * 1024),
    ),
}
ALL_KNOBS = set().union(*[set(v) for v in POLICY_ENV.values()]) | {
    "TTN_MULTICNOT_PARITY_REWRITE"}


def sample_circuit(name, rewrite, policy):
    saved = {}
    env = dict(POLICY_ENV[policy])
    env["TTN_MULTICNOT_PARITY_REWRITE"] = "1" if rewrite else "0"
    for k in ALL_KNOBS:
        saved[k] = os.environ.get(k)
        os.environ.pop(k, None)
    for k, v in env.items():
        os.environ[k] = v
    try:
        with open(f"qec_bench/circuits/{name}.stim") as h:
            src = h.read()
        prog = clifft.compile(src)
        spec = export_backend_spec(prog, strict=False)
        homing = assign_homes_and_classify(spec)
        backend = TTNBackend(spec, homing)
        arr = np.asarray(backend.sample(prog, shots=SHOTS, seed=SEED))
        m = backend.state.metrics if getattr(backend, "state", None) else {}
        info = dict(
            windows=int(m.get("multicnot_parity_rewrite_windows", 0)),
            local=int(m.get("multicnot_parity_local_cnots", 0)),
            crossing=int(m.get("multicnot_parity_crossing_cnots", 0)),
            qr=int(m.get("n_qr", 0)),
            transports=int(m.get("n_transports", 0)),
        )
        return arr, info
    finally:
        for k, old in saved.items():
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old


def main():
    ok = True
    for policy in ("pure", "fused"):
        print(f"=== policy: {policy} ===")
        for name in CIRCUITS:
            base, bi = sample_circuit(name, rewrite=False, policy=policy)
            rw, ri = sample_circuit(name, rewrite=True, policy=policy)
            if base.shape != rw.shape:
                print(f"{name:16s} SHAPE MISMATCH base={base.shape} rw={rw.shape}")
                ok = False
                continue
            identical = bool(np.array_equal(base, rw))
            match = float(np.mean(base == rw))
            status = "EXACT" if identical else "FAIL"
            if not identical:
                ok = False
            print(f"{name:16s} identical={identical} match={match:.4f} "
                  f"status={status} | rewrite windows={ri['windows']} "
                  f"local={ri['local']} crossing={ri['crossing']} "
                  f"| qr {bi['qr']}->{ri['qr']} "
                  f"transports {bi['transports']}->{ri['transports']}")
    print("\nparity_rewrite_correctness:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
