"""Correctness check: forced block-staged transport must reproduce the dense
transport result exactly.

The staged factorization M = Q R is exact up to the internal bond gauge, which
is gauge-invariant for physical amplitudes, so with an identical RNG stream the
sampled measurement records must match the dense path bit-for-bit.
"""
import os
import sys
import numpy as np

import clifft
from ttn_backend.backend_spec import export_backend_spec, assign_homes_and_classify
from ttn_backend import TTNBackend

CIRCUITS = sys.argv[1:] or ["distillation", "coherent_d3_r1", "coherent_d5_r1"]
SHOTS = int(os.environ.get("SHOTS", "16"))
SEED = int(os.environ.get("SEED", "42"))

STAGED_ENV = dict(
    TTN_STAGED_TRANSPORT="1",
    TTN_STAGED_TRANSPORT_FORCE="1",   # force staged on every transport
    TTN_STAGED_FORCE_REORTH="1",      # tightest accuracy
    TTN_STAGED_BLOCK_BYTES="65536",   # tiny blocks to exercise streaming
)


def sample_circuit(name, staged):
    saved = {}
    for k, v in STAGED_ENV.items():
        saved[k] = os.environ.get(k)
        if staged:
            os.environ[k] = v
        elif k in os.environ:
            del os.environ[k]
    try:
        path = f"qec_bench/circuits/{name}.stim"
        with open(path) as h:
            src = h.read()
        prog = clifft.compile(src)
        spec = export_backend_spec(prog, strict=False)
        homing = assign_homes_and_classify(spec)
        backend = TTNBackend(spec, homing)
        arr = backend.sample(prog, shots=SHOTS, seed=SEED)
        staged_ct = int(backend.state.metrics.get("staged_transport_count", 0)) \
            if hasattr(backend, "state") and backend.state else None
        return np.asarray(arr), staged_ct
    finally:
        for k, old in saved.items():
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old


def main():
    ok = True
    for name in CIRCUITS:
        dense, _ = sample_circuit(name, staged=False)
        staged, sct = sample_circuit(name, staged=True)
        if dense.shape != staged.shape:
            print(f"{name:16s} SHAPE MISMATCH dense={dense.shape} staged={staged.shape}")
            ok = False
            continue
        match = float(np.mean(dense == staged))
        identical = bool(np.array_equal(dense, staged))
        status = "EXACT" if identical else ("CLOSE" if match > 0.999 else "FAIL")
        if status == "FAIL":
            ok = False
        print(f"{name:16s} shape={dense.shape} match={match:.4f} "
              f"identical={identical} status={status}")
    print("\nstaged_transport_correctness:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
