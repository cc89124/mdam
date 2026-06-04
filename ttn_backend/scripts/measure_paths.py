"""Measure TTN transport-sweep memory metrics on selected Clifft circuits."""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, ".")

import clifft

from ttn_backend.backend_spec import assign_homes_and_classify, export_backend_spec
from ttn_backend import TTNBackend


CIRCUITS = ["coherent_d5_r1", "coherent_d5_r5", "coherent_d7_r1"]
SEED = 42
CIRC_DIR = "qec_bench/circuits"


def _mb(n):
    return float(n) / 1e6


def run_one(name):
    print(f"\n{name}", flush=True)
    path = os.path.join(CIRC_DIR, name + ".stim")
    with open(path) as f:
        prog = clifft.compile(f.read())

    t0 = time.time()
    spec = export_backend_spec(prog, strict=False)
    homing = assign_homes_and_classify(spec)
    spec_s = time.time() - t0
    union_bytes = int(spec["union"]["sum2"]) * 16

    backend = TTNBackend(spec, homing)
    t0 = time.time()
    rec = backend.run_shot(prog, SEED)
    shot_s = time.time() - t0
    m = backend.last_metrics or {}

    print(f"  ops={prog.num_instructions} peak_rank={prog.peak_rank} measurements={prog.num_measurements}", flush=True)
    print(f"  spec_time={spec_s:.3f}s shot_time={shot_s:.3f}s records={len(rec)}", flush=True)
    print(f"  union_prediction={_mb(union_bytes):.3f} MB", flush=True)
    print(f"  peak_stored_bytes={m.get('peak_stored_bytes', 0)} ({_mb(m.get('peak_stored_bytes', 0)):.3f} MB)", flush=True)
    print(f"  peak_pair_workspace_bytes={m.get('peak_pair_workspace_bytes', 0)} ({_mb(m.get('peak_pair_workspace_bytes', 0)):.3f} MB)", flush=True)
    print(f"  max_bond_dim={m.get('max_bond_dim', 0)}", flush=True)
    print(f"  n_transports={m.get('n_transports', 0)}", flush=True)
    print(f"  n_qr={m.get('n_qr', 0)}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("circuits", nargs="*", default=CIRCUITS)
    args = parser.parse_args()
    for name in args.circuits:
        run_one(name)


if __name__ == "__main__":
    main()
