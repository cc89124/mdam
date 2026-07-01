#!/usr/bin/env python
"""Gate L: AUTHORITATIVE native MDAM wall vs Clifft across ALL benchmarks (FUSED-compiled).

MDAM = nvm_mdam_sample_batch (run_batch -> authoritative run() per shot, fb_mode OFF) -- the localized
near-Clifford path.  Clifft = clifft.sample (full 2^peak_rank register).  Both run the SAME fused prog.

Per the user request: drive MDAM to a high shot count (caching/warmup amortizes; saturates fast on the
authoritative path) and Clifft once (its per-shot cost is flat).  N is adapted per bench to a wall budget
(authoritative path is flat-after-warmup, so a smaller N on a slow bench is representative).  Clifft is
skipped when 2^k is physically infeasible (e.g. k=38 -> 4 TB).

  taskset -c 2 python gate_l_wall_all.py [bench ...]      # default: all native-supported
Emits a TSV line per bench (parseable) plus a human header.
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import sys, ctypes, time, gc
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import clifft
from verify_mdam_oneshot import translate, make_prog, pcg, load_lib, _ROOT

ALL = ["coherent_d3_r1", "coherent_d3_r3", "coherent_d5_r1", "coherent_d5_r5",
       "coherent_d7_r1", "coherent_d7_r7",
       "coherent_rx_d3_r1", "coherent_rx_d3_r3", "coherent_rx_d5_r1", "coherent_rx_d5_r5",
       "coherent_ry_d3_r1", "coherent_ry_d3_r3", "coherent_ry_d5_r1", "coherent_ry_d5_r5",
       "cultivation_d3", "cultivation_d5", "distillation", "surface_d7_r7"]

MDAM_BUDGET_NS = 30e9     # ~30 s of timed MDAM work per bench
MDAM_CAP = 1_000_000      # honor the user's 1M ceiling where it fits the budget
MDAM_MIN = 1              # allow tiny N on very slow benches (authoritative path is flat-after-warmup)
CLIFFT_K_MAX = 26         # 2^26 * 16 B = 1 GB, ~30 s/shot -- beyond this Clifft is skipped (infeasible/too slow)


def measure_mdam(lib, ph, vm, nm, N, seed):
    ab = np.zeros((N, nm), np.uint8)
    eb = ctypes.create_string_buffer(256)
    t0 = time.perf_counter()
    rc = lib.nvm_mdam_sample_batch(ph, vm, N, *pcg(seed), ab.ctypes.data, None, eb, 256)
    dt = time.perf_counter() - t0
    if rc != 0:
        raise RuntimeError(eb.value.decode())
    del ab; gc.collect()
    return dt / N * 1e9


def main():
    targets = sys.argv[1:] or ALL
    lib = load_lib()
    lib.nvm_mdam_sample_batch.restype = ctypes.c_int
    P = ctypes.c_void_p
    lib.nvm_mdam_sample_batch.argtypes = [P, P, ctypes.c_uint64] + [ctypes.c_uint64] * 4 + [P, P, P, ctypes.c_int]

    print("# bench\tk\tnmeas\tmdam_N\tmdam_ns\tclifft_N\tclifft_ns\tspeedup", flush=True)
    for b in targets:
        text = open(f"{_ROOT}/qec_bench/circuits/{b}.stim").read()
        prog = clifft.compile(text)
        k = getattr(prog, "peak_rank", 0)
        try:
            t = translate(prog)
        except Exception as e:
            print(f"{b}\t{k}\t-\t-\tNO-TRANSLATE\t-\t-\t- ({str(e)[:40]})", flush=True); continue
        nm = t["num_meas"]
        ph = make_prog(lib, t); vm = lib.nvm_mdam_vm_create(ph)
        # warmup + per-shot estimate (also exercises lazy-grow / first-shot alloc)
        eb = ctypes.create_string_buffer(256)
        wb = np.zeros((50, nm), np.uint8)
        rc = lib.nvm_mdam_sample_batch(ph, vm, 50, *pcg(11), wb.ctypes.data, None, eb, 256)
        if rc != 0:
            print(f"{b}\t{k}\t{nm}\t-\tNATIVE-ERR\t-\t-\t- ({eb.value.decode()[:40]})", flush=True); continue
        t0 = time.perf_counter()
        lib.nvm_mdam_sample_batch(ph, vm, 50, *pcg(13), wb.ctypes.data, None, eb, 256)
        ps = (time.perf_counter() - t0) / 50 * 1e9
        del wb
        N = int(min(MDAM_CAP, max(MDAM_MIN, MDAM_BUDGET_NS / max(ps, 1.0))))
        mdam_ns = measure_mdam(lib, ph, vm, nm, N, 777)

        # Clifft baseline (full register) -- skip when infeasible
        if k > CLIFFT_K_MAX:
            print(f"{b}\t{k}\t{nm}\t{N}\t{mdam_ns:.1f}\t-\tINFEASIBLE(2^{k})\tMDAM-only", flush=True)
        else:
            clifft.sample(prog, 1)  # warm
            cl_N = 2 if k >= 22 else (20 if k >= 14 else 200)
            t0 = time.perf_counter()
            clifft.sample(prog, cl_N)
            cl_ns = (time.perf_counter() - t0) / cl_N * 1e9
            sp = cl_ns / mdam_ns
            print(f"{b}\t{k}\t{nm}\t{N}\t{mdam_ns:.1f}\t{cl_N}\t{cl_ns:.1f}\t{sp:.2f}x", flush=True)


if __name__ == "__main__":
    main()
