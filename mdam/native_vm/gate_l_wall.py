#!/usr/bin/env python
"""Gate L: AUTHORITATIVE native MDAM wall vs Clifft, on a FUSED-compiled coherent benchmark.

MDAM = nvm_mdam_sample_batch (run_batch -> authoritative run() per shot, fb_mode OFF) -- the localized
near-Clifford path (materializes only the maxM core).  Clifft = clifft.sample (full 2^peak_rank register).
Both run the SAME fused prog.  Reports median ns/shot + ratio.  Run AFTER bit-exactness is confirmed.

  taskset -c 2 python gate_l_wall.py <bench> [mdam_N] [clifft_N]
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import sys, ctypes, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import clifft
from verify_mdam_oneshot import translate, make_prog, pcg, load_lib, _ROOT


def med(fn, reps=3):
    return sorted(fn() for _ in range(reps))[reps // 2]


def main():
    bench = sys.argv[1] if len(sys.argv) > 1 else "coherent_d5_r5"
    mdam_N = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    cl_N = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    text = open(f"{_ROOT}/qec_bench/circuits/{bench}.stim").read()
    prog = clifft.compile(text)                       # FUSED — the authoritative localized algorithm
    k = getattr(prog, "peak_rank", 0)
    t = translate(prog); nm = t["num_meas"]
    lib = load_lib()
    lib.nvm_mdam_sample_batch.restype = ctypes.c_int
    P = ctypes.c_void_p
    lib.nvm_mdam_sample_batch.argtypes = [P, P, ctypes.c_uint64] + [ctypes.c_uint64] * 4 + [P, P, P, ctypes.c_int]
    ph = make_prog(lib, t); vm = lib.nvm_mdam_vm_create(ph)
    eb = ctypes.create_string_buffer(256)

    # warm both paths (first-shot lazy-grow / alloc must not pollute the timed median)
    sb = np.zeros((1, nm), np.uint8)
    rc = lib.nvm_mdam_sample_batch(ph, vm, 1, *pcg(12345), sb.ctypes.data, None, eb, 256)
    if rc != 0:
        print("MDAM native error:", eb.value.decode()); sys.exit(1)
    clifft.sample(prog, 1)

    ab = np.zeros((mdam_N, nm), np.uint8)

    def mdam():
        t0 = time.perf_counter()
        lib.nvm_mdam_sample_batch(ph, vm, mdam_N, *pcg(777), ab.ctypes.data, None, eb, 256)
        return (time.perf_counter() - t0) / mdam_N * 1e9

    def cl():
        t0 = time.perf_counter()
        clifft.sample(prog, cl_N)
        return (time.perf_counter() - t0) / cl_N * 1e9

    print(f"== Gate L authoritative wall: {bench} (FUSED, peak_rank={k}, num_meas={nm}) ==", flush=True)
    m = med(mdam)
    print(f"   MDAM native (localized, authoritative)  : {m:10.1f} ns/shot   (N={mdam_N})", flush=True)
    c = med(cl)
    print(f"   Clifft      (full 2^{k} register)        : {c:10.1f} ns/shot   (N={cl_N})", flush=True)
    print(f"   MDAM / Clifft = {m / c:.3f}x   ({'MDAM WINS' if m < c else 'Clifft wins'} by {max(m, c) / min(m, c):.2f}x)", flush=True)


if __name__ == "__main__":
    main()
