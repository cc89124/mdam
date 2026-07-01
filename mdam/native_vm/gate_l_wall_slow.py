#!/usr/bin/env python
"""Gate L: MDAM-only wall for the benches where Clifft is PHYSICALLY infeasible.
  coherent_rx_d5_* (k=38 -> 2^38*16B = 4 TB), coherent_d7_r7 (k=48 -> 4.5 PB).
MDAM runs (localizes), Clifft cannot allocate the register at all.  Few shots (each is seconds-minutes);
the authoritative path is flat-after-warmup so 1-3 shots is a representative ns/shot.  No Clifft column.

  taskset -c 2 python gate_l_wall_slow.py <bench> [N]
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


def main():
    bench = sys.argv[1]
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    text = open(f"{_ROOT}/qec_bench/circuits/{bench}.stim").read()
    prog = clifft.compile(text)
    k = getattr(prog, "peak_rank", 0)
    t = translate(prog); nm = t["num_meas"]
    lib = load_lib()
    lib.nvm_mdam_sample_batch.restype = ctypes.c_int
    P = ctypes.c_void_p
    lib.nvm_mdam_sample_batch.argtypes = [P, P, ctypes.c_uint64] + [ctypes.c_uint64] * 4 + [P, P, P, ctypes.c_int]
    ph = make_prog(lib, t); vm = lib.nvm_mdam_vm_create(ph)
    eb = ctypes.create_string_buffer(256)
    ab = np.zeros((N, nm), np.uint8)
    print(f"== {bench} (k={k}, nmeas={nm}) MDAM-only; Clifft 2^{k} infeasible ==", flush=True)
    t0 = time.perf_counter()
    rc = lib.nvm_mdam_sample_batch(ph, vm, N, *pcg(777), ab.ctypes.data, None, eb, 256)
    dt = time.perf_counter() - t0
    if rc != 0:
        print(f"   native rc={rc} err={eb.value.decode()}", flush=True); sys.exit(1)
    print(f"   MDAM native = {dt / N * 1e9:.1f} ns/shot   ({dt:.2f} s total, N={N})", flush=True)
    cl_bytes = (1 << k) * 16
    print(f"   Clifft       = INFEASIBLE  (2^{k} amplitudes = {cl_bytes/2**40:.1f} TB register)", flush=True)


if __name__ == "__main__":
    main()
