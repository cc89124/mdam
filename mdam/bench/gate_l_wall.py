#!/usr/bin/env python
"""Focused native-vs-Clifft wall @1M for a single benchmark (default cultivation_d5 -- the FLOP-ratio 0.07x
"MDAM does 14x more FLOP" case).  native = Gate-K FAST (cmode5) MEASURED at real 1M (cross-shot cache
amortization); Clifft = probed + extrapolated (shot-independent).  Bit-exact gate runs before timing."""
import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ.setdefault(v,"1")
import sys, ctypes, time, statistics
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "native_vm")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import clifft
from verify_mdam_oneshot import translate, make_prog, pcg
from gate_k_fast import bind
from wall_compare import setup_native, native_bitexact, time_native, time_clifft, clifft_progD

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SHOTS = 1_000_000


def main():
    bench = sys.argv[1] if len(sys.argv) > 1 else "cultivation_d5"
    text = open(f"{ROOT}/qec_bench/circuits/{bench}.stim").read()
    prog = clifft.compile(text); k = getattr(prog, "peak_rank", 0)
    progD = clifft_progD(text)
    lib = bind()
    lib.nvm_j2e_noise_skip.argtypes = [ctypes.c_int]; lib.nvm_j2e_noise_skip(1)
    lib.nvm_j2e_time.argtypes = [ctypes.c_int]; lib.nvm_j2e_time(0)
    ws = (ctypes.c_long*20)()

    print(f"=== {bench}  (n={prog.num_qubits}, peak_rank={k}) wall @ {SHOTS:,} shots ===")
    cl_ns, cl_n = time_clifft(progD)
    print(f"Clifft : {cl_ns:9.1f} ns/shot  (probe={cl_n}, total@1M = {cl_ns*SHOTS/1e9:.3f} s)")

    t = translate(prog); nm = t["num_meas"]
    ph, cp, vm, jp, eb = setup_native(lib, t)
    if not native_bitexact(lib, ph, cp, vm, jp, nm, ws, eb, n=2000):
        print("native FAST not bit-exact -> aborting"); sys.exit(1)
    print("native FAST cmode5 == authoritative: bit-exact (2000 shots) OK")
    nat_ns = time_native(lib, ph, cp, vm, jp, nm, ws)
    print(f"native : {nat_ns:9.1f} ns/shot  (measured @1M, total@1M = {nat_ns*SHOTS/1e9:.3f} s)")
    print(f"\nnative/clifft = {nat_ns/cl_ns:.3f}x   (>1 native slower, <1 native faster)")


if __name__ == "__main__":
    main()
