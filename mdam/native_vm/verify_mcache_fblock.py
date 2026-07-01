#!/usr/bin/env python
"""Gate N — frame-block superinstruction (`mc_fblock`, default OFF) correctness + timing.

distillation is 81% pure MO_FRAME_* opcodes (1625/1995 ops, in 90 runs, mean 18 / max 52). Each pays a
full big-switch dispatch (6 array loads + jump) for an ~8-cyc XOR body.  `mc_fblock` batches each maximal
run of pure frame opcodes into ONE dispatch + a tight grow-hoisted inner loop — executing the IDENTICAL
ops in the IDENTICAL order, so it is bit-exact by construction.

Ground truth = the native AUTHORITATIVE path (`nvm_mdam_sample_batch` on a no-cache VM == Python).
NEVER a cmode-vs-cmode comparison.  Timing is interleaved (OFF / ON / Clifft per rep) so machine drift
cancels; cold fresh-run (mc_reset each rep, cache built during the run -> real miss rate).

Usage:  verify_mcache_fblock.py [bench1,bench2,...] [shots] [nseed]
Default: distillation,cultivation_d3,cultivation_d5
"""
import os, sys
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import ctypes, time, statistics
import numpy as np

_HERE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE_DIR); sys.path.insert(0, os.path.join(_HERE_DIR, ".."))
from verify_mdam_oneshot import translate, make_prog, pcg, _ROOT, load_lib
import clifft
P = ctypes.c_void_p


def bind():
    lib = load_lib()
    lib.nvm_mcache_batch.restype = ctypes.c_int
    lib.nvm_mcache_batch.argtypes = [P, P, ctypes.c_uint64] + [ctypes.c_uint64] * 4 + [P, P, ctypes.c_int]
    lib.nvm_mdam_sample_batch.restype = ctypes.c_int
    lib.nvm_mdam_sample_batch.argtypes = [P, P, ctypes.c_uint64] + [ctypes.c_uint64] * 4 + [P, P, P, ctypes.c_int]
    for f in ("nvm_mcache_set_mode", "nvm_mcache_set_fblock"):
        getattr(lib, f).argtypes = [P, ctypes.c_int]; getattr(lib, f).restype = None
    lib.nvm_mcache_reset.argtypes = [P]; lib.nvm_mcache_reset.restype = None
    lib.nvm_mcache_stats.argtypes = [P, P]; lib.nvm_mcache_stats.restype = None
    return lib


def run(bench, T, nseed, reps=15):
    text = open(os.path.join(_ROOT, f"qec_bench/circuits/{bench}.stim")).read()
    prog = clifft.compile(text); t = translate(prog); nm = t["num_meas"]
    progD = clifft.compile(text)            # Clifft baseline (same default passes incl. squeeze)
    lib = bind(); ph = make_prog(lib, t)
    va = lib.nvm_mdam_vm_create(ph); vm = lib.nvm_mdam_vm_create(ph)
    eb = ctypes.create_string_buffer(256)

    # --- correctness: fblock ON in EVERY cache mode (1 SHADOW, 2 snapshot, 3 carry) vs authoritative ---
    tot = 0
    for s in range(1, nseed + 1):
        ms = s * 7919
        ba = np.zeros((T, nm), np.uint8)
        lib.nvm_mdam_sample_batch(ph, va, T, *pcg(ms), ba.ctypes.data, None, eb, 256)
        for mode in (1, 2, 3):
            bf = np.zeros((T, nm), np.uint8)
            lib.nvm_mcache_set_mode(vm, mode); lib.nvm_mcache_set_fblock(vm, 1); lib.nvm_mcache_reset(vm)
            lib.nvm_mcache_batch(ph, vm, T, *pcg(ms), bf.ctypes.data, eb, 256)
            d = int((ba != bf).sum()); tot += d
            if d:
                print(f"   !! {bench} seed#{s} mode{mode}: {d} mismatches")
    ok = (tot == 0)
    print(f"== {bench}: fblock modes 1/2/3 vs authoritative, {nseed} seeds x {T} shots: "
          f"mismatch={tot}  {'OK' if ok else 'FAIL'}")

    # --- timing: carry (mode 3) fblock OFF vs ON vs Clifft, interleaved (drift cancels), cold fresh-run ---
    buf = np.zeros((T, nm), np.uint8); ms = 424242

    def carry(fb):
        lib.nvm_mcache_set_mode(vm, 3); lib.nvm_mcache_set_fblock(vm, fb); lib.nvm_mcache_reset(vm)
        t0 = time.perf_counter(); lib.nvm_mcache_batch(ph, vm, T, *pcg(ms), buf.ctypes.data, eb, 256)
        return (time.perf_counter() - t0) / T * 1e9

    def cl():
        t0 = time.perf_counter(); clifft.sample(progD, T)
        return (time.perf_counter() - t0) / T * 1e9

    for _ in range(3):
        carry(0); carry(1); cl()
    OFF, ON, CL = [], [], []
    for _ in range(reps):
        OFF.append(carry(0)); ON.append(carry(1)); CL.append(cl())
    o, n, c = statistics.median(OFF), statistics.median(ON), statistics.median(CL)
    st = (ctypes.c_long * 10)(); lib.nvm_mcache_stats(vm, st)
    bnd = st[0] + st[1] + st[2] + st[3]; hr = st[0] / max(1, bnd)
    rel = "MDAM faster" if n < c else "Clifft faster"
    print(f"   carry ns/shot (interleaved {reps} reps): OFF={o:.0f}  ON={n:.0f}  Clifft={c:.0f}  hit={hr:.1%}")
    print(f"     fblock saves {o - n:.0f} ns ({(o - n) / o:.1%}) | ON/Clifft={n / c:.2f}x ({rel} by {abs(n - c):.0f} ns)")
    print()
    return ok


if __name__ == "__main__":
    benches = sys.argv[1].split(",") if len(sys.argv) > 1 else ["distillation", "cultivation_d3", "cultivation_d5"]
    T = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
    nseed = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    allok = True
    for b in benches:
        Tb = T if b != "cultivation_d5" else min(T, 2000)
        allok &= run(b, Tb, nseed)
    print("ALL FBLOCK CHECKS PASS" if allok else "FBLOCK CHECKS FAILED")
    sys.exit(0 if allok else 1)
