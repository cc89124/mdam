#!/usr/bin/env python
"""Gate K shot-count sweep v2 — FAIR warmup-amortization curve, MDAM-FAST (cmode5) vs Clifft.

Fixes two confounds from v1: (a) output handling — MDAM now uses reuse_buf=0 (store full N×nm records) to
MATCH clifft.sample (which always stores), so neither side gets a free cache-locality win; (b) throttle/output
streaming — controlled by comparing MDAM-COLD vs MDAM-WARM at the SAME N (same duration, same output, same
thermal state; ONLY the cache state differs → clean warmup signal).

Also measures whether the edge/intern caches SATURATE: cumulative cold run, distinct_keys at each checkpoint.
If distinct_keys keeps growing ~linearly with N, there is no warm steady-state at realistic shot counts."""
import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ.setdefault(v,"1")
import sys, ctypes, time, statistics
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),"..")))
from verify_mdam_oneshot import translate, make_prog, pcg, BENCH, _ROOT, _HERE
from gate_k_fast import bind
import clifft

def main():
    text=open(os.path.join(_ROOT,f"qec_bench/circuits/{BENCH}.stim")).read()
    prog=clifft.compile(text); t=translate(prog); nm=t["num_meas"]
    progD=clifft.compile(text, hir_passes=clifft.default_hir_pass_manager(), bytecode_passes=clifft.default_bytecode_pass_manager())
    lib=bind(); ph=make_prog(lib,t); eb=ctypes.create_string_buffer(256)
    lib.nvm_j2e_noise_skip.argtypes=[ctypes.c_int]; lib.nvm_j2e_time.argtypes=[ctypes.c_int]
    lib.nvm_mdam_vm_free.argtypes=[ctypes.c_void_p]; lib.nvm_jphase_free.argtypes=[ctypes.c_void_p]
    info=(ctypes.c_int*5)(); cp=lib.nvm_jcompile(ph); lib.nvm_jcompile_info(cp,info)
    ws=(ctypes.c_long*20)()
    lib.nvm_j2e_noise_skip(1); lib.nvm_j2e_time(0)
    SEED=777

    def new_vm():
        vm=lib.nvm_mdam_vm_create(ph); jpl=lib.nvm_jphase_compile(ph,vm,*pcg(12345))
        lib.nvm_mdam_vm_set_imem(vm,2); lib.nvm_mdam_vm_set_fb(vm,1)
        sb=np.zeros((1,nm),np.uint8); lib.nvm_mdam_sample_batch(ph,vm,1,*pcg(12345),sb.ctypes.data,None,eb,256)
        return vm,jpl
    def warm(vm,jpl):     # the gate_k_noise_skip warmup: ~240k shots across 2f/2g/5 → saturate every cache
        wb=np.zeros(nm,np.uint8)
        for ms in [12345,1,777,2026,99991,3,31337,424242,2024,2025]:
            lib.nvm_jfast2f_batch(ph,cp,jpl,vm,8000,*pcg(ms),wb.ctypes.data,1,ws)
            lib.nvm_jfast2g_batch(ph,cp,jpl,vm,8000,*pcg(ms),wb.ctypes.data,1,ws)
            lib.nvm_jfast5_batch(ph,cp,jpl,vm,8000,*pcg(ms),wb.ctypes.data,1,ws)

    print(f"== Gate K shot-count sweep v2 (FAIR: reuse_buf=0 both sides) — {BENCH} ==")

    # (A) cache saturation: ONE cold vm, cumulative chunks, distinct_keys + cumulative hit% at each checkpoint
    Ns=[1000,8000,32000,128000,1000000]
    vm,jpl=new_vm(); st=(ctypes.c_long*20)(); done=0
    print(f"\n   (A) cache growth (cold, cumulative)")
    print(f"   {'shots':>9} {'distinct_keys':>14} {'cum_lookups':>12} {'cum_hit%':>9}")
    for N in Ns:
        chunk=N-done
        buf=np.zeros((chunk,nm),np.uint8)
        lib.nvm_jfast5_batch(ph,cp,jpl,vm,chunk,*pcg(SEED+done),buf.ctypes.data,0,ws); done=N
        lib.nvm_jkcache_stats(vm,st); lk,hit,dk=st[0],st[1],st[9]
        print(f"   {N:>9} {dk:>14} {lk:>12} {100*hit/max(1,lk):>8.1f}%")
    lib.nvm_jphase_free(jpl); lib.nvm_mdam_vm_free(vm)

    # (B) FAIR timing: MDAM-cold(fresh vm) / MDAM-warm(pre-warmed vm) / Clifft, ALL store full N×nm output.
    # REP-INTERLEAVED: within each rep, measure cold, warm, Clifft back-to-back → shared thermal envelope
    # (kills the multi-second throttle drift that made warm>cold at 128k in the non-interleaved version).
    reps_for=lambda N: 9 if N<=32000 else (5 if N<=128000 else 3)
    vmw,jpw=new_vm(); warm(vmw,jpw)   # one thoroughly-warmed vm, reused for all warm measurements
    def t_cold(N):
        vm,jpl=new_vm(); buf=np.zeros((N,nm),np.uint8)
        t0=time.perf_counter(); lib.nvm_jfast5_batch(ph,cp,jpl,vm,N,*pcg(SEED),buf.ctypes.data,0,ws); el=time.perf_counter()-t0
        lib.nvm_jphase_free(jpl); lib.nvm_mdam_vm_free(vm); return el/N*1e9
    def t_warm(N):
        buf=np.zeros((N,nm),np.uint8)
        t0=time.perf_counter(); lib.nvm_jfast5_batch(ph,cp,jpw,vmw,N,*pcg(SEED),buf.ctypes.data,0,ws); return (time.perf_counter()-t0)/N*1e9
    def t_cl(N):
        t0=time.perf_counter(); clifft.sample(progD,N); return (time.perf_counter()-t0)/N*1e9

    print(f"\n   (B) FAIR timing (ns/shot, full output both sides; rep-interleaved)")
    print(f"   {'shots':>9} {'cold':>8} {'warm':>8} {'Clifft':>8} {'cold/Cl':>8} {'warm/Cl':>8} {'warmup/sh':>10}")
    for N in Ns:
        R=reps_for(N); cs=[]; wsr=[]; cls=[]
        t_cold(N); t_warm(N); t_cl(N)   # one untimed warmup rep at this N
        for _ in range(R):
            cs.append(t_cold(N)); wsr.append(t_warm(N)); cls.append(t_cl(N))
        c=statistics.median(cs); w=statistics.median(wsr); cl=statistics.median(cls)
        print(f"   {N:>9} {c:>8.0f} {w:>8.0f} {cl:>8.0f} {c/cl:>7.2f}x {w/cl:>7.2f}x {c-w:>10.0f}")
    lib.nvm_jphase_free(jpw); lib.nvm_mdam_vm_free(vmw)
    print(f"\n   READ: warm/Cl = steady-state ratio (cache saturated).  cold/Cl = run-from-scratch incl. warmup.")
    print(f"        cold-warm = per-shot warmup penalty amortized at that N (one-time iff distinct_keys saturates).")

if __name__=="__main__":
    main()
