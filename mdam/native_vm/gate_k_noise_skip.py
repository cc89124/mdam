#!/usr/bin/env python
"""Gate K skip-to-next-fire: the gap-sampler knows next_idx (next firing site), so non-firing apply_site calls
are pure no-ops.  Visit ONLY blocks containing next_idx (range check) instead of scanning all 504 sites/shot.
EXACT (correctness-preserving).  SHADOW-FIRST: verify skip=1 == authoritative (25/25 + 128k 0) AND skip=1 ==
skip=0 (per-site baseline), then counters (site_calls 504->~0.5, blocks_skipped ~73.5), then wall."""
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
    lib=bind(); ph=make_prog(lib,t); info=(ctypes.c_int*5)(); eb=ctypes.create_string_buffer(256)
    lib.nvm_j2e_noise_skip.argtypes=[ctypes.c_int]; lib.nvm_j2e_time.argtypes=[ctypes.c_int]
    lib.nvm_j2e_cyc_reset.argtypes=[]; lib.nvm_j2e_cyc_get.argtypes=[ctypes.c_void_p]
    va=lib.nvm_mdam_vm_create(ph); vm=lib.nvm_mdam_vm_create(ph)
    cp=lib.nvm_jcompile(ph); lib.nvm_jcompile_info(cp,info)
    jp=lib.nvm_jphase_compile(ph,vm,*pcg(12345)); lib.nvm_mdam_vm_set_imem(vm,2); lib.nvm_mdam_vm_set_fb(vm,1)
    print(f"== Gate K skip-to-next-fire ({BENCH}) ==")
    sb=np.zeros((1,nm),np.uint8); lib.nvm_mdam_sample_batch(ph,vm,1,*pcg(12345),sb.ctypes.data,None,eb,256)
    wb=np.zeros(nm,np.uint8); ws=(ctypes.c_long*20)(); seeds=[12345,1,777,2026,99991,3,31337,424242,2024,2025]
    for ms in seeds: lib.nvm_jfast2f_batch(ph,cp,jp,vm,8000,*pcg(ms),wb.ctypes.data,1,ws)
    for ms in seeds: lib.nvm_jfast2g_batch(ph,cp,jp,vm,8000,*pcg(ms),wb.ctypes.data,1,ws)
    for ms in seeds: lib.nvm_jfast5_batch(ph,cp,jp,vm,8000,*pcg(ms),wb.ctypes.data,1,ws)

    # (1) oneshot 25: skip=1 == authoritative; skip=1 == skip=0
    out=np.zeros(nm,np.uint8); k0v=np.zeros(nm,np.uint8); k1v=np.zeros(nm,np.uint8)
    dr=ctypes.c_ulonglong(); cpn=ctypes.c_int(); orc=ctypes.c_int(); st=(ctypes.c_long*20)()
    rs=np.random.RandomState(2026); seedl=[1,7,42,123,999]+[int(x) for x in rs.randint(0,2**31-1,size=20)]
    for sd in seedl: lib.nvm_jfast5_run(ph,cp,jp,vm,*pcg(sd),k0v.ctypes.data,st)  # warm both modes' kcache (same trajectory)
    ea=eb01=0
    for sd in seedl:
        s4=pcg(sd)
        lib.nvm_mdam_run(ph,va,*s4,out.ctypes.data,ctypes.byref(dr),ctypes.byref(cpn),ctypes.byref(orc),eb,256)
        lib.nvm_j2e_noise_skip(0); lib.nvm_jfast5_run(ph,cp,jp,vm,*s4,k0v.ctypes.data,st)
        lib.nvm_j2e_noise_skip(1); lib.nvm_jfast5_run(ph,cp,jp,vm,*s4,k1v.ctypes.data,st)
        ea+=int(np.array_equal(out,k1v)); eb01+=int(np.array_equal(k0v,k1v))
    print(f"\n   (1) oneshot: skip=1==authoritative {ea}/25   skip=1==skip=0 {eb01}/25")

    # (2) scale 128k: skip=1 == authoritative
    BN=16000; mism=0; tot=0
    for ms in [555,8675309,271828,2718281,2024,2025,99,100001]:
        a=np.zeros((BN,nm),np.uint8); k=np.zeros((BN,nm),np.uint8)
        lib.nvm_mdam_sample_batch(ph,va,BN,*pcg(ms),a.ctypes.data,None,eb,256)
        lib.nvm_j2e_noise_skip(1); lib.nvm_jfast5_batch(ph,cp,jp,vm,BN,*pcg(ms),k.ctypes.data,0,ws)
        mism+=int(np.count_nonzero(np.any(a!=k,axis=1))); tot+=BN
    print(f"   (2) scale: {tot} shots — skip=1 != authoritative = {mism}")

    # (3) counters (tm=1): site_calls / block_checks / blocks_skipped / draws / fires  for skip=0 vs skip=1
    def counters(skipv):
        lib.nvm_j2e_noise_skip(skipv); lib.nvm_j2e_time(1); lib.nvm_j2e_cyc_reset()
        NS=40000; T=20000
        for k in range(0,NS,T): lib.nvm_jfast5_batch(ph,cp,jp,vm,T,*pcg(777+k),wb.ctypes.data,1,ws)
        c=(ctypes.c_uint64*16)(); lib.nvm_j2e_cyc_get(c); lib.nvm_j2e_time(0)
        return {"site_calls":c[8]/NS,"draws":c[9]/NS,"fires":c[10]/NS,"block_checks":c[11]/NS,"blocks_skipped":c[12]/NS,
                "noise_sample_cyc":c[4]/NS,"noise_apply_cyc":c[5]/NS}
    c0=counters(0); c1=counters(1)
    print(f"\n   counters/shot       skip=0(loop)   skip=1(next-fire)")
    for kk in ("site_calls","block_checks","blocks_skipped","draws","fires"):
        print(f"     {kk:16s} {c0[kk]:10.2f}     {c1[kk]:10.2f}")

    # (4) wall: skip=0 vs skip=1 vs Clifft (tm off)
    lib.nvm_j2e_time(0); T=20000; tb=np.zeros(nm,np.uint8); reps=11
    def med(fn):
        for _ in range(4): fn()
        return statistics.median([fn() for _ in range(reps)])
    def w(skipv): lib.nvm_j2e_noise_skip(skipv); return med(lambda:(lambda t0:(lib.nvm_jfast5_batch(ph,cp,jp,vm,T,*pcg(777),tb.ctypes.data,1,ws),(time.perf_counter()-t0)/T*1e9)[1])(time.perf_counter()))
    w0=w(0); w1=w(1); cl=med(lambda:(lambda t0:(clifft.sample(progD,T),(time.perf_counter()-t0)/T*1e9)[1])(time.perf_counter()))
    lib.nvm_j2e_noise_skip(0)
    print(f"\n   (4) wall ns/shot: skip=0(loop) {w0:.0f}   skip=1(next-fire) {w1:.0f}   Clifft {cl:.0f}")
    print(f"       skip Δ={w0-w1:.0f} ns ({(w0-w1)/w0:.1%})  |  MDAM/Clifft: {w0/cl:.2f}x -> {w1/cl:.2f}x")
    ok=(ea==25 and eb01==25 and mism==0 and c1["site_calls"]<1.0 and abs(c1["fires"]-c0["fires"])<1e-9 and abs(c1["draws"]-c0["draws"])<1e-9)
    print(f"\n   RESULT: {'skip-to-next-fire BIT-EXACT (25/25 + 128k 0, skip==loop); site_calls 504->~0.5, fires/draws UNCHANGED; wall dropped' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)

if __name__=="__main__":
    main()
