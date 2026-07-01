#!/usr/bin/env python
"""Phase-3: authoritative-edge cache (run_mcache) correctness + benefit.
 SHADOW (mode1): always-live + build/verify -> mismatch must be 0, records == authoritative.
 FAST  (mode2): full hit skips measure_z, restores pool snapshot -> records must == authoritative
                on BOTH warmed seeds (high hit) AND fresh seeds (misses handled live)."""
import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ.setdefault(v,"1")
import sys, ctypes, time, statistics
import numpy as np
import os as _os; _HERE_DIR=_os.path.dirname(_os.path.abspath(__file__))
sys.path.insert(0, _HERE_DIR); sys.path.insert(0, _os.path.join(_HERE_DIR, ".."))
from verify_mdam_oneshot import translate, make_prog, pcg, _ROOT, _HERE
import clifft
P=ctypes.c_void_p

def bind():
    lib=ctypes.CDLL(os.path.join(_HERE,"native_mdam_vm.so"))
    lib.nvm_mdam_create.restype=P
    lib.nvm_mdam_create.argtypes=[ctypes.c_int,P,P,P,P,P,P,P,ctypes.c_int,P,ctypes.c_int,P,P,P,P,ctypes.c_int,P,P,ctypes.c_int]+[ctypes.c_int]*5
    lib.nvm_mdam_vm_create.restype=P; lib.nvm_mdam_vm_create.argtypes=[P]
    lib.nvm_mdam_run.restype=ctypes.c_int; lib.nvm_mdam_run.argtypes=[P,P]+[ctypes.c_uint64]*4+[P,P,P,P,P,ctypes.c_int]
    lib.nvm_mdam_run_mcache.restype=ctypes.c_int; lib.nvm_mdam_run_mcache.argtypes=[P,P]+[ctypes.c_uint64]*4+[P,P,ctypes.c_int]
    lib.nvm_mcache_batch.restype=ctypes.c_int; lib.nvm_mcache_batch.argtypes=[P,P,ctypes.c_uint64]+[ctypes.c_uint64]*4+[P,P,ctypes.c_int]
    lib.nvm_mdam_sample_batch.restype=ctypes.c_int; lib.nvm_mdam_sample_batch.argtypes=[P,P,ctypes.c_uint64]+[ctypes.c_uint64]*4+[P,P,P,ctypes.c_int]
    lib.nvm_mcache_set_mode.argtypes=[P,ctypes.c_int]; lib.nvm_mcache_reset.argtypes=[P]; lib.nvm_mcache_stats.argtypes=[P,P]
    return lib

def authoritative(lib,ph,va,sd,nm,eb):
    out=np.zeros(nm,np.uint8); dr=ctypes.c_ulonglong(); cp=ctypes.c_int(); orc=ctypes.c_int()
    lib.nvm_mdam_run(ph,va,*pcg(sd),out.ctypes.data,ctypes.byref(dr),ctypes.byref(cp),ctypes.byref(orc),eb,256)
    return out

def run(bench, warm, test):
    text=open(os.path.join(_ROOT,f"qec_bench/circuits/{bench}.stim")).read()
    prog=clifft.compile(text); t=translate(prog); nm=t["num_meas"]
    progD=clifft.compile(text, hir_passes=clifft.default_hir_pass_manager(), bytecode_passes=clifft.default_bytecode_pass_manager())
    lib=bind(); ph=make_prog(lib,t); va=lib.nvm_mdam_vm_create(ph); vm=lib.nvm_mdam_vm_create(ph)
    eb=ctypes.create_string_buffer(256); rb=np.zeros(nm,np.uint8); st=(ctypes.c_long*10)()
    print(f"== {bench}: warm={warm} test={test} ==")

    # (1) SHADOW: build + verify (records must == authoritative, mismatch 0)
    lib.nvm_mcache_set_mode(vm,1); lib.nvm_mcache_reset(vm)
    eq=0
    for sd in range(warm):
        lib.nvm_mdam_run_mcache(ph,vm,*pcg(sd),rb.ctypes.data,eb,256)
        if np.array_equal(rb, authoritative(lib,ph,va,sd,nm,eb)): eq+=1
    lib.nvm_mcache_stats(vm,st)
    print(f"   SHADOW: records==authoritative {eq}/{warm}  mismatch={st[5]}  verify={st[4]}  pool={st[7]} edges={st[8]} states={st[9]}")

    # (2) FAST on WARMED seeds (high hit) then FRESH seeds (misses handled live) -> both bit-exact
    lib.nvm_mcache_set_mode(vm,2)
    eqw=0
    for sd in range(min(test,warm)):
        lib.nvm_mdam_run_mcache(ph,vm,*pcg(sd),rb.ctypes.data,eb,256)
        if np.array_equal(rb, authoritative(lib,ph,va,sd,nm,eb)): eqw+=1
    nw=min(test,warm)
    eqf=0
    for sd in range(100000, 100000+test):   # fresh seeds never warmed
        lib.nvm_mdam_run_mcache(ph,vm,*pcg(sd),rb.ctypes.data,eb,256)
        if np.array_equal(rb, authoritative(lib,ph,va,sd,nm,eb)): eqf+=1
    s2=(ctypes.c_long*10)(); lib.nvm_mcache_stats(vm,s2)
    tot_b=s2[0]+s2[1]+s2[2]+s2[3]
    print(f"   FAST warmed: {eqw}/{nw} bit-exact   FAST fresh: {eqf}/{test} bit-exact")
    print(f"   FAST counters (cumulative): hit={s2[0]} miss={s2[1]} partial={s2[2]} antis={s2[3]} mismatch={s2[5]} restore={s2[6]}")
    print(f"      hit_rate={s2[0]/max(1,tot_b):.1%}  pool(distinct post-states)={s2[7]}  edges={s2[8]}  interned states={s2[9]}")

    # (3) BATCH correctness + fair timing (ONE C call per batch; same master-seed expansion as sample_batch)
    T=min(max(test,4000),20000); ms=777
    ab=np.zeros((T,nm),np.uint8); mb=np.zeros((T,nm),np.uint8)
    lib.nvm_mdam_sample_batch(ph,va,T,*pcg(ms),ab.ctypes.data,None,eb,256)   # authoritative batch
    lib.nvm_mcache_batch(ph,vm,T,*pcg(ms),mb.ctypes.data,eb,256)             # mcache batch (warms + runs)
    bexact=int(np.count_nonzero(np.any(ab!=mb,axis=1)))
    s3=(ctypes.c_long*10)(); lib.nvm_mcache_stats(vm,s3)
    print(f"   BATCH {T}: mcache!=authoritative = {bexact}   pool={s3[7]} edges={s3[8]} states={s3[9]}")
    reps=7
    def med(fn):
        for _ in range(3): fn()
        return statistics.median([fn() for _ in range(reps)])
    a=med(lambda:(lambda t0:(lib.nvm_mdam_sample_batch(ph,va,T,*pcg(ms),ab.ctypes.data,None,eb,256),(time.perf_counter()-t0)/T*1e9)[1])(time.perf_counter()))
    m=med(lambda:(lambda t0:(lib.nvm_mcache_batch(ph,vm,T,*pcg(ms),mb.ctypes.data,eb,256),(time.perf_counter()-t0)/T*1e9)[1])(time.perf_counter()))
    c=med(lambda:(lambda t0:(clifft.sample(progD,T),(time.perf_counter()-t0)/T*1e9)[1])(time.perf_counter()))
    print(f"   timing ns/shot: authoritative={a:.0f}  mcache_FAST={m:.0f}  Clifft={c:.0f}   (auth/mc={a/m:.2f}x  mc/Clifft={m/c:.2f}x  auth/Clifft={a/c:.2f}x)")
    ok=(eq==warm and st[5]==0 and eqw==nw and eqf==test and s2[5]==0 and bexact==0)
    print(f"   RESULT: {'CORRECT (shadow+fast bit-exact, 0 mismatch)' if ok else 'FAIL'}\n")
    return ok

if __name__=="__main__":
    benches=sys.argv[1].split(",") if len(sys.argv)>1 else ["distillation"]
    warm=int(sys.argv[2]) if len(sys.argv)>2 else 3000
    test=int(sys.argv[3]) if len(sys.argv)>3 else 2000
    allok=True
    for b in benches: allok &= run(b,warm,test)
    sys.exit(0 if allok else 1)
