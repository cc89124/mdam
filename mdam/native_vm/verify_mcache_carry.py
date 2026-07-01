#!/usr/bin/env python
"""Phase-4: mcache_carry (mode 3) — bit-exact vs authoritative + cost decomposition + timing vs snapshot(mode2)."""
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
    lib.nvm_mcache_set_mode.argtypes=[P,ctypes.c_int]; lib.nvm_mcache_reset.argtypes=[P]
    lib.nvm_mcache_set_time.argtypes=[P,ctypes.c_int]; lib.nvm_mcache_cyc_get.argtypes=[P,P]; lib.nvm_mcache_stats.argtypes=[P,P]
    return lib
def auth1(lib,ph,va,sd,nm,eb):
    o=np.zeros(nm,np.uint8); dr=ctypes.c_ulonglong(); cp=ctypes.c_int(); orc=ctypes.c_int()
    lib.nvm_mdam_run(ph,va,*pcg(sd),o.ctypes.data,ctypes.byref(dr),ctypes.byref(cp),ctypes.byref(orc),eb,256); return o
def run(bench, warm, test):
    text=open(os.path.join(_ROOT,f"qec_bench/circuits/{bench}.stim")).read()
    prog=clifft.compile(text); t=translate(prog); nm=t["num_meas"]
    progD=clifft.compile(text, hir_passes=clifft.default_hir_pass_manager(), bytecode_passes=clifft.default_bytecode_pass_manager())
    lib=bind(); ph=make_prog(lib,t); va=lib.nvm_mdam_vm_create(ph); vm=lib.nvm_mdam_vm_create(ph)
    eb=ctypes.create_string_buffer(256); rb=np.zeros(nm,np.uint8); st=(ctypes.c_long*10)()
    print(f"== {bench}: warm={warm} test={test} ==")
    # mode 3 CARRY correctness: warm then per-shot warmed + fresh vs authoritative
    lib.nvm_mcache_set_mode(vm,3); lib.nvm_mcache_reset(vm)
    for sd in range(warm): lib.nvm_mdam_run_mcache(ph,vm,*pcg(sd),rb.ctypes.data,eb,256)
    eqw=sum(np.array_equal((lib.nvm_mdam_run_mcache(ph,vm,*pcg(sd),rb.ctypes.data,eb,256),rb)[1], auth1(lib,ph,va,sd,nm,eb)) for sd in range(min(test,warm)))
    eqf=sum(np.array_equal((lib.nvm_mdam_run_mcache(ph,vm,*pcg(sd),rb.ctypes.data,eb,256),rb)[1], auth1(lib,ph,va,sd,nm,eb)) for sd in range(100000,100000+test))
    lib.nvm_mcache_stats(vm,st)
    print(f"   CARRY(3): warmed {eqw}/{min(test,warm)}  fresh {eqf}/{test}  mismatch={st[5]}  hit={st[0]} miss={st[1]} antis={st[3]}")
    # BATCH carry vs authoritative
    T=min(max(test,4000),20000); ms=777
    ab=np.zeros((T,nm),np.uint8); mb=np.zeros((T,nm),np.uint8)
    lib.nvm_mdam_sample_batch(ph,va,T,*pcg(ms),ab.ctypes.data,None,eb,256)
    lib.nvm_mcache_set_mode(vm,3); lib.nvm_mcache_batch(ph,vm,T,*pcg(ms),mb.ctypes.data,eb,256)
    print(f"   CARRY BATCH {T}: mcache3!=authoritative = {int(np.count_nonzero(np.any(ab!=mb,axis=1)))}")
    # cost decomposition mode 3 (carry) — key-hash should collapse
    wbuf=np.zeros((warm,nm),np.uint8)   # persistent (avoid temporary-array GC -> dangling ctypes.data)
    for mode,nmstr in ((2,"snapshot"),(3,"carry")):
        lib.nvm_mcache_set_mode(vm,mode); lib.nvm_mcache_reset(vm)
        lib.nvm_mcache_batch(ph,vm,warm,*pcg(ms),wbuf.ctypes.data,eb,256)
        s0=(ctypes.c_long*10)(); lib.nvm_mcache_stats(vm,s0); h0=s0[0]
        lib.nvm_mcache_set_time(vm,1); lib.nvm_mcache_batch(ph,vm,warm,*pcg(ms),wbuf.ctypes.data,eb,256); lib.nvm_mcache_set_time(vm,0)
        cyc=(ctypes.c_uint64*8)(); lib.nvm_mcache_cyc_get(vm,cyc)
        s1=(ctypes.c_long*10)(); lib.nvm_mcache_stats(vm,s1); hits=max(1,s1[0]-h0)
        print(f"   [{nmstr}] cyc/hit: key-hash={cyc[0]/(hits or 1):.0f} restore={cyc[2]/hits:.0f} hit-total={cyc[5]/hits:.0f}")
    # timing: auth / snapshot / carry / Clifft (batch)
    reps=7
    def med(fn):
        for _ in range(3): fn()
        return statistics.median([fn() for _ in range(reps)])
    def tb(mode):
        lib.nvm_mcache_set_mode(vm,mode); lib.nvm_mcache_reset(vm); lib.nvm_mcache_batch(ph,vm,T,*pcg(ms),mb.ctypes.data,eb,256)  # warm
        return med(lambda:(lambda t0:(lib.nvm_mcache_batch(ph,vm,T,*pcg(ms),mb.ctypes.data,eb,256),(time.perf_counter()-t0)/T*1e9)[1])(time.perf_counter()))
    a=med(lambda:(lambda t0:(lib.nvm_mdam_sample_batch(ph,va,T,*pcg(ms),ab.ctypes.data,None,eb,256),(time.perf_counter()-t0)/T*1e9)[1])(time.perf_counter()))
    m2=tb(2); m3=tb(3)
    c=med(lambda:(lambda t0:(clifft.sample(progD,T),(time.perf_counter()-t0)/T*1e9)[1])(time.perf_counter()))
    print(f"   timing ns/shot: auth={a:.0f}  snapshot={m2:.0f}  carry={m3:.0f}  Clifft={c:.0f}  (carry/snapshot={m3/m2:.2f}x  carry/Clifft={m3/c:.2f}x)")
    print()
if __name__=="__main__":
    for b in (sys.argv[1].split(",") if len(sys.argv)>1 else ["distillation","cultivation_d3","cultivation_d5"]):
        run(b, 4000 if b!="cultivation_d5" else 1500, 2000 if b!="cultivation_d5" else 800)
