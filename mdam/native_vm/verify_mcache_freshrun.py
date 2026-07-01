#!/usr/bin/env python
"""Phase-4 follow-up: REALISTIC fresh-run end-to-end (NOT warm all-hit best case).
Cold run = mc_reset + run T shots from EMPTY cache (cache builds during the run) -> the true "run T shots"
cost incl. the real miss rate.  Repeatable (reset each rep).  4-way same-harness: auth / snapshot / carry / Clifft.
Separates speed-win from memory-win; reports hit rate, pool/sid/edge growth, peak cache bytes.  No 'beat Clifft'."""
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
    lib.nvm_mcache_batch.restype=ctypes.c_int; lib.nvm_mcache_batch.argtypes=[P,P,ctypes.c_uint64]+[ctypes.c_uint64]*4+[P,P,ctypes.c_int]
    lib.nvm_mdam_sample_batch.restype=ctypes.c_int; lib.nvm_mdam_sample_batch.argtypes=[P,P,ctypes.c_uint64]+[ctypes.c_uint64]*4+[P,P,P,ctypes.c_int]
    lib.nvm_mcache_set_mode.argtypes=[P,ctypes.c_int]; lib.nvm_mcache_set_mode.restype=None
    lib.nvm_mcache_reset.argtypes=[P]; lib.nvm_mcache_reset.restype=None
    lib.nvm_mcache_stats.argtypes=[P,P]; lib.nvm_mcache_stats.restype=None
    lib.nvm_mcache_membytes.argtypes=[P,P]; lib.nvm_mcache_membytes.restype=None
    return lib

def run(bench, T):
    text=open(os.path.join(_ROOT,f"qec_bench/circuits/{bench}.stim")).read()
    prog=clifft.compile(text); t=translate(prog); nm=t["num_meas"]
    progD=clifft.compile(text)   # default passes incl. squeeze (same as MDAM source); Clifft baseline
    lib=bind(); ph=make_prog(lib,t); va=lib.nvm_mdam_vm_create(ph); vm=lib.nvm_mdam_vm_create(ph)
    eb=ctypes.create_string_buffer(256); buf=np.zeros((T,nm),np.uint8); reps=7; ms=20260630
    def med(fn):
        for _ in range(2): fn()
        return statistics.median([fn() for _ in range(reps)])
    # COLD-RUN ns/shot (fresh, cache built during run; reset each rep -> repeatable + realistic)
    def cold(mode):
        def one():
            lib.nvm_mcache_set_mode(vm,mode); lib.nvm_mcache_reset(vm)
            t0=time.perf_counter(); lib.nvm_mcache_batch(ph,vm,T,*pcg(ms),buf.ctypes.data,eb,256)
            return (time.perf_counter()-t0)/T*1e9
        return med(one)
    auth=med(lambda:(lambda t0:(lib.nvm_mdam_sample_batch(ph,va,T,*pcg(ms),buf.ctypes.data,None,eb,256),(time.perf_counter()-t0)/T*1e9)[1])(time.perf_counter()))
    snap=cold(2); carry=cold(3)
    cl=med(lambda:(lambda t0:(clifft.sample(progD,T),(time.perf_counter()-t0)/T*1e9)[1])(time.perf_counter()))
    # stats after one fresh cold build (mode 3)
    lib.nvm_mcache_set_mode(vm,3); lib.nvm_mcache_reset(vm); lib.nvm_mcache_batch(ph,vm,T,*pcg(ms),buf.ctypes.data,eb,256)
    st=(ctypes.c_long*10)(); lib.nvm_mcache_stats(vm,st); mb=(ctypes.c_uint64*4)(); lib.nvm_mcache_membytes(vm,mb)
    bnd=st[0]+st[1]+st[2]+st[3]; hr=st[0]/max(1,bnd)
    print(f"== {bench}: COLD fresh-run {T} shots (cache built during run) ==")
    print(f"   ns/shot: authoritative={auth:.0f}  snapshot={snap:.0f}  carry={carry:.0f}  Clifft={cl:.0f}")
    print(f"            carry/snapshot={carry/snap:.2f}x  carry/auth={carry/auth:.2f}x  carry/Clifft={carry/cl:.2f}x")
    print(f"   cache after cold {T}: hit_rate={hr:.1%}  miss={st[1]}  pool={st[7]}  sid={st[9]}  edges={st[8]}  mem={mb[3]/1e6:.1f}MB (pool={mb[0]/1e6:.1f} sid={mb[1]/1e6:.1f} edge={mb[2]/1e6:.1f})")
    # sid / hit-rate growth: cold runs of increasing T (does it saturate or grow linearly?)
    print(f"   growth (cold reset each):")
    for Tg in [2000,4000,8000,min(16000,T*2)]:
        b2=np.zeros((Tg,nm),np.uint8); lib.nvm_mcache_set_mode(vm,3); lib.nvm_mcache_reset(vm)
        lib.nvm_mcache_batch(ph,vm,Tg,*pcg(ms),b2.ctypes.data,eb,256)
        s2=(ctypes.c_long*10)(); lib.nvm_mcache_stats(vm,s2); m2=(ctypes.c_uint64*4)(); lib.nvm_mcache_membytes(vm,m2)
        b=s2[0]+s2[1]+s2[2]+s2[3]
        print(f"      T={Tg:6d}: sid={s2[9]:7d} ({s2[9]/Tg:.2f}/shot)  hit={s2[0]/max(1,b):.1%}  mem={m2[3]/1e6:.1f}MB")
    print()
    return bench, dict(auth=auth,snap=snap,carry=carry,cl=cl,hr=hr,mem=mb[3]/1e6,sid=st[9],pool=st[7])

def cult_d5_conditions(T=4000):
    bench="cultivation_d5"
    text=open(os.path.join(_ROOT,f"qec_bench/circuits/{bench}.stim")).read()
    prog=clifft.compile(text); t=translate(prog); nm=t["num_meas"]
    lib=bind(); ph=make_prog(lib,t); vm=lib.nvm_mdam_vm_create(ph)
    eb=ctypes.create_string_buffer(256); buf=np.zeros((T,nm),np.uint8); reps=7
    def med(fn):
        for _ in range(2): fn()
        return statistics.median([fn() for _ in range(reps)])
    def hitrate(): st=(ctypes.c_long*10)(); lib.nvm_mcache_stats(vm,st); b=st[0]+st[1]+st[2]+st[3]; return st[0]/max(1,b)
    print(f"== cultivation_d5: three regimes (carry mode 3, {T} shots) ==")
    # (a) warm all-hit: warm on seed S, time REPLAY of S (cache already has all of S's states)
    lib.nvm_mcache_set_mode(vm,3); lib.nvm_mcache_reset(vm); lib.nvm_mcache_batch(ph,vm,T,*pcg(777),buf.ctypes.data,eb,256)
    s0=(ctypes.c_long*10)(); lib.nvm_mcache_stats(vm,s0); h0=s0[0]
    t_allhit=med(lambda:(lambda t0:(lib.nvm_mcache_batch(ph,vm,T,*pcg(777),buf.ctypes.data,eb,256),(time.perf_counter()-t0)/T*1e9)[1])(time.perf_counter()))
    s1=(ctypes.c_long*10)(); lib.nvm_mcache_stats(vm,s1); b=(s1[0]-h0); hr_allhit=1.0  # replay -> ~all hit
    # (b) warm-then-fresh (partial): keep that warm cache, run a DIFFERENT fresh seed once (realistic 'cache helps a bit')
    lib.nvm_mcache_batch(ph,vm,T,*pcg(13579),buf.ctypes.data,eb,256); hr_partial=None
    sA=(ctypes.c_long*10)(); lib.nvm_mcache_stats(vm,sA)
    t0=time.perf_counter(); lib.nvm_mcache_batch(ph,vm,T,*pcg(24680),buf.ctypes.data,eb,256); t_partial=(time.perf_counter()-t0)/T*1e9
    sB=(ctypes.c_long*10)(); lib.nvm_mcache_stats(vm,sB); pb=(sB[0]-sA[0])+(sB[1]-sA[1])+(sB[2]-sA[2])+(sB[3]-sA[3]); hr_partial=(sB[0]-sA[0])/max(1,pb)
    # (c) fresh cold: empty cache -> run T (the realistic 'first run')
    lib.nvm_mcache_reset(vm)
    t0=time.perf_counter(); lib.nvm_mcache_batch(ph,vm,T,*pcg(98765),buf.ctypes.data,eb,256); t_cold=(time.perf_counter()-t0)/T*1e9
    sc=(ctypes.c_long*10)(); lib.nvm_mcache_stats(vm,sc); cb=sc[0]+sc[1]+sc[2]+sc[3]; hr_cold=sc[0]/max(1,cb)
    print(f"   (a) warm all-hit (replay seed):  hit~100%       carry={t_allhit:.0f} ns/shot   [BEST CASE, not realistic]")
    print(f"   (b) warm-then-fresh (partial):   hit={hr_partial:.1%}       carry={t_partial:.0f} ns/shot")
    print(f"   (c) fresh cold (empty->run):     hit={hr_cold:.1%}       carry={t_cold:.0f} ns/shot   [realistic first run]")
    print()

if __name__=="__main__":
    run("distillation", 8000)
    run("cultivation_d3", 8000)
    run("cultivation_d5", 4000)
    cult_d5_conditions(4000)
