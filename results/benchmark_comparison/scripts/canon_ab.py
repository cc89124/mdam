"""mc_canon A/B: canonical (phase + 1e-9 grid) sid interning, flag vs baseline.

modes:
  bitexact <bench> <seeds> <shots_per_seed>
     canon=1 lean-forced (adaptive cal=0, PERSISTENT vm so the cache is warm and
     merge-hits actually fire) vs authoritative sample_batch, per-seed record compare.
     canon=0 control runs too (expect 0/0).
  perf <bench> <N> [R=3]
     cold single-call lean-forced (adaptive entry, cal=0) canon=0 vs canon=1,
     R fresh-VM reps each, report best wall ns/shot + final fb + sids/edges/bytes.
"""
import os, sys, time, ctypes
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ.setdefault(v,"1")
import numpy as np
ROOT="/home/jung/clifft-paper"; NV=ROOT+"/mdam/native_vm"
sys.path.insert(0,NV); sys.path.insert(0,ROOT+"/mdam"); sys.path.insert(0,ROOT)
import clifft
from verify_mdam_oneshot import translate, make_prog, pcg, load_lib
P=ctypes.c_void_p; U=ctypes.c_uint64; C=ctypes.c_int; D=ctypes.c_double; L=ctypes.c_long
lib=load_lib()
lib.nvm_mdam_sample_batch.restype=C; lib.nvm_mdam_sample_batch.argtypes=[P,P,U]+[U]*4+[P,P,P,C]
lib.nvm_run_lean_adapt_batch.restype=C; lib.nvm_run_lean_adapt_batch.argtypes=[P,P,U]+[U]*4+[P,P,C]
for f in ("nvm_mcache_set_mode","nvm_mcache_set_fblock","nvm_sg_shadow","nvm_sg_signs","nvm_mc_canon"):
    getattr(lib,f).argtypes=[P,C]; getattr(lib,f).restype=None
lib.nvm_mcache_reset.argtypes=[P]; lib.nvm_sg_reset.argtypes=[P]; lib.nvm_lean_reset_counts.argtypes=[P]
lib.nvm_adapt_config.argtypes=[P,L,L,L,L,L,D,D,C]; lib.nvm_adapt_config.restype=None
lib.nvm_adapt_cal.argtypes=[P,L,D]; lib.nvm_adapt_cal.restype=None
lib.nvm_lean_reserve.argtypes=[P,L,L]; lib.nvm_lean_reserve.restype=None
lib.nvm_adapt_stats2.argtypes=[P,ctypes.POINTER(D),C]; lib.nvm_adapt_stats2.restype=None
lib.nvm_adapt_trace_n.restype=L; lib.nvm_adapt_trace_n.argtypes=[P]
lib.nvm_adapt_trace.argtypes=[P,ctypes.POINTER(D),L]; lib.nvm_adapt_trace.restype=None
lib.nvm_mcache_stats.argtypes=[P,ctypes.POINTER(L)]; lib.nvm_mcache_stats.restype=None
lib.nvm_mcache_membytes.argtypes=[P,ctypes.POINTER(ctypes.c_uint64)]; lib.nvm_mcache_membytes.restype=None

mode=sys.argv[1]; bench=sys.argv[2]
eb=ctypes.create_string_buffer(256)
t=translate(clifft.compile(open(f"{ROOT}/qec_bench/circuits/{bench}.stim").read())); nm=t["num_meas"]
ph=make_prog(lib,t)

def new_lean_vm(canon, reserve=(4_000_000,16_000_000)):
    vm=lib.nvm_mdam_vm_create(ph)
    lib.nvm_adapt_config(vm,512,0,0,1<<60,-1,1e9,0.0,0)
    lib.nvm_adapt_cal(vm,0,-1.0)
    lib.nvm_mc_canon(vm,canon)
    lib.nvm_rb_static_reset(); lib.nvm_rb_static(1)
    lib.nvm_mcache_set_mode(vm,3); lib.nvm_mcache_set_fblock(vm,1)
    lib.nvm_mcache_reset(vm); lib.nvm_sg_reset(vm); lib.nvm_sg_signs(vm,1)
    lib.nvm_sg_shadow(vm,1); lib.nvm_lean_reset_counts(vm)
    lib.nvm_lean_reserve(vm,*reserve)
    return vm

if mode=="bitexact":
    S=int(sys.argv[3]); NS=int(sys.argv[4])
    rec_l=np.zeros((NS,nm),np.uint8); rec_a=np.zeros((NS,nm),np.uint8)
    for canon in (1,0):
        vm=new_lean_vm(canon); va=lib.nvm_mdam_vm_create(ph)
        lib.nvm_rb_static_reset(); lib.nvm_rb_static(0)
        lib.nvm_mcache_set_mode(va,3); lib.nvm_mcache_set_fblock(va,1)
        mism=0; bits=0
        for s in range(S):
            r=lib.nvm_run_lean_adapt_batch(ph,vm,NS,*pcg(50000+s),rec_l.ctypes.data,eb,256)
            assert r==0, eb.value
            r=lib.nvm_mdam_sample_batch(ph,va,NS,*pcg(50000+s),rec_a.ctypes.data,None,eb,256)
            assert r==0, eb.value
            d=int(np.count_nonzero(rec_l!=rec_a))
            if d: mism+=1; bits+=d
        st=(L*10)(); lib.nvm_mcache_stats(vm,st)
        print(f"{bench} canon={canon}: {S} seeds x {NS} shots -> mismatch seeds={mism}, bits={bits} "
              f"(distinct sids={st[9]:,}, edges={st[8]:,})",flush=True)
elif mode=="perf":
    N=int(sys.argv[3]); R=int(sys.argv[4]) if len(sys.argv)>4 else 3
    big=np.zeros((N,nm),np.uint8)
    for canon in (0,1):
        best=None
        for rep in range(R):
            vm=new_lean_vm(canon); seed=30000
            while True:
                t0=time.perf_counter()
                r=lib.nvm_run_lean_adapt_batch(ph,vm,N,*pcg(seed),big.ctypes.data,eb,256)
                wall=time.perf_counter()-t0
                if r==0: break
                seed+=1; vm=new_lean_vm(canon)
            nr=lib.nvm_adapt_trace_n(vm); tr=np.zeros((max(nr,1),13))
            if nr: lib.nvm_adapt_trace(vm,tr.ctypes.data_as(ctypes.POINTER(D)),nr)
            W=tr[(tr[:,10]==0)&(tr[:,12]>0)]
            fb_tail=float(W[-8:,1].mean()) if len(W)>=8 else float("nan")
            st=(L*10)(); lib.nvm_mcache_stats(vm,st)
            mb=(ctypes.c_uint64*4)(); lib.nvm_mcache_membytes(vm,mb)
            row=(wall/N*1e9, fb_tail, st[9], st[8], mb[1]/1e6, mb[3]/1e6, seed)
            print(f"  canon={canon} rep{rep}: {row[0]:8.1f} ns/shot  fb_tail={row[1]*100:5.2f}%  "
                  f"sids={row[2]:,} edges={row[3]:,} sid_MB={row[4]:.0f} cache_MB={row[5]:.0f}",flush=True)
            if best is None or row[0]<best[0]: best=row
        print(f"{bench} canon={canon} BEST: {best[0]:.1f} ns/shot fb_tail={best[1]*100:.2f}% "
              f"sids={best[2]:,} edges={best[3]:,} cache_MB={best[5]:.0f}",flush=True)
