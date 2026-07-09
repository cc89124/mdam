"""d5_r5 special-case measurement, same protocol as measure_all512 (reserve + R=3
identical-seed reps, per-window min) EXCEPT the LEAN-forced pass runs only
N_LEAN=3072 shots: forced LEAN accumulates 2.64MB/shot of never-reused mode-3
learning state, so N=100k would need ~264GB.  AUTH + adaptive run the full N."""
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
for f in ("nvm_mcache_set_mode","nvm_mcache_set_fblock","nvm_sg_shadow","nvm_sg_signs"):
    getattr(lib,f).argtypes=[P,C]; getattr(lib,f).restype=None
lib.nvm_mcache_reset.argtypes=[P]; lib.nvm_sg_reset.argtypes=[P]; lib.nvm_lean_reset_counts.argtypes=[P]
lib.nvm_rb_static.argtypes=[C]
lib.nvm_adapt_stats2.argtypes=[P,ctypes.POINTER(D),C]; lib.nvm_adapt_stats2.restype=None
lib.nvm_adapt_trace_n.restype=L; lib.nvm_adapt_trace_n.argtypes=[P]
lib.nvm_adapt_trace.argtypes=[P,ctypes.POINTER(D),L]; lib.nvm_adapt_trace.restype=None
lib.nvm_adapt_config.argtypes=[P,L,L,L,L,L,D,D,C]; lib.nvm_adapt_config.restype=None
lib.nvm_adapt_cal.argtypes=[P,L,D]; lib.nvm_adapt_cal.restype=None
lib.nvm_lean_reserve.argtypes=[P,L,L]; lib.nvm_lean_reserve.restype=None

bench="coherent_d5_r5"; N=100_000; N_LEAN=3072; CH=512; NW=N//CH
R=3; RESERVE=(4_000_000,16_000_000)
eb=ctypes.create_string_buffer(256)
t=translate(clifft.compile(open(f"{ROOT}/qec_bench/circuits/{bench}.stim").read())); nm=t["num_meas"]
ph=make_prog(lib,t)

def prep(vm):
    lib.nvm_rb_static_reset(); lib.nvm_rb_static(1)
    lib.nvm_mcache_set_mode(vm,3); lib.nvm_mcache_set_fblock(vm,1)
    lib.nvm_mcache_reset(vm); lib.nvm_sg_reset(vm); lib.nvm_sg_signs(vm,1)
    lib.nvm_sg_shadow(vm,1); lib.nvm_lean_reset_counts(vm)

def trace_windows(vm):
    o=(D*26)(); lib.nvm_adapt_stats2(vm,o,26)
    nr=lib.nvm_adapt_trace_n(vm); tr=np.zeros((max(nr,1),13))
    if nr: lib.nvm_adapt_trace(vm,tr.ctypes.data_as(ctypes.POINTER(D)),nr)
    return o,tr[(tr[:,10]==0)&(tr[:,12]>0)]

def lean_rep():
    vm=lib.nvm_mdam_vm_create(ph)
    lib.nvm_adapt_config(vm,512,0,0,1<<60,-1,1e9,0.0,0)
    lib.nvm_adapt_cal(vm,0,-1.0)
    prep(vm); lib.nvm_lean_reserve(vm,*RESERVE)
    buf=np.zeros((N_LEAN,nm),np.uint8)
    r=lib.nvm_run_lean_adapt_batch(ph,vm,N_LEAN,*pcg(30000),buf.ctypes.data,eb,256); assert r==0, eb.value
    _,W=trace_windows(vm); lib.nvm_sg_shadow(vm,0)
    return W[:,1].copy(), W[:,12].copy()

def auth_rep():
    va=lib.nvm_mdam_vm_create(ph)
    lib.nvm_rb_static_reset(); lib.nvm_rb_static(0)
    lib.nvm_mcache_set_mode(va,3); lib.nvm_mcache_set_fblock(va,1)
    lib.nvm_lean_reserve(va,*RESERVE)
    buf=np.zeros((CH,nm),np.uint8); w=np.zeros(NW)
    for i in range(NW):
        t0=time.perf_counter()
        r=lib.nvm_mdam_sample_batch(ph,va,CH,*pcg(30000+i*CH),buf.ctypes.data,None,eb,256)
        w[i]=(time.perf_counter()-t0)/CH*1e9; assert r==0, eb.value
    return w

def adapt_rep(big):
    vd=lib.nvm_mdam_vm_create(ph)
    lib.nvm_adapt_config(vd,512,0,0,0,-1,-1.0,0.0,0)
    prep(vd); lib.nvm_lean_reserve(vd,*RESERVE)
    seed=40000; aretr=0
    while True:
        t0=time.perf_counter()
        r=lib.nvm_run_lean_adapt_batch(ph,vd,N,*pcg(seed),big.ctypes.data,eb,256)
        awall=time.perf_counter()-t0
        if r==0: break
        aretr+=1; seed+=1; assert aretr<=4, eb.value
        prep(vd); lib.nvm_lean_reserve(vd,*RESERVE)
    o,W=trace_windows(vd); lib.nvm_sg_shadow(vd,0)
    demote=int(o[1]) if int(o[0])==1 else -1
    return W[:,0].copy(),W[:,12].copy(),demote,float(o[21]),awall/N*1e9,seed

lr=[lean_rep() for _ in range(R)]
lr=[x for x in lr if np.array_equal(x[0],lr[0][0])] or lr[:1]
fb=lr[0][0]; wall_lean=np.min(np.stack([x[1] for x in lr]),axis=0)
print(f"lean stub done: {len(lr)} reps, median {np.median(wall_lean)/1e3:.0f}us/shot", flush=True)

wall_auth=np.min(np.stack([auth_rep() for _ in range(R)]),axis=0)
print(f"auth done: {np.sum(wall_auth)*CH/1e9:.1f}s (min of {R})", flush=True)

big=np.zeros((N,nm),np.uint8)
ar=[adapt_rep(big) for _ in range(R)]
ar=[x for x in ar if x[2]==ar[0][2] and x[5]==ar[0][5] and np.array_equal(x[0],ar[0][0])] or ar[:1]
an=ar[0][0]; demote=ar[0][2]
awin=np.min(np.stack([x[1] for x in ar]),axis=0)
a_cal=min(x[3] for x in ar); a_total=min(x[4] for x in ar)
lib.nvm_rb_static(0)

T_auth=float(np.mean(wall_auth))
np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)),"all512_coherent_d5_r5.npz"),
         fb=fb, wall_lean=wall_lean, wall_auth=wall_auth, lean_truncated=1, N_lean=N_LEAN,
         an=an, afb=np.ones_like(an), awin=awin, demote=demote, a_cal=a_cal, a_total_ns=a_total,
         T_auth=T_auth, N=N, reps_lean=len(lr), reps_adapt=len(ar))
print(f"{bench:18s} N={N:,} (lean {N_LEAN}) reps(l/a)={len(lr)}/{len(ar)} fb[0]={fb[0]:.3f} "
      f"T_auth={T_auth/1e3:.1f}us lean_pershot={np.median(wall_lean)/1e3:.1f}us "
      f"auth={np.sum(wall_auth)*CH/1e9:.1f}s adapt={a_total*N/1e9:.1f}s demote={demote}", flush=True)
