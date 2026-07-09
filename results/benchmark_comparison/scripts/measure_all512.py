"""Unified per-bench measurement, N identical across benches, w=512, ALL MEASURED:
  1) LEAN-forced : SINGLE cold call via the adaptive entry with calibration disabled
                   (nvm_adapt_cal(vm,0) -> every demote trigger inert -> pure LEAN from
                   shot 0, records bit-identical to run_lean_fb_batch); per-window fb +
                   wall from the trace.  Same call shape as the adaptive run.
  2) AUTH-forced : per-window wall (chunked sample_batch, same seeds)
  3) adaptive    : single cold call, window=512, per-window wall from the trace
Noise control (both verified to leave records bit-identical):
  - nvm_lean_reserve pre-sizes the tables so no rehash/realloc lands in a timed window
  - each measurement is repeated R=3 times with the SAME seed (identical work) and the
    per-window MIN is kept -> one-off OS hiccups (ms-scale) drop out of 3ms windows
argv: bench N   -> saves all512_<bench>.npz"""
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
lib.nvm_mcache_reset.argtypes=[P]; lib.nvm_sg_reset.argtypes=[P]
lib.nvm_lean_reset_counts.argtypes=[P]
lib.nvm_rb_static.argtypes=[C]
lib.nvm_adapt_stats2.argtypes=[P,ctypes.POINTER(D),C]; lib.nvm_adapt_stats2.restype=None
lib.nvm_adapt_trace_n.restype=L; lib.nvm_adapt_trace_n.argtypes=[P]
lib.nvm_adapt_trace.argtypes=[P,ctypes.POINTER(D),L]; lib.nvm_adapt_trace.restype=None
lib.nvm_adapt_config.argtypes=[P,L,L,L,L,L,D,D,C]; lib.nvm_adapt_config.restype=None
lib.nvm_adapt_cal.argtypes=[P,L,D]; lib.nvm_adapt_cal.restype=None
lib.nvm_lean_reserve.argtypes=[P,L,L]; lib.nvm_lean_reserve.restype=None

bench=sys.argv[1]; N=int(sys.argv[2]); CH=512; NW=N//CH
R=3                                    # identical-seed repetitions; per-window min kept
RESERVE=(4_000_000,16_000_000)
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
    W=tr[(tr[:,10]==0)&(tr[:,12]>0)]
    return o,W

big=np.zeros((N,nm),np.uint8)

def lean_rep():
    """one cold forced-LEAN run -> (fb, wall) per window"""
    vm=lib.nvm_mdam_vm_create(ph)
    lib.nvm_adapt_config(vm,512,0,0,1<<60,-1,1e9,0.0,0)   # mem/node-floor out of reach
    lib.nvm_adapt_cal(vm,0,-1.0)                          # cal=0 -> pure LEAN from shot 0
    prep(vm); lib.nvm_lean_reserve(vm,*RESERVE)
    seed=30000; retries=0
    while True:
        r=lib.nvm_run_lean_adapt_batch(ph,vm,N,*pcg(seed),big.ctypes.data,eb,256)
        if r==0: break
        retries+=1; seed+=1; assert retries<=5, eb.value
        prep(vm); lib.nvm_lean_reserve(vm,*RESERVE)
    _,W=trace_windows(vm); lib.nvm_sg_shadow(vm,0)
    return W[:,1].copy(), W[:,12].copy(), seed

def auth_rep():
    """one cold AUTH-forced chunked pass -> wall per window"""
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

def adapt_rep():
    """one cold adaptive run -> (an, afb, awin, demote, cal, total_ns_per_shot)"""
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
    return W[:,0].copy(),W[:,1].copy(),W[:,12].copy(),demote,float(o[21]),awall/N*1e9,seed

# ---- 1) LEAN-forced, R cold reps, same seed -> per-window min ----
lr=[lean_rep() for _ in range(R)]
lr=[x for x in lr if x[2]==lr[0][2] and np.array_equal(x[0],lr[0][0])] or lr[:1]
fb=lr[0][0]; wall_lean=np.min(np.stack([x[1] for x in lr]),axis=0)

# ---- 2) AUTH-forced, R reps -> per-window min ----
wall_auth=np.min(np.stack([auth_rep() for _ in range(R)]),axis=0)

# ---- 3) adaptive, R cold reps -> per-window min (same decisions verified) ----
ar=[adapt_rep() for _ in range(R)]
ar=[x for x in ar if x[3]==ar[0][3] and x[6]==ar[0][6] and np.array_equal(x[0],ar[0][0])] or ar[:1]
an=ar[0][0]; afb=ar[0][1]; demote=ar[0][3]
awin=np.min(np.stack([x[2] for x in ar]),axis=0)
a_cal=min(x[4] for x in ar); a_total=min(x[5] for x in ar)

lib.nvm_rb_static(0)
T_auth=float(np.mean(wall_auth))
np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)),f"all512_{bench}.npz"),
         fb=fb, wall_lean=wall_lean, wall_auth=wall_auth,
         an=an, afb=afb, awin=awin, demote=demote, a_cal=a_cal, a_total_ns=a_total,
         T_auth=T_auth, N=N, reps_lean=len(lr), reps_adapt=len(ar))
print(f"{bench:18s} N={N:,} reps(l/a)={len(lr)}/{len(ar)} fb[0]={fb[0]:.3f} fb[-1]={fb[-1]:.3f} "
      f"T_auth={T_auth/1e3:.1f}us lean={np.sum(wall_lean)*CH/1e9:.1f}s auth={np.sum(wall_auth)*CH/1e9:.1f}s "
      f"adapt={a_total*N/1e9:.1f}s demote={demote}", flush=True)
