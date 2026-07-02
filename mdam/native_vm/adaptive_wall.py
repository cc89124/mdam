"""Wire the adaptive executor (nvm_run_lean_adapt_batch) into the wall measurement: ONE production method,
DEFAULT config, cold single-call per bench -> the algorithm itself auto-selects LEAN vs sticky SLOW_ONLY.
cold-amortized ns/shot = total_wall / N.  Single contiguous call (window/horizon state is per-call).
N = max(150k so the 100k horizon is reachable, ~18s-worth), capped by 10M and ~400MB buffer.
Clifft REUSED from wall_table.tsv (constant).  Bit-exact spot-checked vs authoritative sample_batch.
Worker mode: argv[1]=bench, argv[2]=results-file (subprocess-per-bench for memory isolation)."""
import os, sys, time, ctypes
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ.setdefault(v,"1")
import numpy as np
ROOT="/home/jung/clifft-paper"; HERE=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,HERE); sys.path.insert(0,os.path.join(ROOT,"mdam")); sys.path.insert(0,ROOT)
import clifft
from verify_mdam_oneshot import translate, make_prog, pcg, load_lib
P=ctypes.c_void_p; U=ctypes.c_uint64; L=ctypes.c_long; D=ctypes.c_double
lib=load_lib()
lib.nvm_mdam_sample_batch.restype=ctypes.c_int; lib.nvm_mdam_sample_batch.argtypes=[P,P,U]+[U]*4+[P,P,P,ctypes.c_int]
lib.nvm_run_lean_adapt_batch.restype=ctypes.c_int; lib.nvm_run_lean_adapt_batch.argtypes=[P,P,U]+[U]*4+[P,P,ctypes.c_int]
lib.nvm_adapt_config.argtypes=[P,L,L,L,L,L,D,D,ctypes.c_int]; lib.nvm_adapt_config.restype=None
lib.nvm_adapt_stats.argtypes=[P,ctypes.POINTER(D)]; lib.nvm_adapt_stats.restype=None
for f in ("nvm_mcache_set_mode","nvm_mcache_set_fblock","nvm_sg_shadow","nvm_sg_signs"):
    getattr(lib,f).argtypes=[P,ctypes.c_int]; getattr(lib,f).restype=None
lib.nvm_mcache_reset.argtypes=[P]; lib.nvm_sg_reset.argtypes=[P]; lib.nvm_lean_reset_counts.argtypes=[P]
lib.nvm_rb_static.argtypes=[ctypes.c_int]; lib.nvm_rb_static_reset.restype=None
def setup_lean(vm): lib.nvm_rb_static_reset(); lib.nvm_rb_static(1); lib.nvm_mcache_set_mode(vm,3); lib.nvm_mcache_set_fblock(vm,1)
def fresh(vm): lib.nvm_mcache_reset(vm); lib.nvm_sg_reset(vm); lib.nvm_sg_signs(vm,1); lib.nvm_sg_shadow(vm,1); lib.nvm_lean_reset_counts(vm)

TSV=f"{ROOT}/results/benchmark_comparison/wall_table.tsv"
old={}
for ln in open(TSV):
    if ln.startswith("#") or not ln.strip(): continue
    f=ln.split("\t"); old[f[0]]=dict(k=int(f[1]),maxM=f[2],nmeas=int(f[3]),clifft_ns=float(f[4]),
        auth_ns=float(f[5]),lean_ns=(None if 'OOM' in f[7] else float(f[7])))   # lean_ns is col 7 (col 8 = speedup_lean)

def bench_adapt(bench):
    d=old[bench]; eb=ctypes.create_string_buffer(256)
    txt=open(f"{ROOT}/qec_bench/circuits/{bench}.stim").read()
    prog=clifft.compile(txt); t=translate(prog); nm=t["num_meas"]; ph=make_prog(lib,t)
    vm=lib.nvm_mdam_vm_create(ph); va=lib.nvm_mdam_vm_create(ph)
    lean_ns=d["lean_ns"] or d["auth_ns"]
    N=int(max(150000, 18e9/max(lean_ns,1))); N=min(N, 10_000_000, int(400e6//max(nm,1)))
    # bit-exact spot check (2 seeds x 2000) vs authoritative
    mism=0; Tv=2000
    for sd in (11,22):
        A=np.zeros((Tv,nm),np.uint8); B=np.zeros((Tv,nm),np.uint8)
        setup_lean(va); lib.nvm_rb_static(0); lib.nvm_mdam_sample_batch(ph,va,Tv,*pcg(sd),A.ctypes.data,None,eb,256)
        setup_lean(vm); fresh(vm); lib.nvm_run_lean_adapt_batch(ph,vm,Tv,*pcg(sd),B.ctypes.data,eb,256)
        mism+=int((A!=B).sum())
    # cold single-call adaptive (DEFAULT config)
    setup_lean(vm); fresh(vm)   # default: window4096 horizon100k node_floor0.02 cost_margin1.10 bad3
    buf=np.zeros((N,nm),np.uint8); t0=time.perf_counter()
    setup_lean(vm); lib.nvm_run_lean_adapt_batch(ph,vm,N,*pcg(40000),buf.ctypes.data,eb,256)
    wall=time.perf_counter()-t0; adapt_ns=wall/N*1e9
    o=(D*16)(); lib.nvm_adapt_stats(vm,o); lib.nvm_sg_shadow(vm,0); lib.nvm_rb_static(0)
    pol="AUTH@%d"%int(o[1]) if int(o[0])==1 else "LEAN"
    return dict(bench=bench,k=d["k"],maxM=d["maxM"],clifft_ns=d["clifft_ns"],auth_ns=d["auth_ns"],lean_ns=d["lean_ns"],
                N=N,adapt_ns=adapt_ns,pol=pol,speedup=d["clifft_ns"]/adapt_ns,mism=mism,
                fb=o[8]*100 if o[8]>=0 else -1)

if __name__=="__main__":
    b=sys.argv[1]; resf=sys.argv[2]
    r=bench_adapt(b)
    auth_sp=r["clifft_ns"]/r["auth_ns"]; lean_sp=(r["clifft_ns"]/r["lean_ns"]) if r["lean_ns"] else float('nan')
    with open(resf,"a") as f:
        f.write(f"{r['bench']}\t{r['k']}\t{r['N']}\t{r['adapt_ns']:.1f}\t{r['pol']}\t{r['speedup']:.2f}\t"
                f"{auth_sp:.2f}\t{lean_sp:.2f}\t{'OK' if r['mism']==0 else 'FAIL'}\n")
    print(f"{r['bench']:18s} k={r['k']:<3d} N={r['N']:>8d} adapt={r['adapt_ns']:>11.1f}ns pol={r['pol']:<10s} "
          f"spd={r['speedup']:.2f}x (auth={auth_sp:.2f} lean={lean_sp:.2f})  {'OK' if r['mism']==0 else 'FAIL'}",flush=True)
