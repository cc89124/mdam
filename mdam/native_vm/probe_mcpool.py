"""Measure mc_pool (dense-core cache) growth for a saturating LEAN winner (cult_d5, keep LEAN)
vs a non-saturating localization circuit (d5_r5, must demote).  Demote is DISABLED (huge caps) so
policy stays LEAN and mc_pool accumulates; we read mc_pool_bytes/size + RSS + window lean/slow/node_rate
at increasing cumulative shot counts.  Stops a bench early if RSS crosses a soft ceiling (no OOM-kill).
taskset -c 2, single-thread."""
import os, sys, ctypes, time, resource
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ.setdefault(v,"1")
import numpy as np
ROOT="/home/jung/clifft-paper"; HERE=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,HERE); sys.path.insert(0,os.path.join(ROOT,"mdam")); sys.path.insert(0,ROOT)
import clifft
from verify_mdam_oneshot import translate, make_prog, pcg, load_lib
P=ctypes.c_void_p; U=ctypes.c_uint64; L=ctypes.c_long; D=ctypes.c_double
lib=load_lib()
lib.nvm_run_lean_adapt_batch.restype=ctypes.c_int; lib.nvm_run_lean_adapt_batch.argtypes=[P,P,U]+[U]*4+[P,P,ctypes.c_int]
lib.nvm_adapt_config.argtypes=[P,L,L,L,L,L,D,D,ctypes.c_int]; lib.nvm_adapt_config.restype=None
lib.nvm_adapt_stats.argtypes=[P,ctypes.POINTER(D)]; lib.nvm_adapt_stats.restype=None
for f in ("nvm_mcache_set_mode","nvm_mcache_set_fblock","nvm_sg_shadow","nvm_sg_signs"):
    getattr(lib,f).argtypes=[P,ctypes.c_int]; getattr(lib,f).restype=None
lib.nvm_mcache_reset.argtypes=[P]; lib.nvm_sg_reset.argtypes=[P]; lib.nvm_lean_reset_counts.argtypes=[P]
lib.nvm_rb_static.argtypes=[ctypes.c_int]; lib.nvm_rb_static_reset.restype=None
def setup_lean(vm): lib.nvm_rb_static_reset(); lib.nvm_rb_static(1); lib.nvm_mcache_set_mode(vm,3); lib.nvm_mcache_set_fblock(vm,1)
def fresh(vm): lib.nvm_mcache_reset(vm); lib.nvm_sg_reset(vm); lib.nvm_sg_signs(vm,1); lib.nvm_sg_shadow(vm,1); lib.nvm_lean_reset_counts(vm)
eb=ctypes.create_string_buffer(256)
BIG=10**15
def rss_mb(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0

def probe(bench, chunk=500, max_chunks=24, rss_ceiling=2600):
    txt=open(f"{ROOT}/qec_bench/circuits/{bench}.stim").read()
    prog=clifft.compile(txt); t=translate(prog); nm=t["num_meas"]; ph=make_prog(lib,t)
    vm=lib.nvm_mdam_vm_create(ph); setup_lean(vm); fresh(vm)
    lib.nvm_adapt_config(vm, chunk, BIG,BIG,BIG, BIG, 0.0, float(BIG), 999999)  # never demote
    o=(D*16)(); tot=0; buf=np.zeros((chunk,nm),np.uint8)
    print(f"\n{bench}  nm={nm}  (demote disabled; LEAN accumulates)")
    print(f"  {'shots':>7} {'mc_pool':>8} {'poolMB':>8} {'nodes':>9} {'node/sh':>8} {'fb%':>6} {'lean_us':>9} {'slow_us':>9} {'RSS_MB':>8}")
    for i in range(max_chunks):
        setup_lean(vm)
        t0=time.perf_counter()
        lib.nvm_run_lean_adapt_batch(ph,vm,chunk,*pcg(1000+i),buf.ctypes.data,eb,256)
        dt=time.perf_counter()-t0; tot+=chunk
        lib.nvm_adapt_stats(vm,o)
        pool=int(o[12]); poolMB=o[13]/1e6; nodes=int(o[9]); nrate=o[5]; fb=o[8]*100 if o[8]>=0 else -1
        lean_us=o[6]/1000 if o[6]>=0 else -1; slow_us=o[7]/1000 if o[7]>=0 else -1; r=rss_mb()
        print(f"  {tot:>7} {pool:>8} {poolMB:>8.1f} {nodes:>9} {nrate:>8.2f} {fb:>6.1f} {lean_us:>9.1f} {slow_us:>9.1f} {r:>8.0f}",flush=True)
        if r>rss_ceiling: print(f"  -> RSS ceiling {rss_ceiling}MB crossed at {tot} shots (would OOM unbounded). STOP."); break
    lib.nvm_sg_shadow(vm,0); lib.nvm_rb_static(0)

if __name__=="__main__":
    for b in (sys.argv[1:] or ["cultivation_d5","coherent_d5_r5"]):
        probe({"d5_r5":"coherent_d5_r5","cult_d5":"cultivation_d5","d7_r1":"coherent_d7_r1","d5_r1":"coherent_d5_r1"}.get(b,b))
