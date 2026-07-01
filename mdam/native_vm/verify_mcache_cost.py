#!/usr/bin/env python
"""Step 3-1: decompose the run_mcache hit cost (rdtsc cycles).  Is the full-engine snapshot RESTORE the
bottleneck, or the per-boundary key HASH?  Reports cyc/boundary, cyc/hit, cyc/miss broken down."""
import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ.setdefault(v,"1")
import sys, ctypes
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
    lib.nvm_mcache_set_mode.argtypes=[P,ctypes.c_int]; lib.nvm_mcache_reset.argtypes=[P]
    lib.nvm_mcache_set_time.argtypes=[P,ctypes.c_int]; lib.nvm_mcache_cyc_get.argtypes=[P,P]
    lib.nvm_mcache_stats.argtypes=[P,P]
    return lib
def run(bench, warm, timeshots):
    text=open(os.path.join(_ROOT,f"qec_bench/circuits/{bench}.stim")).read()
    prog=clifft.compile(text); t=translate(prog); nm=t["num_meas"]
    lib=bind(); ph=make_prog(lib,t); vm=lib.nvm_mdam_vm_create(ph)
    eb=ctypes.create_string_buffer(256); rb=np.zeros((max(warm,timeshots),nm),np.uint8)
    lib.nvm_mcache_set_mode(vm,2); lib.nvm_mcache_reset(vm)
    lib.nvm_mcache_batch(ph,vm,warm,*pcg(777),rb.ctypes.data,eb,256)   # warm
    s0=(ctypes.c_long*10)(); lib.nvm_mcache_stats(vm,s0); h0,m0=s0[0],s0[1]
    lib.nvm_mcache_set_time(vm,1)
    lib.nvm_mcache_batch(ph,vm,timeshots,*pcg(777),rb.ctypes.data,eb,256)   # timed (same seed -> warm hits)
    lib.nvm_mcache_set_time(vm,0)
    cyc=(ctypes.c_uint64*8)(); lib.nvm_mcache_cyc_get(vm,cyc); cyc=[cyc[i] for i in range(8)]
    s1=(ctypes.c_long*10)(); lib.nvm_mcache_stats(vm,s1)
    hits=s1[0]-h0; miss=s1[1]-m0; bnd=hits+miss   # approx boundaries during timed window (antis/partial small)
    print(f"== {bench}: timed {timeshots} shots, hits={hits} miss={miss} ==")
    if hits==0: hits=1
    if miss==0: miss=1
    if bnd==0: bnd=1
    print(f"   per-boundary:  key-hash={cyc[0]/bnd:.0f}  lookup={cyc[1]/bnd:.0f}")
    print(f"   per-HIT:       restore={cyc[2]/hits:.0f}   hit-total={cyc[5]/hits:.0f}  (key+lookup+rng+restore)")
    print(f"   per-LIVE/miss: measure_z={cyc[3]/miss:.0f}  pool_intern={cyc[4]/miss:.0f}  live-total={cyc[6]/miss:.0f}")
    print(f"   SNAPSHOT RESTORE share of hit = {cyc[2]/max(1,cyc[5]):.1%}   KEY-HASH share of hit = {cyc[0]/max(1,cyc[5]):.1%} (key over all bnd / total hit)")
    print()
if __name__=="__main__":
    for b in (sys.argv[1].split(",") if len(sys.argv)>1 else ["distillation","cultivation_d3","cultivation_d5"]):
        run(b, 4000 if b!="cultivation_d5" else 1500, 4000 if b!="cultivation_d5" else 1500)
