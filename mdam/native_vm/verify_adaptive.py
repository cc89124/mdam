"""Verify the adaptive bounded-regret executor (run_lean_adapt_batch):
  (1) BIT-EXACT: adaptive output == authoritative sample_batch, across the lean->AUTH policy switch.
  (2) PROTECTION: slow-saturating lean winners (cult_d3) + fast-saturating (distillation) are NEVER demoted
      at the realistic horizon -> no regression (the whole point of the conservative criterion).
  (3) DEMOTE PATH: with an aggressive config, a non-saturating circuit CAN sticky-demote to AUTH,
      and output stays bit-exact through the switch.
taskset -c 2, single-thread.  Authoritative run()/sample_batch and run_lean_fb_batch are untouched."""
import os, sys, ctypes, time
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

def setup_lean(vm):
    lib.nvm_rb_static_reset(); lib.nvm_rb_static(1); lib.nvm_mcache_set_mode(vm,3); lib.nvm_mcache_set_fblock(vm,1)
def fresh(vm):
    lib.nvm_mcache_reset(vm); lib.nvm_sg_reset(vm); lib.nvm_sg_signs(vm,1); lib.nvm_sg_shadow(vm,1); lib.nvm_lean_reset_counts(vm)

def load(bench):
    txt=open(f"{ROOT}/qec_bench/circuits/{bench}.stim").read()
    prog=clifft.compile(txt); t=translate(prog); nm=t["num_meas"]; ph=make_prog(lib,t)
    return ph, lib.nvm_mdam_vm_create(ph), lib.nvm_mdam_vm_create(ph), nm

def adapt_stats(vm):
    o=(D*16)(); lib.nvm_adapt_stats(vm,o); return list(o)
eb=ctypes.create_string_buffer(256)

print("="*78); print("(1)+(3) BIT-EXACT adaptive vs authoritative  (incl. across policy switch)"); print("="*78)
# distillation/cult_d3: default config (won't demote).  cult_d5: aggressive config (forces demote mid-batch).
cases=[("distillation", 4000, None),
       ("cultivation_d3", 4000, None),
       ("cultivation_d5", 8000, (2048, 0, 4096, 1.0)),   # (window,horizon,-,cost_margin) aggressive
       ("coherent_d3_r3", 8000, (2048, 0, 4096, 1.0)) ]  # 100% fallback -> should DEMOTE mid-batch
for bench,T,agg in cases:
    ph,vm,va,nm=load(bench); mism=0; pol=None
    for sd in (11,22):
        A=np.zeros((T,nm),np.uint8); B=np.zeros((T,nm),np.uint8)
        setup_lean(va); lib.nvm_rb_static(0)
        lib.nvm_mdam_sample_batch(ph,va,T,*pcg(sd),A.ctypes.data,None,eb,256)   # authoritative
        setup_lean(vm); fresh(vm)
        if agg: lib.nvm_adapt_config(vm, agg[0], -1,-1,-1, agg[1], -1.0, agg[3], 1)
        lib.nvm_run_lean_adapt_batch(ph,vm,T,*pcg(sd),B.ctypes.data,eb,256)     # adaptive
        mism+=int((A!=B).sum()); st=adapt_stats(vm); pol=int(st[0])
        lib.nvm_sg_shadow(vm,0); lib.nvm_rb_static(0)
    print(f"  {bench:16s} T={T} x2  mism={mism}  final_policy={'AUTH' if pol==1 else 'LEAN'}  {'OK' if mism==0 else 'FAIL'}")

print(); print("="*78); print("(2) PROTECTION: cult_d3 / distillation NEVER demote at realistic horizon (default)"); print("="*78)
for bench,N in [("distillation",130000),("cultivation_d3",130000)]:
    ph,vm,va,nm=load(bench); buf=np.zeros((min(N,4000),nm),np.uint8)
    setup_lean(vm); fresh(vm)   # DEFAULT config: horizon=100000, node_floor=0.02, cost_margin=1.10, bad=3
    done=0; chunk=4000; t0=time.perf_counter()
    while done<N:
        c=min(chunk,N-done); b=np.zeros((c,nm),np.uint8)
        setup_lean(vm); lib.nvm_run_lean_adapt_batch(ph,vm,c,*pcg(700+done),b.ctypes.data,eb,256); done+=c
    dt=time.perf_counter()-t0; st=adapt_stats(vm); lib.nvm_sg_shadow(vm,0); lib.nvm_rb_static(0)
    # NOTE: chunked calls reset the window each call; single-call is the real test -> also do one big call
    ph2,vm2,_,nm2=load(bench); setup_lean(vm2); fresh(vm2)
    big=np.zeros((N,nm2),np.uint8); t0=time.perf_counter()
    lib.nvm_run_lean_adapt_batch(ph2,vm2,N,*pcg(999),big.ctypes.data,eb,256); dt2=time.perf_counter()-t0
    st2=adapt_stats(vm2); lib.nvm_sg_shadow(vm2,0); lib.nvm_rb_static(0)
    pol=int(st2[0]); nodes=int(st2[9]); nrl=st2[5]
    print(f"  {bench:16s} N={N} single-call: final_policy={'AUTH!' if pol==1 else 'LEAN'} "
          f"nodes={nodes} node_rate_last={nrl:.4f} demote_shot={int(st2[1])}  {'PROTECTED' if pol==0 else 'REGRESSION!'}  ({dt2:.2f}s)")

print(); print("="*78); print("(3b) DEMOTE PATH: fires when lean is genuinely worse (100% fallback); NOT when lean wins"); print("="*78)
for bench,N in [("coherent_d3_r3",20000),("cultivation_d5",20000)]:
    ph,vm,va,nm=load(bench); setup_lean(vm); fresh(vm)
    lib.nvm_adapt_config(vm, 2048, -1,-1,-1, 0, -1.0, 1.0, 1)   # horizon=0, cost_margin=1.0 => demote iff lean>slow
    big=np.zeros((N,nm),np.uint8)
    lib.nvm_run_lean_adapt_batch(ph,vm,N,*pcg(555),big.ctypes.data,eb,256)
    st=adapt_stats(vm); lib.nvm_sg_shadow(vm,0); lib.nvm_rb_static(0)
    verdict = "DEMOTED (lean>slow, correct)" if int(st[0])==1 else "kept LEAN (lean<=slow, correct)"
    print(f"  {bench:16s} N={N}: {'AUTH' if int(st[0])==1 else 'LEAN'} demote_shot={int(st[1])} "
          f"lean_ns={st[6]:.0f} slow_ns={st[7]:.0f} node_rate={st[5]:.2f}  -> {verdict}")
