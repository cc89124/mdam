#!/usr/bin/env python
"""Per-op-category breakdown of the authoritative MDAM run() (PROFILE build) for a benchmark, to answer
'where does the wall go' — is it a sane distribution (frame/measure/noise over n qubits) or pathological?
Uses native_mdam_vm_prof.so (built with -DMDAM_PROFILE; the release .so is byte-unchanged)."""
import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ.setdefault(v,"1")
import sys, ctypes
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),"..","..")))
from verify_mdam_oneshot import translate, make_prog, pcg, BENCH, _ROOT, _HERE
import clifft

LBL=["SEED","RESET","RUN(loop)","MAGIC_PLAN","MAGIC_KERNEL","MAGIC_COMMIT","ORACLE","OUTPUT",
     "OP_FRAME","OP_ACTIVEGATE","OP_ROT","OP_NOISE","OP_DORMANT","OP_OTHER","PLAN_CORE","PULLBACK","LOCALIZER"]

def main():
    N=int(sys.argv[1]) if len(sys.argv)>1 else 20000
    text=open(os.path.join(_ROOT,f"qec_bench/circuits/{BENCH}.stim")).read()
    prog=clifft.compile(text); t=translate(prog); nm=t["num_meas"]; k=getattr(prog,"peak_rank",0)
    # opcode histogram
    from collections import Counter
    hist=Counter(d["op"] for d in t["ops"]) if "ops" in t else None
    lib=ctypes.CDLL(os.path.join(_HERE,"native_mdam_vm_prof.so")); P=ctypes.c_void_p
    lib.nvm_mdam_create.restype=P
    lib.nvm_mdam_create.argtypes=[ctypes.c_int,P,P,P,P,P,P,P,ctypes.c_int,P,ctypes.c_int,P,P,P,P,ctypes.c_int,P,P,ctypes.c_int]+[ctypes.c_int]*5
    lib.nvm_mdam_vm_create.restype=P; lib.nvm_mdam_vm_create.argtypes=[P]
    lib.nvm_mdam_run_batch_prof.restype=ctypes.c_int
    lib.nvm_mdam_run_batch_prof.argtypes=[P,P,ctypes.c_uint64]+[ctypes.c_uint64]*4+[P,P]
    ph=make_prog(lib,t); vm=lib.nvm_mdam_vm_create(ph)
    rec=np.zeros(nm,np.uint8); pr=(ctypes.c_double*17)()
    rc=lib.nvm_mdam_run_batch_prof(ph,vm,N,*pcg(777),rec.ctypes.data,pr)
    if rc!=0: print("not a PROFILE build"); sys.exit(1)
    pr=[pr[i]/N for i in range(17)]   # ns/shot
    run_loop=pr[2]; magic=pr[3]+pr[4]+pr[5]+pr[6]; total=pr[0]+pr[1]+run_loop+pr[7]
    inner=sum(pr[8:14])+magic
    print(f"== authoritative run() per-op breakdown ({BENCH}, n={prog.num_qubits}, peak_rank={k}, num_meas={nm}, N={N}) ==")
    print(f"   total/shot = {total:8.0f} ns   (SEED {pr[0]:.0f} + RESET {pr[1]:.0f} + RUN-loop {run_loop:.0f} + OUTPUT {pr[7]:.0f})")
    print(f"   --- inner categories (sum {inner:.0f} ~= RUN-loop {run_loop:.0f}; residual = per-op timer overhead) ---")
    for i,nm_ in [(8,"OP_FRAME"),(9,"OP_ACTIVEGATE"),(10,"OP_ROT"),(11,"OP_NOISE"),(12,"OP_DORMANT"),(13,"OP_OTHER")]:
        print(f"     {nm_:14} {pr[i]:8.0f} ns/shot  ({pr[i]/total:6.1%})")
    print(f"     {'MAGIC(plan+ker+commit+oracle)':14} {magic:8.0f} ns/shot  ({magic/total:6.1%})  [plan_core {pr[14]:.0f} pullback {pr[15]:.0f} localizer {pr[16]:.0f}]")
    if hist:
        print(f"   --- opcode histogram (ops/shot, static) ---")
        for op,c in sorted(hist.items(), key=lambda x:-x[1])[:14]: print(f"     {op:24} {c}")

if __name__=="__main__":
    main()
