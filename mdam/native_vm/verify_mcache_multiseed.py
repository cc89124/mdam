#!/usr/bin/env python
"""Phase-4 completion: multi-master-seed correctness for mcache_carry (mode 3) + SHADOW fingerprint (mode 1).
Ground truth = native authoritative sample_batch (== Python, per verify_mdam_batch).  NEVER cmode-vs-cmode."""
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
    lib.nvm_mdam_sample_batch.restype=ctypes.c_int; lib.nvm_mdam_sample_batch.argtypes=[P,P,ctypes.c_uint64]+[ctypes.c_uint64]*4+[P,P,P,ctypes.c_int]
    lib.nvm_mcache_set_mode.argtypes=[P,ctypes.c_int]; lib.nvm_mcache_reset.argtypes=[P]; lib.nvm_mcache_stats.argtypes=[P,P]
    return lib
def run(bench, BN):
    text=open(os.path.join(_ROOT,f"qec_bench/circuits/{bench}.stim")).read()
    prog=clifft.compile(text); t=translate(prog); nm=t["num_meas"]
    lib=bind(); ph=make_prog(lib,t); va=lib.nvm_mdam_vm_create(ph); vm=lib.nvm_mcache_set_mode  # placeholder
    vm=lib.nvm_mdam_vm_create(ph)
    eb=ctypes.create_string_buffer(256)
    seeds=[777,12345,2026,99991,31337,424242,555,8675309]
    # SHADOW (mode 1): every boundary re-verified against the stored edge (p0 + post-state) -> mismatch must be 0
    lib.nvm_mcache_set_mode(vm,1); lib.nvm_mcache_reset(vm)
    a=np.zeros((BN,nm),np.uint8); m=np.zeros((BN,nm),np.uint8); sh_bad=0; sh_rec=0
    for s in seeds:
        lib.nvm_mdam_sample_batch(ph,va,BN,*pcg(s),a.ctypes.data,None,eb,256)
        lib.nvm_mcache_batch(ph,vm,BN,*pcg(s),m.ctypes.data,eb,256)
        sh_rec += int(np.count_nonzero(np.any(a!=m,axis=1)))
    st=(ctypes.c_long*10)(); lib.nvm_mcache_stats(vm,st); sh_bad=st[5]
    # CARRY (mode 3): bit-exact vs authoritative, fresh cache per master seed AND warmed (cache persists across seeds)
    lib.nvm_mcache_set_mode(vm,3); lib.nvm_mcache_reset(vm); cold=0; warm=0
    for s in seeds:                              # cold pass: each seed first-seen (cache persists -> later seeds partly warm)
        lib.nvm_mdam_sample_batch(ph,va,BN,*pcg(s),a.ctypes.data,None,eb,256)
        lib.nvm_mcache_batch(ph,vm,BN,*pcg(s),m.ctypes.data,eb,256)
        cold += int(np.count_nonzero(np.any(a!=m,axis=1)))
    for s in seeds:                              # warmed pass: same seeds again (now high hit)
        lib.nvm_mdam_sample_batch(ph,va,BN,*pcg(s),a.ctypes.data,None,eb,256)
        lib.nvm_mcache_batch(ph,vm,BN,*pcg(s),m.ctypes.data,eb,256)
        warm += int(np.count_nonzero(np.any(a!=m,axis=1)))
    st2=(ctypes.c_long*10)(); lib.nvm_mcache_stats(vm,st2)
    tot=len(seeds)*BN
    print(f"== {bench}: {len(seeds)} master seeds x {BN} = {tot} shots/pass ==")
    print(f"   SHADOW(1) record!=auth = {sh_rec}   edge mismatch(fingerprint) = {sh_bad}")
    print(f"   CARRY(3)  cold pass record!=auth = {cold}   warmed pass = {warm}")
    print(f"   carry hit={st2[0]} miss={st2[1]} antis={st2[3]} pool={st2[7]} edges={st2[8]}  mismatch={st2[5]}")
    ok=(sh_rec==0 and sh_bad==0 and cold==0 and warm==0)
    print(f"   {'PASS' if ok else 'FAIL'}\n")
    return ok
if __name__=="__main__":
    allok=True
    for b,BN in (("distillation",8000),("cultivation_d3",8000),("cultivation_d5",1500)):
        allok &= run(b,BN)
    sys.exit(0 if allok else 1)
