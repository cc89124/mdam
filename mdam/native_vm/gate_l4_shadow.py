#!/usr/bin/env python
"""Gate L4c: cmode4 SHADOW for a coherent benchmark.  cmode4 (kshadow) keeps the live forward AND
verifies the snapshot (fb_mismatch) + maintains/verifies the boundary-edge cache (k_mismatch) at every
boundary with NO live skip.  Pass criteria: fb_mismatch=0, k_mismatch=0, collision=0, AND cmode4 records
== authoritative (sample_batch) bit-exact.  Run BEFORE cmode5 FAST.  MDAM_BENCH selects the circuit."""
import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ.setdefault(v,"1")
import sys, ctypes
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),"..","..")))
from verify_mdam_oneshot import translate, make_prog, pcg, BENCH, _ROOT
from gate_k_fast import bind
import clifft

def main():
    text=open(os.path.join(_ROOT,f"qec_bench/circuits/{BENCH}.stim")).read()
    prog=clifft.compile(text); t=translate(prog); nm=t["num_meas"]; k=getattr(prog,"peak_rank",0)
    lib=bind(); ph=make_prog(lib,t); info=(ctypes.c_int*5)(); eb=ctypes.create_string_buffer(256)
    lib.nvm_jfast4_batch.restype=ctypes.c_int
    lib.nvm_jfast4_batch.argtypes=[ctypes.c_void_p]*4+[ctypes.c_uint64]*5+[ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p]
    va=lib.nvm_mdam_vm_create(ph); vm=lib.nvm_mdam_vm_create(ph)
    cp=lib.nvm_jcompile(ph); lib.nvm_jcompile_info(cp,info)
    jp=lib.nvm_jphase_compile(ph,vm,*pcg(12345)); lib.nvm_mdam_vm_set_imem(vm,2); lib.nvm_mdam_vm_set_fb(vm,1)
    ji=(ctypes.c_int*5)(); lib.nvm_jphase_info(cp,ji)
    print(f"== Gate L4c cmode4 SHADOW ({BENCH}, n={prog.num_qubits}, peak_rank={k}) ==")
    print(f"   jcompile fast_ok={info[0]}  nrot={info[1]}  record_cap={info[2]}   jphase nmagic={ji[2]} built={ji[3]}")
    if not info[0]:
        print("   jcompile NOT fast_ok (nrot/record_cap>64) -> cmode5 path unsupported for this bench"); sys.exit(2)
    sb=np.zeros((1,nm),np.uint8); lib.nvm_mdam_sample_batch(ph,vm,1,*pcg(12345),sb.ctypes.data,None,eb,256)  # build fb_snap
    wb=np.zeros(nm,np.uint8); ws=(ctypes.c_long*20)()
    seedlist=[12345,1,777,2026,99991,3,31337,424242,2024,2025,55,99,100001,141421]
    for ms in seedlist: lib.nvm_jfast2f_batch(ph,cp,jp,vm,4000,*pcg(ms),wb.ctypes.data,1,ws)   # warm plan/core/rfd/Imem
    for ms in seedlist: lib.nvm_jfast2g_batch(ph,cp,jp,vm,4000,*pcg(ms),wb.ctypes.data,1,ws)   # warm BoundaryVariant

    # cmode4 SHADOW: live forward + verify snapshot + verify/store edge cache, NO skip.  Compare to authoritative.
    k0=(ctypes.c_long*16)(); lib.nvm_jkcache_stats(vm,k0); k0=[k0[i] for i in range(16)]
    BN=8000; recmis=0; tot=0; fbmis=0
    for ms in [555,8675309,271828,2718281,2024,2025,99,100001]:
        a=np.zeros((BN,nm),np.uint8); c=np.zeros((BN,nm),np.uint8); stb=(ctypes.c_long*20)()
        lib.nvm_mdam_sample_batch(ph,va,BN,*pcg(ms),a.ctypes.data,None,eb,256)
        lib.nvm_jfast4_batch(ph,cp,jp,vm,BN,*pcg(ms),c.ctypes.data,0,stb)
        recmis+=int(np.count_nonzero(np.any(a!=c,axis=1))); tot+=BN; fbmis+=stb[12]
    k1=(ctypes.c_long*16)(); lib.nvm_jkcache_stats(vm,k1); d=[k1[i]-k0[i] for i in range(16)]
    kmis=d[3]+d[8]; coll=d[4]; lookups=d[0]
    print(f"   cmode4 shadow over {tot} shots:")
    print(f"     records != authoritative = {recmis}")
    print(f"     fb_mismatch (snapshot vs live) = {fbmis}")
    print(f"     k_mismatch (edge cache vs live) = {kmis}   collision = {coll}   (edge lookups={lookups}, distinct_keys={k1[9]})")
    ok=(recmis==0 and fbmis==0 and kmis==0 and coll==0)
    print(f"\n   RESULT: {'cmode4 SHADOW CLEAN (mismatch=0) -> proceed to cmode5' if ok else 'FAIL (mismatch != 0)'}")
    sys.exit(0 if ok else 1)

if __name__=="__main__":
    main()
