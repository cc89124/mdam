#!/usr/bin/env python
"""Gate K Step-2: boundary-edge SHADOW cache (cmode=4) — CORRECTNESS lock, NO live skip.

cmode=4 runs identically to 2G (cmode 3) but at every magic boundary also (a) builds the edge key
FNV(mag, M_in, resident_in, rpp, sign, thetas), (b) on a cache HIT verifies p0 + per-outcome post-state
(survivor dense bytes + M_out + phase_pack_out) against the LIVE boundary, with raw-input comparison to
defend against FNV collisions, (c) on a MISS stores it.  Live is NOT skipped, so records must stay bit-exact
with the authoritative path; the cache's job here is only to PROVE the edge is a deterministic function of the
key (k_mismatch=0, k_collision=0) before Step-4 makes hits skip the live boundary.

Success: cmode4 records == authoritative 25/25 + 128k 0; k_mismatch=0; k_collision=0; hit rate reported
(compiled = exact key; oracle = optimistic (state,M) key -> a mismatch there is the 'oracle key insufficient'
diagnosis for Step-3, NOT a bug)."""
import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ.setdefault(v,"1")
import sys, ctypes
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),"..")))
from verify_mdam_oneshot import translate, make_prog, pcg, BENCH, _ROOT, _HERE
import clifft

def bind():
    lib=ctypes.CDLL(os.path.join(_HERE,"native_mdam_vm.so")); P=ctypes.c_void_p
    lib.nvm_mdam_create.restype=P
    lib.nvm_mdam_create.argtypes=[ctypes.c_int,P,P,P,P,P,P,P,ctypes.c_int,P,ctypes.c_int,P,P,P,P,ctypes.c_int,P,P,ctypes.c_int]+[ctypes.c_int]*5
    lib.nvm_mdam_vm_create.restype=P; lib.nvm_mdam_vm_create.argtypes=[P]
    lib.nvm_mdam_run.restype=ctypes.c_int; lib.nvm_mdam_run.argtypes=[P,P]+[ctypes.c_uint64]*4+[P,P,P,P,P,ctypes.c_int]
    lib.nvm_mdam_sample_batch.restype=ctypes.c_int; lib.nvm_mdam_sample_batch.argtypes=[P,P,ctypes.c_uint64]+[ctypes.c_uint64]*4+[P,P,P,ctypes.c_int]
    lib.nvm_mdam_vm_set_fb.argtypes=[P,ctypes.c_int]; lib.nvm_mdam_vm_set_imem.argtypes=[P,ctypes.c_int]
    lib.nvm_jcompile.restype=P; lib.nvm_jcompile.argtypes=[P]; lib.nvm_jcompile_info.argtypes=[P,P]
    lib.nvm_jphase_compile.restype=P; lib.nvm_jphase_compile.argtypes=[P,P]+[ctypes.c_uint64]*4
    lib.nvm_jfast2f_batch.restype=ctypes.c_int; lib.nvm_jfast2f_batch.argtypes=[P,P,P,P,ctypes.c_uint64]+[ctypes.c_uint64]*4+[P,ctypes.c_int,P]
    lib.nvm_jfast2g_batch.restype=ctypes.c_int; lib.nvm_jfast2g_batch.argtypes=[P,P,P,P,ctypes.c_uint64]+[ctypes.c_uint64]*4+[P,ctypes.c_int,P]
    lib.nvm_jfast4_run.restype=ctypes.c_int; lib.nvm_jfast4_run.argtypes=[P,P,P,P]+[ctypes.c_uint64]*4+[P,P]
    lib.nvm_jfast4_batch.restype=ctypes.c_int; lib.nvm_jfast4_batch.argtypes=[P,P,P,P,ctypes.c_uint64]+[ctypes.c_uint64]*4+[P,ctypes.c_int,P]
    lib.nvm_jkcache_reset.argtypes=[P]; lib.nvm_jkcache_stats.argtypes=[P,P]
    return lib

def main():
    text=open(os.path.join(_ROOT,f"qec_bench/circuits/{BENCH}.stim")).read()
    prog=clifft.compile(text); t=translate(prog); nm=t["num_meas"]
    lib=bind(); ph=make_prog(lib,t); info=(ctypes.c_int*5)(); eb=ctypes.create_string_buffer(256)
    va=lib.nvm_mdam_vm_create(ph); vm=lib.nvm_mdam_vm_create(ph)
    cp=lib.nvm_jcompile(ph); lib.nvm_jcompile_info(cp,info)
    jp=lib.nvm_jphase_compile(ph,vm,*pcg(12345)); lib.nvm_mdam_vm_set_imem(vm,2); lib.nvm_mdam_vm_set_fb(vm,1)
    print(f"== Gate K Step-2: boundary-edge SHADOW cache (cmode=4, {BENCH}) ==")
    sb=np.zeros((1,nm),np.uint8); lib.nvm_mdam_sample_batch(ph,vm,1,*pcg(12345),sb.ctypes.data,None,eb,256)
    wb=np.zeros(nm,np.uint8); ws=(ctypes.c_long*20)()
    seedlist=[12345,1,777,2026,99991,3,31337,424242,2024,2025,55,99,100001,141421]
    for ms in seedlist: lib.nvm_jfast2f_batch(ph,cp,jp,vm,8000,*pcg(ms),wb.ctypes.data,1,ws)   # warm plan/core/rfd/Imem
    for ms in seedlist: lib.nvm_jfast2g_batch(ph,cp,jp,vm,8000,*pcg(ms),wb.ctypes.data,1,ws)   # warm BoundaryVariant

    # (1) oneshot: cmode4 records == authoritative; warm kcache first
    out=np.zeros(nm,np.uint8); kout=np.zeros(nm,np.uint8); dr=ctypes.c_ulonglong(); cpn=ctypes.c_int(); orc=ctypes.c_int()
    st=(ctypes.c_long*20)(); fixed=[1,7,42,123,999]; rs=np.random.RandomState(2026); seeds=fixed+[int(x) for x in rs.randint(0,2**31-1,size=20)]
    for sd in seeds: lib.nvm_jfast4_run(ph,cp,jp,vm,*pcg(sd),kout.ctypes.data,st)   # warm kcache
    ea=0
    for sd in seeds:
        s4=pcg(sd)
        lib.nvm_mdam_run(ph,va,*s4,out.ctypes.data,ctypes.byref(dr),ctypes.byref(cpn),ctypes.byref(orc),eb,256)
        lib.nvm_jfast4_run(ph,cp,jp,vm,*s4,kout.ctypes.data,st)
        if np.array_equal(out,kout): ea+=1
    print(f"\n   (1) oneshot: cmode4==authoritative = {ea}/25")

    # (2) scale 128k: cmode4 records == authoritative + kcache stats (reset counters+cache, accumulate fresh)
    lib.nvm_jkcache_reset(vm)
    BN=16000; mism=0; tot=0
    for ms in [555,8675309,271828,2718281,2024,2025,99,100001]:
        a=np.zeros((BN,nm),np.uint8); k=np.zeros((BN,nm),np.uint8); stb=(ctypes.c_long*20)()
        lib.nvm_mdam_sample_batch(ph,va,BN,*pcg(ms),a.ctypes.data,None,eb,256)
        lib.nvm_jfast4_batch(ph,cp,jp,vm,BN,*pcg(ms),k.ctypes.data,0,stb)
        mism+=int(np.count_nonzero(np.any(a!=k,axis=1))); tot+=BN
    ks=(ctypes.c_long*16)(); lib.nvm_jkcache_stats(vm,ks)
    look,hit,miss,mmis,coll,looko,hito,misso,mmiso,dk=[ks[i] for i in range(10)]
    print(f"   (2) scale: {tot} shots — cmode4!=authoritative = {mism}")
    print(f"\n   kcache (cumulative over the {tot}-shot verify):")
    print(f"     lookups={look}  hits={hit} ({hit/max(look,1):.3%})  misses={miss}  distinct_keys={dk}")
    print(f"     COMPILED: hits={hit-hito} mismatch={mmis-mmiso} collision={coll}")
    print(f"     ORACLE:   lookups={looko} hits={hito} mismatch={mmiso}  (key=(state,M), optimistic)")
    print(f"     TOTAL k_mismatch={mmis}  k_collision={coll}")

    comp_ok = (ea==25 and mism==0 and (mmis-mmiso)==0 and coll==0)
    oracle_ok = (mmiso==0)
    print(f"\n   RESULT: compiled edge {'BIT-EXACT (k_mismatch=0, k_collision=0, records 25/25+128k 0)' if comp_ok else 'FAIL'}")
    print(f"           oracle edge   {'also exact (optimistic key sufficed)' if oracle_ok else f'MISMATCH {mmiso} -> Step-3 needs oracle full key (EXPECTED diagnosis, not a bug)'}")
    sys.exit(0 if comp_ok else 1)

if __name__=="__main__":
    main()
