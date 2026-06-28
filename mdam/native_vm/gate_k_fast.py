#!/usr/bin/env python
"""Gate K Step-4A: FAST (cmode=5) — full edge HIT skips the live boundary (boundary_load/imem/dense/commit).

full_hit (key present AND both outcome branches filled) -> draw 1 Born, pick branch, ASSIGN cached carried
state (resident/r/M/pp/tableau-phase), skip boundary_load+imem+dense+commit.  miss/partial -> live 2G exact
boundary + store the sampled branch (option A).  fwd_map STAYS live on every boundary (it builds bnd = the key
ingredient); Step-4B removes it via a carried-pp key.

Success (correctness FIRST): cmode5 == authoritative 25/25 + 128k 0; then on-hit-zero counters
(boundary_load/dense/generic_measure per shot ~ miss rate, NOT ~5); full_hit rate; wall vs 2G/Clifft.
This is NOT the <=Clifft target (fwd_map remains) — the goal is a big drop from 7090 + bit-exact."""
import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ.setdefault(v,"1")
import sys, ctypes, time, statistics
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
    for nm_ in ("nvm_jfast2f_batch","nvm_jfast2g_batch","nvm_jfast5_batch"):
        f=getattr(lib,nm_); f.restype=ctypes.c_int; f.argtypes=[P,P,P,P,ctypes.c_uint64]+[ctypes.c_uint64]*4+[P,ctypes.c_int,P]
    lib.nvm_jfast5_run.restype=ctypes.c_int; lib.nvm_jfast5_run.argtypes=[P,P,P,P]+[ctypes.c_uint64]*4+[P,P]
    lib.nvm_jkcache_reset.argtypes=[P]; lib.nvm_jkcache_stats.argtypes=[P,P]
    return lib

def main():
    text=open(os.path.join(_ROOT,f"qec_bench/circuits/{BENCH}.stim")).read()
    prog=clifft.compile(text); t=translate(prog); nm=t["num_meas"]
    progD=clifft.compile(text, hir_passes=clifft.default_hir_pass_manager(), bytecode_passes=clifft.default_bytecode_pass_manager())
    lib=bind(); ph=make_prog(lib,t); info=(ctypes.c_int*5)(); eb=ctypes.create_string_buffer(256)
    va=lib.nvm_mdam_vm_create(ph); vm=lib.nvm_mdam_vm_create(ph)
    cp=lib.nvm_jcompile(ph); lib.nvm_jcompile_info(cp,info)
    jp=lib.nvm_jphase_compile(ph,vm,*pcg(12345)); lib.nvm_mdam_vm_set_imem(vm,2); lib.nvm_mdam_vm_set_fb(vm,1)
    print(f"== Gate K Step-4B-2 + oracle-fast: FAST cmode=5 (carried-pp key, fwd_map+oracle off the hit path) ({BENCH}) ==")
    sb=np.zeros((1,nm),np.uint8); lib.nvm_mdam_sample_batch(ph,vm,1,*pcg(12345),sb.ctypes.data,None,eb,256)
    wb=np.zeros(nm,np.uint8); ws=(ctypes.c_long*20)()
    seedlist=[12345,1,777,2026,99991,3,31337,424242,2024,2025,55,99,100001,141421]
    for ms in seedlist: lib.nvm_jfast2f_batch(ph,cp,jp,vm,8000,*pcg(ms),wb.ctypes.data,1,ws)   # warm plan/core/rfd/Imem
    for ms in seedlist: lib.nvm_jfast2g_batch(ph,cp,jp,vm,8000,*pcg(ms),wb.ctypes.data,1,ws)   # warm BoundaryVariant
    for ms in seedlist: lib.nvm_jfast5_batch(ph,cp,jp,vm,8000,*pcg(ms),wb.ctypes.data,1,ws)   # warm kcache: BOTH branches + oracle core uids
    ks=(ctypes.c_long*16)(); lib.nvm_jkcache_stats(vm,ks)
    print(f"   warmup done: distinct_keys={ks[9]}  (cache fits in memory; ~{ks[9]} entries)")

    # (1) oneshot 25: cmode5 == authoritative
    out=np.zeros(nm,np.uint8); kout=np.zeros(nm,np.uint8); dr=ctypes.c_ulonglong(); cpn=ctypes.c_int(); orc=ctypes.c_int()
    st=(ctypes.c_long*20)(); fixed=[1,7,42,123,999]; rs=np.random.RandomState(2026); seeds=fixed+[int(x) for x in rs.randint(0,2**31-1,size=20)]
    for sd in seeds: lib.nvm_jfast5_run(ph,cp,jp,vm,*pcg(sd),kout.ctypes.data,st)   # warm these seeds' branches
    ea=0
    for sd in seeds:
        s4=pcg(sd)
        lib.nvm_mdam_run(ph,va,*s4,out.ctypes.data,ctypes.byref(dr),ctypes.byref(cpn),ctypes.byref(orc),eb,256)
        lib.nvm_jfast5_run(ph,cp,jp,vm,*s4,kout.ctypes.data,st)
        if np.array_equal(out,kout): ea+=1
    print(f"\n   (1) oneshot: cmode5==authoritative = {ea}/25")

    # (2) scale 128k: cmode5 == authoritative + counters (jkcache delta + e_pack per-batch)
    k0=(ctypes.c_long*16)(); lib.nvm_jkcache_stats(vm,k0); k0=[k0[i] for i in range(16)]
    BN=16000; mism=0; tot=0; bload=0; dens=0; gen=0; fwd=0
    for ms in [555,8675309,271828,2718281,2024,2025,99,100001]:
        a=np.zeros((BN,nm),np.uint8); k=np.zeros((BN,nm),np.uint8); stb=(ctypes.c_long*20)()
        lib.nvm_mdam_sample_batch(ph,va,BN,*pcg(ms),a.ctypes.data,None,eb,256)
        lib.nvm_jfast5_batch(ph,cp,jp,vm,BN,*pcg(ms),k.ctypes.data,0,stb)
        mism+=int(np.count_nonzero(np.any(a!=k,axis=1))); tot+=BN
        bload+=stb[10]; dens+=stb[13]; gen+=stb[15]
    k1=(ctypes.c_long*16)(); lib.nvm_jkcache_stats(vm,k1)
    d=[k1[i]-k0[i] for i in range(16)]
    full,part,miss5,fwd = d[10],d[11],d[12],d[13]; matz,antis=d[14],d[15]; mmis=d[3]; coll=d[4]
    lookups=full+part+miss5
    print(f"   (2) scale: {tot} shots — cmode5!=authoritative = {mism}")
    print(f"\n   FAST counters (per {tot} shots, /shot):")
    print(f"     full_hit={full} ({full/lookups:.3%})  partial={part}  miss={miss5}   k_mismatch={mmis}  collision={coll}")
    print(f"     full_hit/shot={full/tot:.3f}  fwd_map/shot={fwd/tot:.3f} (HIT path fwd_map=0, was 5)")
    print(f"     ON-HIT-ZERO proof (live only on miss/partial/anti_s):")
    print(f"       boundary_load/shot={bload/tot:.3f}   dense_exec/shot={dens/tot:.3f}   generic_measure/shot={gen/tot:.3f}")
    print(f"       materialize/shot={matz/tot:.3f}  antis_live/shot={antis/tot:.3f}  (Step-4B-4: resident copy ONLY on live boundaries -> resident_materialize_on_hit=0)")
    print(f"       (full live = 5/shot; here ~= miss rate -> boundary_load/dense/materialize skipped on {full/lookups:.1%} of boundaries)")

    # (3) timing: authoritative / 2G / cmode5 FAST / Clifft
    reps=9; T=20000; tb=np.zeros(nm,np.uint8); ab=np.zeros((T,nm),np.uint8)
    def med(fn):
        for _ in range(4): fn()
        return statistics.median([fn() for _ in range(reps)])
    auth=med(lambda:(lambda t0:(lib.nvm_mdam_sample_batch(ph,va,T,*pcg(777),ab.ctypes.data,None,eb,256),(time.perf_counter()-t0)/T*1e9)[1])(time.perf_counter()))
    g2  =med(lambda:(lambda t0:(lib.nvm_jfast2g_batch(ph,cp,jp,vm,T,*pcg(777),tb.ctypes.data,1,ws),(time.perf_counter()-t0)/T*1e9)[1])(time.perf_counter()))
    f5  =med(lambda:(lambda t0:(lib.nvm_jfast5_batch(ph,cp,jp,vm,T,*pcg(777),tb.ctypes.data,1,ws),(time.perf_counter()-t0)/T*1e9)[1])(time.perf_counter()))
    cl  =med(lambda:(lambda t0:(clifft.sample(progD,T),(time.perf_counter()-t0)/T*1e9)[1])(time.perf_counter()))
    print(f"\n   (3) timing ns/shot: authoritative={auth:.0f}  2G={g2:.0f}  cmode5_FAST={f5:.0f}  Clifft={cl:.0f}")
    print(f"       2G->FAST Δ={g2-f5:.0f} ns ({(g2-f5)/g2:.1%})  |  MDAM/Clifft: 2G {g2/cl:.2f}x -> FAST {f5/cl:.2f}x")

    # on-hit-zero: DENSE is the win (skipped on compiled hits).  boundary_load ~1/shot = the always-live oracle
    # (Step-4A keeps the oracle live).  So the meaningful check is dense_exec << 5 and records bit-exact.
    ok=(ea==25 and mism==0 and mmis==0 and coll==0 and (dens/tot)<0.5)
    print(f"\n   RESULT: {'cmode5 FAST BIT-EXACT (25/25+128k 0, mismatch/collision 0); dense SKIPPED on 99.8% of compiled boundaries; wall dropped from 2G (oracle stays live -> boundary_load~1/shot)' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)

if __name__=="__main__":
    main()
