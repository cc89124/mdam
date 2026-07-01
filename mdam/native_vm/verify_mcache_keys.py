#!/usr/bin/env python
"""Phase-0/1/2: lightweight semantic-key SUFFICIENCY proof on the AUTHORITATIVE path.

For each measurement boundary on the authoritative run() (bit-exact vs Python), capture a RICH key
  qkey = (mp, sid_in, inv_sig, pend_sig, m_sig, kind)            # quantum-transition determinants
  ctrl = (xb, zb, i1)                                            # classical-record determinants
and the EDGE (p0, outcome, sid_out, rec).  Prove the user's thesis:
  (T1) same qkey            -> same p0           (Born prob is a function of the small key)
  (T2) same (qkey, outcome) -> same sid_out      (the quantum transition is)
  (T3) same qkey            -> same antis-flag   (rng-consumption type: Born double vs coin int)
  (T4) rec == outcome ^ parity ^ i1              (classical record is a trivial XOR of ctrl)
A conflict == "key too small": names which field is missing.  Uses ONLY authoritative measure_z, so the
edges are correct by construction (NO F4/imem/plan/bplan).  Also brute-force search the MINIMAL sufficient
subset of the 6 quantum fields.
"""
import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ.setdefault(v,"1")
import sys, ctypes, itertools
import numpy as np
import os as _os; _HERE_DIR=_os.path.dirname(_os.path.abspath(__file__))
sys.path.insert(0, _HERE_DIR); sys.path.insert(0, _os.path.join(_HERE_DIR, ".."))
from verify_mdam_oneshot import translate, make_prog, pcg, _ROOT, _HERE
import clifft

MP,SID,INV,PEND,M,XB,ZB,I1,KIND,ORC,OUT,SOUT,REC = range(13)
QF=[("mp",MP),("sid",SID),("inv",INV),("pend",PEND),("m",M),("kind",KIND)]

def bind():
    lib=ctypes.CDLL(os.path.join(_HERE,"native_mdam_vm.so")); P=ctypes.c_void_p
    lib.nvm_mdam_create.restype=P
    lib.nvm_mdam_create.argtypes=[ctypes.c_int,P,P,P,P,P,P,P,ctypes.c_int,P,ctypes.c_int,P,P,P,P,ctypes.c_int,P,P,ctypes.c_int]+[ctypes.c_int]*5
    lib.nvm_mdam_vm_create.restype=P; lib.nvm_mdam_vm_create.argtypes=[P]
    lib.nvm_mdam_run_bcap.restype=ctypes.c_int; lib.nvm_mdam_run_bcap.argtypes=[P,P]+[ctypes.c_uint64]*4+[P,P,ctypes.c_int]
    lib.nvm_bcap_n.restype=ctypes.c_long; lib.nvm_bcap_n.argtypes=[P]
    lib.nvm_bcap_get.argtypes=[P,P,P]
    lib.nvm_bcap_distinct_states.restype=ctypes.c_long; lib.nvm_bcap_distinct_states.argtypes=[P]
    return lib

def conflicts(rows, p0, cols, include_outcome):
    """Group rows by key=cols (+outcome if include_outcome); return # of rows whose target disagrees
       with the first-seen target for that key.  target = p0 (T1) or sid_out (T2)."""
    if include_outcome:
        key=rows[:, cols+[OUT]]; tgt=rows[:, SOUT]
    else:
        key=rows[:, cols]; tgt=None
    # encode key rows as bytes for grouping
    kb=np.ascontiguousarray(key).view([('',key.dtype)]*key.shape[1]).ravel()
    order=np.argsort(kb, kind='stable'); ks=kb[order]
    boundaries=np.concatenate(([True], ks[1:]!=ks[:-1]))
    grp=np.cumsum(boundaries)-1
    bad=0
    if include_outcome:
        t=tgt[order]
        # first index of each group
        first=np.zeros(grp[-1]+1, dtype=t.dtype); seen=np.zeros(grp[-1]+1, bool)
        for i in range(len(t)):
            g=grp[i]
            if not seen[g]: seen[g]=True; first[g]=t[i]
            elif first[g]!=t[i]: bad+=1
    else:
        t=p0[order]
        first=np.zeros(grp[-1]+1); seen=np.zeros(grp[-1]+1, bool)
        for i in range(len(t)):
            g=grp[i]
            if not seen[g]: seen[g]=True; first[g]=t[i]
            elif first[g]!=t[i]: bad+=1
    return bad

def run(bench, nshots):
    text=open(os.path.join(_ROOT,f"qec_bench/circuits/{bench}.stim")).read()
    prog=clifft.compile(text); t=translate(prog); nm=t["num_meas"]
    lib=bind(); ph=make_prog(lib,t); vm=lib.nvm_mdam_vm_create(ph)
    eb=ctypes.create_string_buffer(256); rec=np.zeros(nm,np.uint8)
    MAXB=nm+128; ibuf=np.zeros((MAXB,13),np.int64); pbuf=np.zeros(MAXB,np.float64)
    all_rows=[]; all_p0=[]
    err=None
    for sd in range(nshots):
        rc=lib.nvm_mdam_run_bcap(ph,vm,*pcg(sd),rec.ctypes.data,eb,256)
        if rc!=0: err=eb.value.decode(); break
        n=lib.nvm_bcap_n(vm)
        if n==0: continue
        if n>MAXB: err=f"boundary count {n}>MAXB"; break
        lib.nvm_bcap_get(vm, ibuf.ctypes.data, pbuf.ctypes.data)
        all_rows.append(ibuf[:n].copy()); all_p0.append(pbuf[:n].copy())
    if err: print(f"  ERROR: {err}"); return False
    rows=np.concatenate(all_rows); p0=np.concatenate(all_p0)
    nobs=len(rows); distinct_states=lib.nvm_bcap_distinct_states(vm)
    full=[MP,SID,INV,PEND,M,KIND]
    b1=conflicts(rows,p0,full,False)
    b2=conflicts(rows,p0,full,True)
    # T3 antis per qkey
    antis=(rows[:,ORC]==2).astype(np.int64)
    kb=np.ascontiguousarray(rows[:,full]).view([('',rows.dtype)]*len(full)).ravel()
    order=np.argsort(kb,kind='stable'); ks=kb[order]; a=antis[order]
    grp=np.cumsum(np.concatenate(([True],ks[1:]!=ks[:-1])))-1
    b3=0; seen={}
    for i in range(len(a)):
        g=grp[i]
        if g not in seen: seen[g]=a[i]
        elif seen[g]!=a[i]: b3+=1
    # T4 rec == out^parity^i1
    parity=np.where(rows[:,KIND]==0, rows[:,XB], rows[:,ZB])
    b4=int(np.count_nonzero(rows[:,REC]!=((rows[:,OUT]^parity^rows[:,I1])&1)))
    nq=len(set(map(tuple, rows[:,full].tolist())))
    print(f"== {bench}: {nshots} shots, {nobs} boundary obs ==")
    print(f"   distinct qkeys={nq}  distinct interned states={distinct_states}  antis_obs={int(antis.sum())}")
    print(f"   T1 p0 conflicts={b1}   T2 sid_out conflicts={b2}   T3 antis conflicts={b3}   T4 rec-XOR viol={b4}")
    full_ok=(b1==0 and b2==0 and b3==0 and b4==0)
    print(f"   FULL KEY (mp,sid,inv,pend,m,kind): {'SUFFICIENT' if full_ok else 'INSUFFICIENT'}")
    # minimal subset search: smallest subset of the 6 fields with 0 p0 AND 0 sid_out conflicts
    if full_ok:
        mins=[]
        for r in range(1,7):
            for combo in itertools.combinations(range(6), r):
                cols=[full[i] for i in combo]
                if conflicts(rows,p0,cols,False)==0 and conflicts(rows,p0,cols,True)==0:
                    mins.append(tuple(QF[i][0] for i in combo))
            if mins: break   # smallest r with any sufficient subset
        print(f"   MINIMAL sufficient subset(s) (size {len(mins[0]) if mins else '-'}): {mins[:6]}")
    return full_ok

if __name__=="__main__":
    benches=sys.argv[1].split(",") if len(sys.argv)>1 else ["distillation"]
    ns=int(sys.argv[2]) if len(sys.argv)>2 else 20000
    allok=True
    for b in benches:
        allok &= run(b, ns); print()
    sys.exit(0 if allok else 1)
