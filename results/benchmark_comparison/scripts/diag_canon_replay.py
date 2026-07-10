"""Step B: key/edge-level effect of sid canonicalization (cultivation_d5).

Capture the full boundary-event stream over NS authoritative shots
(nvm_mdam_run_bcap: rows = {mp,sid_in,inv,pend,m,...,kind,outcome,...}),
then replay the automaton twice offline:
  RAW  : key = (mp,kind,sid,inv,pend,m)            [today's key]
  CANON: key = (mp,kind,cls[sid],inv,pend,m)       [phase-canonical sid]
and count (a) distinct keys, (b) distinct edges (key,outcome),
(c) shot-level fb (shot contains >=1 first-seen edge) per 1024-shot window.
The fb ratio CANON/RAW is the honest upper bound of what sid-canonicalization
can recover (inv/pend/m stay raw - cannot be canonicalized offline).
argv: bench NS   (default cultivation_d5 30000)
"""
import os, sys, time, ctypes
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ.setdefault(v,"1")
import numpy as np
ROOT="/home/jung/clifft-paper"; NV=ROOT+"/mdam/native_vm"
sys.path.insert(0,NV); sys.path.insert(0,ROOT+"/mdam"); sys.path.insert(0,ROOT)
import clifft
from verify_mdam_oneshot import translate, make_prog, pcg, load_lib
P=ctypes.c_void_p; U=ctypes.c_uint64; C=ctypes.c_int; D=ctypes.c_double; L=ctypes.c_long; LL=ctypes.c_longlong
lib=load_lib()
lib.nvm_mdam_run_bcap.restype=C; lib.nvm_mdam_run_bcap.argtypes=[P,P]+[U]*4+[P,P,C]
lib.nvm_bcap_n.restype=L; lib.nvm_bcap_n.argtypes=[P]
lib.nvm_bcap_get.argtypes=[P,ctypes.POINTER(LL),ctypes.POINTER(D)]; lib.nvm_bcap_get.restype=None
lib.nvm_bcap_distinct_states.restype=L; lib.nvm_bcap_distinct_states.argtypes=[P]
lib.nvm_diag_canon_map.restype=L; lib.nvm_diag_canon_map.argtypes=[P,C,ctypes.POINTER(L),L]

bench=sys.argv[1] if len(sys.argv)>1 else "cultivation_d5"
NS=int(sys.argv[2]) if len(sys.argv)>2 else 30000
eb=ctypes.create_string_buffer(256)
t=translate(clifft.compile(open(f"{ROOT}/qec_bench/circuits/{bench}.stim").read())); nm=t["num_meas"]
ph=make_prog(lib,t)
vm=lib.nvm_mdam_vm_create(ph)
rec=np.zeros(nm,np.uint8)

rows=[]; shot_of=[]
t0=time.perf_counter()
for i in range(NS):
    r=lib.nvm_mdam_run_bcap(ph,vm,*pcg(30000+i),rec.ctypes.data,eb,256)
    if r!=0:   # reduce_full-class: skip shot (counting run)
        continue
    n=lib.nvm_bcap_n(vm)
    bi=np.zeros((n,13),np.int64); bp=np.zeros(n)
    lib.nvm_bcap_get(vm,bi.ctypes.data_as(ctypes.POINTER(LL)),bp.ctypes.data_as(ctypes.POINTER(D)))
    rows.append(bi.copy()); shot_of.append(np.full(n,i,np.int64))
    if (i+1)%5000==0: print(f"  {i+1:,}/{NS:,} shots  ({time.perf_counter()-t0:.0f}s)",flush=True)
E=np.concatenate(rows); S=np.concatenate(shot_of)
nsid=lib.nvm_bcap_distinct_states(vm)
cmap1=np.zeros(nsid,dtype=np.int64)
n1=lib.nvm_diag_canon_map(vm,1,cmap1.ctypes.data_as(ctypes.POINTER(L)),nsid)
cmap0=np.zeros(nsid,dtype=np.int64)
n0=lib.nvm_diag_canon_map(vm,0,cmap0.ctypes.data_as(ctypes.POINTER(L)),nsid)
assert n1==nsid and n0==nsid,(n1,n0,nsid)
print(f"\n{bench}: {NS:,} shots, {len(E):,} boundary events, {nsid:,} distinct sids "
      f"-> round {len(np.unique(cmap0)):,}, phase-canon {len(np.unique(cmap1)):,}",flush=True)

# columns: 0 mp,1 sid_in,2 inv,3 pend,4 m, 8 kind, 10 outcome, 11 sid_out
def keys(sidcol):
    K=np.stack([E[:,0],E[:,8],sidcol,E[:,2],E[:,3],E[:,4]],axis=1)
    return K
def stats(name,sidcol):
    K=keys(sidcol)
    uk =len(np.unique(K,axis=0))
    KE=np.concatenate([K,E[:,10:11]],axis=1)
    ue =len(np.unique(KE,axis=0))
    # temporal shot-level fb: shot has >=1 first-seen edge
    seen=set(); fb=np.zeros(NS,bool)
    for r,sh in zip(map(tuple,KE),S):
        if r not in seen: seen.add(r); fb[sh]=True
    W=1024; nw=NS//W
    fbw=fb[:nw*W].reshape(nw,W).mean(axis=1)
    print(f"{name:6s} keys={uk:8,d} edges={ue:8,d} fb(last4win)={fbw[-4:].mean()*100:5.2f}% "
          f"fb curve %: {np.round(fbw[::max(1,nw//8)]*100,1)}",flush=True)
    return uk,ue,fbw
uk_r,ue_r,fw_r=stats("RAW",E[:,1])
uk_c,ue_c,fw_c=stats("CANON",cmap1[E[:,1]])
uk_0,ue_0,fw_0=stats("ROUND",cmap0[E[:,1]])
print(f"\nkey merge  : {1-uk_c/uk_r:6.2%}   edge merge: {1-ue_c/ue_r:6.2%}")
print(f"fb steady  : RAW {fw_r[-4:].mean()*100:.2f}% -> CANON {fw_c[-4:].mean()*100:.2f}% "
      f"(x{fw_c[-4:].mean()/fw_r[-4:].mean():.3f})")
np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)),f"canonreplay_{bench}.npz"),
         E=E,S=S,cmap0=cmap0,cmap1=cmap1,fw_r=fw_r,fw_c=fw_c,fw_0=fw_0)
