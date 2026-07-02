"""Confirm the AUTH-demote fix: (a) d5_r5 no longer OOMs (peak RSS bounded, demote frees tables),
(b) localization circuits demote to AUTH and per-shot approaches auth speed, (c) still bit-exact.
Default config.  Bounded N so it finishes; we report demote_shot, nodes(after free), peak RSS, and the
post-demote per-shot time (AUTH steady state) vs auth_ns from the tsv.  taskset -c 2, single-thread."""
import os, sys, ctypes, time, resource
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
def setup_lean(vm): lib.nvm_rb_static_reset(); lib.nvm_rb_static(1); lib.nvm_mcache_set_mode(vm,3); lib.nvm_mcache_set_fblock(vm,1)
def fresh(vm): lib.nvm_mcache_reset(vm); lib.nvm_sg_reset(vm); lib.nvm_sg_signs(vm,1); lib.nvm_sg_shadow(vm,1); lib.nvm_lean_reset_counts(vm)
eb=ctypes.create_string_buffer(256)

# tsv reference (clifft_ns, auth_ns)
ref={}
for ln in open(f"{ROOT}/results/benchmark_comparison/wall_table.tsv"):
    if ln.startswith("#") or not ln.strip(): continue
    f=ln.split("\t"); ref[f[0]]=(float(f[4]),float(f[5]))   # clifft_ns, auth_ns

def run(bench, N, cfg=None, spot=1500):
    txt=open(f"{ROOT}/qec_bench/circuits/{bench}.stim").read()
    prog=clifft.compile(txt); t=translate(prog); nm=t["num_meas"]; ph=make_prog(lib,t)
    vm=lib.nvm_mdam_vm_create(ph); va=lib.nvm_mdam_vm_create(ph)
    # bit-exact spot check vs authoritative (across whatever policy switch happens in the first `spot` shots)
    A=np.zeros((spot,nm),np.uint8); B=np.zeros((spot,nm),np.uint8)
    setup_lean(va); lib.nvm_rb_static(0); lib.nvm_mdam_sample_batch(ph,va,spot,*pcg(7),A.ctypes.data,None,eb,256)
    setup_lean(vm); fresh(vm)
    if cfg: lib.nvm_adapt_config(vm,*cfg)
    lib.nvm_run_lean_adapt_batch(ph,vm,spot,*pcg(7),B.ctypes.data,eb,256)
    mism=int((A!=B).sum())
    # timed full run (fresh vm)
    vm2=lib.nvm_mdam_vm_create(ph); setup_lean(vm2); fresh(vm2)
    if cfg: lib.nvm_adapt_config(vm2,*cfg)
    buf=np.zeros((N,nm),np.uint8); t0=time.perf_counter()
    lib.nvm_run_lean_adapt_batch(ph,vm2,N,*pcg(40000),buf.ctypes.data,eb,256)
    wall=time.perf_counter()-t0
    o=(D*16)(); lib.nvm_adapt_stats(vm2,o); lib.nvm_sg_shadow(vm2,0); lib.nvm_rb_static(0)
    pol="AUTH@%d"%int(o[1]) if int(o[0])==1 else "LEAN"
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0  # MB (peak, whole process)
    cl,au=ref[bench]
    return dict(bench=bench,N=N,nm=nm,wall=wall,adapt_ns=wall/N*1e9,pol=pol,demote=int(o[1]),
                nodes_after=int(o[9]),edges_after=int(o[10]),mem_est=o[11]/1e6,rss=rss,mism=mism,
                clifft_ns=cl,auth_ns=au,sp_adapt=cl/(wall/N*1e9),sp_auth=cl/au)

if __name__=="__main__":
    which=sys.argv[1] if len(sys.argv)>1 else "d5_r5"
    N=int(sys.argv[2]) if len(sys.argv)>2 else 12000
    bench={"d5_r5":"coherent_d5_r5","d7_r1":"coherent_d7_r1","d5_r1":"coherent_d5_r1"}[which]
    r=run(bench,N)
    print(f"\n{r['bench']}  N={r['N']}  nm={r['nm']}")
    print(f"  bit-exact spot: mism={r['mism']}  {'OK' if r['mism']==0 else 'FAIL'}")
    print(f"  policy={r['pol']}  demote_shot={r['demote']}")
    print(f"  nodes_after_free={r['nodes_after']}  edges_after_free={r['edges_after']}  mem_est={r['mem_est']:.1f}MB")
    print(f"  PEAK RSS (whole proc) = {r['rss']:.0f} MB   <-- OOM check")
    print(f"  adapt_ns={r['adapt_ns']:.0f}  (wall {r['wall']:.1f}s)")
    print(f"  speedup: adapt={r['sp_adapt']:.2f}x   auth(steady-state target)={r['sp_auth']:.2f}x   clifft_ns={r['clifft_ns']:.0f}")
