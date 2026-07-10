"""Step A of the BoundaryKey canonicalization experiment (cultivation_d5).

Question: how many of the stored distinct dense blocks (bcap_amp, the `sid`
component of the boundary key) are REPRESENTATION variants of the same physical
state?  nvm_diag_compress counts distinct blocks under 4 equivalences:
  [0] exact raw bits            (= what the key uses today, baseline)
  [1] rounded to 1e-9 grid      (ulp / FP-path noise collapsed)
  [2] global-phase-canonical + rounded  (phase AND ulp collapsed)
  [3] |amp|^2 rounded           (over-merge floor: modulus only, NOT physical)
A big [0]->[2] drop = canonicalization has real headroom; [0]~=[2] = the
novelty is physical and the hypothesis dies.

Runs forced-LEAN (adaptive entry, cal=0, mode 3, pool off = production build
path) in chunks, printing the 4 counts + edge count at each checkpoint so we
also see whether duplication GROWS with N.
argv: bench total_shots chunk   (default cultivation_d5 300000 100000)
"""
import os, sys, time, ctypes
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ.setdefault(v,"1")
import numpy as np
ROOT="/home/jung/clifft-paper"; NV=ROOT+"/mdam/native_vm"
sys.path.insert(0,NV); sys.path.insert(0,ROOT+"/mdam"); sys.path.insert(0,ROOT)
import clifft
from verify_mdam_oneshot import translate, make_prog, pcg, load_lib
P=ctypes.c_void_p; U=ctypes.c_uint64; C=ctypes.c_int; D=ctypes.c_double; L=ctypes.c_long
lib=load_lib()
lib.nvm_run_lean_adapt_batch.restype=C; lib.nvm_run_lean_adapt_batch.argtypes=[P,P,U]+[U]*4+[P,P,C]
for f in ("nvm_mcache_set_mode","nvm_mcache_set_fblock","nvm_sg_shadow","nvm_sg_signs"):
    getattr(lib,f).argtypes=[P,C]; getattr(lib,f).restype=None
lib.nvm_mcache_reset.argtypes=[P]; lib.nvm_sg_reset.argtypes=[P]
lib.nvm_lean_reset_counts.argtypes=[P]
lib.nvm_adapt_config.argtypes=[P,L,L,L,L,L,D,D,C]; lib.nvm_adapt_config.restype=None
lib.nvm_adapt_cal.argtypes=[P,L,D]; lib.nvm_adapt_cal.restype=None
lib.nvm_lean_reserve.argtypes=[P,L,L]; lib.nvm_lean_reserve.restype=None
lib.nvm_diag_compress.argtypes=[P,ctypes.POINTER(L)]; lib.nvm_diag_compress.restype=None
lib.nvm_mcache_stats.argtypes=[P,ctypes.POINTER(L)]; lib.nvm_mcache_stats.restype=None
lib.nvm_lean_stats.argtypes=[P,ctypes.POINTER(L)]; lib.nvm_lean_stats.restype=None

bench=sys.argv[1] if len(sys.argv)>1 else "cultivation_d5"
NT=int(sys.argv[2]) if len(sys.argv)>2 else 300000
CH=int(sys.argv[3]) if len(sys.argv)>3 else 100000
eb=ctypes.create_string_buffer(256)
t=translate(clifft.compile(open(f"{ROOT}/qec_bench/circuits/{bench}.stim").read())); nm=t["num_meas"]
ph=make_prog(lib,t)

vm=lib.nvm_mdam_vm_create(ph)
lib.nvm_rb_static_reset(); lib.nvm_rb_static(1)
lib.nvm_adapt_config(vm,512,0,0,1<<60,-1,1e9,0.0,0)
lib.nvm_adapt_cal(vm,0,-1.0)                       # pure LEAN from shot 0
lib.nvm_mcache_set_mode(vm,3); lib.nvm_mcache_set_fblock(vm,1)
lib.nvm_mcache_reset(vm); lib.nvm_sg_reset(vm); lib.nvm_sg_signs(vm,1)
lib.nvm_lean_reset_counts(vm)
lib.nvm_lean_reserve(vm,4_000_000,16_000_000)

buf=np.zeros((CH,nm),np.uint8)
done=0; seed=30000
print(f"{bench}: total={NT:,} chunk={CH:,}  (mode3, pool off, forced-LEAN)", flush=True)
print(f"{'shots':>9s} {'exact':>9s} {'round1e-9':>9s} {'phase-can':>9s} {'modulus':>9s} {'edges':>10s} {'fb%':>6s} {'chunk_s':>8s}", flush=True)
while done<NT:
    t0=time.perf_counter()
    r=lib.nvm_run_lean_adapt_batch(ph,vm,CH,*pcg(seed),buf.ctypes.data,eb,256)
    dt=time.perf_counter()-t0
    if r!=0:
        # reduce_full-class retry: new seed, tables kept (counting run, records unused)
        print(f"  retry: {eb.value.decode()[:80]}", flush=True); seed+=1; continue
    done+=CH; seed+=1
    o=(L*4)(); lib.nvm_diag_compress(vm,o)
    st=(L*10)(); lib.nvm_mcache_stats(vm,st)
    ls=(L*3)(); lib.nvm_lean_stats(vm,ls)
    fb=100.0*ls[2]/done
    print(f"{done:9,d} {o[0]:9,d} {o[1]:9,d} {o[2]:9,d} {o[3]:9,d} {st[8]:10,d} {fb:6.2f} {dt:8.1f}", flush=True)
ratio=o[2]/o[0] if o[0] else 1.0
print(f"\nphase-canonical/exact = {ratio:.4f}  ({o[0]-o[2]:,} of {o[0]:,} blocks are representation duplicates)", flush=True)
