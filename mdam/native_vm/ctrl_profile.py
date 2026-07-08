"""control-plane profile (step 1 before codegen): decompose warm run_lean per-shot into
frame / noise / measure(hash+Born) / rot / residual(dispatch+dorm+feedback+bookkeeping) via skip masks.
Goal: quantify how much of the per-shot control is codegen-removable (dispatch+static structure) vs an
irreducible floor (RNG draws + automaton hash probe).  Warm the automaton first so run_lean is the warm path.
CAVEAT: skipping frame perturbs the trajectory (frame->sign->edge), so the split is approximate; the point is
the ORDER of magnitude (is dispatch a big removable chunk or not).  taskset -c 2, single thread, min-of-5."""
import os, sys, ctypes, time
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ.setdefault(v,"1")
import numpy as np
ROOT="/home/jung/clifft-paper"; HERE=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,HERE); sys.path.insert(0,os.path.join(ROOT,"mdam")); sys.path.insert(0,ROOT)
import clifft
from verify_mdam_oneshot import translate, make_prog, pcg, load_lib
P=ctypes.c_void_p; U=ctypes.c_uint64; C=ctypes.c_int
lib=load_lib()
lib.nvm_run_lean_fb_batch.restype=C; lib.nvm_run_lean_fb_batch.argtypes=[P,P,U]+[U]*4+[P,P,C]
lib.nvm_run_lean_batch.restype=C; lib.nvm_run_lean_batch.argtypes=[P,P,U]+[U]*4+[P,P,P,C]
for f in ("nvm_mcache_set_mode","nvm_mcache_set_fblock","nvm_sg_shadow","nvm_sg_signs","nvm_mcache_set_skip"):
    getattr(lib,f).argtypes=[P,C]; getattr(lib,f).restype=None
lib.nvm_mcache_reset.argtypes=[P]; lib.nvm_sg_reset.argtypes=[P]; lib.nvm_lean_reset_counts.argtypes=[P]; lib.nvm_rb_static.argtypes=[C]
eb=ctypes.create_string_buffer(256); N=12800; SEED=99
def load(b):
    prog=clifft.compile(open(f"{ROOT}/qec_bench/circuits/{b}.stim").read()); t=translate(prog); return make_prog(lib,t), t["num_meas"]
def warm(ph,vm,seed,nm):
    lib.nvm_rb_static(1); lib.nvm_mcache_set_mode(vm,3); lib.nvm_mcache_set_fblock(vm,0)   # fblock OFF: profile the raw per-op loop (fblock is a partial compile already)
    lib.nvm_mcache_reset(vm); lib.nvm_sg_reset(vm); lib.nvm_sg_signs(vm,1); lib.nvm_sg_shadow(vm,1); lib.nvm_lean_reset_counts(vm)
    w=np.zeros((N,nm),np.uint8); lib.nvm_run_lean_fb_batch(ph,vm,N,*pcg(seed),w.ctypes.data,eb,256); lib.nvm_sg_shadow(vm,0); lib.nvm_rb_static(0)
def tlean(ph,vm,nm,skip,reps=5):
    rec=np.zeros((N,nm),np.uint8); inc=np.zeros(N,np.uint8); lib.nvm_mcache_set_skip(vm,skip)
    lib.nvm_run_lean_batch(ph,vm,N,*pcg(SEED),rec.ctypes.data,inc.ctypes.data,eb,256)
    best=1e30
    for _ in range(reps):
        t0=time.perf_counter(); lib.nvm_run_lean_batch(ph,vm,N,*pcg(SEED),rec.ctypes.data,inc.ctypes.data,eb,256); best=min(best,(time.perf_counter()-t0)/N*1e9)
    lib.nvm_mcache_set_skip(vm,0); return best
out=[]
def emit(s): print(s); out.append(s)
emit("# control-plane profile: warm run_lean per-shot decomposition (fblock OFF = raw op loop)\n")
emit("skip: &1=frame Clifford, &8=noise, &32=boundary measure(hash+Born), &4=rotation-sign.  DORM/feedback/")
emit("bookkeeping/dispatch always run.  residual = full with frame+noise+measure+rot all skipped = the loop/")
emit("dispatch + DORM(incl coin RNG) + feedback + boundary-bookkeeping floor.\n")
hdr=f"{'bench':17s} {'full_ns':>8s} | {'frame':>7s} {'noise':>7s} {'measure':>7s} {'rot':>6s} {'residual':>8s} | {'frame%':>6s} {'resid%':>6s}"
emit("```"); emit(hdr); emit("-"*len(hdr))
for b in ["cultivation_d3","distillation","coherent_d3_r1","coherent_rx_d3_r1"]:
    ph,nm=load(b); vm=lib.nvm_mdam_vm_create(ph); warm(ph,vm,SEED,nm)
    t0=tlean(ph,vm,nm,0); t1=tlean(ph,vm,nm,1); t18=tlean(ph,vm,nm,1|8); t1832=tlean(ph,vm,nm,1|8|32); t45=tlean(ph,vm,nm,1|8|32|4)
    frame=t0-t1; noise=t1-t18; meas=t18-t1832; rot=t1832-t45; resid=t45
    emit(f"{b:17s} {t0:8.1f} | {frame:7.1f} {noise:7.1f} {meas:7.1f} {rot:6.1f} {resid:8.1f} | {100*frame/t0:5.1f}% {100*resid/t0:5.1f}%")
emit("```\n")
emit("Reading for codegen: codegen removes DISPATCH + compiles the STATIC structure of frame/feedback/dorm/")
emit("bookkeeping (indices+ops as immediates, bit-packed frame).  It does NOT remove: noise-RNG draws, the Born")
emit("draw + automaton hash probe (inside 'measure'), dorm coin.  So codegen-removable ~ frame + rot + the")
emit("dispatch part of residual; irreducible floor ~ noise-RNG (part of 'noise') + measure(hash+Born) + dorm coin.")
with open(f"{ROOT}/results/benchmark_comparison/ctrl_profile.md","w") as f: f.write("\n".join(out)+"\n")
print(f"\n[written] {ROOT}/results/benchmark_comparison/ctrl_profile.md")
