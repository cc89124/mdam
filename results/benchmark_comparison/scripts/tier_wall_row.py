"""Measure the codegen-tier production entry (codegen_exec.run_batch) for ONE bench -> append a row.
Protocol: .so cache PRE-WARMED (circuit-lifetime cached scenario; compile_s disclosed separately),
cold single-call run_batch, N = adaptive_wall policy (max(150k, 18s/lean_ns), cap 10M & 400MB buf),
except cultivation_d3 N=1e6 (native reduce_full gap ~1/2M shots; seed-retry, attempts disclosed) and
coherent_d5_r5 N=8000 (heavy AUTH regime, same N as the adapt column).
Bit-exact spot: run_batch(2000) vs authoritative sample_batch(2000), 2 seeds (skipped for coherent_d5_r5:
2000-shot probe IS the verified adaptive path; marked OK*).
argv: bench lean_ns clifft_ns resfile [pool_off]"""
import os, sys, time
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ.setdefault(v,"1")
import numpy as np, ctypes
ROOT=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","..","..")); NV=ROOT+"/mdam/native_vm"
sys.path.insert(0,NV); sys.path.insert(0,ROOT+"/mdam"); sys.path.insert(0,ROOT)
import mdam_run as ce
from verify_mdam_oneshot import pcg
import clifft

bench=sys.argv[1]; lean_ns=float(sys.argv[2]); clifft_ns=float(sys.argv[3]); resf=sys.argv[4]
flags=(sys.argv[5].split(",") if len(sys.argv)>5 else [])
pool_off="pool_off" in flags; noprewarm="noprewarm" in flags   # noprewarm: tier routes AUTH before any
                                                               # compile at production N -> faithful = no .so
force_lean="force_lean" in flags                               # LEAN-forced ablation: AUTH route disabled,
                                                               # compile gate/race criterion unchanged
text=open(f"{ROOT}/qec_bench/circuits/{bench}.stim").read()
from verify_mdam_oneshot import translate, make_prog
prog=clifft.compile(text); nm=translate(prog)["num_meas"]
cfg={"POOL_OFF":True} if pool_off else {}
if force_lean: cfg["FORCE_LEAN"]=True

# prewarm .so (cached-lifetime scenario; not counted in wall, disclosed)
cw=0.0
if not noprewarm: so,hit,cw=ce.prewarm_so(text)
# N policy
if bench=="cultivation_d3": N=1_000_000
elif bench=="coherent_d5_r5": N=8000
else:
    N=int(max(150000, 18e9/max(lean_ns,1))); N=min(N,10_000_000,int(400e6//max(nm,1)))
# bit-exact spot
mism=0; spot="OK"
if bench=="coherent_d5_r5": spot="OK*"
else:
    eb=ctypes.create_string_buffer(256)
    lib=ce.lib
    t=translate(prog); ph=make_prog(lib,t)
    for sd in (11,22):
        va=lib.nvm_mdam_vm_create(ph)
        A=np.zeros((2000,nm),np.uint8)
        r=lib.nvm_mdam_sample_batch(ph,va,2000,*pcg(sd),A.ctypes.data,None,eb,256); assert r==0, eb.value
        B,_=ce.run_batch(text,2000,seed=sd,cfg=cfg)
        mism+=int((A!=B).sum())
    spot="OK" if mism==0 else "FAIL"
# timed cold single-call (retry seeds for the pre-existing reduce_full gap)
attempts=0; err=""
for sd in (40000,40001,40002,40003,40004):
    attempts+=1
    try:
        t0=time.perf_counter(); rec,info=ce.run_batch(text,N,seed=sd,cfg=cfg); wall=time.perf_counter()-t0
        break
    except RuntimeError as e:
        err=str(e); rec=None
if rec is None: raise SystemExit(f"{bench}: all seeds failed: {err}")
cg_ns=wall/N*1e9
paths={}
for pth,n,ns in info["plan"]: paths[pth]=paths.get(pth,0)+n
route="+".join(f"{k}:{v}" for k,v in paths.items())
pol=info["policy"]; eng=info["engaged"]
fbo=info.get("fb_overall",-1.0)
with open(resf,"a") as f:
    f.write(f"{bench}\t{N}\t{cg_ns:.1f}\t{clifft_ns/cg_ns:.2f}\t{pol}\t{int(eng)}\t{cw:.2f}\t{attempts}\t{spot}\t{route}\t{100*fbo:.1f}\n")
print(f"{bench:18s} N={N:>8d} cg={cg_ns:>10.1f}ns spd={clifft_ns/cg_ns:.2f}x pol={pol} engaged={eng} "
      f"prewarm_compile={cw:.1f}s tries={attempts} {spot}\n  route: {route}",flush=True)
