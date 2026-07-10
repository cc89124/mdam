"""wall_table v2 measurement (canon-integrated production, 1 core).

Per bench:
  clifft : compile once (timed) -> R=3 x clifft.sample(prog, Nc) walls
  MDAM   : front-end compile once (clifft.compile+translate+make_prog, timed)
           -> R=3 x run_batch(text, Nm, seed=40000); runtime = info['total_ns']
           (segment walls + canon probe + retry waste; EXCLUDES codegen g++ wall,
           which rep1 pays into a fresh shared cache dir and is reported in the
           compile column: mdam_compile_s = front_end + rep1 codegen compile)
  bitexact: run_batch(2000, seed 11/22) vs authoritative composition per plan
Shot policy (user-approved): cache/LEAN benches N=1M BOTH tools; AUTH benches
flat-cost so N differs (d7_r1 clifft 20 / mdam 100k; d5_r5 clifft 20 / mdam 8k).
Stats: mean/min/max over the 3 reps (same seed -> identical work).
Rows appended to wt2_rows.tsv as they finish."""
import os, sys, time, ctypes, shutil
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ.setdefault(v,"1")
import numpy as np
SC=os.path.dirname(os.path.abspath(__file__))
ROOT="/home/jung/clifft-paper"
sys.path.insert(0,ROOT+"/mdam/native_vm"); sys.path.insert(0,ROOT+"/mdam"); sys.path.insert(0,ROOT)
import clifft
import mdam_run as M
from verify_mdam_oneshot import translate, make_prog, pcg
D=ctypes.c_double
CG=os.path.join(SC,"cgcache_wt2")

# (bench, N_clifft, N_mdam, prewarm) — cheap first.  prewarm=True (LEAN benches): compile the
# codegen .so up front so the runtime column is pure runtime for BOTH tools (clifft's sample also
# excludes its compile); the prewarm g++ wall goes into the compile column.  AUTH-routed benches
# never reach the codegen stage -> no prewarm, compile = front-end only.
BENCHES=[("coherent_d3_r1",1_000_000,1_000_000,True),
         ("cultivation_d3",1_000_000,1_000_000,True),
         ("coherent_d3_r3",1_000_000,1_000_000,True),
         ("distillation",1_000_000,1_000_000,True),
         ("surface_d7_r7",1_000_000,1_000_000,True),
         ("coherent_d5_r1",100_000,100_000,False),
         ("cultivation_d5",1_000_000,1_000_000,True),
         ("coherent_d7_r1",20,100_000,False),
         ("coherent_rx_d3_r1",1_000_000,1_000_000,True),
         ("coherent_d5_r5",20,8_000,False),
         ("coherent_rx_d3_r3",1_000_000,1_000_000,True)]
R=3
out=open(os.path.join(SC,"wt2_rows.tsv"),"a")

def bitexact(text, seed):
    rec,info=M.run_batch(text,2000,seed=seed,cache_dir=CG)
    t=translate(clifft.compile(text)); ph=make_prog(M.lib,t)
    va=M.lib.nvm_mdam_vm_create(ph); eb=ctypes.create_string_buffer(256)
    M.lib.nvm_rb_static_reset(); M.lib.nvm_rb_static(0)
    M.lib.nvm_mcache_set_mode(va,3); M.lib.nvm_mcache_set_fblock(va,1)
    ref=np.zeros_like(rec); done=0
    for pth,n,ns in info["plan"]:
        if "retry" in pth: return -1          # seed-bumped segment: composition ref not applicable
        r=M.lib.nvm_mdam_sample_batch(ph,va,n,*pcg(seed+done),ref[done:done+n].ctypes.data,None,eb,256)
        assert r==0, eb.value; done+=n
    return int(np.count_nonzero(rec!=ref))

for bench,Nc,Nm,prewarm in BENCHES:
    text=open(f"{ROOT}/qec_bench/circuits/{bench}.stim").read()
    # ---- clifft ----
    t0=time.perf_counter(); prog=clifft.compile(text); c_comp=time.perf_counter()-t0
    cw=[]
    for _ in range(R):
        t0=time.perf_counter(); clifft.sample(prog,Nc); cw.append((time.perf_counter()-t0)/Nc*1e9)
    # ---- MDAM front-end compile ----
    t0=time.perf_counter(); prog2=clifft.compile(text); t=translate(prog2); ph=make_prog(M.lib,t)
    fe=time.perf_counter()-t0
    # ---- MDAM codegen prewarm (LEAN benches): pure-runtime reps, g++ wall -> compile column ----
    gen_comp=0.0
    if prewarm:
        _,hit,gw=M.prewarm_so(text,cache_dir=CG); gen_comp=0.0 if hit else gw
    # ---- MDAM production reps ----
    mw=[]; info=None
    for rep in range(R):
        rec,info=M.run_batch(text,Nm,seed=40000,cache_dir=CG)
        mw.append(info["total_ns"])
        if rep==0: gen_comp+=info["compile_s"]      # 0 on prewarmed/declined paths
    m_comp=fe+gen_comp
    be=[bitexact(text,11),bitexact(text,22)]
    bes="OK" if be==[0,0] else ("OK*" if all(b<=0 for b in be) else f"FAIL{be}")
    route=("AUTH" if info["policy"]=="AUTH" else ("LEAN(compiled)" if info["engaged"] else "LEAN(interp)"))
    cm=info.get("canon_merge"); cm=-1 if cm is None else cm
    row=(f"{bench}\t{Nc}\t{Nm}\t{c_comp:.4f}\t{np.mean(cw):.1f}\t{min(cw):.1f}\t{max(cw):.1f}\t"
         f"{m_comp:.4f}\t{np.mean(mw):.1f}\t{min(mw):.1f}\t{max(mw):.1f}\t"
         f"{np.mean(cw)/np.mean(mw):.2f}\t{route}\t{int(info['canon'])}\t{cm:.3f}\t"
         f"{info['probe_fb']*100:.2f}\t{info['retry_waste_s']:.2f}\t{bes}")
    out.write(row+"\n"); out.flush()
    print(f"{bench:20s} clifft {np.mean(cw):>12.1f} [{min(cw):.1f},{max(cw):.1f}]  "
          f"mdam {np.mean(mw):>11.1f} [{min(mw):.1f},{max(mw):.1f}]  x{np.mean(cw)/np.mean(mw):.2f} "
          f"{route} canon={int(info['canon'])} be={bes} comp(c/m)={c_comp:.2f}/{m_comp:.2f}s",flush=True)
print("ALL DONE",flush=True)
