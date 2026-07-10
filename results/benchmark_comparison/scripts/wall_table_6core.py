"""6-core (all physical cores) throughput measurement — clifft-paper harness protocol mirror.

Protocol (= qec_bench/bench_common.py): T=6 worker processes (spawn), each pinned 1:1 to a
physical core (CPUs 0-5; HT siblings 6-11 left idle), each worker independently compiles and
samples the FULL per-worker N (total shots = 6N).  Reported wall = pool.map wall (includes
per-worker compile, negligible for clifft; MDAM codegen .so comes from the SHARED prewarmed
cache dir so workers never g++).  R=3 reps, fresh workers each rep (cold caches — honest).
effective ns/shot = wall / (T*N)  -> same unit as the 1-core table.
MDAM worker seeds: 40000 + wid*10_000_000 (independent streams).
cult_d5 memory guard: 3.8GB cache/worker; if available RAM < workers*4+4 GB, shrink workers
for that bench and record the actual count in the row.
Rows appended to wt6_rows.tsv."""
import os, sys, time, multiprocessing

SC=os.path.dirname(os.path.abspath(__file__))
ROOT="/home/jung/clifft-paper"
CG=os.path.join(SC,"cgcache_wt2")           # shared, prewarmed by the 1-core run
T=6; CPUS=list(range(6)); R=3

BENCHES=[("coherent_d3_r1",1_000_000,1_000_000),
         ("cultivation_d3",1_000_000,1_000_000),
         ("coherent_d3_r3",1_000_000,1_000_000),
         ("distillation",1_000_000,1_000_000),
         ("surface_d7_r7",1_000_000,1_000_000),
         ("coherent_d5_r1",100_000,100_000),
         ("cultivation_d5",1_000_000,1_000_000),
         ("coherent_d7_r1",20,100_000),
         ("coherent_rx_d3_r1",1_000_000,1_000_000),
         ("coherent_d5_r5",20,8_000),
         ("coherent_rx_d3_r3",1_000_000,1_000_000)]

def _pin_env(cpu):
    os.sched_setaffinity(0,{cpu})
    for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        os.environ[v]="1"

def clifft_worker(args):
    text,N,cpu=args
    _pin_env(cpu)
    import clifft
    prog=clifft.compile(text)
    t0=time.perf_counter(); clifft.sample(prog,N)
    return time.perf_counter()-t0

def mdam_worker(args):
    text,N,cpu,seed=args
    _pin_env(cpu)
    sys.path.insert(0,ROOT+"/mdam/native_vm"); sys.path.insert(0,ROOT+"/mdam"); sys.path.insert(0,ROOT)
    import mdam_run as M
    rec,info=M.run_batch(text,N,seed=seed,cache_dir=CG)
    return info["total_ns"]

def avail_gb():
    for ln in open("/proc/meminfo"):
        if ln.startswith("MemAvailable"): return int(ln.split()[1])/1e6
    return 0.0

def pool_run(fn, argss):
    ctx=multiprocessing.get_context("spawn")
    with ctx.Pool(len(argss)) as pool:
        t0=time.perf_counter(); pool.map(fn,argss)
        return time.perf_counter()-t0

if __name__=="__main__":
    out=open(os.path.join(SC,"wt6_rows.tsv"),"a")
    for bench,Nc,Nm in BENCHES:
        text=open(f"{ROOT}/qec_bench/circuits/{bench}.stim").read()
        # ---- clifft, T workers x Nc each ----
        cw=[]
        for _ in range(R):
            w=pool_run(clifft_worker,[(text,Nc,CPUS[i]) for i in range(T)])
            cw.append(w/(T*Nc)*1e9)
        # ---- MDAM, memory-guarded worker count ----
        Tm=T
        if bench=="cultivation_d5":
            while Tm>2 and avail_gb() < Tm*4.0+4.0: Tm-=1
        mw=[]
        for _ in range(R):
            w=pool_run(mdam_worker,[(text,Nm,CPUS[i],40000+i*10_000_000) for i in range(Tm)])
            mw.append(w/(Tm*Nm)*1e9)
        import numpy as np
        row=(f"{bench}\t{Nc}\t{Nm}\t{T}\t{Tm}\t"
             f"{np.mean(cw):.1f}\t{min(cw):.1f}\t{max(cw):.1f}\t"
             f"{np.mean(mw):.1f}\t{min(mw):.1f}\t{max(mw):.1f}\t{np.mean(cw)/np.mean(mw):.2f}")
        out.write(row+"\n"); out.flush()
        print(f"{bench:20s} T={T}/{Tm}  clifft_eff {np.mean(cw):>12.1f} [{min(cw):.1f},{max(cw):.1f}]  "
              f"mdam_eff {np.mean(mw):>11.1f} [{min(mw):.1f},{max(mw):.1f}]  x{np.mean(cw)/np.mean(mw):.2f}",flush=True)
    print("ALL DONE 6CORE",flush=True)
