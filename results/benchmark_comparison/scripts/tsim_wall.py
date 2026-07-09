"""tsim wall measurement for ONE bench, same conditions as the clifft_ns column:
taskset single core (by caller), single-thread XLA/BLAS, sampling wall / N.
Matches clifft_ns semantics: program build (compile_sampler) and JAX JIT are
excluded from the timed wall (clifft_ns also excludes clifft.compile); both are
disclosed in the output row.  Batched sampling with a fixed warmed batch shape
(JAX re-JITs per shape, so N is a multiple of B and B is warmed first).
argv: bench [target_s]   -> appends a row to tsim_rows.tsv"""
import os, sys, time
os.environ["XLA_FLAGS"]="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
os.environ["JAX_PLATFORMS"]="cpu"
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ.setdefault(v,"1")
import numpy as np
SC=os.path.dirname(os.path.abspath(__file__))
bench=sys.argv[1]; target_s=float(sys.argv[2]) if len(sys.argv)>2 else 20.0
text=open(f"/home/jung/clifft-paper/qec_bench/circuits/{bench}.stim").read()

import tsim
t0=time.perf_counter(); samp=tsim.Circuit(text).compile_sampler(); build_s=time.perf_counter()-t0

# JIT warm + probe, escalating the batch size (JAX re-JITs per shape; heavy benches
# stay at a small batch so the probe itself cannot blow the budget)
B=16
t0=time.perf_counter(); samp.sample(shots=B); jit_s=time.perf_counter()-t0
t0=time.perf_counter(); samp.sample(shots=B); probe=(time.perf_counter()-t0)/B
for nb in (256,8192):
    if probe*nb>10.0: break
    B=nb
    t0=time.perf_counter(); samp.sample(shots=B); jit_s+=time.perf_counter()-t0
    t0=time.perf_counter(); samp.sample(shots=B); probe=(time.perf_counter()-t0)/B

calls=max(1,min(int(target_s/max(probe*B,1e-9)), 10_000_000//B))
t0=time.perf_counter()
for _ in range(calls): samp.sample(shots=B)
wall=time.perf_counter()-t0
N=calls*B; ns=wall/N*1e9
with open(f"{SC}/tsim_rows.tsv","a") as f:
    f.write(f"{bench}\t{ns:.1f}\t{N}\t{B}\t{build_s:.2f}\t{jit_s:.2f}\n")
print(f"{bench:18s} tsim={ns:>12.1f} ns/shot  N={N:>8d} B={B} build={build_s:.2f}s jit={jit_s:.2f}s", flush=True)
