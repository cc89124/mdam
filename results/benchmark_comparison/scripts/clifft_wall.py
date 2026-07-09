"""clifft baseline wall for ONE bench -> appends a row to clifft_rows.tsv.
Protocol (same as the wall_table clifft_ns column): default compile settings
(squeeze on), clifft.sample record sampling, cold-amortized total_wall/N on a
single pinned core.  N grows in chunks until target_s of wall (cap 10M shots),
so light benches amortize well and heavy benches still finish.
argv: bench [target_s=20]"""
import os, sys, time
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ.setdefault(v,"1")
SC=os.path.dirname(os.path.abspath(__file__))
ROOT=os.path.abspath(os.path.join(SC,"..","..",".."))
bench=sys.argv[1]; target_s=float(sys.argv[2]) if len(sys.argv)>2 else 20.0
import clifft
prog=clifft.compile(open(f"{ROOT}/qec_bench/circuits/{bench}.stim").read())
# probe -> chunk size for ~1s per call (memory-bounded); then loop to the budget
t0=time.perf_counter(); clifft.sample(prog,8); probe=(time.perf_counter()-t0)/8
chunk=max(8,min(200_000,int(1.0/max(probe,1e-12))))
N=0; wall=0.0
while wall<target_s and N<10_000_000:
    t0=time.perf_counter(); clifft.sample(prog,chunk); wall+=time.perf_counter()-t0; N+=chunk
ns=wall/N*1e9
with open(f"{SC}/clifft_rows.tsv","a") as f: f.write(f"{bench}\t{ns:.1f}\t{N}\t{clifft.version()}\n")
print(f"{bench:18s} clifft={ns:>13.1f} ns/shot  N={N:>9,}  ({clifft.version()}, {clifft.svm_backend()})", flush=True)
