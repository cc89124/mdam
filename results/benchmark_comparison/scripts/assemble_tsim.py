"""Merge tsim_rows.tsv into wall_table.csv: adds tsim_ns, tsim_vs_clifft, mdam_vs_tsim.
Benches with no tsim row (crash/timeout) get 'infeasible'."""
import csv, os
SC=os.path.dirname(os.path.abspath(__file__))
CSV="/home/jung/clifft-paper/results/benchmark_comparison/wall_table.csv"
tsim={}
for ln in open(f"{SC}/tsim_rows.tsv"):
    f=ln.split("\t")
    if len(f)>=2: tsim[f[0]]=float(f[1])
rows=list(csv.DictReader(open(CSV)))
out=[]
for r in rows:
    b=r["bench"]; cl=float(r["clifft_ns"]); md=float(r["mdam_ns"])
    if b in tsim:
        ts=tsim[b]
        r["tsim_ns"]=f"{ts:.1f}"; r["tsim_vs_clifft"]=f"{cl/ts:.2f}x"; r["mdam_vs_tsim"]=f"{ts/md:.2f}x"
    else:
        r["tsim_ns"]="infeasible"; r["tsim_vs_clifft"]="-"; r["mdam_vs_tsim"]="-"
    out.append(r)
# keep column order: insert the tsim columns right after mdam_vs_clifft
base=list(rows[0].keys())[:list(rows[0].keys()).index("route")]
rest=list(rows[0].keys())[list(rows[0].keys()).index("route"):]
cols=[c for c in base if not c.startswith("tsim") and c!="mdam_vs_tsim"]+ \
     ["tsim_ns","tsim_vs_clifft","mdam_vs_tsim"]+[c for c in rest if not c.startswith("tsim") and c!="mdam_vs_tsim"]
with open(CSV,"w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=cols); w.writeheader()
    for r in out: w.writerow({c:r.get(c,"") for c in cols})
print("updated",CSV)
for r in out: print(f"{r['bench']:20s} clifft={r['clifft_ns']:>14s} tsim={r['tsim_ns']:>14s} mdam={r['mdam_ns']:>12s} mdam_vs_tsim={r['mdam_vs_tsim']}")
