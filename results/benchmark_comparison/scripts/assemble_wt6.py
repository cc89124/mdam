"""Assemble wall_table_6core.csv from wt6_rows.tsv.
Same column layout as wall_table.csv (1-core v5); effective ns/shot = pool wall / (T*N).
k/maxM/nmeas/regime + compile_s carried from the 1-core table (compile is a one-time cost,
independent of worker count); compile per-shot = compile_s / (T*N).  scaling_eff_* =
1-core mean ns / 6-core effective mean ns / T (1.0 = ideal linear)."""
import csv, os
SC=os.path.dirname(os.path.abspath(__file__))
BASE="/home/jung/clifft-paper/results/benchmark_comparison"
one={r["bench"]:r for r in csv.DictReader(open(f"{BASE}/wall_table.csv"))}
cols=("bench,k,maxM,nmeas,regime,threads_clifft,threads_mdam,shots_clifft_per_worker,shots_mdam_per_worker,"
      "clifft_compile_s,clifft_ns_mean,clifft_ns_min,clifft_ns_max,clifft_compile_ns_per_shot,"
      "mdam_compile_s,mdam_ns_mean,mdam_ns_min,mdam_ns_max,mdam_compile_ns_per_shot,"
      "mdam_vs_clifft,scaling_eff_clifft,scaling_eff_mdam").split(",")
rows=[]
for ln in open(f"{SC}/wt6_rows.tsv"):
    f=ln.rstrip("\n").split("\t")
    bench,Nc,Nm,T,Tm=f[0],int(f[1]),int(f[2]),int(f[3]),int(f[4])
    cwm,cwn,cwx,mwm,mwn,mwx,ratio=map(float,f[5:12])
    o=one[bench]
    cc=float(o["clifft_compile_s"]); mc=float(o["mdam_compile_s"])
    se_c=float(o["clifft_ns_mean"])/cwm/T; se_m=float(o["mdam_ns_mean"])/mwm/Tm
    rows.append(dict(zip(cols,[bench,o["k"],o["maxM"],o["nmeas"],o["regime"],T,Tm,Nc,Nm,
        f"{cc:.4f}",f"{cwm:.1f}",f"{cwn:.1f}",f"{cwx:.1f}",f"{cc/(T*Nc)*1e9:.2f}",
        f"{mc:.4f}",f"{mwm:.1f}",f"{mwn:.1f}",f"{mwx:.1f}",f"{mc/(Tm*Nm)*1e9:.2f}",
        f"{ratio:.2f}x",f"{se_c:.2f}",f"{se_m:.2f}"])))
with open(f"{BASE}/wall_table_6core.csv","w",newline="") as fo:
    w=csv.DictWriter(fo,fieldnames=cols); w.writeheader()
    for r in rows: w.writerow(r)
print("wrote wall_table_6core.csv")
for r in rows:
    print(f"{r['bench']:20s} clifft {float(r['clifft_ns_mean']):>13.1f} (eff {r['scaling_eff_clifft']})  "
          f"mdam {float(r['mdam_ns_mean']):>12.1f} (eff {r['scaling_eff_mdam']})  {r['mdam_vs_clifft']:>10s}")
