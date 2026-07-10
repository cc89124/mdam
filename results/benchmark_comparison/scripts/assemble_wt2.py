"""Assemble wall_table.csv (v5 format) from wt2_rows.tsv.
- k/maxM/nmeas/regime carried from the previous csv (circuit properties).
- compile correction: benches that never entered the codegen stage (route LEAN(interp)
  AND final fb > FB_MAX=2%) never load the prewarmed .so -> mdam_compile_s = front-end only
  (cultivation_d5 0.0702s, coherent_rx_d3_r3 0.0035s, measured).
- per-shot compile = compile_s / shots * 1e9 (ns/shot)."""
import csv, os
SC=os.path.dirname(os.path.abspath(__file__))
CSV="/home/jung/clifft-paper/results/benchmark_comparison/wall_table.csv"
old={r["bench"]:r for r in csv.DictReader(open(CSV))}
FE_FIX={"cultivation_d5":0.0702,"coherent_rx_d3_r3":0.0035}
cols=("bench,k,maxM,nmeas,regime,threads,shots_clifft,shots_mdam,"
      "clifft_compile_s,clifft_ns_mean,clifft_ns_min,clifft_ns_max,clifft_compile_ns_per_shot,"
      "mdam_compile_s,mdam_ns_mean,mdam_ns_min,mdam_ns_max,mdam_compile_ns_per_shot,"
      "mdam_vs_clifft,route,canon,canon_sid_merge_pct,final_fb_pct,retry_waste_s,bitexact").split(",")
rows=[]
for ln in open(f"{SC}/wt2_rows.tsv"):
    f=ln.rstrip("\n").split("\t")
    (bench,Nc,Nm,c_comp,cm_,cmin,cmax,m_comp,mm_,mmin,mmax,ratio,route,canon,cmerge,fb,waste,be)=f
    Nc,Nm=int(Nc),int(Nm)
    m_comp=float(m_comp)
    if bench in FE_FIX and route=="LEAN(interp)" and float(fb)>2.0: m_comp=FE_FIX[bench]
    o=old.get(bench,{})
    rows.append(dict(zip(cols,[bench,o.get("k",""),o.get("maxM",""),o.get("nmeas",""),o.get("regime",""),
        1,Nc,Nm,
        f"{float(c_comp):.4f}",cm_,cmin,cmax,f"{float(c_comp)/Nc*1e9:.2f}",
        f"{m_comp:.4f}",mm_,mmin,mmax,f"{m_comp/Nm*1e9:.2f}",
        f"{float(ratio):.2f}x",route,canon,
        f"{float(cmerge)*100:.1f}" if float(cmerge)>=0 else "-",fb,waste,be])))
with open(CSV,"w",newline="") as fo:
    w=csv.DictWriter(fo,fieldnames=cols); w.writeheader()
    for r in rows: w.writerow(r)
print("wrote",CSV)
for r in rows:
    print(f"{r['bench']:20s} clifft {float(r['clifft_ns_mean']):>13.1f}  mdam {float(r['mdam_ns_mean']):>12.1f}  "
          f"{r['mdam_vs_clifft']:>9s}  comp c/m {r['clifft_compile_s']}/{r['mdam_compile_s']}s  {r['route']}")
