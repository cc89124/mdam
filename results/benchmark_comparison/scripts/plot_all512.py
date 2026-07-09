"""Final per-bench 2-panel figure, uniform N, w=512, ALL MEASURED:
  (top)    per-window fallback rate + break-even line (AUTH/LEAN words only)
  (bottom) time per 512-shot window: LEAN-only / AUTH-only / adaptive
           (adaptive: dashed at AUTH level after a demote; LEAN curve may be
            truncated when LEAN-forced N=100k is memory-infeasible, e.g. d5_r5)
argv: bench [bench...]  (default: every all512_*.npz)"""
import os, sys, glob
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
SC=os.path.dirname(os.path.abspath(__file__))
OUT=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","adaptive_algorithm_results")
BLUE="#1f77b4"; RED="#c44e52"; GREEN="#2ca02c"

def break_even(fb,wl,T_auth):
    """fb_be from measured windows.  Per-window wall = T_hit + fb*(T_miss-T_hit),
    so an OLS line over the (fb, wall) points measures T_hit (intercept) and
    T_miss (intercept+slope) whenever fb actually varies."""
    if fb.max()==0: return float("inf")           # no miss ever -> no criterion to draw
    if fb.max()-fb.min()>=0.03:
        A=np.vstack([np.ones_like(fb),fb]).T
        (T_hit,slope),*_=np.linalg.lstsq(A,wl,rcond=None)
        if slope>0: return (T_auth-T_hit)/slope
    # fb ~ constant: hit cost unmeasurable -> generous upper bound (T_hit=0)
    m=fb>0.5
    return float(T_auth/np.median(wl[m]/fb[m])) if m.any() else float("inf")

benches=sys.argv[1:] or [os.path.basename(p)[7:-4] for p in sorted(glob.glob(f"{SC}/all512_*.npz"))]
for bench in benches:
    z=np.load(f"{SC}/all512_{bench}.npz")
    fb=z["fb"]; wl=z["wall_lean"]; wa=z["wall_auth"]; N=int(z["N"])
    fb_be=break_even(fb,wl,float(np.mean(wa)))
    an=z["an"]; awin=z["awin"]; demote=int(z["demote"]); a_cal=float(z["a_cal"])
    NW=len(wa); w=np.arange(1,NW+1); wlean=np.arange(1,len(wl)+1)

    fig,(p1,p2)=plt.subplots(2,1,figsize=(9.4,7.8),sharex=True)
    fig.subplots_adjust(hspace=0.12,left=0.09,right=0.97,top=0.93,bottom=0.09)
    fig.suptitle(f"{bench}   (N = {N:,})",fontsize=12,fontweight="bold")

    # ---- (top) fallback rate ----
    show_line=np.isfinite(fb_be) and fb_be<=1.02
    ytop=max(fb.max()*1.10,(fb_be*1.18 if show_line else 0),0.02)
    p1.plot(wlean,fb,"-",lw=1.2,color=BLUE,zorder=5)
    if show_line:
        p1.axhline(fb_be,color=RED,lw=1.6)
        p1.text(NW,fb_be+ytop*0.02,"AUTH ",color=RED,fontsize=11,ha="right",va="bottom")
        p1.text(NW,fb_be-ytop*0.02,"LEAN ",color=RED,fontsize=11,ha="right",va="top")
    p1.set_ylim(-ytop*0.04,ytop); p1.set_ylabel("fallback rate",fontsize=10.5); p1.grid(alpha=0.2)

    # ---- (bottom) time per 512-shot window, all measured ----
    ms_l=wl*512/1e6; ms_a=wa*512/1e6
    p2.plot(wlean,ms_l,"-",lw=1.3,color=BLUE,label="LEAN only",zorder=4)
    p2.plot(w,ms_a,"-",lw=1.3,color=RED,label="AUTH only",zorder=3)
    gx=np.concatenate([[1.0],an/512.0]); gy=np.concatenate([[a_cal*512/1e6],awin*512/1e6])
    p2.plot(gx,gy,"-",lw=1.3,color=GREEN,alpha=0.9,label="adaptive",zorder=5)
    if demote>0:
        for ax in (p1,p2): ax.axvline(demote/512.0,color=GREEN,ls=":",lw=1.4,zorder=6)
    ymax=max(np.percentile(ms_l,99),ms_a.max(),np.percentile(gy,99))
    p2.set_xlim(0,NW*1.02); p2.set_ylim(0,ymax*1.12)
    p2.set_ylabel("time per window  (ms)",fontsize=10.5)
    p2.set_xlabel("window index  (512 shots per window)",fontsize=11)
    p2.grid(alpha=0.2); p2.legend(fontsize=9.5,loc="center right",framealpha=0.95)
    plt.savefig(f"{OUT}/fb_{bench}.png",dpi=130); plt.close(fig)
    print(f"[written] fb_{bench}.png  fb_be={fb_be:.3f} demote={demote} "
          f"lean_w1={ms_l[0]:.2f}ms late={np.median(ms_l[-min(50,len(ms_l)):]):.2f}ms auth_med={np.median(ms_a):.2f}ms")
