import os,sys
os.chdir("/home/jung/clifft-paper"); sys.path.insert(0,"/home/jung/clifft-paper")
sys.setrecursionlimit(400000)
import numpy as np
import clifft, nearclifford_backend.backend as bk
import nearclifford_backend.block_magic as bm
import nearclifford_backend.virtual_axis.fused_integrate as fi
import nearclifford_backend.virtual_axis.virtual_engine as ve
import nearclifford_backend.virtual_axis.flop_meter as fm
from nearclifford_backend.virtual_axis.fused_single_frame import FusedSingleFrame, fused_ws_single
from nearclifford_backend.virtual_axis.bench_memory import clifft_k

# ---- (0) regression: source instrumentation must NOT change fused_ws ----
ref={'coherent_d3_r1':1,'distillation':3,'cultivation_d3':3,'cultivation_d5':9,'coherent_d3_r3':4}
print("[0] regression (instrumented code, meter OFF) — fused_ws must be unchanged:")
ok=True
for c,exp in ref.items():
    got=fused_ws_single(c)
    flag='OK' if got==exp else f'CHANGED(exp {exp})'
    if got!=exp: ok=False
    print(f"    {c:16} fused_ws={got}  {flag}")
print("    REGRESSION", "PASS" if ok else "FAIL")
if not ok: sys.exit("instrumentation changed behaviour — abort")

# ---- primitive wrappers -> same flop_meter buckets ----
_apl=bm._apply_pauli_local; _vh=bm._vec_h; _vs=bm._vec_s
_kron=np.kron; _vdot=np.vdot; _norm=np.linalg.norm
def apl(qubits,vec,xm,zm,ph): fm.ak(vec.size,6.0); return _apl(qubits,vec,xm,zm,ph)
def vh(vec,j): fm.el(vec.size,4.0); return _vh(vec,j)
def vs(vec,j,dag): fm.el(vec.size,1.0); return _vs(vec,j,dag)
def kron(a,b): out=_kron(a,b); fm.ak(out.size,6.0); return out
def vdot(a,b): fm.vn(np.asarray(a).size,8.0); return _vdot(a,b)
def norm(*a,**k): fm.vn(np.asarray(a[0]).size,4.0); return _norm(*a,**k)

def patch_on():
    fi._apply_pauli_local=apl; ve._apply_pauli_local=apl
    fi._vec_h=vh; fi._vec_s=vs; bm._vec_h=vh; bm._vec_s=vs
    np.kron=kron; np.vdot=vdot; np.linalg.norm=norm
def patch_off():
    fi._apply_pauli_local=_apl; ve._apply_pauli_local=_apl
    fi._vec_h=_vh; fi._vec_s=_vs; bm._vec_h=_vh; bm._vec_s=_vs
    np.kron=_kron; np.vdot=_vdot; np.linalg.norm=_norm

def measure(circ,seed=1):
    fm.reset(); fm.enable(); patch_on()
    prog=clifft.compile(open(f'qec_bench/circuits/{circ}.stim').read())
    orig=bk.LazyNearClifford; bk.LazyNearClifford=FusedSingleFrame
    try:
        be=bk.NearCliffordBackend(lazy=True,drop_dead=False,structure_once=False)
        be.run_shot(prog,seed); ws=be.nc.max_fused_ws
    finally:
        bk.LazyNearClifford=orig; patch_off(); fm.disable()
    return fm.snapshot(), ws

def hf(x):
    for u,d in [('T',1e12),('G',1e9),('M',1e6),('K',1e3)]:
        if x>=d: return f"{x/d:.1f}{u}"
    return f"{x:.0f}"

CLIFFT_TOTAL={'coherent_d3_r1':20.8e3,'coherent_d3_r3':564.3e3,'coherent_d5_r1':12.8e6,
              'coherent_d5_r5':209.3e9,'distillation':25.1e3,'cultivation_d3':26.4e3,
              'cultivation_d5':3.5e6,'surface_d7_r7':4.7e3}
CIRCS=['coherent_d3_r1','coherent_d3_r3','coherent_d5_r1','coherent_d5_r5',
       'cultivation_d3','cultivation_d5','distillation','surface_d7_r7']
print("\n[FULL FLOP]")
hdr=f"{'circuit':16}{'k':>4}{'ws':>4}{'apply/kron':>11}{'vdot/norm':>11}{'elementwise':>12}{'FLOOR':>9}{'FULL':>9}{'clifft':>9}{'redux':>8}"
print(hdr)
rows=[]
for c in CIRCS:
    try: k=clifft_k(c)
    except Exception: k='?'
    snap,ws=measure(c)
    ak=snap['apply_kron']; vn=snap['vdot_norm']; el=snap['elementwise']
    floor=ak+vn; full=ak+vn+el; ct=CLIFFT_TOTAL.get(c,0)
    rdx=(ct/full) if full>0 else float('inf')
    rows.append((c,k,ws,ak,vn,el,floor,full,ct,rdx))
    rdxs='inf' if full==0 else f'{rdx:.0f}x'
    print(f"{c:16}{str(k):>4}{ws:>4}{hf(ak):>11}{hf(vn):>11}{hf(el):>12}{hf(floor):>9}{hf(full):>9}{hf(ct):>9}{rdxs:>8}")

import json
with open('reports/per_step_flops/fused_va_full_flops.csv','w') as f:
    f.write("circuit,clifft_k,fused_ws,apply_or_kron,vdot_norm,elementwise,floor,full,clifft_total,reduction_vs_clifft\n")
    for (c,k,ws,ak,vn,el,fl,fu,ct,rd) in rows:
        f.write(f"{c},{k},{ws},{ak:.0f},{vn:.0f},{el:.0f},{fl:.0f},{fu:.0f},{ct:.0f},{rd:.2f}\n")
print("\nWROTE reports/per_step_flops/fused_va_full_flops.csv")
