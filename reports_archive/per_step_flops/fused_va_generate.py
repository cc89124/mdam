import os,sys
os.chdir("/home/jung/clifft-paper"); sys.path.insert(0,"/home/jung/clifft-paper")
sys.setrecursionlimit(400000)
import numpy as np
import clifft, nearclifford_backend.backend as bk
from nearclifford_backend.backend import count_idents
import nearclifford_backend.block_magic as bm
import nearclifford_backend.virtual_axis.fused_integrate as fi
import nearclifford_backend.virtual_axis.virtual_engine as ve
from nearclifford_backend.virtual_axis.fused_single_frame import FusedSingleFrame
from nearclifford_backend.virtual_axis.bench_memory import clifft_k

CIRCS=['coherent_d3_r1','coherent_d3_r3','coherent_d5_r1','coherent_d5_r5',
       'cultivation_d3','cultivation_d5','distillation','surface_d7_r7']
OUT='reports/per_step_flops'
C={'mm':0.0,'norm':0.0}

# ---- FLOP instrumentation (same convention as block_magic flop_mm/flop_norm) ----
_apl=bm._apply_pauli_local
def apl(qubits,vec,xm,zm,ph):
    C['mm']+=6.0*vec.size                 # Pauli apply (phase mult per amplitude)
    return _apl(qubits,vec,xm,zm,ph)
_vdot=np.vdot
def vdot(a,b):
    C['norm']+=8.0*np.asarray(a).size; return _vdot(a,b)
_norm=np.linalg.norm
def norm(*a,**k):
    C['norm']+=4.0*np.asarray(a[0]).size; return _norm(*a,**k)
_kron=np.kron
def kron(a,b):
    out=_kron(a,b); C['mm']+=6.0*out.size; return out

def measure_flop(circ, seed=1):
    C['mm']=0.0; C['norm']=0.0
    prog=clifft.compile(open(f'qec_bench/circuits/{circ}.stim').read())
    orig=bk.LazyNearClifford; bk.LazyNearClifford=FusedSingleFrame
    bm._apply_pauli_local=apl; fi._apply_pauli_local=apl; ve._apply_pauli_local=apl
    o_vdot,o_norm,o_kron=np.vdot,np.linalg.norm,np.kron
    np.vdot=vdot; np.linalg.norm=norm; np.kron=kron
    fi.np.vdot=vdot; fi.np.linalg.norm=norm; fi.np.kron=kron; ve.np.linalg.norm=norm
    try:
        be=bk.NearCliffordBackend(lazy=True,drop_dead=False,structure_once=False)
        be.run_shot(prog,seed)
        ws=be.nc.max_fused_ws
    finally:
        bk.LazyNearClifford=orig
        bm._apply_pauli_local=_apl; fi._apply_pauli_local=_apl; ve._apply_pauli_local=_apl
        np.vdot,np.linalg.norm,np.kron=o_vdot,o_norm,o_kron
        fi.np.vdot=o_vdot; fi.np.linalg.norm=o_norm; fi.np.kron=o_kron; ve.np.linalg.norm=o_norm
    return C['mm'],C['norm'],ws

# clifft analytic model TOTAL (16*2^k dense, matmul-only proxy) -- reuse the repo convention:
# clifft pays a dense 2^k matmul per non-Clifford op; here we report the published FLOPS_TABLE
# clifft TOTAL for context, and compute fused as MEASURED.
def hf(x):
    for u,d in [('T',1e12),('G',1e9),('M',1e6),('K',1e3)]:
        if x>=d: return f"{x/d:.1f}{u}"
    return f"{x:.0f}"

print(f"{'circuit':16}{'clifft_k':>9}{'fused_ws':>9}{'fused_mm':>11}{'fused_norm':>12}{'fused_TOTAL':>12}")
rows=[]
for c in CIRCS:
    try: k=clifft_k(c)
    except Exception: k='?'
    mm,nr,ws=measure_flop(c)
    rows.append((c,k,ws,mm,nr,mm+nr))
    print(f"{c:16}{str(k):>9}{ws:>9}{hf(mm):>11}{hf(nr):>12}{hf(mm+nr):>12}")

with open(f'{OUT}/FUSED_VA_FLOPS.md','w') as f:
    f.write("# Compute (FLOP) of the fused virtual-axis LIVE backend\n\n")
    f.write("MEASURED FLOP of the dense-free single-frame fused backend (`FusedSingleFrame`), "
            "summed over the whole run. Same complex-arith convention as `FLOPS_TABLE.md` / "
            "`block_magic` counters: **Pauli-apply = 6·size, kron = 6·size, vdot = 8·size, "
            "norm = 4·size**.\n\n")
    f.write("* **fused matmul** = state-evolution (Pauli applies in the core contractions + kron).\n")
    f.write("* **fused norm** = the projection/Born scans (vdot + norm).\n")
    f.write("* **fused TOTAL** = matmul + norm (the FLOP floor; excludes the shared polynomial "
            "Clifford bit-op work, identical to the block NC accounting).\n")
    f.write("* The fused contraction never materialises clifft's 2^k vector, so its FLOP scales "
            "with 2^fused_ws, not 2^k -- e.g. coherent_d5_r5 works on 2^12, not 2^24.\n\n")
    f.write("| circuit | clifft_k | fused_ws | fused matmul | fused norm | fused TOTAL |\n")
    f.write("|---|--:|--:|--:|--:|--:|\n")
    for (c,k,ws,mm,nr,tot) in rows:
        f.write(f"| {c} | {k} | {ws} | {hf(mm)} | {hf(nr)} | {hf(tot)} |\n")
    # raw csv inline
print("WROTE",OUT+"/FUSED_VA_FLOPS.md")
with open(f'{OUT}/fused_va_flops.csv','w') as f:
    f.write("circuit,clifft_k,fused_ws,fused_matmul_flop,fused_norm_flop,fused_total_flop\n")
    for (c,k,ws,mm,nr,tot) in rows:
        f.write(f"{c},{k},{ws},{mm:.0f},{nr:.0f},{tot:.0f}\n")
