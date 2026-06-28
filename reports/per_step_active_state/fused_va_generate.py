import os,sys
os.chdir("/home/jung/clifft-paper"); sys.path.insert(0,"/home/jung/clifft-paper")
sys.setrecursionlimit(400000)
import clifft, nearclifford_backend.backend as bk
from nearclifford_backend.backend import count_idents
from nearclifford_backend.virtual_axis.fused_single_frame import FusedSingleFrame
from nearclifford_backend.virtual_axis.bench_memory import clifft_k

CIRCS=['coherent_d3_r1','coherent_d3_r3','coherent_d5_r1','coherent_d5_r5',
       'cultivation_d3','cultivation_d5','distillation','surface_d7_r7']
OUT='reports/per_step_active_state'

def extract(circ, seed=1):
    prog=clifft.compile(open(f'qec_bench/circuits/{circ}.stim').read()); n=count_idents(prog)
    orig=bk.LazyNearClifford; bk.LazyNearClifford=FusedSingleFrame
    rows=[]; cur={'s':-1}
    omz=FusedSingleFrame.measure_z
    def mz(self,q):
        out=omz(self,q)
        w=self.core_log[-1][1] if self.core_log else len(self.magic)
        cur['W']=max(w,len(self.magic))   # transient at THIS measurement step
        return out
    FusedSingleFrame.measure_z=mz
    def rec(step,be):
        nc=be.nc
        na=len(be.slot2id)
        res=len(nc.magic)
        tr=cur.pop('W',res)               # W set if a measurement happened since last rec
        rows.append((step,na,res,tr))
    try:
        be=bk.NearCliffordBackend(lazy=True,drop_dead=False,structure_once=False)
        be.run_shot(prog,seed,step_recorder=rec)
        finalmag=len(be.nc.magic); peak_ws=be.nc.max_fused_ws; peak_res=be.nc.max_M
    finally:
        bk.LazyNearClifford=orig; FusedSingleFrame.measure_z=omz
    return n,rows,peak_ws,peak_res

print(f"{'circuit':16}{'clifft_k':>9}{'fused_transient':>16}{'fused_resident':>15}{'steps':>7}")
summ=[]
for c in CIRCS:
    try:
        k=clifft_k(c)
    except Exception:
        k='?'
    n,rows,pw,pr=extract(c)
    # write per-step CSV
    with open(f'{OUT}/fused_va_{c}_per_step.csv','w') as f:
        f.write('step,n_active,fused_resident_qubits,fused_transient_qubits,fused_resident_dim,fused_transient_dim\n')
        for (s,na,res,tr) in rows:
            f.write(f'{s},{na},{res},{tr},{1<<res},{1<<tr}\n')
    summ.append((c,k,pw,pr,len(rows)))
    print(f"{c:16}{str(k):>9}{pw:>16}{pr:>15}{len(rows):>7}")

# summary markdown
with open(f'{OUT}/FUSED_VA_SUMMARY.md','w') as f:
    f.write("# Per-step ACTIVE-STATE: fused virtual-axis LIVE backend\n\n")
    f.write("Peak active-state size of the dense-free single-frame fused backend "
            "(`FusedSingleFrame`). **transient** = peak fused workspace during a core "
            "contraction (= `fused_ws`); **resident** = settled magic rank between "
            "measurements. Per-step traces: `fused_va_<circuit>_per_step.csv`.\n\n")
    f.write("| circuit | Clifft k | fused transient (qubits) | fused resident (qubits) | fused transient dim | saving 2^(k-transient) |\n")
    f.write("|---|--:|--:|--:|--:|--:|\n")
    for (c,k,pw,pr,ns) in summ:
        sav=f"2^{k-pw}" if isinstance(k,int) else "?"
        f.write(f"| {c} | {k} | {pw} | {pr} | 2^{pw} | {sav} |\n")
print("WROTE", OUT+"/FUSED_VA_SUMMARY.md and per-step CSVs")
