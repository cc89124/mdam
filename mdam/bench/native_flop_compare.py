"""Updated FLOP table: NATIVE one-factor MDAM dense FLOP (NOT the old Python fused Pauli-sum backend)
vs Clifft's own dense-op schedule.  SAME convention both sides (offdiag=12, diag=6, perm=0, meas=12 per 2^r).
Native split: rot (core rotation factors) | collapse (Born+project+norm) | loc (localizer Cliffords).
Clifft split: gate (offdiag+diag) | meas.  This pinpoints WHERE native exceeds Clifft when r~=k."""
import os, sys, ctypes, csv
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ.setdefault(v,"1")
import numpy as np
_NV="/home/jung/clifft-paper/mdam/native_vm"
sys.path.insert(0,_NV); sys.path.insert(0,"/home/jung/clifft-paper/mdam"); sys.path.insert(0,"/home/jung/clifft-paper")
from verify_mdam_oneshot import translate, make_prog, pcg, _ROOT, load_lib
import clifft

_CL_OFFD={"ARRAY_ROT","ARRAY_H","ARRAY_U2","ARRAY_U4","EXPAND_ROT","EXPAND_T","EXPAND_T_DAG"}
_CL_DIAG={"ARRAY_T","ARRAY_T_DAG","ARRAY_S","ARRAY_CZ","ARRAY_MULTI_CZ"}
_CL_MEAS={"MEAS_ACTIVE_DIAGONAL","MEAS_ACTIVE_INTERFERE","SWAP_MEAS_INTERFERE"}
def clifft_flop_split(cprog):
    akh=list(cprog.active_k_history); gate=0.0; meas=0.0
    for i in range(len(cprog)):
        op=str(cprog[i].opcode).replace("Opcode.OP_",""); k=akh[i] if i<len(akh) else 0
        if op in _CL_OFFD:   gate+=12.0*(1<<k)
        elif op in _CL_DIAG: gate+= 6.0*(1<<k)
        elif op in _CL_MEAS: meas+=12.0*(1<<k)
    return gate, meas

def axis_of(c):
    if 'rx' in c: return 'R_X'
    if 'ry' in c: return 'R_Y'
    if c.startswith('cultivation') or c.startswith('distill'): return 'T'
    return 'R_Z'

P=ctypes.c_void_p
lib=load_lib()
lib.nvm_mdam_sample_batch.restype=ctypes.c_int
lib.nvm_mdam_sample_batch.argtypes=[P,P,ctypes.c_uint64]+[ctypes.c_uint64]*4+[P,P,P,ctypes.c_int]
for f in ("nvm_dense_flop_rot","nvm_dense_flop_collapse","nvm_dense_flop_loc"): getattr(lib,f).restype=ctypes.c_ulonglong
lib.nvm_dense_peak_r.restype=ctypes.c_int
lib.nvm_dense_flop_reset.restype=None

CIRCUITS=["coherent_d3_r1","coherent_d3_r3","coherent_d5_r1","coherent_d5_r5","surface_d7_r7",
"coherent_rx_d3_r1","coherent_rx_d3_r3","coherent_rx_d5_r1","coherent_rx_d5_r5",
"coherent_ry_d3_r1","coherent_ry_d3_r3","coherent_ry_d5_r1","coherent_ry_d5_r5",
"cultivation_d3","cultivation_d5","distillation"]
KMAX_NATIVE=26   # native lazy-grows to 2^max_M (small for r<<k); but k>26 circuits also Clifft-infeasible
SEEDS=[1,2,3,4,5]; T=1500

def native_run(circ, k):
    text=open(os.path.join(_ROOT,f"qec_bench/circuits/{circ}.stim")).read()
    prog=clifft.compile(text); t=translate(prog); nm=t["num_meas"]
    ph=make_prog(lib,t); vm=lib.nvm_mdam_vm_create(ph)
    eb=ctypes.create_string_buffer(256); buf=np.zeros((T,nm),np.uint8)
    lib.nvm_dense_flop_reset(); tot=0; peak=0
    for s in SEEDS:
        rc=lib.nvm_mdam_sample_batch(ph,vm,T,*pcg(s*104729),buf.ctypes.data,None,eb,256); tot+=T
        if eb.value: return dict(err=eb.value.decode()[:60])
        if rc!=0: return dict(err=f"rc={rc}")
    rot=lib.nvm_dense_flop_rot()/tot; col=lib.nvm_dense_flop_collapse()/tot
    loc=lib.nvm_dense_flop_loc()/tot; peak=lib.nvm_dense_peak_r()
    return dict(rot=rot,col=col,loc=loc,peak=peak,err=None)

rows=[]
print(f"{'circuit':17}{'ax':5}{'k':>3}{'maxM':>5} | {'nat ROT':>10}{'COLLAPSE':>10}{'LOC':>9}{'TOTAL':>11} | {'cl GATE':>11}{'MEAS':>11}{'TOTAL':>11} | {'tot/Cl':>8}{'core/Cl':>8}")
INFEASIBLE={"coherent_rx_d5_r1","coherent_rx_d5_r5","coherent_ry_d5_r1","coherent_ry_d5_r5"}
for c in CIRCUITS:
    text=open(os.path.join(_ROOT,f"qec_bench/circuits/{c}.stim")).read()
    cprog=clifft.compile(text); k=cprog.peak_rank
    cg,cm=clifft_flop_split(cprog); cl=cg+cm
    if c in INFEASIBLE or k>KMAX_NATIVE:
        rows.append([c,axis_of(c),k,"",  "","","","",  f"{cg:.0f}",f"{cm:.0f}",f"{cl:.0f}",  "",""])
        print(f"{c:17}{axis_of(c):5}{k:>3}{'--':>5} | {'NATIVE/CLIFFT INFEASIBLE (2^%d)'%k:>40} | {cg:>11.0f}{cm:>11.0f}{cl:>11.0f} |"); continue
    r=native_run(c,k)
    if r.get('err'):
        rows.append([c,axis_of(c),k,"ERR","","","","",f"{cg:.0f}",f"{cm:.0f}",f"{cl:.0f}","",""])
        print(f"{c:17}{axis_of(c):5}{k:>3}{'ERR':>5} | native unsupported: {r['err']}"); continue
    nt=r['rot']+r['col']+r['loc']; core=r['rot']+r['col']
    tr=nt/cl if cl else 0.0; cr=core/cl if cl else 0.0
    rows.append([c,axis_of(c),k,r['peak'], f"{r['rot']:.0f}",f"{r['col']:.0f}",f"{r['loc']:.0f}",f"{nt:.0f}",
                 f"{cg:.0f}",f"{cm:.0f}",f"{cl:.0f}", f"{tr:.3f}",f"{cr:.3f}"])
    print(f"{c:17}{axis_of(c):5}{k:>3}{r['peak']:>5} | {r['rot']:>10.0f}{r['col']:>10.0f}{r['loc']:>9.0f}{nt:>11.0f} | {cg:>11.0f}{cm:>11.0f}{cl:>11.0f} | {tr:>8.2f}{cr:>8.2f}")

out="results/benchmark_comparison/flop_table_native.csv"
with open(out,"w",newline="") as fh:
    w=csv.writer(fh)
    w.writerow(["circuit","axis","k_clifft","max_M_native","native_rot_FLOP","native_collapse_FLOP","native_localizer_FLOP",
                "native_total_FLOP","clifft_gate_FLOP","clifft_meas_FLOP","clifft_total_FLOP","native_total_over_clifft","native_core_over_clifft"])
    for row in rows: w.writerow(row)
print(f"\n-> {out}  (native one-factor; SEEDS={SEEDS} x {T} shots; supersedes the Python-fused flop_table.csv)")
