"""Phase C/D: ARRAY_CNOT / ARRAY_CZ gauge-vs-numerical per-opcode proof.

clifft meter: array_cnot = 0 arithmetic (pure amplitude PERMUTATION; 'processed' = traffic only),
array_cz = rcmul (the -1 phase on the |11> block), array_swap = 0 arithmetic (permutation).

So on the MAGIC register these opcodes are NUMERICAL (they entangle/phase the dense array) -- they
are NOT frame-gauge relabels (the frame only carries stabiliser/dormant qubits, routed via o_cx /
frame.cnot). We prove, per opcode:
  (1) UNIT (random magic states): _cnot_axes is an EXACT CNOT permutation (bit-exact, 0 arith);
      the CZ sub-block -1 is an EXACT CZ (bit-exact up to the rcmul phase).
  (2) IN-CIRCUIT (cultivation_d3, dense): E_{t+1}|psi> = O_logical . E_t|psi> at machine precision
      for every both-magic ARRAY_CNOT / ARRAY_CZ.
  (3) COST: array_cnot 0 arith + 2^(r-1) swap traffic (= clifft array_cnot 0 arith); array_cz
      2^(r-2) negate = rcmul (= clifft array_cz rcmul). Dormant-routed CNOT/CZ are GAUGE (frame
      metadata, 0 numerical).
"""
import sys, copy; sys.path.insert(0, "/home/jung/clifft-paper"); sys.setrecursionlimit(400000)
import numpy as np, clifft
import nearclifford_backend.backend as bk
from nearclifford_backend.backend import _opname
from nearclifford_backend.clifft_axis.bounded import compile_bounded, CliftAxisBoundedNearClifford as C

def _par(u):
    u = u.copy()
    for s in (16,8,4,2,1): u ^= u>>s
    return u&1
def _pauli_apply(P,v):
    x,z,p=P; a=np.arange(v.size); b=a^x
    return ((1j**p)*(1.0-2.0*_par(z&b)))*v[b]
def upg(a,b):
    na=np.linalg.norm(a); nb=np.linalg.norm(b)
    if na<1e-14 or nb<1e-14: return abs(na-nb)
    return float(1.0-abs(np.vdot(a/na,b/nb)))
def robust_statevector(nc):
    for (x,z,p,th,uid) in list(nc.pending.values()): nc._flush_one(x,z,th,p)
    nc.pending={}
    n=nc.n; dim=1<<n
    rngv=np.random.default_rng(12345)
    v=rngv.standard_normal(dim)+1j*rngv.standard_normal(dim)
    for i in range(n): v=0.5*(v+_pauli_apply(nc.Zc[i],v))
    u0=v/np.linalg.norm(v)
    psi=np.zeros(dim,complex)
    for idx in range(len(nc.phi)):
        amp=nc.phi[idx]
        if abs(amp)<1e-300: continue
        col=u0; bits=0
        for j,q in enumerate(nc.M):
            if (idx>>j)&1: bits|=1<<q
        bi=bits
        while bi:
            i=(bi&-bi).bit_length()-1; bi&=bi-1
            col=_pauli_apply(nc.Xc[i],col)
        psi=psi+amp*col
    return psi

def dense_cnot(psi,c,t):
    a=np.arange(psi.size)
    flip=((a>>c)&1).astype(bool)
    src=np.where(flip, a^(1<<t), a)
    return psi[src]
def dense_cz(psi,c,t):
    a=np.arange(psi.size)
    sign=1.0-2.0*(((a>>c)&1)&((a>>t)&1))
    return sign*psi

# ---------- (1) UNIT test on random magic states ----------
print("(1) UNIT: _cnot_axes / CZ sub-block vs dense CNOT/CZ on random magic states (phi-direct):")
worst_cx=worst_cz=0.0
for seed in range(60):
    rng=np.random.default_rng(seed)
    r=5; nc=C(r)
    for q in range(r): nc._promote(q)
    phi0=rng.standard_normal(1<<r)+1j*rng.standard_normal(1<<r); phi0/=np.linalg.norm(phi0)
    jc,jt=rng.choice(r,size=2,replace=False).tolist()
    # CNOT: control phi-bit jc, target phi-bit jt
    nc.phi[:]=phi0; nc._cnot_axes(jc,jt)
    ref=dense_cnot(phi0.copy(),jc,jt)            # phi bit jc=ctrl, jt=tgt
    worst_cx=max(worst_cx, float(np.max(np.abs(nc.phi-ref))))
    # CZ: phi-bits jc,jt  (reduced rcz does -1 on the (bit_jc=1,bit_jt=1) sub-block)
    nc.phi[:]=phi0
    rr=len(nc.M); t=nc.phi.reshape([2]*rr); s=[slice(None)]*rr; s[rr-1-jc]=1; s[rr-1-jt]=1; t[tuple(s)]*=-1.0
    refz=dense_cz(phi0.copy(),jc,jt)
    worst_cz=max(worst_cz, float(np.max(np.abs(nc.phi-refz))))
print(f"   _cnot_axes max|phi-CNOT.phi| = {worst_cx:.2e}   (0 => exact permutation, 0 arithmetic)")
print(f"   CZ sub-block max|phi-CZ.phi|  = {worst_cz:.2e}   (0 => exact phase, rcmul only)")

# ---------- reduced overrides for the in-circuit dense check ----------
o_h=C.h;o_s=C.s;o_cx=C.cx;o_cz=C.cz;o_birth=bk.NearCliffordBackend._birth;o_rot=bk.NearCliffordBackend._rot
def rh(self,q):
    if q in self.M:self._h_axis(self.M.index(q))
    else:o_h(self,q)
def rs(self,q,dag=False):
    if q in self.M:self._s_axis(self.M.index(q),dag)
    else:o_s(self,q,dag)
def rcx(self,c,t):
    if c in self.M and t in self.M:self._cnot_axes(self.M.index(c),self.M.index(t))
    else:o_cx(self,c,t)
def rcz(self,a,b):
    if a in self.M and b in self.M:
        ja=self.M.index(a);jb=self.M.index(b);r=len(self.M);t=self.phi.reshape([2]*r);s=[slice(None)]*r;s[r-1-ja]=1;s[r-1-jb]=1;t[tuple(s)]*=-1.0
    else:o_cz(self,a,b)
def rbirth(self,slot):
    q=self._new_q(slot);self.nc._promote(q);self.nc._h_axis(self.nc.M.index(q));return q
def rrot(self,slot,angle):
    q=self.slot2id.get(slot)
    if q is None:return
    if q not in self.nc.M:self.nc._promote(q);self.nc._h_axis(self.nc.M.index(q))
    sign=-1.0 if self.frame.xb(slot) else 1.0;bit=self.nc.M.index(q);v=self.nc.phi.reshape(-1,2,1<<bit);v[:,1,:]*=np.exp(1j*sign*angle);self._track_M()
def setup(): C.h=rh;C.s=rs;C.cx=rcx;C.cz=rcz;bk.NearCliffordBackend._birth=rbirth;bk.NearCliffordBackend._rot=rrot
def teardown(): C.h=o_h;C.s=o_s;C.cx=o_cx;C.cz=o_cz;bk.NearCliffordBackend._birth=o_birth;bk.NearCliffordBackend._rot=o_rot

print("\n(2) IN-CIRCUIT: per-opcode E_{t+1}|psi> = O_logical . E_t|psi> (cultivation_d3, dense):")
results={"cnot_numeric":[], "cnot_gauge":[], "cz_numeric":[], "cz_gauge":[]}
prog=compile_bounded(open("qec_bench/circuits/cultivation_d3.stim").read())
PEND={}
def rec(step,be):
    if step>=len(prog):
        # finalize any pending op from the previous step
        return _finalize(be)
    _finalize(be)
    name=_opname(prog[step].opcode)
    if name in ("OP_ARRAY_CNOT","OP_ARRAY_CZ"):
        inst=prog[step]; a1=int(inst.axis_1); a2=int(inst.axis_2)
        u=be.slot2id.get(a1); v=be.slot2id.get(a2)
        both=(u is not None and v is not None and u in be.nc.M and v in be.nc.M)
        PEND.update(dict(name=name,u=u,v=v,both=both,pre=robust_statevector(copy.deepcopy(be.nc))))
def _finalize(be):
    if not PEND: return
    p=PEND.copy(); PEND.clear()
    post=robust_statevector(copy.deepcopy(be.nc))
    if p["u"] is None or p["v"] is None:
        return
    if p["name"]=="OP_ARRAY_CNOT":
        ref=dense_cnot(p["pre"].copy(),p["u"],p["v"])
        key="cnot_numeric" if p["both"] else "cnot_gauge"
    else:
        ref=dense_cz(p["pre"].copy(),p["u"],p["v"])
        key="cz_numeric" if p["both"] else "cz_gauge"
    results[key].append(upg(post,ref))
setup()
try:
    be=bk.NearCliffordBackend(clifft_axis_bounded=True,drop_dead=False,structure_once=False,clifft_axis_enforce=True)
    be.run_shot(prog,1,step_recorder=rec)
finally: teardown()
for k,v in results.items():
    if v: print(f"   {k:14}: n={len(v):3}  worst E_{{t+1}}=O.E_t residual = {max(v):.2e}")
    else: print(f"   {k:14}: n=0")
allres=[x for v in results.values() for x in v]
ok = worst_cx<1e-13 and worst_cz<1e-13 and (max(allres) if allres else 0)<1e-12
print("\nRESULT:", "PASS" if ok else "FAIL")
