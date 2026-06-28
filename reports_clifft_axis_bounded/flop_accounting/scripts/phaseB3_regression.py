"""Phase B3 regression: EXPLICIT clean INTERFERE consumer (no _pullback / _localize_to_Z),
direct diagonal Born on the H-folded array bit + project + drop + frame X-fold (= _drop_localized).
Verify BIT-IDENTICAL records / peak-rank / per-measurement p0 vs the AUTHORITATIVE path, on
cultivation_d3 AND cultivation_d5 over many seeds, and measure actual FLOP + wall.

x!=0 FAIL-FAST: the clean path asserts the measured Z_q pulls back to pure-Z on the array bit
(no X-content) — if a measurement were genuinely off-diagonal it would raise (Phase C/D regime),
NOT silently relocalize.
"""
import sys, time; sys.path.insert(0, "/home/jung/clifft-paper"); sys.setrecursionlimit(400000)
from collections import defaultdict
import numpy as np, clifft
from clifft import _clifft_core as cc
import nearclifford_backend.backend as bk
from nearclifford_backend.simulator import pauli_commute
from nearclifford_backend.clifft_axis import budget as _bud
from nearclifford_backend.clifft_axis.bounded import compile_bounded, CliftAxisBoundedNearClifford as C

BCOEF={'rot:offdiag':12,'rot:offdiag-scalar':12,'rot:diaghalf':3,'rot:diag':6,'rot:diag0':6,'rot:diag-scalar':6,
       'meas':10,'exp':10,'sqnorm':2,'normalize':2,'purge:h':4,'purge:s':2,'purge:cnot':0,'reduce:cnot':0,'reduce:cz':0,
       'reduce:gf2scan':0,'reduce:verify':0,'drop':0,'promote':0,'init':0,'post-reduce':0,'expand':0}
CONV={'cmul':6,'rcmul':2,'cadd':2,'sqmag':4,'vdot':8}
def Hn(x):
    a=abs(x)
    for u,s in ((1e9,'G'),(1e6,'M'),(1e3,'k')):
        if a>=u: return f"{x/u:.2f}{s}"
    return f"{x:.0f}"

o_h=C.h;o_s=C.s;o_cx=C.cx;o_cz=C.cz;o_birth=bk.NearCliffordBackend._birth;o_rot=bk.NearCliffordBackend._rot;o_mz=C.measure_z
def rh(self,q):
    if q in self.M:self._h_axis(self.M.index(q))
    else:o_h(self,q)
def rs(self,q,dag=False):
    if q in self.M:self._s_axis(self.M.index(q),dag)
    else:o_s(self,q,dag)
def rcx(self,c,t):
    if c in self.M and t in self.M:self._cnot_axes(self.M.index(c),self.M.index(t));self.budget.charge(self.phi.size,0,"reduce:cnot")
    else:o_cx(self,c,t)
def rcz(self,a,b):
    if a in self.M and b in self.M:
        ja=self.M.index(a);jb=self.M.index(b);r=len(self.M);t=self.phi.reshape([2]*r);s=[slice(None)]*r;s[r-1-ja]=1;s[r-1-jb]=1;t[tuple(s)]*=-1.0;self.budget.charge(self.phi.size,0,"reduce:cz")
    else:o_cz(self,a,b)
def rbirth(self,slot):
    q=self._new_q(slot);self.nc._promote(q);self.nc._h_axis(self.nc.M.index(q));return q
def rrot(self,slot,angle):
    q=self.slot2id.get(slot)
    if q is None:return
    if q not in self.nc.M:self.nc._promote(q);self.nc._h_axis(self.nc.M.index(q))
    sign=-1.0 if self.frame.xb(slot) else 1.0;bit=self.nc.M.index(q);v=self.nc.phi.reshape(-1,2,1<<bit);v[:,1,:]*=np.exp(1j*sign*angle);self.nc.budget.charge(self.nc.phi.size,0,"rot:diaghalf");self._track_M()

STAT = defaultdict(int)
def mz_clean(self, q):
    """EXPLICIT clean consumer. B0: _pullback(Z_q) = FP-free frame pullback -> reduced Z-parity
    (xp,zp,pp). x!=0 FAIL-FAST (no hidden relocalising H/butterfly -> Phase C/D regime). B2:
    contract the Z-parity onto a single pivot via FP-free CNOT-folds (purge:cnot=0, NO H), the
    user's affine substitution. B1: diagonal branch Born + project. Drop = _drop_localized."""
    self._flush_core(0, 1 << q)                      # FP-free here (eager rrot => pending empty)
    Pm = (0, 1 << q, 0)
    magset = set(self.M)
    anti_s = [i for i in range(self.n) if i not in magset and not pauli_commute(self.Zc[i], Pm)]
    M_before = len(self.M); p0 = None
    if anti_s:                                        # Gottesman-Knill stabilizer (FP-free, no array)
        STAT["ag"] += 1
        out = self._ag_measure(Pm, anti_s); branch = "stabilizer"
    else:
        xp, zp, pp = self._pullback(0, 1 << q)        # B0: physical Z_q -> reduced Pauli (FP-free)
        if xp != 0:                                   # FAIL-FAST: off-diagonal -> NOT a hidden H
            raise AssertionError(f"clean-consumer: measured Z_q pulls back OFF-DIAGONAL xp={xp:#x} "
                                 f"(q={q}) -> Phase C/D gauge-vs-numerical regime, NOT a hidden H")
        supp = [s for s in range(self.n) if (zp >> s) & 1 and s in self.M]
        if len(supp) > 1: STAT["parity_fold"] += 1
        r, sign = self._localize_to_Z(xp, zp, pp, prefer=q)   # B2: FP-free CNOT-fold (no H, no butterfly)
        if r is None:                                 # deterministic (no magic support)
            p0 = max(0.0, min(1.0, (1.0 + sign) / 2.0))
            out = 0 if float(self.rng.random()) < p0 else 1
            STAT["det"] += 1
        else:
            jr = self.M.index(r)
            s0 = self._branch_sqnorm(jr, 0); s1 = self._branch_sqnorm(jr, 1); tot = s0 + s1
            p0 = ((s0 if sign > 0 else s1) / tot) if tot > 1e-300 else 0.5
            p0 = max(0.0, min(1.0, p0))
            out = 0 if float(self.rng.random()) < p0 else 1
            plus_bit = 0 if sign > 0 else 1
            keepbit = plus_bit if out == 0 else (1 - plus_bit)
            v = self.phi.reshape(-1, 2, 1 << jr); v[:, 1 - keepbit, :] = 0.0
            nrm2 = s0 if keepbit == 0 else s1
            if nrm2 > 1e-24:
                self.budget.charge(self.phi.size, 0, "normalize"); self.phi /= nrm2 ** 0.5
            self._drop_localized(r, keepbit)
            STAT["magic"] += 1
        branch = "magic"
    self._reduce_full()
    if len(self.M) > self.max_M: self.max_M = len(self.M)
    self.budget.note_resident(self.phi.size, "post-reduce")
    if self.log_cores:
        self.core_log.append(dict(meas=self._meas_log_ctr, branch=branch, M_before=M_before,
                                  M_after=len(self.M), p0=p0, peak_live_words=self.budget.peak))
    self._meas_log_ctr += 1
    return out

def setup(clean):
    C.h=rh;C.s=rs;C.cx=rcx;C.cz=rcz;bk.NearCliffordBackend._birth=rbirth;bk.NearCliffordBackend._rot=rrot
    if clean: C.measure_z=mz_clean
def teardown():
    C.h=o_h;C.s=o_s;C.cx=o_cx;C.cz=o_cz;bk.NearCliffordBackend._birth=o_birth;bk.NearCliffordBackend._rot=o_rot;C.measure_z=o_mz

def run(circ, seed, clean):
    prog=compile_bounded(open(f"qec_bench/circuits/{circ}.stim").read()); setup(clean)
    try:
        be=bk.NearCliffordBackend(clifft_axis_bounded=True,drop_dead=False,structure_once=False,clifft_axis_enforce=True)
        rec=tuple(be.run_shot(prog,seed)); pk=be.nc.budget.peak_resident.bit_length()-1
        p0=tuple(round(c["p0"],10) for c in be.nc.core_log if c.get("p0") is not None)
    finally: teardown()
    return rec,pk,p0

def flop(circ, seed, clean):
    prog=compile_bounded(open(f"qec_bench/circuits/{circ}.stim").read())
    agg=defaultdict(int); orig=_bud.DenseMemoryBudget.charge
    def ch(self,rr,t=0,where=""):agg[where]+=int(rr);return orig(self,rr,t,where)
    _bud.DenseMemoryBudget.charge=ch; setup(clean)
    try:
        be=bk.NearCliffordBackend(clifft_axis_bounded=True,drop_dead=False,structure_once=False,clifft_axis_enforce=True);be.run_shot(prog,seed)
    finally:_bud.DenseMemoryBudget.charge=orig; teardown()
    return sum(BCOEF.get(w,0)*s for w,s in agg.items())

def clifft_flop(circ,seed=1):
    prog=clifft.compile(open(f"qec_bench/circuits/{circ}.stim").read(),bytecode_passes=None)
    cc.cost_meter_reset(); cc.cost_meter_enable(True); clifft.sample(prog,1,seed); cc.cost_meter_enable(False)
    return sum(sum(CONV[k]*s[k] for k in CONV) for s in cc.cost_meter_snapshot().values())

def wall(circ, seed, clean):
    prog=compile_bounded(open(f"qec_bench/circuits/{circ}.stim").read()); setup(clean)
    try:
        be=bk.NearCliffordBackend(clifft_axis_bounded=True,drop_dead=False,structure_once=False,clifft_axis_enforce=True);be.run_shot(prog,seed)
        t0=time.perf_counter()
        for _ in range(10): be.run_shot(prog,seed)
        dt=(time.perf_counter()-t0)/10*1000
    finally: teardown()
    return dt

for circ in ["cultivation_d3","cultivation_d5"]:
    STAT.clear()
    rmis=pmis=p0mis=0; NS=60
    for s in range(1, NS+1):
        ra,ka,qa=run(circ,s,False)     # authoritative (measure_z)
        rc,kc,qc=run(circ,s,True)      # explicit clean consumer
        if ra!=rc: rmis+=1
        if ka!=kc: pmis+=1
        if qa!=qc: p0mis+=1
    fc=clifft_flop(circ); fa=flop(circ,1,False); fcl=flop(circ,1,True)
    wa=wall(circ,1,False); wcl=wall(circ,1,True)
    print(f"\n{circ}:  clifft FLOP={Hn(fc)}")
    print(f"   bit-identity clean vs authoritative over {NS} seeds:  records_mis={rmis}  peakrank_mis={pmis}  p0_mis={p0mis}")
    print(f"   FLOP  authoritative={Hn(fa)} ({fa/fc:.2f}x)   clean={Hn(fcl)} ({fcl/fc:.2f}x)")
    print(f"   wall  authoritative={wa:.1f}ms   clean={wcl:.1f}ms")
    print(f"   measurement routing (clean): {dict(STAT)}")
