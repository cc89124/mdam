"""Free-born actual-cost measurement (the user's §3): replace the born H-butterfly (_h_axis,
purge:h=4/elem) with a COPY-born matching clifft's expand (0 arithmetic; the 1/v2 deferred to the
measurement's tot renormalisation).  Measure -- not project -- FP-FLOP, allocation, copy bytes,
normalisation, memory traffic, peak workspace, wall; verify records/rank/p0 bit-identical.

Variants:
  current   : _promote + _h_axis            (purge:h butterfly, 4/elem)
  free_scale: _promote + copy + *(1/v2)      (copy 0-arith + rcmul scale 2/elem)
  free_defer: _promote + copy                (copy 0-arith only; = clifft expand; norm deferred)
"""
import sys, time, tracemalloc; sys.path.insert(0, "/home/jung/clifft-paper"); sys.setrecursionlimit(400000)
from collections import defaultdict
import numpy as np, clifft
from clifft import _clifft_core as cc
import nearclifford_backend.backend as bk
from nearclifford_backend.clifft_axis import budget as _bud
from nearclifford_backend.clifft_axis.bounded import compile_bounded, CliftAxisBoundedNearClifford as C

INV_SQRT2 = 0.7071067811865476
BCOEF={'rot:offdiag':12,'rot:diaghalf':3,'rot:diag':6,'meas':10,'sqnorm':2,'normalize':2,
       'purge:h':4,'purge:s':2,'purge:cnot':0,'reduce:cnot':0,'reduce:cz':0,'reduce:gf2scan':0,
       'reduce:verify':0,'drop':0,'promote':0,'init':0,'post-reduce':0,'expand':0,
       'born:copy':0,'born:scale':2}
CONV={'cmul':6,'rcmul':2,'cadd':2,'sqmag':4,'vdot':8}
def Hn(x):
    a=abs(x)
    for u,s in ((1e9,'G'),(1e6,'M'),(1e3,'k')):
        if a>=u: return f"{x/u:.2f}{s}"
    return f"{x:.0f}"

o_h=C.h;o_s=C.s;o_cx=C.cx;o_cz=C.cz;o_birth=bk.NearCliffordBackend._birth;o_rot=bk.NearCliffordBackend._rot
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
def rrot(self,slot,angle):
    q=self.slot2id.get(slot)
    if q is None:return
    if q not in self.nc.M:self.nc._promote(q);self.nc._h_axis(self.nc.M.index(q))
    sign=-1.0 if self.frame.xb(slot) else 1.0;bit=self.nc.M.index(q);v=self.nc.phi.reshape(-1,2,1<<bit);v[:,1,:]*=np.exp(1j*sign*angle);self.nc.budget.charge(self.nc.phi.size,0,"rot:diaghalf");self._track_M()

IN_BIRTH={"f":False}
# born variants
def birth_current(self,slot):
    q=self._new_q(slot);IN_BIRTH["f"]=True
    try:self.nc._promote(q);self.nc._h_axis(self.nc.M.index(q))
    finally:IN_BIRTH["f"]=False
    return q
def birth_free_scale(self,slot):
    q=self._new_q(slot);IN_BIRTH["f"]=True
    try:
        self.nc._promote(q)
        half=self.nc.phi.size>>1
        self.nc.phi[half:]=self.nc.phi[:half]            # COPY low->high (0 arithmetic)
        self.nc.budget.charge(self.nc.phi.size,0,"born:copy")
        self.nc.phi*=INV_SQRT2                            # scale (rcmul)
        self.nc.budget.charge(self.nc.phi.size,0,"born:scale")
    finally:IN_BIRTH["f"]=False
    return q
def birth_free_defer(self,slot):
    q=self._new_q(slot);IN_BIRTH["f"]=True
    try:
        self.nc._promote(q)
        half=self.nc.phi.size>>1
        self.nc.phi[half:]=self.nc.phi[:half]            # COPY only; 1/v2 deferred to meas tot (= clifft)
        self.nc.budget.charge(self.nc.phi.size,0,"born:copy")
    finally:IN_BIRTH["f"]=False
    return q

BIRTHS={"current":birth_current,"free_scale":birth_free_scale,"free_defer":birth_free_defer}
def setup(birth):
    C.h=rh;C.s=rs;C.cx=rcx;C.cz=rcz;bk.NearCliffordBackend._birth=birth;bk.NearCliffordBackend._rot=rrot
def teardown():
    C.h=o_h;C.s=o_s;C.cx=o_cx;C.cz=o_cz;bk.NearCliffordBackend._birth=o_birth;bk.NearCliffordBackend._rot=o_rot

def make(): return bk.NearCliffordBackend(clifft_axis_bounded=True,drop_dead=False,structure_once=False,clifft_axis_enforce=True)

def records(circ,seed,birth):
    prog=compile_bounded(open(f"qec_bench/circuits/{circ}.stim").read());setup(birth)
    try:
        be=make();rec=tuple(be.run_shot(prog,seed));pk=be.nc.budget.peak_resident.bit_length()-1
        p0=tuple(round(c["p0"],10) for c in be.nc.core_log if c.get("p0") is not None)
    finally:teardown()
    return rec,pk,p0

def costs(circ,seed,birth):
    """FP-FLOP (BCOEF) + born-specific element counts (copy/scale) + peak resident words."""
    prog=compile_bounded(open(f"qec_bench/circuits/{circ}.stim").read())
    agg=defaultdict(int);bornagg=defaultdict(int);orig=_bud.DenseMemoryBudget.charge
    def ch(self,rr,t=0,where=""):
        agg[where]+=int(rr)
        if IN_BIRTH["f"]:bornagg[where]+=int(rr)
        return orig(self,rr,t,where)
    _bud.DenseMemoryBudget.charge=ch;setup(birth)
    try:
        be=make();be.run_shot(prog,seed);peak=be.nc.budget.peak_resident
    finally:_bud.DenseMemoryBudget.charge=orig;teardown()
    flop=sum(BCOEF.get(w,0)*s for w,s in agg.items())
    return flop,agg,bornagg,peak

def wall(circ,seed,birth,reps=15):
    prog=compile_bounded(open(f"qec_bench/circuits/{circ}.stim").read());setup(birth)
    try:
        be=make();be.run_shot(prog,seed)
        t0=time.perf_counter()
        for _ in range(reps):be.run_shot(prog,seed)
        dt=(time.perf_counter()-t0)/reps*1000
    finally:teardown()
    return dt

def peak_alloc(circ,seed,birth):
    prog=compile_bounded(open(f"qec_bench/circuits/{circ}.stim").read());setup(birth)
    try:
        be=make();be.run_shot(prog,seed)  # warm
        tracemalloc.start()
        be.run_shot(prog,seed)
        cur,pk=tracemalloc.get_traced_memory();tracemalloc.stop()
    finally:teardown()
    return pk

def clifft_flop(circ,seed=1):
    prog=clifft.compile(open(f"qec_bench/circuits/{circ}.stim").read(),bytecode_passes=None)
    cc.cost_meter_reset();cc.cost_meter_enable(True);clifft.sample(prog,1,seed);cc.cost_meter_enable(False)
    return sum(sum(CONV[k]*s[k] for k in CONV) for s in cc.cost_meter_snapshot().values())

for circ in ["cultivation_d3","cultivation_d5"]:
    fc=clifft_flop(circ)
    print(f"\n================  {circ}   (clifft FLOP={Hn(fc)})  ================")
    # bit-identity of each variant vs current, over seeds
    base=[records(circ,s,birth_current) for s in range(1,41)]
    print(f"{'variant':11} {'FLOP':>9} {'xclifft':>8} {'bornFLOP':>9} {'copy_elems':>10} {'peak_words':>10} {'tracemB':>9} {'wall_ms':>8}  bit-ident(rec/rank/p0)")
    for name,birth in BIRTHS.items():
        f,agg,bornagg,peak=costs(circ,1,birth)
        bornflop=sum(BCOEF.get(w,0)*s for w,s in bornagg.items())   # ISOLATED born-only FP-FLOP
        copye=bornagg.get('born:copy',0)
        tm=peak_alloc(circ,1,birth); w=wall(circ,1,birth)
        # bit-identity
        rmis=pmis=qmis=0
        for s in range(1,41):
            r,k,q=records(circ,s,birth); rb,kb,qb=base[s-1]
            if r!=rb:rmis+=1
            if k!=kb:pmis+=1
            if q!=qb:qmis+=1
        print(f"{name:11} {Hn(f):>9} {f/fc:>7.2f}x {Hn(bornflop):>9} {Hn(copye):>10} {peak:>10} {tm/1e6:>8.2f}M {w:>7.1f}ms  {rmis}/{pmis}/{qmis}")
