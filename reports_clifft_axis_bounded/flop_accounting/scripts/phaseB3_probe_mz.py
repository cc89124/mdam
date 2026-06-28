"""Does measure_z fire ANY relocalize/butterfly/_localize_to_Z on the reduced INTERFERE path?
Instrument the budget 'where' tags emitted DURING measure_z calls (vs unitary ops)."""
import sys; sys.path.insert(0, "/home/jung/clifft-paper"); sys.setrecursionlimit(400000)
from collections import defaultdict
import numpy as np, clifft
import nearclifford_backend.backend as bk
from nearclifford_backend.clifft_axis import budget as _bud
from nearclifford_backend.clifft_axis.bounded import compile_bounded, CliftAxisBoundedNearClifford as C

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

IN_MZ = {"flag": False}
def mz_probe(self, q):
    IN_MZ["flag"] = True
    try:
        return o_mz(self, q)
    finally:
        IN_MZ["flag"] = False

def setup():
    C.h=rh;C.s=rs;C.cx=rcx;C.cz=rcz;bk.NearCliffordBackend._birth=rbirth;bk.NearCliffordBackend._rot=rrot;C.measure_z=mz_probe
def teardown():
    C.h=o_h;C.s=o_s;C.cx=o_cx;C.cz=o_cz;bk.NearCliffordBackend._birth=o_birth;bk.NearCliffordBackend._rot=o_rot;C.measure_z=o_mz

for circ in ["cultivation_d3","cultivation_d5"]:
    prog=compile_bounded(open(f"qec_bench/circuits/{circ}.stim").read())
    inside=defaultdict(int); outside=defaultdict(int)
    orig=_bud.DenseMemoryBudget.charge
    def ch(self,rr,t=0,where=""):
        (inside if IN_MZ["flag"] else outside)[where]+=1
        return orig(self,rr,t,where)
    _bud.DenseMemoryBudget.charge=ch; setup()
    try:
        be=bk.NearCliffordBackend(clifft_axis_bounded=True,drop_dead=False,structure_once=False,clifft_axis_enforce=True)
        be.run_shot(prog,1)
    finally:
        _bud.DenseMemoryBudget.charge=orig; teardown()
    print(f"\n{circ}: budget 'where' tags emitted INSIDE measure_z:")
    for w,c in sorted(inside.items(), key=lambda x:-x[1]):
        print(f"    {w:24} x{c}")
    flagged=[w for w in inside if any(k in w for k in ("localize","butterfly","offdiag","relocal","gf2","reduce:verify","pullback"))]
    print(f"  -> relocalize/butterfly/localize tags inside measure_z: {flagged if flagged else 'NONE (clean diagonal path)'}")
