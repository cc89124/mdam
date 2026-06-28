import sys
sys.path.insert(0,"/home/jung/clifft-paper"); sys.setrecursionlimit(400000)
import clifft  # noqa
import nearclifford_backend.backend as bk
from nearclifford_backend.clifft_axis.bounded import compile_bounded, CliftAxisBoundedNearClifford as C
from collections import defaultdict

# Track promote (M.append) and drop (M shrink) per qubit -> incarnations. For each T flush,
# record the pulled-back single-qubit Pauli on each support qubit, tagged by that qubit's current
# incarnation id. 0-H feasible iff within each (qubit, incarnation) all generators share one
# single-qubit Pauli direction (consistent born-basis).
prog=compile_bounded(open("qec_bench/circuits/cultivation_d5.stim").read())
of1=C._flush_one; ofc=C._flush_core
incarn=defaultdict(int)             # qubit -> current incarnation counter
active=set()
gens=defaultdict(list)              # (qubit, incarn) -> list of single-qubit pauli types
nflush=[0]; nmeas=[0]; agcount=[0]
oag=C._ag_measure
def sync(self):
    cur=set(self.M)
    for q in cur-active:            # newly promoted
        incarn[q]+=1
    active.clear(); active.update(cur)
def f1(self,x,z,theta,phase=0):
    sync(self)
    xp,zp,pp=self._pullback(x,z)
    for q in range(16):
        xb=(xp>>q)&1; zb=(zp>>q)&1
        if xb or zb:
            t={(1,0):'X',(0,1):'Z',(1,1):'Y'}[(xb,zb)]
            gens[(q,incarn[q])].append(t)
    nflush[0]+=1
    return of1(self,x,z,theta,phase)
def fc(self,qx,qz):
    r=ofc(self,qx,qz); nmeas[0]+=1; sync(self); return r
def ag(self,Pm,anti):
    agcount[0]+=1; return oag(self,Pm,anti)
C._flush_one=f1; C._flush_core=fc; C._ag_measure=ag
try:
    bk.NearCliffordBackend(clifft_axis_bounded=True,drop_dead=False,structure_once=False,clifft_axis_enforce=True).run_shot(prog,1)
finally:
    C._flush_one=of1; C._flush_core=ofc; C._ag_measure=oag

print(f"flushes={nflush[0]}  measurements={nmeas[0]}  AG-measures(Clifford-frame updates)={agcount[0]}")
print(f"distinct (qubit,incarnation) cells with generators = {len(gens)}")
conflict=0; total_incarn=defaultdict(set)
for (q,inc),ts in sorted(gens.items()):
    total_incarn[q].add(inc)
    s=set(ts)
    if len(s)>1:
        conflict+=1
        print(f"  q{q} incarnation#{inc}: types={sorted(s)} ({len(ts)} gens)  <-- MIXED -> needs >=1 runtime H")
print(f"\nincarnations per conflict qubit: q5={sorted(total_incarn.get(5,[]))} q14={sorted(total_incarn.get(14,[]))}")
print(f"(qubit,incarnation) cells with MIXED single-qubit Pauli = {conflict}")
print(f"=> minimum runtime H under per-incarnation born-basis (consistency lower bound) = {conflict}")
print(f"   if 0: each axis incarnation is single-basis -> H absorbed at promote -> 0 runtime H (matches clifft)")
