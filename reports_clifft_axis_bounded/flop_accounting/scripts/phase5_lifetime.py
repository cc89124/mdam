import sys
sys.path.insert(0,"/home/jung/clifft-paper"); sys.setrecursionlimit(400000)
import clifft  # noqa
import nearclifford_backend.backend as bk
from nearclifford_backend.clifft_axis.bounded import compile_bounded, CliftAxisBoundedNearClifford as C
from collections import defaultdict

ev=[]; batch=[0]
prog=compile_bounded(open("qec_bench/circuits/cultivation_d5.stim").read())
of1=C._flush_one; ofc=C._flush_core; opr=C._promote
promotes=[]; drops=[]
def f1(self,x,z,theta,phase=0):
    xp,zp,pp=self._pullback(x,z)
    ev.append((int(xp),int(zp),batch[0]))
    return of1(self,x,z,theta,phase)
def fc(self,qx,qz):
    r=ofc(self,qx,qz); batch[0]+=1; return r
def pr(self,q):
    was = q in self.M
    r=opr(self,q)
    if not was: promotes.append((q,batch[0],len(self.M)))
    return r
C._flush_one=f1; C._flush_core=fc; C._promote=pr
try:
    bk.NearCliffordBackend(clifft_axis_bounded=True,drop_dead=False,structure_once=False,clifft_axis_enforce=True).run_shot(prog,1)
finally:
    C._flush_one=of1; C._flush_core=ofc; C._promote=opr

# per physical qubit: the single-qubit reduced Pauli (x,z) of every generator it appears in
perq=defaultdict(list)
for xp,zp,b in ev:
    for q in range(16):
        xb=(xp>>q)&1; zb=(zp>>q)&1
        if xb or zb:
            perq[q].append((xb,zb,b))

def ptype(xb,zb):
    return {(1,0):'X',(0,1):'Z',(1,1):'Y'}[(xb,zb)]

print("per-qubit single-qubit Pauli type across lifetime (does its OWN basis need to flip?):")
conflict_qubits=0
for q in sorted(perq):
    types=set(ptype(xb,zb) for xb,zb,_ in perq[q])
    batches=sorted(set(b for _,_,b in perq[q]))
    flip = len(types)>1
    conflict_qubits += flip
    print(f"  q{q:2}: {len(perq[q]):3} appearances, types={sorted(types)}, batches={batches[:8]}{'...' if len(batches)>8 else ''}  {'<-- BASIS FLIP' if flip else 'single-basis OK'}")
print(f"\nqubits whose own single-qubit basis FLIPS mid-life = {conflict_qubits}/{len(perq)}")
print("(0 flips => a fixed promote-basis per qubit keeps ALL its T's diagonal => 0 runtime H feasible,")
print(" matching clifft's measured 0 array_h. >0 => that many qubits force >=1 runtime basis change.)")
