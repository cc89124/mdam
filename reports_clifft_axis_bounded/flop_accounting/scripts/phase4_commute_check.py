import sys
sys.path.insert(0, "/home/jung/clifft-paper")
sys.setrecursionlimit(400000)
import clifft  # noqa
import nearclifford_backend.backend as bk
from nearclifford_backend.clifft_axis.bounded import compile_bounded, CliftAxisBoundedNearClifford as C

def symp(p, q):
    x1,z1=p; x2,z2=q
    return ((x1&z2).bit_count()+(x2&z1).bit_count())&1

events=[]; batch=[0]
prog=compile_bounded(open("qec_bench/circuits/cultivation_d5.stim").read())
of1=C._flush_one; ofc=C._flush_core
def f1(self,x,z,theta,phase=0):
    xp,zp,pp=self._pullback(x,z)
    events.append((int(xp),int(zp),batch[0],len(self.M)))
    return of1(self,x,z,theta,phase)
def fc(self,qx,qz):
    r=ofc(self,qx,qz); batch[0]+=1; return r
C._flush_one=f1; C._flush_core=fc
try:
    bk.NearCliffordBackend(clifft_axis_bounded=True,drop_dead=False,structure_once=False,clifft_axis_enforce=True).run_shot(prog,1)
finally:
    C._flush_one=of1; C._flush_core=fc if False else ofc

n=len(events)
# full pairwise commuting graph
anti=[]
for i in range(n):
    for j in range(i+1,n):
        if symp(events[i][:2],events[j][:2]):
            anti.append((i,j))
print(f"generators={n}")
print(f"total pairs = {n*(n-1)//2}")
print(f"ANTICOMMUTING pairs = {len(anti)}  -> all-91-pairwise-commute = {len(anti)==0}")
# cross-batch vs within-batch
within=sum(1 for i,j in anti if events[i][2]==events[j][2])
cross=len(anti)-within
print(f"  within-batch anticommuting = {within}   (must be 0 if batches commute)")
print(f"  cross-batch  anticommuting = {cross}")
# which batches anticommute
from collections import Counter
bc=Counter((min(events[i][2],events[j][2]),max(events[i][2],events[j][2])) for i,j in anti)
print(f"  batch-pairs with anticommutation (top): {dict(list(sorted(bc.items()))[:12])}")
# greedy ordered runs (pairwise-commuting)
runs=[]; cur=[]
for idx,e in enumerate(events):
    if all(symp(e[:2],events[g][:2])==0 for g in cur): cur.append(idx)
    else: runs.append(cur); cur=[idx]
if cur: runs.append(cur)
print(f"greedy ordered pairwise-commuting runs = {len(runs)}  sizes={[len(r) for r in runs]}")
print(f"run boundaries at indices: {[r[0] for r in runs]}")
