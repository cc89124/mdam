"""Decisive engine-vs-dense test on a COMPLEX circuit (many R_Y deferred + propagated through
many CX/H, like surface-code syndrome extraction depth).  If the engine diverges from the dense
oracle here, bug #2 is in the ENGINE's multi-gate pending conjugation/core (not the backend).
Forced outcomes; exact Born comparison at every measurement."""
import sys
import numpy as np
sys.path.insert(0, "/home/jung/clifft-paper")
from nearclifford_backend.clifft_axis.bounded import CliftAxisBoundedNearClifford as B

I2=np.eye(2,dtype=complex); Xm=np.array([[0,1],[1,0]],dtype=complex); Zm=np.array([[1,0],[0,-1]],dtype=complex)
Ym=np.array([[0,-1j],[1j,0]],dtype=complex); Hm=np.array([[1,1],[1,-1]],dtype=complex)/np.sqrt(2)
Sm=np.array([[1,0],[0,1j]],dtype=complex)
class Dense:
    def __init__(s,n): s.n=n; s.v=np.zeros(1<<n,dtype=complex); s.v[0]=1.0
    def _op1(s,g,q):
        s.v=s.v.reshape([2]*s.n); s.v=np.moveaxis(np.tensordot(g,np.moveaxis(s.v,s.n-1-q,0),axes=(1,0)),0,s.n-1-q); s.v=s.v.reshape(-1)
    def h(s,q): s._op1(Hm,q)
    def sg(s,q,dag=False): s._op1(Sm.conj().T if dag else Sm,q)
    def rz(s,q,t): s._op1(np.cos(t/2)*I2-1j*np.sin(t/2)*Zm,q)
    def ry(s,q,t): s.sg(q,True); s.h(q); s.rz(q,t); s.h(q); s.sg(q,False)
    def cx(s,c,t):
        d=1<<s.n; out=np.empty_like(s.v)
        for i in range(d): out[(i^(1<<t)) if (i>>c)&1 else i]=s.v[i]
        s.v=out
    def cz(s,a,b):
        for i in range(1<<s.n):
            if (i>>a)&1 and (i>>b)&1: s.v[i]*=-1
    def born_p0(s,q):
        s.v=s.v.reshape([2]*s.n); idx=[slice(None)]*s.n; idx[s.n-1-q]=0
        p=float(np.sum(np.abs(s.v[tuple(idx)])**2)); s.v=s.v.reshape(-1); return p
    def proj(s,q,o):
        s.v=s.v.reshape([2]*s.n); idx=[slice(None)]*s.n; idx[s.n-1-q]=1-o
        s.v[tuple(idx)]=0; s.v=s.v.reshape(-1); n=np.linalg.norm(s.v); s.v/= (n if n>1e-15 else 1)
class _R:
    def __init__(s,b): s.b=b
    def random(s): return 0.0 if s.b==0 else 1.0
def eng_ry(e,q,t): e.s(q,dag=True); e.h(q); e.apply_rotation(0,1<<q,t); e.h(q); e.s(q,dag=False)

# deterministic pseudo-random circuit (no Math.random; fixed seed via numpy default_rng with int)
rng=np.random.default_rng(12345)
N=6; TH=0.0628
ops=[]
# emulate surface-code-like depth: 6 layers of {RY on all, random CX/CZ entangling, RY on all}
for layer in range(6):
    for q in range(N): ops.append(('ry',q,TH*(1 if rng.random()<0.5 else -1)))
    pairs=list(range(N)); rng.shuffle(pairs)
    for k in range(0,N-1,2):
        a,b=pairs[k],pairs[k+1]
        ops.append(('cx',a,b) if rng.random()<0.5 else ('cz',a,b))
    for q in range(N):
        if rng.random()<0.5: ops.append(('h',q))
    for q in range(N): ops.append(('ry',q,TH*(1 if rng.random()<0.5 else -1)))
# measure all qubits (forced to a fixed pattern)
forced=[int(rng.random()<0.5) for _ in range(N)]
for q in range(N): ops.append(('m',q,forced[q]))

e=B(N); e.set_clifft_budget(N+2,enforce=False); e.log_cores=True; e.core_log=[]
d=Dense(N); worst=0.0; mi=0
for op in ops:
    k=op[0]
    if k=='ry': eng_ry(e,op[1],op[2]); d.ry(op[1],op[2])
    elif k=='h': e.h(op[1]); d.h(op[1])
    elif k=='cx': e.cx(op[1],op[2]); d.cx(op[1],op[2])
    elif k=='cz': e.cz(op[1],op[2]); d.cz(op[1],op[2])
    elif k=='m':
        q,o=op[1],op[2]; dp0=d.born_p0(q); e.rng=_R(o); e.measure_z(q)
        ep0=e.core_log[-1]['p0']; dd=abs(ep0-dp0); worst=max(worst,dd)
        print(f"  meas q={q} engine_p0={ep0:.6f} dense_p0={dp0:.6f} |Δ|={dd:.2e}{'  <== DIVERGES' if dd>1e-6 else ''}")
        d.proj(q,o); mi+=1
print(f"\nworst|Δ|={worst:.2e}  -> {'ENGINE BUG (multi-gate pending)' if worst>1e-6 else 'ENGINE EXACT even on complex depth -> bug #2 is in BACKEND'}")
