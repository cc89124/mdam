"""Reduce the engine-level R_Y divergence to its minimal form (exact vs dense, forced outcomes).
Test increasingly complex small circuits to find the smallest that diverges -> pinpoints the
mechanism (multiple pending on one qubit? propagation through 2 entanglers? core of 2 rotations?)."""
import sys
import numpy as np
sys.path.insert(0, "/home/jung/clifft-paper")
from nearclifford_backend.clifft_axis.bounded import CliftAxisBoundedNearClifford as B

I2=np.eye(2,dtype=complex); Xm=np.array([[0,1],[1,0]],dtype=complex); Zm=np.array([[1,0],[0,-1]],dtype=complex)
Hm=np.array([[1,1],[1,-1]],dtype=complex)/np.sqrt(2); Sm=np.array([[1,0],[0,1j]],dtype=complex); Ym=np.array([[0,-1j],[1j,0]],dtype=complex)
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
        s.v[tuple(idx)]=0; s.v=s.v.reshape(-1); n=np.linalg.norm(s.v); s.v/=(n if n>1e-15 else 1)
class _R:
    def __init__(s,b): s.b=b
    def random(s): return 0.0 if s.b==0 else 1.0
def eng_ry(e,q,t): e.s(q,dag=True); e.h(q); e.apply_rotation(0,1<<q,t); e.h(q); e.s(q,dag=False)
T=0.6
def runcase(name,n,ops):
    e=B(n); e.set_clifft_budget(n+2,enforce=False); e.log_cores=True; e.core_log=[]; d=Dense(n); worst=0.0
    for op in ops:
        k=op[0]
        if k=='ry': eng_ry(e,op[1],op[2]); d.ry(op[1],op[2])
        elif k=='h': e.h(op[1]); d.h(op[1])
        elif k=='s': e.s(op[1],op[2]); d.sg(op[1],op[2])
        elif k=='cx': e.cx(op[1],op[2]); d.cx(op[1],op[2])
        elif k=='cz': e.cz(op[1],op[2]); d.cz(op[1],op[2])
        elif k=='m':
            q,o=op[1],op[2]; dp0=d.born_p0(q); e.rng=_R(o); e.measure_z(q)
            ep0=e.core_log[-1]['p0']; worst=max(worst,abs(ep0-dp0)); d.proj(q,o)
    print(f"  {name:42} worst|Δ|={worst:.2e}  {'DIVERGES' if worst>1e-6 else 'ok'}")
    return worst

# escalating complexity
runcase("RY0;CX01;RY0;M0;M1", 2, [('ry',0,T),('cx',0,1),('ry',0,T),('m',0,0),('m',1,0)])
runcase("RY0;RY1;CX01;M0;M1", 2, [('ry',0,T),('ry',1,T),('cx',0,1),('m',0,0),('m',1,0)])
runcase("RY0;RY1;CX01;RY0;RY1;M0;M1", 2, [('ry',0,T),('ry',1,T),('cx',0,1),('ry',0,T),('ry',1,T),('m',0,0),('m',1,0)])
runcase("RY0;CX01;RY1;CX10;M0;M1", 2, [('ry',0,T),('cx',0,1),('ry',1,T),('cx',1,0),('m',0,0),('m',1,0)])
runcase("RY0;RY1;CZ01;RY0;RY1;Mx2", 2, [('ry',0,T),('ry',1,T),('cz',0,1),('ry',0,T),('ry',1,T),('m',0,0),('m',1,0)])
runcase("3q: RY all;CX01;CX12;RY all;M all",3,
        [('ry',0,T),('ry',1,T),('ry',2,T),('cx',0,1),('cx',1,2),('ry',0,T),('ry',1,T),('ry',2,T),('m',0,0),('m',1,0),('m',2,0)])
runcase("2q two-pending same qubit then entangle",2,
        [('ry',0,T),('ry',0,T),('cx',0,1),('m',1,0),('m',0,0)])
runcase("RY0;H1;CX1->0;RY0;H1;M1(synX);M0",2,
        [('ry',0,T),('h',1),('cx',1,0),('ry',0,T),('h',1),('m',1,0),('m',0,0)])
