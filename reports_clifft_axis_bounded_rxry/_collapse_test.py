"""Decisive test for bug #2 (coherent-tilt loss): drive the bounded ENGINE with primitives and
compare Born p0 at EVERY measurement to a self-contained dense statevector oracle (same
convention: apply_rotation(x,z,theta)=exp(-i theta P/2), radians), with FORCED outcomes so both
follow the same trajectory.  Reproduce the surface-code structure: R_Y on several qubits,
entangling Cliffords, an intermediate (syndrome-style) measurement + collapse, then a coherence-
sensitive data measurement.  If the engine's later Born diverges -> the collapse corrupts
surviving coherence (projection/drop bug).  No clifft tooling (avoids reset/detector limits)."""
import sys, itertools
import numpy as np
sys.path.insert(0, "/home/jung/clifft-paper")
from nearclifford_backend.clifft_axis.bounded import CliftAxisBoundedNearClifford as B

I2 = np.eye(2, dtype=complex)
Xm = np.array([[0,1],[1,0]], dtype=complex); Zm = np.array([[1,0],[0,-1]], dtype=complex)
Ym = np.array([[0,-1j],[1j,0]], dtype=complex); Hm = np.array([[1,1],[1,-1]], dtype=complex)/np.sqrt(2)
Sm = np.array([[1,0],[0,1j]], dtype=complex)

class Dense:
    """statevector oracle, same gate semantics as the engine (RY via S†HRZ HS; exp(-i th P/2))."""
    def __init__(s, n): s.n=n; s.v=np.zeros(1<<n, dtype=complex); s.v[0]=1.0
    def _op1(s,g,q):
        s.v=s.v.reshape([2]*s.n)
        s.v=np.moveaxis(np.tensordot(g, np.moveaxis(s.v,s.n-1-q,0),axes=(1,0)),0,s.n-1-q)
        s.v=s.v.reshape(-1)
    def h(s,q): s._op1(Hm,q)
    def s(s,q,dag=False): s._op1(Sm.conj().T if dag else Sm,q)
    def rz(s,q,th): s._op1(np.cos(th/2)*I2-1j*np.sin(th/2)*Zm,q)
    def ry(s,q,th): s.s(q,True); s.h(q); s.rz(q,th); s.h(q); s.s(q,False)
    def cx(s,c,t):
        s.v=s.v.reshape([2]*s.n); idx=[slice(None)]*s.n; idx[s.n-1-c]=1
        sub=s.v[tuple(idx)]; s.v[tuple(idx)]=np.flip(sub, s.n-1-t if (s.n-1-t)<(s.n-1) else 0)
        # simpler robust CX below
        s.v=s.v.reshape(-1)
    def cx2(s,c,t):
        d=1<<s.n; out=np.empty_like(s.v)
        for i in range(d):
            j=i^(1<<t) if (i>>c)&1 else i; out[j]=s.v[i]
        s.v=out
    def cz(s,a,b):
        d=1<<s.n
        for i in range(d):
            if (i>>a)&1 and (i>>b)&1: s.v[i]*=-1
    def born_p0(s,q):
        s.v=s.v.reshape([2]*s.n); idx=[slice(None)]*s.n; idx[s.n-1-q]=0
        p0=float(np.sum(np.abs(s.v[tuple(idx)])**2)); s.v=s.v.reshape(-1); return p0
    def project(s,q,o):
        s.v=s.v.reshape([2]*s.n); idx=[slice(None)]*s.n; idx[s.n-1-q]=1-o
        s.v[tuple(idx)]=0.0; s.v=s.v.reshape(-1); nrm=np.linalg.norm(s.v)
        if nrm>1e-15: s.v/=nrm

# engine driver with forced outcome
class _R:
    def __init__(s,b): s.b=b
    def random(s): return 0.0 if s.b==0 else 1.0
def eng_ry(e,q,th): e.s(q,dag=True); e.h(q); e.apply_rotation(0,1<<q,th); e.h(q); e.s(q,dag=False)

def run(prog, n, forced):
    """prog = list of ops; forced = dict meas_index->outcome. Returns list of (engine_p0, dense_p0)."""
    e=B(n); e.set_clifft_budget(n+2, enforce=False); e.log_cores=True; e.core_log=[]
    d=Dense(n); res=[]; mi=0
    for op in prog:
        k=op[0]
        if k=='ry': eng_ry(e,op[1],op[2]); d.ry(op[1],op[2])
        elif k=='h': e.h(op[1]); d.h(op[1])
        elif k=='s': e.s(op[1],op[2]); d.s(op[1],op[2])
        elif k=='cx': e.cx(op[1],op[2]); d.cx2(op[1],op[2])
        elif k=='cz': e.cz(op[1],op[2]); d.cz(op[1],op[2])
        elif k=='m':
            q=op[1]; o=forced[mi]
            dp0=d.born_p0(q)
            e.rng=_R(o); out=e.measure_z(q)
            ep0=e.core_log[-1]['p0'] if e.core_log else None
            # align: engine returns its own bit `out`; we forced rng so out follows p0 vs threshold.
            # engine p0 is P(engine_bit=0). dense p0 is P(qubit Z=0). compare directly.
            res.append((mi,q,ep0,dp0,out))
            d.project(q,out)        # project dense with the SAME outcome the engine took
            mi+=1
    return res

# surface-code-like: 3 data + 1 ancilla; RY noise; X-syndrome of data; then data Z-measurements
def make_prog():
    p=[]
    for q in (0,1,2): p.append(('ry',q,0.3))         # coherent RY on data
    # X-stabiliser of data 0,1,2 onto ancilla 3
    p.append(('h',3))
    for q in (0,1,2): p.append(('cx',3,q))           # ancilla controls? use CX(anc->data) for X-stab
    p.append(('h',3)); p.append(('m',3))             # measure ancilla (syndrome)
    for q in (0,1,2): p.append(('ry',q,0.3))         # more coherent RY
    for q in (0,1,2): p.append(('m',q))              # data Z-measurements
    return p,4

prog,n=make_prog()
print("=== engine vs dense, forced outcomes all 0 (then 1 for variety) ===")
for forced_default in (0,1):
    forced={i:forced_default for i in range(4)}
    res=run(prog,n,forced)
    print(f"  forced={forced_default}:")
    worst=0.0
    for (mi,q,ep0,dp0,out) in res:
        dd=abs((ep0 if ep0 is not None else dp0)-dp0)
        worst=max(worst,dd)
        flag='  <== DIVERGES' if dd>1e-6 else ''
        print(f"    meas{mi} q={q} engine_p0={ep0}  dense_p0={dp0:.6f}  |Δ|={dd:.2e}{flag}")
    print(f"    worst |Δ| = {worst:.2e}  {'BUG REPRODUCED at engine level' if worst>1e-6 else 'engine matches dense'}")
