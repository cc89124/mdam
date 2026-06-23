"""Phase 2-5: validate a budget.charge() FLOP hook against DIRECT kernel-event instrumentation
on the clifft_axis_bounded engine, at ranks r=1..6, per kernel.  Neither meter changes engine
numerics (monkeypatch records at entry, then calls the original).  FLOP convention:
  complex mult=6, complex add/sub=2, real*complex (scale)=2, |z|^2 (sqmag)=4, vdot~ as derived.
Permutations (CNOT swap, drop-compaction, promote zero-fill) = 0 algorithmic FLOP -> memory only.

Exact per-kernel FLOP (derived from engine.py arithmetic; note alpha=cos/0.5 is REAL):
  lincomb offdiag : 12N   (2 outputs x (alpha*a[2] + sk*b[2] + bph*(.)[6] + add[2]) over N/2 pairs)
  lincomb diag/0  : 6N    (phi *= complex scalar)
  expectation     : 10N   (sgn*conj(g)*src then sum: 2+6+2)
  branch_sqnorm   : 2N    (4 * (N/2): one branch only -- charge passes full N, processes N/2)
  norm+renorm     : 6N    (4N norm + 2N divide; NOT charged by the engine -> modeled per collapse)
  h_axis          : 5N    (a+=b;b*=-2;b+=a;a*=v2;b*=v2 on the two N/2 halves)
  s_axis          : 3N    (v[:,1]*=+-i : complex mult on N/2)
  cnot/drop/promote: 0     (permutation / zero-fill -> traffic only)
"""
import sys
sys.path.insert(0, "/home/jung/clifft-paper")
sys.setrecursionlimit(400000)
import numpy as np
import nearclifford_backend.backend as bk
from nearclifford_backend.clifft_axis import budget as _bud
from nearclifford_backend.clifft_axis.engine import CliftAxisNearClifford as _Eng
from nearclifford_backend.clifft_axis.bounded import CliftAxisBoundedNearClifford as B


# ============================ DIRECT kernel-event meter ============================
class Direct:
    def __init__(self):
        self.flop = {}; self.R = {}; self.W = {}; self.cnt = {}
        self._saved = {}
    def add(self, k, flop, r, w):
        self.flop[k] = self.flop.get(k, 0.0) + flop
        self.R[k] = self.R.get(k, 0.0) + r
        self.W[k] = self.W.get(k, 0.0) + w
        self.cnt[k] = self.cnt.get(k, 0) + 1
    def total_flop(self): return sum(self.flop.values())

    def enable(self):
        D = self
        S = self._saved
        S['lc'] = _Eng._pauli_lincomb_inplace
        S['ex'] = _Eng._pauli_expectation
        S['bs'] = _Eng._branch_sqnorm
        S['ha'] = _Eng._h_axis
        S['sa'] = _Eng._s_axis
        S['ca'] = _Eng._cnot_axes
        S['ci'] = _Eng._cnot_inplace
        S['da'] = _Eng._drop_axis_inplace
        S['pr'] = _Eng._promote
        S['nm'] = np.linalg.norm

        def lc(self, mx, mz, pp, alpha, beta, where=""):
            N = self.phi.size
            if mx != 0:
                D.add('lincomb:offdiag', 12.0 * N, N, N)
            elif mz == 0:
                D.add('lincomb:diag0', 6.0 * N, N, N)
            else:
                D.add('lincomb:diag', 6.0 * N, N, N)
            return S['lc'](self, mx, mz, pp, alpha, beta, where)

        def ex(self, mx, mz, pp, where="exp"):
            N = self.phi.size
            D.add('expectation', 10.0 * N, (2 * N if mx else N), 0)
            return S['ex'](self, mx, mz, pp, where)

        def bs(self, j, branch):
            N = self.phi.size
            D.add('branch_sqnorm', 4.0 * (N // 2), N // 2, 0)
            return S['bs'](self, j, branch)

        def ha(self, j):
            N = self.phi.size
            D.add('purge:h', 5.0 * N, N, N)
            return S['ha'](self, j)

        def sa(self, j, dag):
            N = self.phi.size
            D.add('purge:s', 6.0 * (N // 2), N // 2, N // 2)   # complex mult on N/2 half
            return S['sa'](self, j, dag)

        def ca(self, jc, jt):
            N = self.phi.size
            D.add('purge:cnot(perm)', 0.0, N // 2, N // 2)
            return S['ca'](self, jc, jt)

        def ci(self, jc, jt):
            N = self.phi.size
            D.add('reduce:cnot(perm)', 0.0, N // 2, N // 2)
            return S['ci'](self, jc, jt)

        def da(self, j, fold_x_qubit=None):
            N = self.phi.size
            D.add('drop(perm)', 0.0, N // 2, N // 2)
            return S['da'](self, j, fold_x_qubit)

        def pr(self, q):
            N = self.phi.size
            D.add('promote(zero-fill)', 0.0, 0, N)             # zero-fill the new half
            return S['pr'](self, q)

        def nm(x, *a, **kw):
            try:
                n = int(np.asarray(x).size)
                if np.asarray(x).dtype.kind == 'c':
                    D.add('norm+renorm', 6.0 * n, n, n)        # 4N norm + 2N divide
            except Exception:
                pass
            return S['nm'](x, *a, **kw)

        _Eng._pauli_lincomb_inplace = lc
        _Eng._pauli_expectation = ex
        _Eng._branch_sqnorm = bs
        _Eng._h_axis = ha
        _Eng._s_axis = sa
        _Eng._cnot_axes = ca
        _Eng._cnot_inplace = ci
        _Eng._drop_axis_inplace = da
        _Eng._promote = pr
        np.linalg.norm = nm

    def disable(self):
        S = self._saved
        _Eng._pauli_lincomb_inplace = S['lc']
        _Eng._pauli_expectation = S['ex']
        _Eng._branch_sqnorm = S['bs']
        _Eng._h_axis = S['ha']
        _Eng._s_axis = S['sa']
        _Eng._cnot_axes = S['ca']
        _Eng._cnot_inplace = S['ci']
        _Eng._drop_axis_inplace = S['da']
        _Eng._promote = S['pr']
        np.linalg.norm = S['nm']


# ============================ budget.charge HOOK meter ============================
# corrected coeff per element OF THE CHARGED N (= phi.size at the charge), and the modeled
# post-collapse norm (engine does NOT charge it).
HOOK_COEFF = {
    'rot:offdiag': 12.0, 'rot:offdiag-scalar': 12.0, 'collapse:offdiag': 12.0,
    'rot:diag': 6.0, 'rot:diag0': 6.0, 'rot:diag-scalar': 6.0,
    'collapse:diag': 6.0, 'collapse:diag0': 6.0, 'collapse:diag-scalar': 6.0,
    'meas': 10.0, 'exp': 10.0, 'reduce:verify': 10.0,
    'sqnorm': 2.0,                 # charge passes full N; kernel processes N/2 -> 4*(N/2)/N = 2
    'purge:h': 5.0, 'purge:s': 3.0,            # s: 6*(N/2)/N = 3
    'purge:cnot': 0.0, 'reduce:cnot': 0.0, 'drop': 0.0, 'promote': 0.0,
    'reduce:gf2scan': 0.0,         # support scan (np.abs/compare) -> not counted as algorithmic FLOP
    'init': 0.0, 'post-reduce': 0.0,
}


class Hook:
    def __init__(self):
        self.flop = {}; self.cnt = {}
        self._orig = None
    def enable(self):
        self._orig = _bud.DenseMemoryBudget.charge
        H = self
        orig = self._orig
        def charge(self, resident, transient=0, where=""):
            coeff = HOOK_COEFF.get(where, 0.0)
            if coeff:
                H.flop[where] = H.flop.get(where, 0.0) + coeff * int(resident)
                H.cnt[where] = H.cnt.get(where, 0) + 1
            if where.startswith('collapse'):     # modeled post-collapse norm+renorm (6N, uncharged)
                H.flop['norm+renorm(modeled)'] = H.flop.get('norm+renorm(modeled)', 0.0) + 6.0 * int(resident)
            return orig(self, resident, transient, where)
        _bud.DenseMemoryBudget.charge = charge
    def disable(self):
        _bud.DenseMemoryBudget.charge = self._orig
    def total_flop(self): return sum(self.flop.values())


# ============================ micro driver: force rank r ============================
class _R:
    def __init__(s, b): s.b = b
    def random(s): return 0.0 if s.b == 0 else 1.0
    def integers(s, lo, hi): return s.b % hi

def eng_ry(e, q, th):
    e.s(q, dag=True); e.h(q); e.apply_rotation(0, 1 << q, th); e.h(q); e.s(q, dag=False)

def micro(r, seed_out=0):
    """r off-axis R_Y on r qubits, CX chain to entangle, measure q0 (forced) -> rank-r core."""
    e = B(r); e.set_clifft_budget(r + 2, enforce=False)
    for q in range(r):
        eng_ry(e, q, 0.2 + 0.03 * q)
    for q in range(r - 1):
        e.cx(q, q + 1)
    for q in range(r):
        eng_ry(e, q, 0.1 + 0.02 * q)
    e.rng = _R(seed_out)
    e.measure_z(0)
    return e


def micro_magic(r):
    """Force the MAGIC-Born path (expectation/collapse/norm/promote/drop) via the
    _complex_engine recipe that provably hits it: layers of R_Y + CX/CZ + H + R_Y, then
    measure all qubits with forced outcomes."""
    rng = np.random.default_rng(1234 + r)
    e = B(r); e.set_clifft_budget(r + 4, enforce=False)
    TH = 0.0628
    for layer in range(3):
        for q in range(r):
            eng_ry(e, q, TH * (1 if rng.random() < 0.5 else -1))
        pairs = list(range(r)); rng.shuffle(pairs)
        for k in range(0, r - 1, 2):
            a, b = pairs[k], pairs[k + 1]
            (e.cx if rng.random() < 0.5 else e.cz)(a, b)
        for q in range(r):
            if rng.random() < 0.5:
                e.h(q)
        for q in range(r):
            eng_ry(e, q, TH * (1 if rng.random() < 0.5 else -1))
    for q in range(r):
        e.rng = _R(int(rng.random() < 0.5))
        e.measure_z(q)
    return e


if __name__ == "__main__":
    print("=== Phase 4: DIRECT vs HOOK FLOP cross-validation, ranks r=1..6 ===")
    print(f"{'r':>2} {'direct FLOP':>13} {'hook FLOP':>13} {'match?':>8}   mismatched kernels")
    allmatch = True
    for r in range(1, 7):
        D = Direct(); Hk = Hook()
        D.enable(); Hk.enable()
        try:
            micro(r, 0)
        finally:
            Hk.disable(); D.disable()
        df = D.total_flop(); hf = Hk.total_flop()
        ok = abs(df - hf) < 1e-6 * max(df, 1.0)
        allmatch &= ok
        print(f"{r:>2} {df:>13.0f} {hf:>13.0f} {'YES' if ok else 'NO':>8}   "
              f"{'' if ok else f'(d-h={df-hf:+.0f})'}")
    # ---- magic-Born micro: force expectation/collapse/norm/promote/drop/reduce ----
    print(f"\n=== MAGIC-Born micro (force expectation/collapse/norm/promote/drop), r=2..5 ===")
    print(f"{'r':>2} {'direct FLOP':>13} {'hook FLOP':>13} {'match?':>8}")
    magicmatch = True
    magic_kernels = set()
    for r in range(2, 6):
        D = Direct(); Hk = Hook(); D.enable(); Hk.enable()
        try: micro_magic(r)
        finally: Hk.disable(); D.disable()
        magic_kernels |= set(D.flop)
        df, hf = D.total_flop(), Hk.total_flop()
        ok = abs(df - hf) < 1e-6 * max(df, 1.0)
        magicmatch &= ok
        print(f"{r:>2} {df:>13.0f} {hf:>13.0f} {'YES' if ok else 'NO':>8}"
              f"{'' if ok else f'  d-h={df-hf:+.0f}'}")
    print(f"magic-path kernels exercised: {sorted(magic_kernels)}")

    # ---- comprehensive: REAL circuits (exercise expectation/collapse/norm/diag/purge/cnot/drop) ----
    from nearclifford_backend.clifft_axis.bounded import compile_bounded
    print(f"\n=== REAL-circuit DIRECT vs HOOK cross-validation (all kernels) ===")
    print(f"{'circuit':20}{'direct FLOP':>14}{'hook FLOP':>14}{'match?':>8}")
    realmatch = True
    seen_kernels = set()
    for c in ['coherent_ry_d3_r1', 'coherent_rx_d3_r1', 'coherent_d3_r3',
              'cultivation_d3', 'distillation']:
        prog = compile_bounded(open(f'qec_bench/circuits/{c}.stim').read())
        D = Direct(); Hk = Hook(); D.enable(); Hk.enable()
        try:
            be = bk.NearCliffordBackend(clifft_axis_bounded=True, drop_dead=False,
                                        structure_once=False, clifft_axis_enforce=True)
            be.run_shot(prog, 1)
        finally:
            Hk.disable(); D.disable()
        seen_kernels |= set(D.flop)
        df, hf = D.total_flop(), Hk.total_flop()
        ok = abs(df - hf) < 1e-6 * max(df, 1.0)
        realmatch &= ok
        print(f"{c:20}{df:>14.0f}{hf:>14.0f}{'YES' if ok else 'NO':>8}"
              f"{'' if ok else f'  d-h={df-hf:+.0f}'}")
    print(f"\nkernels exercised across reals: {sorted(seen_kernels)}")
    # per-kernel breakdown on coherent_ry_d3_r1 (the corrected R_Y)
    prog = compile_bounded(open('qec_bench/circuits/coherent_ry_d3_r1.stim').read())
    D = Direct(); Hk = Hook(); D.enable(); Hk.enable()
    try:
        bk.NearCliffordBackend(clifft_axis_bounded=True, drop_dead=False,
                               structure_once=False, clifft_axis_enforce=True).run_shot(prog, 1)
    finally:
        Hk.disable(); D.disable()
    print(f"\n=== per-kernel @ coherent_ry_d3_r1 (DIRECT) ===")
    for k in sorted(D.flop):
        print(f"  {k:22} cnt={D.cnt[k]:>4} FLOP={D.flop[k]:>12.0f}  R={D.R[k]:>10.0f} W={D.W[k]:>10.0f}")
    print(f"  {'TOTAL FLOP':22}            {D.total_flop():>12.0f}   "
          f"bytes moved={16*(sum(D.R.values())+sum(D.W.values())):.3e}")

    print(f"\nMICRO ranks: {'MATCH' if allmatch else 'MISMATCH'}    "
          f"REAL circuits: {'MATCH' if realmatch else 'MISMATCH'}")
    print("VERDICT:", "budget-hook VALIDATED vs direct events"
          if (allmatch and realmatch) else "MISMATCH -> hook needs fixing")
