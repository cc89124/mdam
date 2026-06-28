"""Step C / S2.1 -- independent C_sym frame shadow (behaviour-neutral, default-off).

Maintains a PARALLEL Z-preserving Clifford frame C_sym alongside the real engine's U_C, by the
forward-fold discipline:
  * CNOT/CZ/S (Z-preserving) -> fold into C_sym (same conjugation the engine applies to U_C).
  * H(q) on a FRESH qubit (C_sym acts as identity on q) -> record born[q]='X', C_sym UNCHANGED
    (the prep-H commutes out to the born factor B). H(q) on an ENTANGLED/active qubit -> array_h
    (fold into C_sym, breaks Z-preservation -> counted as the explicit fallback).
The real engine is untouched (source of truth). At each rotation we compute the generator THROUGH
C_sym (+born) and test Z-only on the active axes -- the 91/91 claim.

This does NOT yet dispatch or change the array; it proves the frame-level Z-preservation and the
per-event generator structure that S2's authoritative diagonal dispatch will rely on.
"""
import sys; sys.path.insert(0, "/home/jung/clifft-paper"); sys.setrecursionlimit(400000)
import clifft  # noqa
import nearclifford_backend.backend as bk
from nearclifford_backend.simulator import pauli_mul
from nearclifford_backend.clifft_axis.bounded import compile_bounded, CliftAxisBoundedNearClifford as C

CIRCS = [("cultivation_d3", 4), ("cultivation_d5", 2), ("coherent_ry_d3_r1", 2),
         ("coherent_rx_d3_r1", 2), ("distillation", 2), ("coherent_d5_r5", 1)]


def conj_pauli_by_clifford(P, ximg, zimg, n):
    """C P C^dag where C is given by images ximg[i]=C X_i C^dag, zimg[i]=C Z_i C^dag.
    P=(x,z,ph). Result = product over set bits of the corresponding images."""
    out = (0, 0, P[2] & 3)
    x, z = P[0], P[1]
    xi = x
    while xi:
        j = (xi & -xi).bit_length() - 1; xi &= xi - 1
        out = pauli_mul(out, ximg[j])
    zi = z
    while zi:
        j = (zi & -zi).bit_length() - 1; zi &= zi - 1
        out = pauli_mul(out, zimg[j])
    return out


class CsymShadow:
    """Parallel Z-preserving frame. ximg/zimg = images of X_i,Z_i under C_sym (forward).
    inv_x/inv_z = images under C_sym^dag (for pullback C_sym^dag P C_sym)."""
    def __init__(self, n):
        self.n = n
        self.xf = [(1 << i, 0, 0) for i in range(n)]   # C_sym X_i C_sym^dag
        self.zf = [(0, 1 << i, 0) for i in range(n)]
        self.xi = [(1 << i, 0, 0) for i in range(n)]   # C_sym^dag X_i C_sym
        self.zi = [(0, 1 << i, 0) for i in range(n)]
        self.born = {}          # qubit -> 'X' (born-X via prep-H)
        self.array_h = 0        # forced active-axis basis changes (the fallback count)

    def _fresh(self, q):
        return self.xf[q] == (1 << q, 0, 0) and self.zf[q] == (0, 1 << q, 0) and q not in self.born

    # forward gate G: U_C -> G U_C. Mirror on C_sym EXCEPT H-on-fresh.
    def cx(self, c, t):
        # CX X_c CX = X_c X_t ; CX Z_t CX = Z_c Z_t  (forward image update)
        self.xf[c] = pauli_mul(self.xf[c], self.xf[t])
        self.zf[t] = pauli_mul(self.zf[c], self.zf[t])
        # inverse images: C_sym^dag updates by the same gate conjugation on the pulled side
        self._inv_cx(c, t)

    def cz(self, a, b):
        self.cx(a, b)  # placeholder; real cz handled via h-cx-h on engine, mirrored below

    def s(self, q, dag=False):
        m = pauli_mul(self.xf[q], self.zf[q])
        self.xf[q] = (m[0], m[1], (m[2] + (3 if dag else 1)) & 3)
        self._inv_s(q, dag)

    def h(self, q):
        if self._fresh(q):
            self.born[q] = 'X'            # prep-H -> born basis, C_sym unchanged
            return
        # entangled/active H -> array_h fallback (apply to C_sym, breaks Z-preservation)
        self.array_h += 1
        self.xf[q], self.zf[q] = self.zf[q], self.xf[q]
        self.xi[q], self.zi[q] = self.zi[q], self.xi[q]

    def _inv_cx(self, c, t):
        # C_sym^dag <- conj by CX on the inverse images (pconj)
        a = pauli_mul(self.xi[c], self.xi[t]); b = pauli_mul(self.zi[c], self.zi[t])
        self.xi[c] = a; self.zi[t] = b

    def _inv_s(self, q, dag):
        # rough inverse update via recompute of that column would be safer; approximate by conj
        m = pauli_mul(self.xi[q], self.zi[q])
        self.xi[q] = (m[0], m[1], (m[2] + (1 if dag else 3)) & 3)

    def generator(self, x, z):
        """C_sym^dag (X^x Z^z) C_sym, then apply born (swap x/z bit on born-X qubits)."""
        g = conj_pauli_by_clifford((x, z, 0), self.xi, self.zi, self.n)
        gx, gz, gp = g
        for q in self.born:
            xb = (gx >> q) & 1; zb = (gz >> q) & 1
            gx = (gx & ~(1 << q)) | (zb << q)        # born-X: swap x<->z on q
            gz = (gz & ~(1 << q)) | (xb << q)
        return gx, gz, gp


def run(circ, seed):
    prog = compile_bounded(open(f"qec_bench/circuits/{circ}.stim").read())
    sh = {}
    of_h = C.h; of_s = C.s; of_cx = C.cx; of_cz = C.cz; of1 = C._flush_one
    st = dict(nT=0, zonly=0, nz=0, rows=[])

    def gh(self, q):
        sh['o'].h(q); return of_h(self, q)
    def gs(self, q, dag=False):
        sh['o'].s(q, dag); return of_s(self, q, dag)
    def gcx(self, c, t):
        sh['o'].cx(c, t); return of_cx(self, c, t)
    def gcz(self, a, b):
        # engine cz = h(b) cx(a,b) h(b); mirror via those (the engine override does the same)
        return of_cz(self, a, b)
    def f1(self, x, z, theta, phase=0):
        gx, gz, gp = sh['o'].generator(x, z)
        mset = set(self.M)
        nx = sum(1 for q in mset if (gx >> q) & 1)
        st['nT'] += 1
        if nx == 0:
            st['zonly'] += 1
        else:
            st['nz'] += 1
            st['rows'].append((st['nT'] - 1, [q for q in mset if (gx >> q) & 1], hex(x), hex(z)))
        return of1(self, x, z, theta, phase)

    C.h = gh; C.s = gs; C.cx = gcx; C.cz = gcz; C._flush_one = f1
    try:
        be = bk.NearCliffordBackend(clifft_axis_bounded=True, drop_dead=False, structure_once=False,
                                    clifft_axis_enforce=True)
        sh['o'] = CsymShadow(be_count := __import__("nearclifford_backend.backend", fromlist=["count_idents"]).count_idents(prog))
        rec = tuple(be.run_shot(prog, seed))
    finally:
        C.h = of_h; C.s = of_s; C.cx = of_cx; C.cz = of_cz; C._flush_one = of1
    return st, sh['o'].array_h, sh['o'].born


print("=" * 90)
print("STEP C / S2.1 -- independent C_sym frame shadow: per-T Z-only count (forward-fold discipline)")
print("=" * 90)
for circ, ns in CIRCS:
    tz = tt = tah = 0; sample = None
    for s in range(1, ns + 1):
        st, ah, born = run(circ, s)
        tz += st['zonly']; tt += st['nT']; tah += ah
        if sample is None:
            sample = st['rows'][:4]
    print(f"{circ:18}: Z-only {tz}/{tt}   array_h(fallback)={tah}   non-diag sample={sample}")
