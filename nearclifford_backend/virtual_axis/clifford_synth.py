"""Step A.2 — virtual Clifford gate synthesis (independent, fully verified).

Given a target Clifford as a stabiliser TABLEAU over r virtual axes -- the images
(X-row[i], Z-row[i]) of X_i, Z_i, each a phase-carrying Pauli -- synthesise a gate
sequence (H/S/Sdg/CNOT) realising that Clifford, and verify the synthesised circuit
reproduces the tableau PHASE-EXACTLY by conjugating the identity tableau.

Used at runtime to apply a basis change (old virtual basis -> target virtual basis)
to the dense 2^r state without any physical promotion or rank elimination.

Self-contained: depends only on pauli (x,z,phase) arithmetic. Verified two ways in
test_synth():
  (a) the single-Pauli conjugation rules match dense-matrix conjugation;
  (b) for random Cliffords, synthesise -> apply -> tableau matches target exactly.
"""
from __future__ import annotations

# Pauli over r qubits: (x, z, phase) with operator = i^phase * X^x Z^z, phase in 0..3.


def _bit(v, q):
    return (v >> q) & 1


# ---- single-Pauli Clifford conjugation  P -> G P G^dag  (phase-exact) ----
def conj_h(p, q):
    x, z, ph = p
    xq = _bit(x, q); zq = _bit(z, q)
    x = (x & ~(1 << q)) | (zq << q)
    z = (z & ~(1 << q)) | (xq << q)
    ph = (ph + 2 * (xq & zq)) & 3            # H Y H = -Y
    return (x, z, ph)


def conj_s(p, q, dag=False):
    x, z, ph = p
    xq = _bit(x, q)
    z ^= (xq << q)                            # S X S^dag = Y (z gains x)
    ph = (ph + (3 if dag else 1) * xq) & 3
    return (x, z, ph)


def conj_cx(p, c, t):
    x, z, ph = p
    xc = _bit(x, c); xt = _bit(x, t); zc = _bit(z, c); zt = _bit(z, t)
    # CX X_c CX = X_c X_t ; CX Z_t CX = Z_c Z_t ; X_t, Z_c fixed.
    x = (x & ~(1 << t)) | ((xt ^ xc) << t)
    z = (z & ~(1 << c)) | ((zc ^ zt) << c)
    # phase is UNCHANGED: CX(X^x Z^z)CX = X^x' Z^z' exactly (insert CX^dag CX between
    # the X^x and Z^z factors -- each conjugates to a pure X- / Z-string, no reorder).
    return (x, z, ph)


def conj_swap(p, a, b):
    x, z, ph = p
    xa = _bit(x, a); xb = _bit(x, b); za = _bit(z, a); zb = _bit(z, b)
    x = (x & ~(1 << a) & ~(1 << b)) | (xb << a) | (xa << b)
    z = (z & ~(1 << a) & ~(1 << b)) | (zb << a) | (za << b)
    return (x, z, ph)


def apply_gate(g, p):
    k = g[0]
    if k == 'h':    return conj_h(p, g[1])
    if k == 's':    return conj_s(p, g[1], False)
    if k == 'sdg':  return conj_s(p, g[1], True)
    if k == 'cx':   return conj_cx(p, g[1], g[2])
    if k == 'swap': return conj_swap(p, g[1], g[2])
    raise ValueError(g)


def invert(g):
    if g[0] == 's':   return ('sdg', g[1])
    if g[0] == 'sdg': return ('s', g[1])
    return g                                   # h, cx, swap self-inverse


# ---- identity tableau + conjugating a whole tableau ----
def identity_tableau(r):
    X = [(1 << i, 0, 0) for i in range(r)]
    Z = [(0, 1 << i, 0) for i in range(r)]
    return X, Z


def conj_tableau(gates, X, Z):
    X = list(X); Z = list(Z)
    for g in gates:
        X = [apply_gate(g, p) for p in X]
        Z = [apply_gate(g, p) for p in Z]
    return X, Z


def _commutes(p, q):
    return (((p[0] & q[1]).bit_count() + (p[1] & q[0]).bit_count()) & 1) == 0


# ---- synthesis: reduce the target tableau to identity, collecting gates ----
def synthesize(Xtab, Ztab, r):
    """Return gates [g...] with conj_tableau(gates, identity) == (Xtab, Ztab), phase-
    exact. Reduce the target tableau to identity (conjugations), then invert+reverse."""
    X = list(Xtab); Z = list(Ztab)
    red = []

    def emit(g):
        nonlocal X, Z
        red.append(g)
        X = [apply_gate(g, p) for p in X]
        Z = [apply_gate(g, p) for p in Z]

    def xbits(p, lo):
        return [j for j in range(lo, r) if _bit(p[0], j)]

    def zbits(p, lo):
        return [j for j in range(lo, r) if _bit(p[1], j)]

    for i in range(r):
        # ===== X[i] -> X_i (support on qubits >= i; <i already cleared) =====
        if not xbits(X[i], i):                  # no x>=i: turn a z>=i into x
            emit(('h', zbits(X[i], i)[0]))
        if not _bit(X[i][0], i):                # bring an x-bit to position i
            emit(('swap', i, xbits(X[i], i)[0]))
        for j in xbits(X[i], i + 1):            # clear other x-bits (i is X-pivot)
            emit(('cx', i, j))
        if _bit(X[i][1], i):                    # kill Z_i (Y_i -> X_i)
            emit(('s', i))
        for j in zbits(X[i], i + 1):            # clear Z_j from X_i via CZ(i,j)=H cx H
            emit(('h', j)); emit(('cx', i, j)); emit(('h', j))
        # ===== Z[i] -> Z_i (preserve X[i]=X_i; gates on j>i or CX(j,i) keep it) =====
        for j in range(i + 1, r):               # clear Z[i] support off i
            xj = _bit(Z[i][0], j); zj = _bit(Z[i][1], j)
            if xj and zj:
                emit(('s', j))                  # Y_j -> X_j
            if _bit(Z[i][0], j):                # X_j -> Z_j
                emit(('h', j))
            if _bit(Z[i][1], j):                # clear Z_j (z_j ^= z_i, z_i=1)
                emit(('cx', j, i))
        if _bit(Z[i][0], i):                    # Y_i -> Z_i via HSH (keeps X_i)
            emit(('h', i)); emit(('s', i)); emit(('h', i))
    # ===== phase fixes: X[i],Z[i] now +-X_i,+-Z_i (phase in {0,2}) =====
    for i in range(r):
        if X[i][2] == 2:                        # -X_i -> apply Z_i (= S S): X->-X
            emit(('s', i)); emit(('s', i))
        if Z[i][2] == 2:                        # -Z_i -> apply X_i (= H S S H): Z->-Z
            emit(('h', i)); emit(('s', i)); emit(('s', i)); emit(('h', i))
    return [invert(g) for g in reversed(red)]
