"""Near-Clifford simulator: Clifford frame (stabilizer tableau) + on-demand dense
magic register. Exploits the fact (measured exactly) that the coherent-error QEC
circuits have tiny anticommutation rank k -- so only ~k qubits ever need to be
"magic" (dense), the rest stay stabilizer (free in the frame).

State representation:
    |psi> = U_C ( (x)_{i not in M} |0>_i  (x)  |phi>_M )
  * U_C  : a Clifford, tracked as a tableau of the images
           Xc[i] = U_C X_i U_C^dag,  Zc[i] = U_C Z_i U_C^dag   (Paulis with phase)
  * M    : the set of "magic" qubits (those taken out of |0> by a rotation)
  * |phi>_M : a dense complex vector over the magic qubits (dim 2^|M|)

Gates:
  Clifford G  -> conjugate the tableau (free; M, |phi> untouched)
  rotation exp(-i theta P /2) on logical P:
       P' = U_C^dag P U_C   (pull back through the frame, via the tableau)
       on |0> qubits P' must act as I/Z (else promote that qubit into M);
       then apply exp(-i theta P'_M /2) to |phi>_M  (dense, 2^|M|).
  measure Z_q -> measure Pauli  Pm = U_C^dag Z_q U_C  on  (|0>(x)|phi>):
       reduces to a Pauli measurement on a product (stabilizer (x) magic) state.

Everything is phase-exact, verified against a dense statevector reference.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Pauli over n qubits: (x, z) bit-masks + phase p in {0,1,2,3} meaning i^p.
# Operator = i^p * prod_q  X_q^{x_q} Z_q^{z_q}   (X before Z per qubit).
# ---------------------------------------------------------------------------
def pauli_mul(a, b):
    """(xa,za,pa)*(xb,zb,pb). Returns (x,z,p). Uses X^x Z^z convention."""
    xa, za, pa = a
    xb, zb, pb = b
    x = xa ^ xb
    z = za ^ zb
    # phase from moving Z_a past X_b: Z X = -X Z, i.e. each (za & xb) bit gives -1=i^2
    p = (pa + pb + 2 * (za & xb).bit_count()) & 3
    return (x, z, p)


def pauli_commute(a, b):
    """True if a,b commute (symplectic inner product even)."""
    xa, za, _ = a
    xb, zb, _ = b
    return (((xa & zb).bit_count() + (za & xb).bit_count()) & 1) == 0


# ---------------------------------------------------------------------------
# Pauli conjugations  P -> G P G^dag  (tuples over all qubits).  Used by the
# incremental inverse-frame (each rule exhaustively verified vs _pullback in
# scripts/phase2_invframe_derive.py on n=1..4).
# ---------------------------------------------------------------------------
def _pconj_h(P, q):
    x, z, p = P
    xq = (x >> q) & 1; zq = (z >> q) & 1; b = 1 << q
    return ((x & ~b) | (zq << q), (z & ~b) | (xq << q), (p + 2 * (xq & zq)) & 3)


def _pconj_s(P, q, dag):
    x, z, p = P
    xq = (x >> q) & 1
    return (x, z ^ (xq << q), (p + xq * (3 if dag else 1)) & 3)


def _pconj_cx(P, c, t):
    x, z, p = P
    xc = (x >> c) & 1; zt = (z >> t) & 1; bc = 1 << c; bt = 1 << t
    x2 = (x & ~bt) | ((((x >> t) & 1) ^ xc) << t)        # X_c -> X_c X_t
    z2 = (z & ~bc) | ((((z >> c) & 1) ^ zt) << c)        # Z_t -> Z_c Z_t
    return (x2, z2, p)


def _pconj_x(P, q):
    x, z, p = P
    return (x, z, (p + 2 * ((z >> q) & 1)) & 3)          # X Z X = -Z


class NearClifford:
    def __init__(self, n):
        self.n = n
        # tableau: images of X_i, Z_i under U_C
        self.Xc = [(1 << i, 0, 0) for i in range(n)]
        self.Zc = [(0, 1 << i, 0) for i in range(n)]
        self.M = []                 # ordered list of magic qubits
        self.phi = np.array([1.0 + 0j])   # dense over M (initially empty -> scalar 1)
        self.rng = np.random.default_rng(0)
        # frame-change version (bumped by every Xc/Zc mutation) + pullback-basis cache:
        # the GF(2) elimination of the frame columns depends only on the frame, so it is
        # reused across _pullback calls until a Clifford gate changes the frame.
        self._frame_ver = 0
        self._pb_cache = None
        # ---- incremental inverse-frame (shadow): Ax[i]=U_C^dag X_i U_C, Az[i]=U_C^dag Z_i U_C.
        # Maintained O(1)/gate (forward) or O(n)/gate (right-fold) so _pullback is an O(1) lookup
        # instead of an O(n^2) GF(2) recompute.  OFF by default; clifft_axis enables it. A frame
        # mutation we do not have an incremental rule for (e.g. _ag_measure projection) sets
        # _inv_dirty, and the next pullback rebuilds the images ONCE from the basis method.
        self._inv_enabled = False
        self._inv_verify = False           # cross-check every pullback vs the basis (test mode)
        self._inv_dirty = False
        self._inv_ax = [(1 << i, 0, 0) for i in range(n)]
        self._inv_az = [(0, 1 << i, 0) for i in range(n)]
        self._inv_recompute = 0            # full rebuilds (should be ~ #stabilizer-measurements)
        self._inv_update = 0               # incremental gate updates
        self._inv_lookup = 0               # O(1) pullback lookups

    # ---- Clifford gates: conjugate the tableau (U_C -> G U_C) ----
    # new image of P_i = G (old image) G^dag. We update by applying G's action to
    # every stored Pauli's q-th components.
    def _apply_clifford_to_all(self, fn):
        for i in range(self.n):
            self.Xc[i] = fn(self.Xc[i])
            self.Zc[i] = fn(self.Zc[i])
        self._frame_ver += 1

    def h(self, q):
        bit = 1 << q
        def fn(P):
            x, z, p = P
            xq = (x >> q) & 1; zq = (z >> q) & 1
            # H X H = Z, H Z H = X, H Y H = -Y
            x2 = (x & ~bit) | (zq << q)
            z2 = (z & ~bit) | (xq << q)
            p2 = (p + 2 * (xq & zq)) & 3      # Y -> -Y
            return (x2, z2, p2)
        self._apply_clifford_to_all(fn)
        if self._inv_enabled:
            self._inv_fwd_h(q)

    def s(self, q, dag=False):
        bit = 1 << q
        def fn(P):
            x, z, p = P
            xq = (x >> q) & 1; zq = (z >> q) & 1
            # S X S^dag = Y, S Z S^dag = Z ; S^dag X S = -Y
            z2 = z ^ (xq << q)               # X gains Z (X->Y=iXZ)
            p2 = (p + (xq * (1 if not dag else 3))) & 3   # +i for S, -i(=+3) for Sdag
            return (x, z2, p2)
        self._apply_clifford_to_all(fn)
        if self._inv_enabled:
            self._inv_fwd_s(q, dag)

    def cx(self, c, t):
        bc = 1 << c; bt = 1 << t
        def fn(P):
            x, z, p = P
            xc = (x >> c) & 1; xt = (x >> t) & 1
            zc = (z >> c) & 1; zt = (z >> t) & 1
            # CX: X_c->X_cX_t, Z_t->Z_cZ_t  (X_t,Z_c unchanged)
            x2 = x | (xc << t) if xc else x
            x2 = (x2 & ~bt) | (((xt ^ xc) & 1) << t)
            z2 = (z & ~bc) | (((zc ^ zt) & 1) << c)
            return (x2, z2, p)
        self._apply_clifford_to_all(fn)
        if self._inv_enabled:
            self._inv_fwd_cx(c, t)

    def cz(self, a, b):
        self.h(b); self.cx(a, b); self.h(b)

    # ---- RIGHT-multiplication: U_C <- U_C G (conjugate the BASE, not the state) ----
    # The forward gate methods above do U_C <- G U_C (left-mult: conjugate every
    # image by G). For folding a base-frame Clifford back out (block_magic's measured
    # -magic purge: |psi> = U_C base = (U_C W^dag)(W base)), we need U_C <- U_C G.
    # New image of X_i = U_C (G X_i G^dag) U_C^dag = forward-image of (G X_i G^dag).
    # G acts on few qubits, so this is a recombination of the stored image COLUMNS
    # (Xc[i]/Zc[i]) via pauli_mul -- O(n) per gate, zero floating point.
    def right_h(self, s):
        """U_C <- U_C H_s. H X_s H = Z_s, H Z_s H = X_s -> swap the X/Z image columns."""
        self.Xc[s], self.Zc[s] = self.Zc[s], self.Xc[s]
        self._frame_ver += 1
        if self._inv_enabled:
            self._inv_right(lambda P: _pconj_h(P, s))

    def right_s(self, s, dag=False):
        """U_C <- U_C S_s (or S_s^dag). S X_s S^dag = Y_s, S Z_s S^dag = Z_s, so only
        the X-column changes: Xc[s] <- image(Y_s) = i^(+/-1) * Xc[s]*Zc[s]
        (Y = i XZ for S; -Y for S^dag)."""
        m = pauli_mul(self.Xc[s], self.Zc[s])      # image(X_s Z_s)
        self.Xc[s] = (m[0], m[1], (m[2] + (3 if dag else 1)) & 3)
        self._frame_ver += 1
        if self._inv_enabled:
            self._inv_right(lambda P: _pconj_s(P, s, not dag))

    def right_cx(self, c, t):
        """U_C <- U_C CNOT(c,t). CNOT X_c CNOT = X_c X_t and CNOT Z_t CNOT = Z_c Z_t
        (X_t, Z_c fixed) -> Xc[c] *= Xc[t], Zc[t] *= Zc[c]."""
        self.Xc[c] = pauli_mul(self.Xc[c], self.Xc[t])
        self.Zc[t] = pauli_mul(self.Zc[c], self.Zc[t])
        self._frame_ver += 1
        if self._inv_enabled:
            self._inv_right(lambda P: _pconj_cx(P, c, t))

    # ---- pull a logical Pauli P back through the frame: P' = U_C^dag P U_C ----
    # If true-P = prod over set bits of (X_i, Z_i), then U_C^dag (true-P) U_C is
    # the product of the corresponding *generators* X_i, Z_i in tableau-space...
    # but we store U_C X_i U_C^dag (forward). For pullback we need U_C^dag P U_C.
    # We instead express P in the frame basis: since {Xc[i],Zc[i]} are the images
    # of the basis, and they generate the Pauli group, decompose P over them.
    # Simpler: maintain forward tableau and invert by solving the symplectic system.
    # For our use (P = Z_q single logical), pullback Z_q = the Pauli M s.t.
    #   U_C M U_C^dag = Z_q  ->  M = U_C^dag Z_q U_C. We get M by expressing Z_q in
    #   terms of {Xc[i],Zc[i]} and reading the coefficients as M's (x,z).
    def _pullback_basis(self):
        """GF(2) elimination basis of the frame columns (Xc|Zc), each as a 2n-bit
        vector (x bits then z bits). Depends ONLY on the frame, so it is cached and
        rebuilt only when a Clifford gate bumps `_frame_ver`."""
        c = self._pb_cache
        if c is not None and c[0] == self._frame_ver:
            return c[1]
        n = self.n
        cvec = [self.Xc[i][0] | (self.Xc[i][1] << n) for i in range(n)]
        cvec += [self.Zc[i][0] | (self.Zc[i][1] << n) for i in range(n)]
        basis = []           # (pivotbit, vec, coeffmask)
        for j, v in enumerate(cvec):
            cur = v; cm = 1 << j
            for (pb, bv, bcm) in basis:
                if (cur >> pb) & 1:
                    cur ^= bv; cm ^= bcm
            if cur:
                pb = (cur & -cur).bit_length() - 1
                basis.append((pb, cur, cm))
        self._pb_cache = (self._frame_ver, basis)
        return basis

    # ------------------------------------------------------------------ #
    #  Incremental inverse-frame: O(1) pullback lookup (no GF(2) recompute) #
    # ------------------------------------------------------------------ #
    def _inv_rebuild(self):
        """Rebuild the inverse images from the basis method (the ONLY full recompute; happens
        after a mutation with no incremental rule, e.g. _ag_measure)."""
        self._inv_recompute += 1
        self._inv_ax = [self._pullback_via_basis(1 << i, 0) for i in range(self.n)]
        self._inv_az = [self._pullback_via_basis(0, 1 << i) for i in range(self.n)]
        self._inv_dirty = False

    def _inv_subst(self, x, z, p=0):
        """U_C^dag P U_C via X_j->Ax[j], Z_j->Az[j] (an O(weight) product of stored images)."""
        out = (0, 0, p)
        ax = self._inv_ax; az = self._inv_az
        xi = x
        while xi:
            j = (xi & -xi).bit_length() - 1; xi &= xi - 1
            out = pauli_mul(out, ax[j])
        zi = z
        while zi:
            j = (zi & -zi).bit_length() - 1; zi &= zi - 1
            out = pauli_mul(out, az[j])
        return out

    # ---- incremental inverse-frame updates per mutation (rules verified in derive script) ----
    def _inv_fwd_h(self, q):                            # forward H_q: U_C -> H_q U_C
        self._inv_ax[q], self._inv_az[q] = self._inv_az[q], self._inv_ax[q]
        self._inv_update += 1

    def _inv_fwd_s(self, q, dag):                       # forward S_q^(dag): only Ax[q] changes
        Q = _pconj_s((1 << q, 0, 0), q, not dag)        # G^dag X_q G with G = S^(dag)
        self._inv_ax[q] = self._inv_subst(Q[0], Q[1], Q[2])
        self._inv_update += 1

    def _inv_fwd_cx(self, c, t):                        # forward CX(c,t)
        a = pauli_mul(self._inv_ax[c], self._inv_ax[t])
        b = pauli_mul(self._inv_az[c], self._inv_az[t])
        self._inv_ax[c] = a; self._inv_az[t] = b
        self._inv_update += 1

    def _inv_right(self, fn):                           # right-fold: conjugate every image by G^dag
        self._inv_ax = [fn(P) for P in self._inv_ax]
        self._inv_az = [fn(P) for P in self._inv_az]
        self._inv_update += 1

    def _inv_fold_x(self, q):                           # Pauli fold U_C -> U_C X_q
        self._inv_right(lambda P: _pconj_x(P, q))

    def _pullback(self, x, z):
        """P' = U_C^dag P U_C as (x',z',phase) for logical P=(x,z,0).  Uses the incremental
        inverse-frame (O(weight) lookup) when enabled; else the GF(2) basis method.  In verify
        mode the lookup is cross-checked against the basis method on every call."""
        if self._inv_enabled:
            if self._inv_dirty:
                self._inv_rebuild()
            self._inv_lookup += 1
            res = self._inv_subst(x, z)
            if self._inv_verify:
                truth = self._pullback_via_basis(x, z)
                if res != truth:
                    raise AssertionError(
                        f"inverse-frame pullback({x},{z})={res} != basis {truth} "
                        f"(frame_ver {self._frame_ver})")
            return res
        return self._pullback_via_basis(x, z)

    def _pullback_via_basis(self, x, z):
        """Return P' = U_C^dag P U_C as (x',z',phase) for logical P=(x,z,phase0=0).
        Solve for coefficients c s.t. P = prod_i Xc[i]^{ax_i} Zc[i]^{az_i}."""
        n = self.n
        basis = self._pullback_basis()         # cached per frame version
        target = x | (z << n)
        # reduce target
        curt = target; coeff = 0
        for (pb, bv, bcm) in basis:
            if (curt >> pb) & 1:
                curt ^= bv; coeff ^= bcm
        if curt != 0:
            raise RuntimeError("pullback: target not in Pauli span (bug)")
        # Phase-exact pullback. The chosen generators, multiplied as IMAGES
        # (Xc/Zc with their stored phases) give Q = i^qp (X^x Z^z); as COMPUTATIONAL
        # generators (X_i/Z_i, phase 0) give R = i^rp (X^x Z^z). Since the images
        # are U_C(computational) U_C^dag, one gets U_C^dag P U_C = i^(rp-qp)(X^x Z^z).
        Q = (0, 0, 0)   # product of images
        R = (0, 0, 0)   # product of computational generators (same order)
        for j in range(2 * n):
            if (coeff >> j) & 1:
                if j < n:
                    Q = pauli_mul(Q, self.Xc[j])
                    R = pauli_mul(R, (1 << j, 0, 0))
                else:
                    Q = pauli_mul(Q, self.Zc[j - n])
                    R = pauli_mul(R, (0, 1 << (j - n), 0))
        return (R[0], R[1], (R[2] - Q[2]) & 3)

    # ---- promote a |0> qubit into the magic register ----
    def _promote(self, q):
        if q in self.M:
            return
        # new magic qubit appended as most-significant; it is in |0>
        self.M.append(q)
        self.phi = np.kron(np.array([1.0 + 0j, 0.0]), self.phi)  # |0> (x) old

    def _magic_index(self, q):
        return self.M.index(q)

    # ---- apply exp(-i theta P /2) for logical Pauli P=(x,z) ----
    def _apply_magic_pauli(self, xp, zp, pp):
        """Apply i^pp * (X^xp Z^zp) restricted to the magic register, DIRECTLY to
        phi (no dense matrix): O(2^|M|). Non-magic |0> qubits: Z->+1 (must have no X)."""
        k = len(self.M)
        mx = mz = 0
        for j, q in enumerate(self.M):
            if (xp >> q) & 1:
                mx |= 1 << j
            if (zp >> q) & 1:
                mz |= 1 << j
        idx = np.arange(1 << k, dtype=np.int64)
        v = idx & mz
        for sh in (32, 16, 8, 4, 2, 1):    # XOR-fold to get bit-parity of v
            v ^= v >> sh
        par = v & 1
        sign = (1j ** pp) * (1 - 2 * par)
        out = np.empty_like(self.phi)
        out[idx ^ mx] = sign * self.phi[idx]
        return out

    def apply_rotation(self, x, z, theta):
        Pp = self._pullback(x, z)
        xp, zp, pp = Pp
        # qubits where P' has X or Y (xp bit set) and are NOT magic must be promoted
        for q in range(self.n):
            if (xp >> q) & 1 and q not in self.M:
                self._promote(q)
        Pphi = self._apply_magic_pauli(xp, zp, pp)
        c = np.cos(theta / 2.0); s = np.sin(theta / 2.0)
        self.phi = c * self.phi - 1j * s * Pphi

    def _magic_pauli_matrix(self, xp, zp, pp, nonmagic_z_sign=True):
        """Dense 2^|M| matrix of i^pp * (X^xp Z^zp) restricted to magic qubits,
        with non-magic qubits in |0>: Z|0>=+|0> (so non-magic z bits give +1, and
        non-magic x bits should have been promoted already)."""
        k = len(self.M)
        dim = 1 << k
        # per magic qubit (ordered as in self.M, with M[0] = least significant in kron above)
        # our kron prepends new qubit as MOST significant; index bit for M[j]:
        # phi index bit layout: bit (k-1-j) corresponds to M[j]? We appended via
        # kron([|0>],old) => new qubit is MOST significant. So M[-1] is MSB.
        mats = []
        I2 = np.eye(2, dtype=complex)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        # phi index: bit j <-> M[j] (M[0]=LSB). kron(A,B): A is MSB. So build with
        # M[0] first (LSB) growing up to M[k-1] (MSB): op = kron(m_j, op) for j=0..k-1.
        op = np.array([[1.0 + 0j]])
        for j in range(k):
            q = self.M[j]
            xq = (xp >> q) & 1; zq = (zp >> q) & 1
            if xq and zq:
                m = Y  # XZ = -iY ... handle phase via pp separately; use X@Z
                m = X @ Z
            elif xq:
                m = X
            elif zq:
                m = Z
            else:
                m = I2
            op = np.kron(m, op)
        op = (1j ** pp) * op
        return op

    def _ag_measure(self, Pm, anti_s):
        """Gottesman-Knill measurement of Pauli Pm that anticommutes with non-magic
        stabilizers anti_s. Uniform-random outcome; update frame; magic untouched."""
        p = anti_s[0]
        out = int(self.rng.integers(0, 2))
        Sp = self.Zc[p]                      # pivot stabilizer
        # all OTHER rows (stab Zc[i] and destab Xc[i]) anticommuting with Pm: *= Sp
        for i in range(self.n):
            if i != p and not pauli_commute(self.Zc[i], Pm):
                self.Zc[i] = pauli_mul(self.Zc[i], Sp)
            if not pauli_commute(self.Xc[i], Pm):
                self.Xc[i] = pauli_mul(self.Xc[i], Sp)
        # pivot: old stabilizer becomes destabilizer; new stabilizer = (-1)^out Pm
        self.Xc[p] = Sp
        self.Zc[p] = (Pm[0], Pm[1], (Pm[2] + 2 * out) & 3)
        self._frame_ver += 1                 # frame changed -> invalidate pullback cache
        if self._inv_enabled:                # AG projection has no incremental rule -> lazy rebuild
            self._inv_dirty = True
        return out

    # ---- measure Z_q ; returns 0/1 outcome (samples), collapses state ----
    def measure_z(self, q):
        # Pure stabilizer measurement first (Gottesman-Knill) vs NON-magic
        # stabilizers Zc[i]; if Z_q anticommutes with one -> random, frame update,
        # NO magic promotion.
        Pm = (0, 1 << q, 0)
        magset = set(self.M)
        anti_s = [i for i in range(self.n)
                  if i not in magset and not pauli_commute(self.Zc[i], Pm)]
        if anti_s:
            return self._ag_measure(Pm, anti_s)
        # commutes with all non-magic stabilizers -> pull back (no non-magic X) and
        # measure on the magic register only.
        Pp = self._pullback(0, 1 << q)
        xp, zp, pp = Pp
        for qq in range(self.n):
            if (xp >> qq) & 1 and qq not in self.M:
                self._promote(qq)
        v = self._apply_magic_pauli(xp, zp, pp)
        # outcome operator eigenvalue +-1 ; projectors (I +- op)/2 on phi
        # <phi| op |phi>
        exp = np.real(np.vdot(self.phi, v))
        p0 = max(0.0, min(1.0, (1.0 + exp) / 2.0))
        r = float(self.rng.random())
        out = 0 if r < p0 else 1
        sign = 1.0 if out == 0 else -1.0
        proj = 0.5 * (self.phi + sign * v)
        nrm = np.linalg.norm(proj)
        if nrm > 1e-12:
            self.phi = proj / nrm
        self._compress_magic()
        return out

    def _compress_magic(self):
        """Remove magic qubits that are in a product |0>/|1> state (disentangled),
        keeping the magic register minimal. Idempotent."""
        changed = True
        while changed and self.M:
            changed = False
            k = len(self.M)
            phi = self.phi.reshape([2] * k)  # axis j (numpy) corresponds to M[k-1-j]?
            # phi index layout: flat index bit j <-> M[j] (M[0]=LSB). reshape([2]*k)
            # gives axis 0 = MSB = bit k-1 = M[k-1]. So numpy axis a <-> M[k-1-a].
            for a in range(k):
                q = self.M[k - 1 - a]
                sl0 = [slice(None)] * k; sl0[a] = 0
                sl1 = [slice(None)] * k; sl1[a] = 1
                b0 = phi[tuple(sl0)]; b1 = phi[tuple(sl1)]
                n0 = np.linalg.norm(b0); n1 = np.linalg.norm(b1)
                if n1 < 1e-10:          # qubit q is |0>
                    self.phi = b0.reshape(-1); self.M.pop(k - 1 - a); changed = True; break
                if n0 < 1e-10:          # qubit q is |1>
                    self.phi = b1.reshape(-1); self.M.pop(k - 1 - a); changed = True; break

    def statevector(self):
        """Dense full statevector (for verification only): U_C (|0..0> with magic phi)."""
        n = self.n
        # base state over n qubits: |0> except magic carry phi
        psi = np.zeros(1 << n, dtype=complex)
        for idx in range(len(self.phi)):
            full = 0
            for j, q in enumerate(self.M):       # M[0] LSB of phi
                if (idx >> j) & 1:
                    full |= (1 << q)
            psi[full] = self.phi[idx]
        # apply U_C: U_C|b> ; build U_C as product? Use the tableau to apply.
        # Easiest correct: U_C maps Z_i -> Zc[i]; the state |0..0>_logical is the
        # +1 eigenstate of all Z_i, so U_C|0> is the +1 eigenstate of all Zc[i].
        # For verification we instead reconstruct U_C as a matrix from the tableau.
        U = self._clifford_matrix()
        return U @ psi

    def _clifford_matrix(self):
        """Dense 2^n x 2^n Clifford matrix from the tableau (verification only)."""
        n = self.n
        dim = 1 << n
        # find U s.t. U X_i U^dag = Xc[i], U Z_i U^dag = Zc[i]. Build by stabilizer
        # state method: columns U|b>. Use that U|0> is +1 eigenstate of Zc[i].
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        I2 = np.eye(2, dtype=complex)

        def pauli_matrix(P):
            x, z, p = P
            op = np.array([[1.0 + 0j]])
            for q in range(n):
                xq = (x >> q) & 1; zq = (z >> q) & 1
                m = (X @ Z) if (xq and zq) else (X if xq else (Z if zq else I2))
                op = np.kron(m, op)
            return (1j ** p) * op
        # U|0> = projector onto +1 of all Zc[i], normalized
        v = np.zeros(dim, dtype=complex); v[0] = 1.0
        proj = np.eye(dim, dtype=complex)
        for i in range(n):
            Pi = pauli_matrix(self.Zc[i])
            proj = proj @ (0.5 * (np.eye(dim) + Pi))
        u0 = proj[:, 0]
        u0 = u0 / np.linalg.norm(u0)
        cols = [None] * dim
        cols[0] = u0
        # generate other columns by applying Xc[i] (which flips logical bit i)
        for b in range(1, dim):
            # find a set bit, build from lower
            lb = (b & -b).bit_length() - 1
            prev = b ^ (1 << lb)
            Xi = pauli_matrix(self.Xc[lb])
            cols[b] = Xi @ cols[prev]
        return np.column_stack(cols)
