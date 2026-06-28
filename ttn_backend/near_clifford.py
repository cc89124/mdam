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
    p = (pa + pb + 2 * (bin(za & xb).count("1"))) & 3
    return (x, z, p)


def pauli_commute(a, b):
    """True if a,b commute (symplectic inner product even)."""
    xa, za, _ = a
    xb, zb, _ = b
    return ((bin(xa & zb).count("1") + bin(za & xb).count("1")) & 1) == 0


class NearClifford:
    def __init__(self, n):
        self.n = n
        # tableau: images of X_i, Z_i under U_C
        self.Xc = [(1 << i, 0, 0) for i in range(n)]
        self.Zc = [(0, 1 << i, 0) for i in range(n)]
        self.M = []                 # ordered list of magic qubits
        self.phi = np.array([1.0 + 0j])   # dense over M (initially empty -> scalar 1)
        self.rng = np.random.default_rng(0)

    # ---- Clifford gates: conjugate the tableau (U_C -> G U_C) ----
    # new image of P_i = G (old image) G^dag. We update by applying G's action to
    # every stored Pauli's q-th components.
    def _apply_clifford_to_all(self, fn):
        for i in range(self.n):
            self.Xc[i] = fn(self.Xc[i])
            self.Zc[i] = fn(self.Zc[i])

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

    def cz(self, a, b):
        self.h(b); self.cx(a, b); self.h(b)

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
    def _pullback(self, x, z):
        """Return P' = U_C^dag P U_C as (x',z',phase) for logical P=(x,z,phase0=0).
        Solve for coefficients c s.t. P = prod_i Xc[i]^{ax_i} Zc[i]^{az_i}."""
        # Build 2n x 2n symplectic matrix whose columns are (Xc[i] | Zc[i]) in
        # (x|z) coordinates; solve for the combination giving target (x|z).
        n = self.n
        cols = []
        for i in range(n):
            cols.append((self.Xc[i][0], self.Xc[i][1]))
        for i in range(n):
            cols.append((self.Zc[i][0], self.Zc[i][1]))
        # GF(2) solve: find coeff bits b (len 2n) with sum b_j col_j = (x,z)
        rows = []   # each row: 2n-bit colmask folded? We do elimination over 2n eqs.
        # represent each column as a 2n-bit vector (x bits then z bits)
        cvec = []
        for (cx_, cz_) in cols:
            v = cx_ | (cz_ << n)
            cvec.append(v)
        target = x | (z << n)
        # Gaussian elimination to express target as XOR of cvec subset
        basis = []           # (pivotbit, vec, coeffmask)
        for j, v in enumerate(cvec):
            cur = v; cm = 1 << j
            for (pb, bv, bcm) in basis:
                if (cur >> pb) & 1:
                    cur ^= bv; cm ^= bcm
            if cur:
                pb = (cur & -cur).bit_length() - 1
                basis.append((pb, cur, cm))
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
