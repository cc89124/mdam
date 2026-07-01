"""Step C-3: streaming virtual-axis engine for the magic register (active-rank).

A CHP-style symplectic frame over the magic register's qubits -- n rows, each a
stabiliser/destabiliser Pauli pair (phase-tracked) -- with a MAGIC subset whose rows are
DENSE axes (a 2^k vector `phi`). Non-magic rows are |0> stabilisers. This is "clifft for
the magic register": a pulled-back rotation that opens a genuinely NEW magic direction
PROMOTES one stabiliser row to a dense axis (k+1); a rotation expressible in the existing
axes + stabilisers does NOT promote (no redundant growth); a measurement projects `phi`
and (C-3b) drops the collapsed axis. So k tracks the genuine independent rank, never the
raw physical support B.

C-3a (this file, verified by test_c3.py): promotion + rotation + in-place measurement +
|0>/|1> compress -- STATE-EXACT vs a dense reference. C-3b adds the synthesised
measurement basis change so the measured axis collapses to |0>/|1> and drops (minimal k).
"""
from __future__ import annotations

import numpy as np

from mdam.backend.simulator import pauli_mul, pauli_commute
from mdam.backend.block_magic import _apply_pauli_local, _vec_cx, _gf2_solve
from mdam.backend.virtual_axis.virtual_axis import _herm
from mdam.backend.virtual_axis import flop_meter as _fm

_I = (0, 0, 0)


def _symp(a, b):
    return 0 if pauli_commute(a, b) else 1


class TableauEngine:
    def __init__(self, n):
        self.n = n
        self.stab = [(0, 1 << j, 0) for j in range(n)]      # Z_j  (axis Z / |0> stabiliser)
        self.destab = [(1 << j, 0, 0) for j in range(n)]    # X_j  (axis X / destabiliser)
        self.magic = []                                     # row ids that are dense axes
        self.phi = np.array([1.0 + 0j])                     # dense over axes (order = magic)
        self.max_k = 0                                      # transient peak (before drop)
        self.max_k_res = 0                                  # resident peak (after reduction)
        self.promote_calls = 0                              # forbidden-op audit (C-4)
        self.reduce_parities = False                        # C-4: drop parity-slaved axes

    # ---- helpers -----------------------------------------------------------
    def _magicset(self):
        return set(self.magic)

    def _stab_decompose(self, R):
        """Express R (commuting with all magic axes) as a product of STABILISER rows;
        return (rows, ok). ok=False means R has a non-stabiliser (new-direction) part."""
        n = self.n
        target = R[0] | (R[1] << n)
        rows = []
        cur = target
        # GF(2) elimination over the stabiliser vectors (non-magic AND magic axis-Z's are
        # stabilisers of the |0>-along-axis state too, but axis rows are dense -> exclude;
        # only NON-magic stab rows are genuine stabilisers). Use non-magic stabs.
        basis = []                                          # (pivotbit, vec, rowid)
        ms = self._magicset()
        for row in range(n):
            if row in ms:
                continue
            v = self.stab[row][0] | (self.stab[row][1] << n)
            cm = [row]
            cv = v
            for (pb, bv, brows) in basis:
                if (cv >> pb) & 1:
                    cv ^= bv
                    cm = cm + brows
            if cv:
                basis.append(((cv & -cv).bit_length() - 1, cv, cm))
        # reduce target
        used = []
        cv = cur
        for (pb, bv, brows) in basis:
            if (cv >> pb) & 1:
                cv ^= bv
                used = used + brows
        return used, (cv == 0)

    def _express(self, P):
        """Reduce P over the magic axes. Returns (mx, mz, Acc, Bcc, R):
        mx/mz = virtual mask bits, Acc/Bcc = physical/virtual generator products (for the
        exact phase), R = residual (commutes with all magic axes)."""
        mx = mz = 0
        Acc = _I
        Bcc = _I
        R = P
        for i, row in enumerate(self.magic):
            xi = _symp(P, self.stab[row])       # X_i component (anticommutes with axis Z)
            zi = _symp(P, self.destab[row])     # Z_i component (anticommutes with axis X)
            if xi:
                mx |= 1 << i
                Acc = pauli_mul(Acc, self.destab[row])
                Bcc = pauli_mul(Bcc, (1 << i, 0, 0))
                R = pauli_mul(R, self.destab[row])
            if zi:
                mz |= 1 << i
                Acc = pauli_mul(Acc, self.stab[row])
                Bcc = pauli_mul(Bcc, (0, 1 << i, 0))
                R = pauli_mul(R, self.stab[row])
        return mx, mz, Acc, Bcc, R

    def _promote(self, R):
        """R commutes with all magic axes but anticommutes with a non-magic stabiliser ->
        open a new dense axis. Returns the new axis index, or None if R is a stabiliser."""
        ms = self._magicset()
        antis = [row for row in range(self.n)
                 if row not in ms and _symp(R, self.stab[row])]
        if not antis:
            return None
        p = antis[0]
        for row in range(self.n):               # make every other generator commute with R
            if row == p:
                continue
            if _symp(self.stab[row], R):
                self.stab[row] = pauli_mul(self.stab[row], self.stab[p])
            if _symp(self.destab[row], R):
                self.destab[row] = pauli_mul(self.destab[row], self.stab[p])
        # new axis X = R, normalised to a Hermitian observable (R is an operator product
        # so it may carry an i; storing the non-Hermitian rep poisons later phase products
        # -- the i is absorbed into the rotation mask's phase in _mask_for).
        self.destab[p] = _herm(R)
        self.magic.append(p)
        if self.phi is not None:                         # phi=None => TABLEAU-ONLY promote
            self.phi = np.kron([1.0 + 0j, 0.0], self.phi)   # new axis |0> (= +1 of stab[p])
        self.promote_calls += 1
        return len(self.magic) - 1

    # ---- physics -----------------------------------------------------------
    def _mask_for(self, P):
        """Build the exact virtual mask (x, z, phase) for P over the CURRENT axes,
        promoting if P opens a new direction. Returns (mx, mz, mphase)."""
        mx, mz, Acc, Bcc, R = self._express(P)
        newax = self._promote(R)
        if newax is not None:
            mx |= 1 << newax
            Acc = pauli_mul(Acc, self.destab[self.magic[newax]])   # Hermitian rep of R
            Bcc = pauli_mul(Bcc, (1 << newax, 0, 0))
        else:
            rows, ok = self._stab_decompose(R)
            assert ok, "residual not in stabiliser span (engine bug)"
            for row in rows:
                Acc = pauli_mul(Acc, self.stab[row])
        assert (Acc[0], Acc[1]) == (P[0], P[1]), "frame product != P (x,z) bug"
        mphase = (P[2] - Acc[2] + Bcc[2]) & 3
        return Bcc[0], Bcc[1], mphase

    def apply_rotation(self, P, theta):
        mx, mz, mph = self._mask_for(P)
        k = len(self.magic)
        Pphi = _apply_pauli_local(list(range(k)), self.phi, mx, mz, mph)
        self.phi = np.cos(theta / 2.0) * self.phi - 1j * np.sin(theta / 2.0) * Pphi
        self.max_k = max(self.max_k, k)          # transient: the 2^k vector materialised
        self._compress()                         # peel product axes as soon as they form
        if self.reduce_parities:                 # drop parities as they form (low resident)
            self._reduce_parities()
        self.max_k_res = max(self.max_k_res, len(self.magic))

    def measure(self, P, forced=None, rng=None):
        """Measure +-1 Pauli P on the magic register. Returns (out, p0)."""
        # First: does P (reduced over magic axes) anticommute with a non-magic stabiliser?
        # Then it probes a |0> direction -> uniform-random outcome, phi untouched (the
        # engine analogue of a Gottesman-Knill stabiliser measurement).
        _, _, _, _, R = self._express(P)
        ms = self._magicset()
        antis = [row for row in range(self.n)
                 if row not in ms and _symp(R, self.stab[row])]
        if antis:
            p = antis[0]
            Sp = self.stab[p]
            for row in range(self.n):
                if row != p and _symp(self.stab[row], P):
                    self.stab[row] = pauli_mul(self.stab[row], Sp)
                if _symp(self.destab[row], P):
                    self.destab[row] = pauli_mul(self.destab[row], Sp)
            out = int(forced) if forced is not None else (
                int(rng.integers(0, 2)) if rng is not None else 0)
            self.destab[p] = Sp
            self.stab[p] = (P[0], P[1], (P[2] + 2 * out) & 3)
            return out, 0.5
        # else: P = (mask over axes) * (stabiliser sign) -> measure on phi
        mx, mz, mph = self._mask_for(P)
        return self._measure_mask(mx, mz, mph, forced, rng)

    def _measure_mask(self, mx, mz, mph, forced, rng):
        k = len(self.magic)
        self.max_k = max(self.max_k, k)          # transient peak (just before any drop)
        Pphi = _apply_pauli_local(list(range(k)), self.phi, mx, mz, mph)
        exp = float(np.real(np.vdot(self.phi, Pphi)))
        p0 = min(1.0, max(0.0, 0.5 * (1.0 + exp)))
        if forced is not None:
            out = int(forced)
        elif rng is not None:
            out = 0 if float(rng.random()) < p0 else 1
        else:
            out = 0 if p0 >= 0.5 else 1
        sign = 1.0 if out == 0 else -1.0
        proj = 0.5 * (self.phi + sign * Pphi)
        _fm.el(self.phi.size, 6.0)               # sign*Pphi (2) + phi+ (2) + 0.5* (2)
        nrm = np.linalg.norm(proj)
        if nrm > 1e-12:
            self.phi = proj / nrm
            _fm.el(self.phi.size, 2.0)           # proj / nrm (real scale)
        self._compress()
        if self.reduce_parities:
            self._reduce_parities()
        self.max_k_res = max(self.max_k_res, len(self.magic))
        return out, p0

    # ---- C-4: drop a parity-slaved axis (Z^mz stabilises phi -> redundant) ----
    def _find_axis_z_stab(self, q):
        """If axis q is parity-slaved -- some Z-parity Z_q (x) Z^mz over the OTHER axes
        stabilises phi -- return that z-mask over axes, else None. (Virtual-axis port of
        the block backend's _find_z_stabilizer: GF(2) candidate then numeric verify.)"""
        k = len(self.magic)
        if k - 1 > 22 or k == 0:
            return None
        arr = self.phi.reshape(-1, 2, 1 << q)
        a = arr[:, 0, :].ravel(); b = arr[:, 1, :].ravel()
        sa = np.nonzero(np.abs(a) > 1e-9)[0]
        sb = np.nonzero(np.abs(b) > 1e-9)[0]
        if len(sa) == 0 or len(sb) == 0:
            return None                          # q already a product -> _compress handles it
        x0 = int(sa[0])
        rows = [(int(x) ^ x0, 0) for x in sa[1:]] + [(int(y) ^ x0, 1) for y in sb]
        mz = _gf2_solve(rows, k - 1)
        if not mz:
            return None
        zmask = 1 << q                           # rest-bit t -> axis index (skip q)
        for t in range(k - 1):
            if (mz >> t) & 1:
                zmask |= 1 << (t if t < q else t + 1)
        Pv = _apply_pauli_local(list(range(k)), self.phi, 0, zmask, 0)
        if abs(abs(complex(np.vdot(self.phi, Pv))) - 1.0) > 1e-6:
            return None                          # not a genuine stabiliser
        return zmask

    def _slaved_basis(self, q):
        """Is axis q stabiliser-slaved in SOME local basis? Try q in the Z / X / Y basis
        (local Clifford on q): if after it q is Z-parity-slaved, return the gate list to
        apply (folded), else None for each. Catches general (not just Z) single-qubit
        correlations -- e.g. an X_q X_r or Y-type stabiliser, the cultivation redundancy
        a pure-Z search misses."""
        from mdam.backend.block_magic import _vec_h, _vec_s
        for gates in ([], [('h', q)], [('s', q, True), ('h', q)]):
            test = self.phi
            for g in gates:
                test = _vec_h(test, g[1]) if g[0] == 'h' else _vec_s(test, g[1], g[2])
            saved, self.phi = self.phi, test
            zmask = self._find_axis_z_stab(q)
            self.phi = saved
            if zmask is not None:
                return gates
        return None

    def _find_stabilizer(self):
        """EXACT: return (x, z) of a non-identity Pauli stabilising phi (P|phi>=+-|phi>),
        or None. Derived from the amplitudes (NOT random sampling): for a candidate X-part
        x, the support must be ^x-invariant and the ratios phi[j^x]/phi[j] must be a unit-
        modulus +-(-1)^{z.j} pattern; z is then the GF(2) solution of parity(z.j)=sign bit.
        Finds the genuine stabiliser quotient (incl. the weight-3 X3X4Z1 the single-qubit
        reduction misses), so the reduction is COMPLETE for the Pauli-stabiliser rep."""
        k = len(self.magic)
        if k == 0 or k > 14:
            return None
        phi = self.phi
        idx = np.arange(1 << k)
        nz = np.abs(phi) > 1e-9
        js = idx[nz]
        j0 = int(np.argmax(np.abs(phi)))
        for x in range(1 << k):
            perm = idx ^ x
            if not np.array_equal(nz, nz[perm]):
                continue                         # support not ^x-invariant
            with np.errstate(all="ignore"):
                r = np.zeros(1 << k, dtype=complex)
                r[nz] = phi[perm][nz] / phi[nz]
            if not np.allclose(np.abs(r[nz]), 1.0, atol=1e-6):
                continue
            s = r[nz] / r[j0]
            if not (np.allclose(s.imag, 0, atol=1e-6)
                    and np.allclose(np.abs(s.real), 1.0, atol=1e-6)):
                continue
            bits = (s.real < 0).astype(int)
            z = _gf2_solve(list(zip([int(j) for j in js], [int(b) for b in bits])), k)
            if z is None:
                continue
            Pv = _apply_pauli_local(list(range(k)), phi, x, int(z), 0)
            if abs(abs(complex(np.vdot(phi, Pv))) - 1.0) < 1e-6 and (x, int(z)) != (0, 0):
                return x, int(z)
        return None

    def _reduce_parities(self):
        """COMPLETE stabiliser-quotient reduction. _compress drops single-qubit stabiliser
        axes; then the exact finder pulls out any remaining (multi-qubit) Pauli stabiliser,
        a Clifford rotates it to a single Z_t (folded into the tableau), and that axis -- now
        a Z eigenstate -- drops. Loops to a fully magic (stabiliser-free) phi. Exact identity
        insertion throughout -> state-exact."""
        self._compress()
        while self.magic:
            P = self._find_stabilizer()
            if P is None:
                break
            x, z = P
            self._rotate_pauli_to_z(x, z)        # stabiliser -> +-Z_t (axis t now definite)
            self._compress()                     # drops the Z-eigenstate axis t

    # ---- C-3b: rotate the measured Pauli to a single Z-axis, measure, DROP it ----
    def _right_h(self, i):
        row = self.magic[i]
        self.destab[row], self.stab[row] = self.stab[row], self.destab[row]

    def _right_s(self, i, dag):
        row = self.magic[i]
        m = pauli_mul(self.destab[row], self.stab[row])     # destab <- image(X Z) = Y
        self.destab[row] = (m[0], m[1], (m[2] + (3 if dag else 1)) & 3)

    def _right_cx(self, i, j):
        ri, rj = self.magic[i], self.magic[j]
        self.destab[ri] = pauli_mul(self.destab[ri], self.destab[rj])   # AX_i *= AX_j
        self.stab[rj] = pauli_mul(self.stab[ri], self.stab[rj])         # AZ_j *= AZ_i

    def measure_drop(self, P, forced=None, rng=None):
        """C-3b measurement: if P collapses a genuine magic direction (mask has X on an
        axis), rotate the mask to a single Z_t with a synthesised Clifford V (applied to
        phi, folded into the tableau), measure Z_t, and DROP axis t -- so k decreases by 1
        per such measurement (the engine stays at the active rank, not the raw support).
        A pure-Z (parity) / stabiliser measurement has no clean single-axis collapse and
        falls back to the in-place measure (its redundancy is C-4's dead-axis reduction)."""
        from mdam.backend.virtual_axis.clifford_synth import conj_h, conj_s, conj_cx
        from mdam.backend.block_magic import _vec_h, _vec_s, _vec_cx

        # antis (|0>-direction) branch -- identical to measure()
        _, _, _, _, R = self._express(P)
        ms = self._magicset()
        antis = [row for row in range(self.n)
                 if row not in ms and _symp(R, self.stab[row])]
        if antis:
            return self.measure(P, forced=forced, rng=rng)

        mx, mz, mph = self._mask_for(P)
        k = len(self.magic)
        if mx == 0 and mz == 0:                  # trivial (+-I): deterministic, no drop
            return self._measure_mask(mx, mz, mph, forced, rng)

        t, gates = self._rotate_pauli_to_z(mx, mz)
        Q = (mx, mz, mph)                        # track the residual phase (sign of +-Z_t)
        for g in gates:
            if g[0] == 'h':
                Q = conj_h(Q, g[1])
            elif g[0] == 's':
                Q = conj_s(Q, g[1], g[2])
            else:
                Q = conj_cx(Q, g[1], g[2])
        assert Q[0] == 0 and Q[1] == (1 << t), "mask did not rotate to a single Z_t (bug)"
        out, p0 = self._measure_mask(0, 1 << t, Q[2], forced, rng)   # measure +-Z_t, drops t
        return out, p0

    def _rotate_pauli_to_z(self, mx, mz):
        """Apply a Clifford to phi (folded into the tableau) turning the Pauli (mx,mz) over
        the axes into a single Z_t; return (t, gates). Turn each support axis X->Z (H),
        Y->Z (S^dag,H), then CNOT-collapse the Z-string onto t. Reused by the measurement
        DROP and by the general stabiliser reduction."""
        from mdam.backend.block_magic import _vec_h, _vec_s, _vec_cx
        k = len(self.magic)
        support = [s for s in range(k) if ((mx >> s) & 1) or ((mz >> s) & 1)]
        xaxes = [s for s in support if (mx >> s) & 1]
        t = xaxes[0] if xaxes else support[0]
        gates = []
        for s in support:
            xb = (mx >> s) & 1; zb = (mz >> s) & 1
            if xb and zb:
                gates.append(('s', s, True)); gates.append(('h', s))     # Y -> Z
            elif xb:
                gates.append(('h', s))                                    # X -> Z
        for s in support:
            if s != t:
                gates.append(('cx', s, t))                                # collapse onto t
        for g in gates:
            if g[0] == 'h':
                self.phi = _vec_h(self.phi, g[1]); self._right_h(g[1])
            elif g[0] == 's':
                self.phi = _vec_s(self.phi, g[1], g[2]); self._right_s(g[1], dag=(not g[2]))
            else:
                self.phi = _vec_cx(self.phi, g[1], g[2]); self._right_cx(g[1], g[2])
        return t, gates

    def _compress(self):
        """Drop every axis that is an unentangled SINGLE-QUBIT STABILISER state, i.e. a
        product in SOME local Pauli basis (|0>/|1> for Z, |+>/|-> for X, |+i>/|-i> for Y).
        Such an axis carries no magic -- a local Clifford (H / S^dag,H) turns it into a Z
        eigenstate which folds into the frame, dropping the dense dimension EXACTLY. (The
        earlier version dropped only Z eigenstates, so a Y/X-eigenstate axis -- e.g.
        cultivation's |+-i> stabiliser qubit -- leaked through. A genuinely magic
        single-qubit state is a product in NO Pauli basis, so it is correctly kept.)"""
        from mdam.backend.block_magic import _vec_h, _vec_s
        changed = True
        while changed and self.magic:
            changed = False
            for q in range(len(self.magic)):     # q = magic index = phi bit position
                for gates in ([], [('h', q)], [('s', q, True), ('h', q)]):
                    test = self.phi
                    for g in gates:
                        test = _vec_h(test, g[1]) if g[0] == 'h' else _vec_s(test, g[1], g[2])
                    arr = test.reshape(-1, 2, 1 << q)
                    n0 = np.linalg.norm(arr[:, 0, :]); n1 = np.linalg.norm(arr[:, 1, :])
                    if n1 >= 1e-10 and n0 >= 1e-10:
                        continue                 # entangled in this basis -> try next
                    for g in gates:              # product here: rotate into it for real + fold
                        if g[0] == 'h':
                            self.phi = _vec_h(self.phi, g[1]); self._right_h(g[1])
                        else:
                            self.phi = _vec_s(self.phi, g[1], g[2]); self._right_s(g[1], dag=(not g[2]))
                    arr = self.phi.reshape(-1, 2, 1 << q)
                    b0 = arr[:, 0, :].ravel(); b1 = arr[:, 1, :].ravel()
                    if np.linalg.norm(b1) < 1e-10:
                        self._drop_axis(q, b0, flip=False)    # |0> -> stabiliser (+)
                    else:
                        self._drop_axis(q, b1, flip=True)     # |1> -> stabiliser (-)
                    changed = True
                    break
                if changed:
                    break

    def _drop_axis(self, idx, newphi, flip):
        row = self.magic[idx]
        if flip:
            self.stab[row] = (self.stab[row][0], self.stab[row][1], (self.stab[row][2] + 2) & 3)
        self.magic.pop(idx)
        self.phi = newphi

    # ---- verification: magic-register statevector over n qubits ------------
    def statevector(self):
        """Dense 2^n magic-register state |psi> = sum_a phi[a] (prod AX_i^{a_i}) |s0>,
        |s0> = +1 eigenstate of every stab row. Matrix-free (Pauli applies + Gray-code
        accumulation): O(n 2^n) for |s0|, O(2^k 2^n) for the sum. Verification only."""
        n = self.n
        dim = 1 << n
        allq = list(range(n))
        # |s0> : project a fixed random vector onto the +1 eigenspace of every stab row
        rng = np.random.default_rng(0)
        v = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
        for row in range(n):
            P = self.stab[row]
            Pv = _apply_pauli_local(allq, v, P[0], P[1], P[2])
            v = 0.5 * (v + Pv)
        nv = np.linalg.norm(v)
        s0 = v / nv if nv > 1e-12 else v
        # sum over axes in Gray-code order: one AX apply per step
        k = len(self.magic)
        psi = self.phi[0] * s0
        cur = s0
        prev_g = 0
        for a in range(1, 1 << k):
            g = a ^ (a >> 1)
            b = (g ^ prev_g).bit_length() - 1          # the single axis bit that flipped
            row = self.magic[b]
            AX = self.destab[row]
            cur = _apply_pauli_local(allq, cur, AX[0], AX[1], AX[2])
            psi = psi + self.phi[g] * cur
            prev_g = g
        return psi
