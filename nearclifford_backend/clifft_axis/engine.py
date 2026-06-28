"""Clifft-axis near-Clifford engine: canonical active-axis dense state with
STRICTLY IN-PLACE pairwise kernels and a HARD memory budget.

Inherits all the verified structure from ``VirtualAxisNearClifford``:
  * Clifford frame (tableau) absorbs every Clifford for free,
  * rotations are deferred (lazy) and only the anticommuting core is flushed,
  * pullback P' = U_C^dag P U_C maps a lab Pauli to the bare register,
  * promote-on-demand grows the magic register only when P' has X-support,
  * a full-register PARITY REDUCTION after every magic measurement peels every
    parity-slaved qubit, holding |M| at clifft's active rank k.

It REPLACES only the dense kernels with in-place pairwise versions:
  * ``_pauli_lincomb_inplace``  : phi <- alpha*phi + beta*(P phi) IN PLACE,
        chunked so the transient scratch is a bounded chunk, never a second
        full-length vector (no `tmp = P @ phi`, no chi0/chi1, no v0/v1 branch).
  * ``_pauli_expectation``      : <phi|P|phi> streamed in chunks (no full Pphi).
  * Born collapse is the SAME in-place primitive with alpha=beta=1/2.
and adds:
  * a hard ``DenseMemoryBudget`` (peak live complex words <= 2^k_clifft, raises),
  * a residual certificate at every pullback (non-magic X must vanish; non-magic
    Z is dormant = +1; logged, never silently promoted past the certificate),
  * a per-measurement ``core_log`` (branch, |M| before/after, residual class,
    Born p0, live-word high-water).

No Pauli-sum / MPS / projected-TN dispatch exists in this engine; the only
exponential object is the single dense register `phi` of size 2^|M|.
"""
from __future__ import annotations

import numpy as np

from nearclifford_backend.simulator import pauli_commute
from nearclifford_backend.virtual_axis.virtual_axis_runtime import (
    VirtualAxisNearClifford)
from nearclifford_backend.block_magic import _gf2_solve
from nearclifford_backend.clifft_axis.budget import DenseMemoryBudget


class CliftAxisResidualError(Exception):
    """Pullback left a residual the Clifft-axis certificate forbids (X-support on a
    qubit that the active-axis schedule did not provide)."""


_FOLDS = (32, 16, 8, 4, 2, 1)


def _parity(a):
    """XOR-fold an int64 array to its per-element bit parity (0/1). Folds IN PLACE
    on the passed array (callers pass a fresh `idx & mask`)."""
    for sh in _FOLDS:
        a ^= a >> sh
    a &= 1
    return a


class CliftAxisNearClifford(VirtualAxisNearClifford):
    # chunk size (complex words) for the in-place pairwise sweeps. Bounds the
    # transient scratch so peak-live = resident + O(_CHUNK), never 2*resident.
    _CHUNK = 1 << 11
    # PHASE-2 STEP-1: strided single-axis-Z half-array rotation fast path. ON by default;
    # the verifier flips it OFF to reconstruct the pre-Step-1 parity kernel for bit-exact A/B.
    _step1_diaghalf = True
    # STAGE B (default OFF): run the in-place dense Pauli linear combination in C++ for the
    # full-formula branches (off-diagonal butterfly + non-diaghalf diagonal) -- the Python
    # scalar/vectorized hot path (the ry/rx bottleneck).  Bit-identical; the diaghalf global-
    # phase fast path stays in Python.  Toggled by the backend `compiled_numerical` flag.
    _compiled_numerical = False

    def __init__(self, n):
        super().__init__(n)
        # clifft_axis ENABLES the incremental inverse-frame (global NearClifford default is OFF):
        # every frame mutation updates Ax[i]=U_C^dag X_i U_C / Az[i]=U_C^dag Z_i U_C in O(1)..O(n)
        # so _pullback is an O(weight) lookup instead of the O(n^2) GF(2) basis recompute. A fresh
        # engine is built per shot (_reset), so the inverse images re-init to identity each shot and
        # this flag is re-armed on every warmed shot; the _pullback_via_basis path stays as fallback.
        self._inv_enabled = True
        # placeholder budget (cap = 2^n, NOT enforced) until the backend tightens it
        # to the genuine clifft active rank via set_clifft_budget(prog.peak_rank).
        self.budget = DenseMemoryBudget(n, enforce=False)
        self.log_cores = True
        self.core_log = []          # per-measurement certificate records
        self._last_cert = {}
        self._meas_log_ctr = 0

    # ---- budget wiring -----------------------------------------------------
    def set_clifft_budget(self, k_clifft, enforce=True):
        """Set the HARD cap = 2^k_clifft and (re)arm enforcement. Called by the
        backend once it knows prog.peak_rank (= clifft's active rank)."""
        self.budget = DenseMemoryBudget(k_clifft, enforce=enforce)
        self.budget.note_resident(self.phi.size, "init")
        return self.budget

    # ---- promote: charge the kron growth ----------------------------------
    def _promote(self, q):
        if q in self.M:
            return
        new_size = self.phi.size * 2
        # new qubit appended as MOST-significant in |0>: phi -> [phi, 0...0]. GROW THE
        # BUFFER IN PLACE (ndarray.resize zero-fills the new tail = bit_new=1 block empty),
        # so there is NEVER a second amplitude-sized buffer (np.kron/concatenate would hold
        # old 2^r + new 2^(r+1) simultaneously). Charge resident=new, transient=0.
        self.budget.charge(new_size, 0, "promote")
        self.M.append(q)
        if self.phi.flags["OWNDATA"]:
            self.phi.resize(new_size, refcheck=False)
        else:
            self.phi = np.concatenate([self.phi, np.zeros_like(self.phi)])

    # ---- residual certificate + (mx, mz) over the phi-bit layout ----------
    def _masks(self, xp, zp, promote, where):
        """Map a pulled-back Pauli (xp,zp) over PHYSICAL qubits to (mx,mz) over the
        magic-register bit layout (bit j <-> self.M[j], M[0] = LSB), promoting any
        X-support qubit into M first. Certifies that AFTER promotion no X-support
        remains on a non-magic qubit (else the active-axis schedule is inconsistent)
        and records the dormant (non-magic Z = +1) residual."""
        n_promote = 0
        for qq in range(self.n):
            if (xp >> qq) & 1 and qq not in self.M:
                if promote:
                    self._promote(qq)
                    n_promote += 1
                else:
                    raise CliftAxisResidualError(
                        f"{where}: X-support on non-magic qubit {qq} with promote=False")
        mx = mz = 0
        mmask = 0
        for j, qq in enumerate(self.M):
            b = 1 << qq
            mmask |= b
            if (xp >> qq) & 1:
                mx |= 1 << j
            if (zp >> qq) & 1:
                mz |= 1 << j
        x_resid = xp & ~mmask
        if x_resid != 0:
            raise CliftAxisResidualError(
                f"{where}: residual X {x_resid:#x} on non-magic qubits after promote")
        z_dormant = int(zp & ~mmask).bit_count()
        self._last_cert = dict(where=where, n_promote=n_promote, z_dormant=z_dormant,
                               mx_weight=int(mx).bit_count(),
                               mz_weight=int(mz).bit_count(), k=len(self.M))
        return mx, mz

    # ---- squared norms with ZERO amplitude-sized temporary -----------------
    @staticmethod
    def _sqnorm_1d(arr):
        """sum |arr|^2 for a 1-D array via einsum on the .real/.imag STRIDED VIEWS -- a
        full reduction with O(1) scratch (measured ~110 words, r-independent), no copy."""
        return float(np.einsum('i,i->', arr.real, arr.real)
                     + np.einsum('i,i->', arr.imag, arr.imag))

    def _branch_sqnorm(self, j, branch):
        """sum_{s: bit_j(s)=branch} |phi[s]|^2  (the squared norm of one measurement branch).
        np.linalg.norm(v[:, branch, :]) ravels the NON-CONTIGUOUS half-view into a 2^(r-1)
        contiguous complex128 copy (the measured Theta(2^(r-1)) temporary). einsum reduces
        the strided .real/.imag VIEWS directly: O(1) scratch, no amplitude-sized copy, exact
        -- and independent of the budget slack, so it holds even at r=k (slack 0)."""
        v = self.phi.reshape(-1, 2, 1 << j)
        seg = v[:, branch, :]                         # strided half-VIEW (no copy)
        self.budget.charge(self.phi.size, 0, "sqnorm")  # O(1) view reduction -> 0 amplitude temp
        return float(np.einsum('ij,ij->', seg.real, seg.real)
                     + np.einsum('ij,ij->', seg.imag, seg.imag))

    def _lincomb_chunk(self, slack):
        """rows per vectorised chunk so the kernel's working arrays (gathered pairs + index
        / sign arrays + RHS temporaries, ~8 chunk-sized arrays) stay within the live budget
        slack; returns 0 to request the SCALAR no-array path when there is no head-room (the
        register is at the cap, slack 0 -- e.g. a rotation flushed exactly at r=k)."""
        if slack < 16:
            return 0
        return max(1, min(self._CHUNK, slack // 8))

    # ---- IN-PLACE pairwise linear combination: phi <- alpha*phi + beta*(P phi) ----
    def _pauli_lincomb_inplace(self, mx, mz, pp, alpha, beta, where=""):
        """phi <- alpha*phi + beta * (P phi)  IN PLACE, with P = i^pp X^mx Z^mz on the
        magic register. Convention (identical to NearClifford._apply_magic_pauli):
            (P phi)[j] = i^pp * (-1)^{parity((j^mx) & mz)} * phi[j^mx].
        Off-diagonal (mx!=0): each pair (j, j^mx) is updated together from saved values --
        a 2x2 in-place butterfly. Diagonal (mx==0): an element-wise parity scale. The
        vectorised chunk is SLACK-AWARE (working arrays <= budget slack); at the cap
        (slack 0, e.g. a rotation at r=k) it falls to a SCALAR Python loop that holds no
        numpy work array at all (resident + O(1)), so peak amplitude words stay <= 2^k."""
        phi = self.phi
        N = phi.size
        ph = (1j ** pp)
        bph = beta * ph
        # STAGE B: route the FULL-FORMULA branches (off-diagonal, and non-diaghalf diagonal) to
        # the C++ kernel.  Keep the diaghalf global-phase fast path AND the mz==0 global scalar in
        # Python so the state stays bit-identical to the authoritative path.
        if self._compiled_numerical and (mx != 0 or mz != 0):
            is_diaghalf = (self._step1_diaghalf and where.startswith("rot") and mx == 0
                           and (mz & (mz - 1)) == 0 and abs(abs(alpha + bph) - 1.0) < 1e-9)
            if not is_diaghalf:
                from nearclifford_backend.clifft_axis import compiled_numerical as CN
                self.budget.charge(N, 0, where + ":cpp")     # in-place: no transient
                CN.lincomb(phi, int(mx), int(mz), complex(alpha), complex(bph))
                return
        slack = self.budget.cap - N
        CH = self._lincomb_chunk(slack)
        if mx == 0:
            if mz == 0:                      # global scalar: pure in-place, no scratch
                self.budget.charge(N, 0, where + ":diag0")
                phi *= (alpha + bph)
                return
            m_even = alpha + bph
            m_odd = alpha - bph
            # PHASE-2 STEP-1 strided single-axis-Z fast path (rotations only). A Hermitian
            # Z-generator rotation (pp in {0,2}) has |m_even| = 1, so factor m_even out as an
            # UNOBSERVABLE global phase and multiply ONLY the bit=1 half by m_odd/m_even on a
            # CONTIGUOUS strided VIEW -- half the amplitudes, fully vectorised (no arange /
            # parity / boolean-mask gather).  This is clifft's array_rot form (3*2^k half-array)
            # and the kernel the localized rotations of Step 2/3 will land in.  Records and Born
            # probabilities are global-phase invariant, so dropping m_even is bit-exact on every
            # observable (verified by the per-seed record/p0 suite).
            if (self._step1_diaghalf and where.startswith("rot") and (mz & (mz - 1)) == 0
                    and abs(abs(m_even) - 1.0) < 1e-9):
                self.budget.charge(N, 0, where + ":diaghalf")
                jbit = mz.bit_length() - 1
                v = phi.reshape(-1, 2, 1 << jbit)
                v[:, 1, :] *= (m_odd / m_even)        # bit_j=1 half; bit_j=0 half = dropped phase
                return
            if CH == 0:                      # strict scalar (slack 0): no numpy work array
                self.budget.charge(N, 0, where + ":diag-scalar")
                for s in range(N):
                    phi[s] *= (m_odd if ((s & mz).bit_count() & 1) else m_even)
                return
            self.budget.charge(N, 3 * CH, where + ":diag")
            for s in range(0, N, CH):
                e = min(s + CH, N)
                par = np.arange(s, e, dtype=np.int64)
                par &= mz
                _parity(par)
                oddb = par.astype(bool)
                seg = phi[s:e]               # view into phi
                seg[~oddb] *= m_even
                seg[oddb] *= m_odd
            return
        # off-diagonal: pair j (pivot bit 0) with k = j^mx
        pivot = mx & (-mx)
        if CH == 0:                          # strict scalar (slack 0): no numpy work array
            self.budget.charge(N, 0, where + ":offdiag-scalar")
            for j in range(N):
                if j & pivot:
                    continue
                kk = j ^ mx
                a = phi[j]                   # numpy 0-d scalars (no array temp)
                b = phi[kk]
                sj = 1 - 2 * ((j & mz).bit_count() & 1)
                sk = 1 - 2 * ((kk & mz).bit_count() & 1)
                phi[j] = alpha * a + bph * (sk * b)
                phi[kk] = alpha * b + bph * (sj * a)
            return
        self.budget.charge(N, 8 * CH, where + ":offdiag")
        for s in range(0, N, CH):
            e = min(s + CH, N)
            idx = np.arange(s, e, dtype=np.int64)
            j = idx[(idx & pivot) == 0]
            if j.size == 0:
                continue
            kk = j ^ mx
            a = phi[j]                       # saved old values (copies)
            b = phi[kk]
            sj = 1 - 2 * _parity(j & mz)     # +-1 for the k-update
            sk = 1 - 2 * _parity(kk & mz)    # +-1 for the j-update
            phi[j] = alpha * a + bph * (sk * b)
            phi[kk] = alpha * b + bph * (sj * a)

    # ---- <phi| P |phi>, streamed (no full P phi materialised) -------------
    def _pauli_expectation(self, mx, mz, pp, where="exp"):
        """Real expectation <phi|P|phi> for Hermitian P = i^pp X^mx Z^mz, computed in
        chunks: <P> = i^pp * sum_m (-1)^{parity(m&mz)} conj(phi[m^mx]) phi[m]."""
        phi = self.phi
        N = phi.size
        ph = (1j ** pp)
        CH = min(self._CHUNK, N)
        self.budget.charge(N, 2 * CH, where)
        acc = 0.0 + 0.0j
        for s in range(0, N, CH):
            e = min(s + CH, N)
            idx = np.arange(s, e, dtype=np.int64)
            sgn = 1 - 2 * _parity(idx & mz)
            src = phi[s:e]
            gathered = src if mx == 0 else phi[(np.arange(s, e, dtype=np.int64)) ^ mx]
            acc += np.sum(sgn * np.conjugate(gathered) * src)
        acc *= ph
        return float(np.real(acc))

    # ---- flush one pending rotation: in-place ------------------------------
    def _flush_one(self, x, z, theta, phase=0):
        # `phase` = the pending generator's ACCUMULATED i^phase (lazy h/s/cz conjugations
        # carry e.g. +Y = i^1 X Z).  lazy._do_flush drops it; `_pullback(x,z)` returns only
        # the FRAME phase pp of the bare X^x Z^z, so the flushed operator would be i^pp X Z
        # (= -iY for a Y generator) instead of i^(phase+pp) X Z (= Y).  The pullback is a
        # homomorphism + i^phase is a scalar, so the correct generator is i^(phase+pp) X^xp
        # Z^zp.  phase==0 (R_Z / R_X flushes) -> identical to before (no regression).
        xp, zp, pp = self._pullback(x, z)
        pp = (pp + phase) & 3
        mx, mz = self._masks(xp, zp, promote=True, where="rot")
        if len(self.M) > self.max_M:
            self.max_M = len(self.M)
        if self.cap is not None and len(self.M) > self.cap:
            from nearclifford_backend.backend import MagicCapExceeded
            raise MagicCapExceeded(-1, len(self.M))
        c = np.cos(theta / 2.0)
        s = np.sin(theta / 2.0)
        self._pauli_lincomb_inplace(mx, mz, pp, alpha=c, beta=(-1j * s), where="rot")

    # ---- clifft_axis-SCOPED overrides of the lazy flush drivers: forward the pending
    #      generator's phase r[2] (which lazy._do_flush / lazy.statevector drop) to
    #      _flush_one.  These shadow lazy.py for the clifft_axis engines ONLY; lazy.py,
    #      virtual_axis/ (fused VA, whose 3-arg _flush_one monkeypatch is shadowed by THIS
    #      class's _flush_one), and block_magic are untouched.  Body matches lazy._do_flush
    #      verbatim except the final _flush_one call passes the phase.
    def _do_flush(self, qx, qz, flush):
        if not flush:
            return
        for r in flush:
            del self.pending[r[4]]
        if self._flushed_uids is not None:
            for r in flush:
                self._flushed_uids.add(r[4])
        if self.resource_only:
            supp = 0
            qxp, qzp, _ = self._pullback(qx, qz)
            supp |= qxp
            for (x, z, p, theta, uid) in flush:
                xp, zp, pp = self._pullback(x, z)
                supp |= xp
            self.max_M = max(self.max_M, supp.bit_count())
            return
        for (x, z, p, theta, uid) in flush:        # p = pending phase, forwarded
            self._flush_one(x, z, theta, p)

    def statevector(self):
        from nearclifford_backend.simulator import NearClifford
        for (x, z, p, theta, uid) in list(self.pending.values()):
            self._flush_one(x, z, theta, p)        # forward pending phase (was dropped)
        self.pending = {}
        return NearClifford.statevector(self)      # the real array builder (skips lazy re-flush)

    def cz(self, a, b):
        # BUG #2 FIX (clifft_axis-scoped): conjugate PENDING by CZ EXACTLY ONCE.  CZ = H_b ·
        # CX(a,b) · H_b, so routing through self.h/self.cx (the lazy overrides) conjugates
        # each pending rotation once AND updates the tableau (their super() calls).
        # LazyNearClifford.cz does super().cz(=this same H/CX/H, conjugating pending once) AND
        # THEN re-conjugates pending by _conj_h/_conj_cx/_conj_h -- a DOUBLE conjugation.  It
        # is a no-op for Z-diagonal (R_Z) pending (CZ commutes with Z -> conj is identity, so
        # R_Z was unaffected) but CORRUPTS off-axis (R_X/R_Y) pending: an R_Y generator
        # conjugated twice by CZ ends up on the wrong axis, silently losing the coherent
        # contribution (the dominant R_Y QEC bias).  Verified exact vs dense (incl. multi-CZ
        # depth).  Overriding here leaves lazy.py / virtual_axis (fused VA) / block untouched.
        self.h(b); self.cx(a, b); self.h(b)

    # ---- parity-stabiliser search with STREAMED verification (no full copy) ----
    def _find_z_stab(self, q):
        """As VirtualAxis._find_z_stab, but the final 'does Z^zmask stabilise phi?' check
        is done with the STREAMED expectation <Z^mz> (a pure-Z diagonal observable) instead
        of materialising P phi via _apply_pauli_local (which allocates a full sign vector +
        a full gathered copy). The GF(2) support scan still reads the two half-slices."""
        M = self.M
        k = len(M)
        if k <= 1 or k - 1 > self._reduce_cap:
            return None
        j = M.index(q)
        arr = self.phi.reshape(-1, 2, 1 << j)
        a = arr[:, 0, :].ravel()
        b = arr[:, 1, :].ravel()
        self.budget.charge(self.phi.size, self.phi.size, "reduce:gf2scan")
        sa = np.nonzero(np.abs(a) > 1e-9)[0]
        sb = np.nonzero(np.abs(b) > 1e-9)[0]
        if len(sa) == 0 or len(sb) == 0:
            return None
        x0 = int(sa[0])
        rows = [(int(x) ^ x0, 0) for x in sa[1:]] + [(int(y) ^ x0, 1) for y in sb]
        mz = _gf2_solve(rows, k - 1)
        if not mz:
            return None
        zmask = 1 << q
        for t in range(k - 1):
            if (mz >> t) & 1:
                zmask |= 1 << M[t if t < j else t + 1]
        # streamed stabiliser check: |<phi|Z^zmask|phi>| ~ 1  (mz over phi-bit layout)
        mz_phi = 0
        for jj, qq in enumerate(M):
            if (zmask >> qq) & 1:
                mz_phi |= 1 << jj
        exp = self._pauli_expectation(0, mz_phi, 0, "reduce:verify")
        if abs(abs(exp) - 1.0) > 1e-6:
            return None
        return zmask

    # ================================================================= #
    #  TRUE in-place single-qubit Cliffords on the magic register.       #
    #  Operate on STRIDED VIEWS with in-place arithmetic -- ZERO          #
    #  amplitude-sized temporary (no full/half copy). `j` is the flat     #
    #  phi bit position (= self.M.index(qubit)).                          #
    # ================================================================= #
    _INV_SQRT2 = 0.7071067811865476

    def _h_axis(self, j):
        """H on axis j IN PLACE via the butterfly a'=(a+b)/v2, b'=(a-b)/v2 on the two
        strided half-views -- no temporary (a += b; b *= -2; b += a; scale)."""
        v = self.phi.reshape(-1, 2, 1 << j)
        a = v[:, 0, :]
        b = v[:, 1, :]
        a += b                       # a = a0 + b0
        b *= -2.0
        b += a                       # b = a0 - b0
        a *= self._INV_SQRT2
        b *= self._INV_SQRT2
        self.budget.charge(self.phi.size, 0, "purge:h")

    def _s_axis(self, j, dag):
        """S (or S^dag) on axis j IN PLACE: scale the bit_j=1 strided half-view by +-i."""
        v = self.phi.reshape(-1, 2, 1 << j)
        v[:, 1, :] *= (-1j if dag else 1j)
        self.budget.charge(self.phi.size, 0, "purge:s")

    def _cnot_axes(self, jc, jt):
        """CNOT(control bit jc, target bit jt) IN PLACE: swap the (jc=1,jt=0) and
        (jc=1,jt=1) strided sub-blocks with the temp-free 3-op swap. ZERO amplitude temp.
        Index the two bits with length-1 SLICES (not integers) so the sub-blocks stay
        VIEWS -- integer-indexing both bits on a rank-2 register yields a scalar COPY, and
        the in-place swap would silently no-op (caused the M=[1,2] reduction to loop)."""
        r = len(self.M)
        t = self.phi.reshape([2] * r)          # axis a <-> bit (r-1-a)
        ac = r - 1 - jc
        at = r - 1 - jt
        s0 = [slice(None)] * r; s0[ac] = slice(1, 2); s0[at] = slice(0, 1)
        s1 = [slice(None)] * r; s1[ac] = slice(1, 2); s1[at] = slice(1, 2)
        a = t[tuple(s0)]
        b = t[tuple(s1)]
        a += b; b *= -1.0; b += a; a -= b      # swap a,b in place (no temp)
        self.budget.charge(self.phi.size, 0, "purge:cnot")

    def _drop_axis_inplace(self, j, fold_x_qubit=None):
        """Drop axis j when it is a product Z-eigenstate, keeping the non-zero slice, IN
        PLACE (compact within the same buffer, then truncate -- never a second
        amplitude-sized buffer). If the |1> slice is kept, the caller has folded X into the
        frame. Returns the kept eigenvalue bit (0/1), or None if not a product axis."""
        sq0 = self._branch_sqnorm(j, 0)              # no 2^(r-1) contiguity copy (einsum view)
        sq1 = self._branch_sqnorm(j, 1)
        if sq1 < 1e-20:
            keep = 0
        elif sq0 < 1e-20:
            keep = 1
        else:
            return None
        half = self.phi.size >> 1
        # compact the kept slice to the front of the SAME buffer in fixed-size chunks
        # (forward copy; dest index <= src index, so no clobber), then truncate.
        bit = 1 << j
        CH = min(self._CHUNK, half)
        self.budget.charge(self.phi.size, CH, "drop")
        phi = self.phi
        for s in range(0, half, CH):
            e = min(s + CH, half)
            n = np.arange(s, e, dtype=np.int64)
            low = n & (bit - 1)
            high = n >> j
            src = (high << (j + 1)) | (keep << j) | low      # insert `keep` bit at pos j
            phi[s:e] = phi[src]
        phi.resize(half, refcheck=False)
        return keep

    # ---- in-place CNOT on the register (a pair-swap permutation, no full copy) ----
    def _cnot_inplace(self, jc, jt):
        """phi <- CNOT(control bit jc, target bit jt) IN PLACE. The block-magic _vec_cx
        does `vec[perm]` (a full copy); a CNOT is just a permutation that SWAPS each pair
        (i, i^(1<<jt)) with control bit set, so we swap chunk-by-chunk with O(_CHUNK)
        scratch instead of a second full-length vector."""
        phi = self.phi
        N = phi.size
        tbit = 1 << jt
        CH = min(self._CHUNK, N)
        self.budget.charge(N, CH, "reduce:cnot")
        for s in range(0, N, CH):
            e = min(s + CH, N)
            idx = np.arange(s, e, dtype=np.int64)
            i0 = idx[(((idx >> jc) & 1) == 1) & (((idx >> jt) & 1) == 0)]
            if i0.size == 0:
                continue
            i1 = i0 ^ tbit
            tmp = phi[i0].copy()
            phi[i0] = phi[i1]
            phi[i1] = tmp

    def _reduce_full(self):
        """Parity-reduction (peel every parity-slaved magic qubit), as VirtualAxis but
        applying each collapsing CNOT IN PLACE (no _vec_cx full copy)."""
        changed = True
        while changed:
            changed = False
            for q in list(self.M):
                zmask = self._find_z_stab(q)
                if zmask is None:
                    continue
                pos = {s: self.M.index(s) for s in range(self.n) if (zmask >> s) & 1}
                for s in list(pos):
                    if s != q:
                        self._cnot_axes(pos[s], pos[q])        # CNOT(s->q), temp-free
                        self.right_cx(s, q)
                self._compress_magic()
                changed = True
                break

    # ---- measurement: lazy core flush -> stabilizer/magic -> in-place collapse ----
    def measure_z(self, q):
        self._flush_core(0, 1 << q)                 # lazy anticommuting-core flush
        Pm = (0, 1 << q, 0)
        magset = set(self.M)
        anti_s = [i for i in range(self.n)
                  if i not in magset and not pauli_commute(self.Zc[i], Pm)]
        M_before = len(self.M)
        p0 = None
        cert = {}
        if anti_s:
            out = self._ag_measure(Pm, anti_s)      # Gottesman-Knill, magic untouched
            branch = "stabilizer"
        else:
            xp, zp, pp = self._pullback(0, 1 << q)
            mx, mz = self._masks(xp, zp, promote=True, where="meas")
            cert = dict(self._last_cert)
            exp = self._pauli_expectation(mx, mz, pp, where="meas")
            p0 = max(0.0, min(1.0, (1.0 + exp) / 2.0))
            r = float(self.rng.random())
            out = 0 if r < p0 else 1
            sign = 1.0 if out == 0 else -1.0
            # IN-PLACE Born collapse: phi <- (1/2)(phi + sign * P phi), then renormalise
            self._pauli_lincomb_inplace(mx, mz, pp, alpha=0.5, beta=0.5 * sign,
                                        where="collapse")
            nrm = float(np.linalg.norm(self.phi))
            if nrm > 1e-12:
                self.phi /= nrm                     # in-place normalise (no copy)
            self._compress_magic()                  # drop disentangled product axes
            branch = "magic"
        # parity reduction holds |M| at the active rank k (the localize-and-drop step);
        # its CNOTs are applied IN PLACE (self._cnot_inplace charges its own scratch).
        self._reduce_full()
        if len(self.M) > self.max_M:
            self.max_M = len(self.M)
        self.budget.note_resident(self.phi.size, "post-reduce")
        if self.log_cores:
            self.core_log.append(dict(
                meas=self._meas_log_ctr, branch=branch, M_before=M_before,
                M_after=len(self.M), p0=p0,
                n_promote=cert.get("n_promote"), z_dormant=cert.get("z_dormant"),
                mx_weight=cert.get("mx_weight"), mz_weight=cert.get("mz_weight"),
                peak_live_words=self.budget.peak))
        self._meas_log_ctr += 1
        return out

    # ---- certificate summary ----------------------------------------------
    def certificate(self):
        """Final certificate: budget summary + peak |M| + per-core log."""
        return dict(budget=self.budget.summary(), peak_M=self.max_M,
                    n_meas=self._meas_log_ctr, core_log=self.core_log)
