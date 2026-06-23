"""clifft_axis_bounded: the canonical bounded engine.

Adds to the dense Clifft-axis engine the MEASUREMENT LOCALIZE-AND-DROP that the lazy
dense engine lacked (the cause of coherent_d3_r3's monotonic 3->5->7->...->12 growth):
after a magic measurement, the measured Pauli P' = U_C^dag Z_q U_C (restricted to the
magic register) is localized to a single Z_r by a block-local Clifford W (H/S^dag,H to
turn each support Pauli into Z, then CNOTs to collapse the Z-string onto r), W is applied
to phi and W^dag folded into the frame (an EXACT identity insertion -- the physical state
is unchanged), so qubit r is left in a definite Z-eigenstate (the measurement fixed it)
and compress drops it. This RELEASES the measured degree of freedom every magic
measurement, so the active rank tracks clifft's active_k_history (which DECREASES after
each measurement) instead of accumulating.

This is the dense port of block_magic.BlockLazyNearClifford._purge_redundant (verified
there); applied after the Born collapse it consumes no RNG and is state-exact, so the
measurement record is identical to the lazy_magic_dense oracle.
"""
from __future__ import annotations

import numpy as np

from nearclifford_backend.simulator import pauli_commute
from nearclifford_backend.block_magic import _support
from nearclifford_backend.lazy import _conj_h, _conj_s, _conj_cx
from nearclifford_backend.clifft_axis.engine import CliftAxisNearClifford


def compile_bounded(stim_text, **kwargs):
    """Compile a circuit for the clifft_axis_bounded backend WITHOUT gate fusion.

    The bounded engine defers magic rotations (lazy `pending`) and reconciles them with
    the Clifford frame only through frame.h/s/cz, which conjugate `pending`.  clifft's
    default fused compile emits OP_ARRAY_U2 / OP_ARRAY_U4 nodes whose de-fusion in
    backend._apply_u2/_apply_u4 advances the frame with a raw frame.set_xz that does NOT
    conjugate `pending`; for off-axis (R_X/R_Y) noise this corrupts the deferred rotations
    -> silently wrong results (measured: coherent_d3_r* R_Y peak rank collapses 10->3,
    prob |D|~0.99).  R_Z noise is unaffected (its pending rotations are Z-diagonal), so the
    default compile happens to work for the Z-axis benchmarks -- but bytecode_passes=None
    is verified bit-for-bit identical on every Z benchmark (rank/resident/records) AND is
    the only correct path for X/Y noise, so it is the canonical compile here.

    This forbids the rotation . entangling-Clifford fusion (and all other bytecode fusion);
    `clifft.compile`'s HIR passes still run, so peak_rank is unchanged.
    """
    import clifft
    return clifft.compile(stim_text, bytecode_passes=None, **kwargs)


class CliftAxisBoundedNearClifford(CliftAxisNearClifford):
    # PHASE-2 STEP-2: localize a single-axis X/Y rotation generator to a diagonal Z_a (one
    # H/S, frame-folded) instead of the off-diagonal butterfly. ON by default; verifier flips
    # it OFF to reconstruct the pre-Step-2 off-diagonal flush for bit-exact A/B.
    _step2_localize = True
    _loc_undo = False             # False = frame-fold (1 H, FLOP-optimal); the incremental inverse-frame
    #                               makes the post-fold _pullback an O(1) lookup, so the old cache-recompute
    #                               thrash (the d5_r5 wall regression) is gone. True = undo (2 H, no frame
    #                               change) retained as fallback if the inverse-frame is ever disabled.
    _loc_min_size = 1 << 14       # localize only at rank >= 14 (phi.size >= 2^14). Below this the
    #                               localizer's strided sweeps + O(n) incremental inverse-frame update
    #                               per fold lose wall to the butterfly -- decisively so at large n / low
    #                               rank (e.g. d5_r5: n=72, peak rank 13 -> excluded, wall 224->194ms back
    #                               to butterfly). The RY rank-16 regime stays localized (ry_d3_r1 FLOP
    #                               win fully retained at 12.85M = 1.05x clifft-unfused). Measured-tuned
    #                               (phase2_gate_sweep.py); both fallbacks (undo / butterfly) preserved.

    def _flush_one(self, x, z, theta, phase=0):
        """Flush one pending rotation. STEP-2: if the pulled-back generator is single-axis with
        X-character (X_a or Y_a -- mx one bit, mz subset {a}), localize it to sign*Z_a with the
        VERIFIED measurement localizer `_localize_to_Z` (applies V=H/S to phi, folds V^dag into
        the Clifford frame, conjugates the generator -- the exact RY/CZ-safe machinery), then
        apply the diagonal R_{Z_a}(sign*theta) via the Step-1 strided half-array kernel.  Else
        fall back to the off-diagonal butterfly."""
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
        # STEP-3 (gated): apply an off-axis rotation as V^dag . diagonal R_Z . V on phi, with V a
        # local Clifford mapping the generator to Z_a -- and UNDO V on the array (V . R_Z . V^dag)
        # rather than folding V^dag into the frame.  The frame is UNCHANGED, so the cached
        # _pullback_basis is NOT invalidated (the 94%-of-overhead recompute is avoided) and there
        # is NO frame/sign-into-frame path (the RY/CZ bug class is out of scope) -- it is literally
        # the same off-diagonal rotation computed with strided H/CNOT kernels instead of a fancy
        # butterfly.  Gate on rank so small registers (where the butterfly is already cheap) keep it.
        if (self._step2_localize and mx != 0 and self.phi.size >= self._loc_min_size
                and self._flush_offdiag_localized(xp, zp, pp, mx, mz, c, s)):
            return
        self._pauli_lincomb_inplace(mx, mz, pp, alpha=c, beta=(-1j * s), where="rot")

    @staticmethod
    def _conj(P, g):
        if g[0] == "h":
            return _conj_h(P, g[1])
        if g[0] == "s":
            return _conj_s(P, g[1], g[2])
        return _conj_cx(P, g[1], g[2])

    def _flush_offdiag_localized(self, xp, zp, pp, mx, mz, c, s):
        """Apply R_{P'}(theta) for off-diagonal P' = i^pp X^xp Z^zp via V . R_{Z_a} . V^dag on phi,
        with V the COLLAPSE-FIRST localizer (ONE H, weight-independent): CNOT-collapse the X-string
        onto pivot a (free permutations), one S^dag if a is Y, ONE H (X_a->Z_a), CNOT-collapse the
        Z-string onto a (free).  V is built SYMBOLICALLY and verified to map P' -> +-Z_a before any
        phi touch; on failure return False (caller uses the butterfly).  V is UNDONE on the array
        (frame untouched -> no _pullback_basis recompute, no frame/sign-into-frame bug class).
        Returns True iff applied."""
        P = (xp, zp, pp)
        xsupp = [ss for ss in _support(xp, zp) if ss in self.M and (xp >> ss) & 1]
        if not xsupp:
            return False
        a = xsupp[0]
        W = []
        for b in xsupp:                                     # 1. collapse X-string onto a (free)
            if b != a:
                g = ("cx", a, b); W.append(g); P = self._conj(P, g)
        if (P[0] >> a) & 1 and (P[1] >> a) & 1:             # 2. a is Y -> S^dag makes it pure X
            g = ("s", a, True); W.append(g); P = self._conj(P, g)
        g = ("h", a); W.append(g); P = self._conj(P, g)     # 3. the ONE H: X_a -> Z_a
        for b in [ss for ss in self.M if ss != a and (P[1] >> ss) & 1]:
            g = ("cx", b, a); W.append(g); P = self._conj(P, g)   # 4. collapse Z-string onto a
        if P[0] != 0 or P[1] != (1 << a):                   # verify P' -> +-Z_a (else bail)
            return False
        sign = 1.0 if (P[2] & 3) == 0 else -1.0
        idx = self.M.index
        if self._loc_undo:
            for g in W:                                     # apply V to phi (strided; CNOTs free)
                if g[0] == "h":
                    self._h_axis(idx(g[1]))
                elif g[0] == "s":
                    self._s_axis(idx(g[1]), g[2])
                else:
                    self._cnot_axes(idx(g[1]), idx(g[2]))
            self._pauli_lincomb_inplace(0, 1 << idx(a), 0, alpha=c, beta=(-1j * s * sign), where="rot")
            for g in reversed(W):                           # UNDO V (frame untouched): 2 H, no cache hit
                if g[0] == "h":
                    self._h_axis(idx(g[1]))
                elif g[0] == "s":
                    self._s_axis(idx(g[1]), not g[2])
                else:
                    self._cnot_axes(idx(g[1]), idx(g[2]))
        else:
            for g in W:                                     # apply V to phi AND fold V^dag to frame
                if g[0] == "h":
                    self._h_axis(idx(g[1])); self.right_h(g[1])
                elif g[0] == "s":
                    self._s_axis(idx(g[1]), g[2]); self.right_s(g[1], dag=(not g[2]))
                else:
                    self._cnot_axes(idx(g[1]), idx(g[2])); self.right_cx(g[1], g[2])
            self._pauli_lincomb_inplace(0, 1 << idx(a), 0, alpha=c, beta=(-1j * s * sign), where="rot")
        return True

    # ================================================================= #
    #  CAPACITY-BUFFER STORAGE.  phi is ALWAYS storage[:sz] -- a contiguous prefix view.
    #  promote grows the logical size in place (zero the new MSB block); drop shrinks via
    #  size/memmove after moving the dropped axis to the MSB.  NO ndarray.resize, which
    #  reallocs a fresh buffer (measured: grow -> 2^(r+1), shrink -> 2^(r-1) transient).
    #  The backend retains the buffer across shots so its capacity settles at the high-water
    #  2^r_max <= 2^k_clifft and warmed shots NEVER realloc -> the storage is the only
    #  exponential object and peak amplitude memory = 2^r_max.
    # ================================================================= #
    def _adopt_storage(self, retained=None):
        """Establish the capacity buffer at shot start, reusing the backend's high-water
        buffer when its capacity is in [sz, cap] (so warmed shots never realloc)."""
        sz = self.phi.size
        cap = self.budget.cap
        if retained is not None and sz <= retained.size <= cap:
            self._storage = retained
        else:
            self._storage = np.empty(max(1, sz), dtype=np.complex128)
        self._storage[:sz] = self.phi
        self._sz = sz
        self.phi = self._storage[:sz]

    def _ensure_inited(self):
        """Lazy adopt: a fresh engine (no backend wiring, e.g. unit tests) or any path that
        reassigned phi to a non-storage array re-establishes the prefix-view invariant."""
        st = getattr(self, "_storage", None)
        if st is None or getattr(self.phi, "base", None) is not st:
            self._adopt_storage(st)

    def _grow_capacity(self, need):
        """High-water grow of the BACKING buffer -- one transient (old+new), only when the
        logical size first exceeds capacity (shot 1 / a new peak rank). Warmed shots skip."""
        if need > self._storage.size:
            bigger = np.empty(need, dtype=np.complex128)
            bigger[:self._sz] = self._storage[:self._sz]
            self._storage = bigger
            self.phi = self._storage[:self._sz]

    def _promote(self, q):
        """Grow logical size 2^r -> 2^(r+1) IN PLACE: the new qubit is appended as the MSB in
        |0>, so the existing amplitudes stay in the low half and the new high block is zeroed
        in storage (no resize realloc, no second buffer)."""
        if q in self.M:
            return
        self._ensure_inited()
        new_size = self._sz * 2
        self.budget.charge(new_size, 0, "promote")        # resident 2^(r+1) <= 2^k (enforced)
        self._grow_capacity(new_size)                     # realloc only on a new high-water
        self._storage[self._sz:new_size] = 0.0            # new MSB qubit = |0>: high block zero
        self._sz = new_size
        self.M.append(q)
        self.phi = self._storage[:self._sz]

    def _swap_axes(self, j1, j2):
        """Exchange magic-register bits j1, j2 IN PLACE (swap the (bit_j1=0,bit_j2=1) and
        (bit_j1=1,bit_j2=0) strided sub-blocks via the temp-free 3-op swap) -- a pure
        permutation, no amplitude temporary."""
        if j1 == j2:
            return
        r = self._sz.bit_length() - 1
        t = self.phi.reshape([2] * r)
        a1 = r - 1 - j1
        a2 = r - 1 - j2
        s01 = [slice(None)] * r; s01[a1] = slice(0, 1); s01[a2] = slice(1, 2)
        s10 = [slice(None)] * r; s10[a1] = slice(1, 2); s10[a2] = slice(0, 1)
        a = t[tuple(s01)]; b = t[tuple(s10)]
        a += b; b *= -1.0; b += a; a -= b

    def _drop_axis_inplace(self, j, fold_x_qubit=None):
        """Drop a product axis with NO resize and NO compaction gather: move it to the MSB
        (in-place strided swap), so the kept branch is a CONTIGUOUS half and the drop is
        pure size bookkeeping -- branch 0 keeps the prefix; branch 1 does ONE disjoint
        in-storage memmove of the high half down to the prefix (no new array). Pops M (the
        dropped/localized axis) internally; returns the kept eigen-bit (0/1) or None."""
        self._ensure_inited()
        sq0 = self._branch_sqnorm(j, 0)
        sq1 = self._branch_sqnorm(j, 1)
        if sq1 < 1e-20:
            keep = 0
        elif sq0 < 1e-20:
            keep = 1
        else:
            return None
        r = self._sz.bit_length() - 1
        msb = r - 1
        if j != msb:
            self._swap_axes(j, msb)                        # dropped axis -> MSB (contiguous half)
            self.M[j], self.M[msb] = self.M[msb], self.M[j]
        half = self._sz >> 1
        if keep == 1:                                      # kept = high half -> memmove to prefix
            self._storage[:half] = self._storage[half:self._sz]   # disjoint ranges, no temp
        self.budget.charge(half, 0, "drop")
        self._sz = half
        self.M.pop()                                       # remove the MSB = localized/product axis
        self.phi = self._storage[:self._sz]
        return keep

    # ---- compress that correctly handles |1> product axes, fully in place ----
    def _compress_magic(self):
        """Drop every product magic axis IN PLACE (no amplitude-sized copy). A qubit found
        in product |1> is dropped by folding X_q into the FRAME (U_C <- U_C X_q, i.e. negate
        Zc[q]'s phase) so the bare |0>_q represents the physical |1>_q -- the base version
        rejoins a |1> qubit to |0>_{notM}, SILENTLY LOSING the |1> (harmless in the lazy
        engine where measured qubits stay entangled, but exposed by the purge's deliberate
        localization of the measured dof to a product Z-eigenstate that may be |1>)."""
        changed = True
        while changed and self.M:
            changed = False
            for a in range(len(self.M)):
                q = self.M[a]                         # phi bit a <-> qubit M[a] (M[0]=LSB)
                keep = self._drop_axis_inplace(a)     # swaps a->MSB, drops, pops M internally
                if keep is None:
                    continue
                if keep == 1:                         # |1> -> fold X_q into the frame
                    zr = self.Zc[q]
                    self.Zc[q] = (zr[0], zr[1], (zr[2] + 2) & 3)
                    self._frame_ver += 1
                    if self._inv_enabled:
                        self._inv_fold_x(q)
                changed = True
                break

    # ---- PHASE-1: drop the just-localized measured axis directly (no O(k) product rescan) ----
    _purge_verify = False        # set True in verification to assert no residual product axis

    def _drop_localized_core(self, q, keep):
        """Swap axis q to the MSB, drop it (branch `keep` survives), pop M, and -- for a kept
        |1> branch -- fold X_q into the frame.  Mirrors _compress_magic's drop + |1>-X-fold
        EXACTLY, but takes a KNOWN keep (no _branch_sqnorm product test)."""
        self._ensure_inited()
        j = self.M.index(q)
        r = self._sz.bit_length() - 1
        msb = r - 1
        if j != msb:
            self._swap_axes(j, msb)                            # dropped axis -> MSB (contiguous)
            self.M[j], self.M[msb] = self.M[msb], self.M[j]
        half = self._sz >> 1
        if keep == 1:                                          # kept = high half -> memmove down
            self._storage[:half] = self._storage[half:self._sz]   # disjoint ranges, no temp
        self.budget.charge(half, 0, "drop")
        self._sz = half
        self.M.pop()                                           # remove MSB = q
        self.phi = self._storage[:self._sz]
        if keep == 1:                                          # |1> product -> fold X_q into frame
            zr = self.Zc[q]
            self.Zc[q] = (zr[0], zr[1], (zr[2] + 2) & 3)
            self._frame_ver += 1
            if self._inv_enabled:
                self._inv_fold_x(q)

    def _support_bits(self):
        """OR and AND over the indices of all nonzero amplitudes, in bounded chunks (O(_CHUNK)
        transient -- no 2^r index array).  Axis a's branch_1 is empty iff bit a is clear in OR;
        its branch_0 is empty iff bit a is set in AND.  This is the CHEAP gate (one pass) that
        flags candidate product axes; the exact branch-sqnorm threshold then confirms each."""
        phi = self.phi
        N = phi.size
        CH = min(self._CHUNK, N)
        self.budget.charge(N, CH, "sqnorm")        # one |.|>tol pass, charge like a sqnorm
        or_bits = 0
        and_bits = -1                              # all-ones; AND narrows it
        any_nz = False
        for s in range(0, N, CH):
            e = min(s + CH, N)
            nz = np.nonzero(np.abs(phi[s:e]) > 1e-10)[0]
            if nz.size == 0:
                continue
            nz = nz + s
            or_bits |= int(np.bitwise_or.reduce(nz))
            and_bits &= int(np.bitwise_and.reduce(nz))
            any_nz = True
        return (or_bits, and_bits) if any_nz else (0, 0)

    def _drop_residual_products(self):
        """Drop every remaining product Z-eigenstate axis (a measurement can disentangle a
        SECOND qubit beyond the localized r -- e.g. distillation).  The product-axis set is
        invariant under dropping a product axis, so the cheap _support_bits gate finds every
        candidate in one pass; each candidate is confirmed with the EXACT branch-sqnorm < 1e-20
        threshold (matching _drop_axis_inplace) before dropping.  Common case: one gate pass,
        no candidate, zero branch-sqnorms -- no O(k) rescan."""
        while self.M:
            if self._sz <= 1:
                break
            or_bits, and_bits = self._support_bits()
            target = None
            for a in range(len(self.M)):
                bit = 1 << a
                if not (or_bits & bit):
                    empty = 1                      # branch_1 empty -> keep 0
                elif and_bits & bit:
                    empty = 0                      # branch_0 empty -> keep 1
                else:
                    continue
                if self._branch_sqnorm(a, empty) < 1e-20:     # exact confirm
                    target = (self.M[a], 1 - empty)           # keep = surviving branch
                    break
            if target is None:
                break
            self._drop_localized_core(*target)

    def _drop_localized(self, q, keep):
        """Drop the just-localized measured axis q (KNOWN product, branch `keep` survives),
        then sweep for any residual product axes.  Replaces the O(k)-sqnorm-per-rank
        _compress_magic rescan: the guaranteed axis r is dropped with no sqnorm, and the cheap
        support gate handles the rare second product axis."""
        self._drop_localized_core(q, keep)
        self._drop_residual_products()

    def _assert_no_residual_product(self):
        """Verification-only guard: after _drop_localized (direct drop of r + residual sweep),
        the original O(k) _compress_magic scan must drop NOTHING -- i.e. the cheap support gate
        found exactly what the full scan would.  A non-zero drop here means a missed product
        axis, so raise loudly instead of silently leaving a dead resident axis."""
        before = len(self.M)
        self._compress_magic()
        if len(self.M) != before:
            raise AssertionError(
                f"_drop_localized left {before - len(self.M)} residual product axis/axes at "
                f"measurement {self._meas_log_ctr}: drops/meas==1 invariant violated")

    # ---- the localize-and-drop releases the measured dof every magic measurement, so
    #      the cross-register parity reduction (_reduce_full, whose _find_z_stab .ravel()s
    #      two half-size COMPLEX slices -- the last amplitude-sized transient) is redundant:
    #      the active rank already tracks clifft WITHOUT it (verified: rank/correctness
    #      identical with _reduce_full disabled on every benchmark). Drop it. ----
    def _reduce_full(self):
        return

    # ---- LOCALIZE the measured Pauli to a single Z_r BEFORE the Born ----------
    def _localize_to_Z(self, xp, zp, pp, prefer=None):
        """Apply a block-local Clifford W (the view-based temp-free H/S/CNOT) that maps the
        magic-register Pauli P' = i^pp X^xp Z^zp to sign*Z_r, applying W to phi and folding
        W^dag into the frame (EXACT identity insertion), and tracking the sign by conjugating
        P through W (phase-exact, the R_Y-critical part). Returns (r, sign in {+1,-1}); or
        (None, ev) when P' has no magic support (a deterministic +-1 outcome). Doing this
        BEFORE the Born makes the measurement single-axis -> the Born is a strided-slice norm
        and the collapse a strided-slice zero, both with ZERO amplitude-sized temporary."""
        supp = [s for s in _support(xp, zp) if s in self.M]
        if not supp:
            ev = float(np.real(1j ** (pp & 3)))      # P' = i^pp on |0>_{notM} -> +-1
            return None, (1.0 if ev >= 0 else -1.0)
        r = prefer if (prefer is not None and prefer in supp) else supp[0]
        W = []
        for s in supp:
            xb = (xp >> s) & 1
            zb = (zp >> s) & 1
            if xb and zb:                            # local Y (=XZ) -> S^dag then H -> Z
                W += [("s", s, True), ("h", s)]
            elif xb:                                 # local X -> H -> Z
                W += [("h", s)]
        for s in supp:
            if s != r:
                W.append(("cx", s, r))               # CNOT(ctrl=s,tgt=r): Z_s Z_r -> Z_r
        P = (xp, zp, pp)
        for g in W:                                  # apply to phi, conjugate P, fold frame
            if g[0] == "h":
                self._h_axis(self.M.index(g[1])); P = _conj_h(P, g[1]); self.right_h(g[1])
            elif g[0] == "s":
                self._s_axis(self.M.index(g[1]), g[2]); P = _conj_s(P, g[1], g[2])
                self.right_s(g[1], dag=(not g[2]))
            else:
                self._cnot_axes(self.M.index(g[1]), self.M.index(g[2]))
                P = _conj_cx(P, g[1], g[2]); self.right_cx(g[1], g[2])
        sign = 1.0 if (P[2] & 3) == 0 else -1.0      # P now (0, 1<<r, pp' in {0,2}) = +-Z_r
        return r, sign

    def measure_z(self, q):
        self._flush_core(0, 1 << q)
        Pm = (0, 1 << q, 0)
        magset = set(self.M)
        anti_s = [i for i in range(self.n)
                  if i not in magset and not pauli_commute(self.Zc[i], Pm)]
        M_before = len(self.M)
        p0 = None
        if anti_s:
            out = self._ag_measure(Pm, anti_s)
            branch = "stabilizer"
        else:
            xp, zp, pp = self._pullback(0, 1 << q)
            r, sign = self._localize_to_Z(xp, zp, pp, prefer=q)
            if r is None:                            # deterministic (no magic support)
                p0 = max(0.0, min(1.0, (1.0 + sign) / 2.0))
                out = 0 if float(self.rng.random()) < p0 else 1
            else:
                jr = self.M.index(r)
                # PHASE-1 Born: BOTH branch sqnorms = exactly ONE full sweep (two strided
                # half-views), and tot = s0+s1 is the state's TRUE current norm^2 recomputed
                # each measurement (drift self-correcting, like the old _sqnorm_1d).  This
                # REPLACES the old (half-sweep Born p0r) + (separate full _sqnorm_1d) = 1.5
                # sweeps with one full sweep whose two halves serve BOTH the Born and the
                # post-projection renormalization (reuse, no third pass).
                s0 = self._branch_sqnorm(jr, 0)                  # ‖bit_r = 0‖^2  (strided view)
                s1 = self._branch_sqnorm(jr, 1)                  # ‖bit_r = 1‖^2  (strided view)
                tot = s0 + s1                                    # exact current norm^2
                p0 = ((s0 if sign > 0 else s1) / tot) if tot > 1e-300 else 0.5
                p0 = max(0.0, min(1.0, p0))                      # = (1 + <P'>)/2
                out = 0 if float(self.rng.random()) < p0 else 1
                plus_bit = 0 if sign > 0 else 1                  # +1 eigenvalue <-> this bit
                keepbit = plus_bit if out == 0 else (1 - plus_bit)
                v = self.phi.reshape(-1, 2, 1 << jr)
                v[:, 1 - keepbit, :] = 0.0                       # project (strided view = 0)
                nrm2 = s0 if keepbit == 0 else s1                # REUSE: surviving branch norm^2
                if nrm2 > 1e-24:
                    self.budget.charge(self.phi.size, 0, "normalize")   # 2^r real*complex divide
                    self.phi /= nrm2 ** 0.5                      # in-place normalise (no sweep)
                # PHASE-1 direct drop: drop the localized axis r (KNOWN product, keepbit known)
                # with NO sqnorm, then a cheap support-gate sweep for the rare second product
                # axis -- replacing _compress_magic's O(k)-sqnorm-per-rank rescan (purge sqnorm
                # 0).  _purge_verify asserts the sweep == the original full scan.
                self._drop_localized(r, keepbit)
                if self._purge_verify:
                    self._assert_no_residual_product()
            branch = "magic"
        self._reduce_full()
        if len(self.M) > self.max_M:
            self.max_M = len(self.M)
        self.budget.note_resident(self.phi.size, "post-reduce")
        if self.log_cores:
            self.core_log.append(dict(meas=self._meas_log_ctr, branch=branch,
                                      M_before=M_before, M_after=len(self.M), p0=p0,
                                      peak_live_words=self.budget.peak))
        self._meas_log_ctr += 1
        return out
