"""Policy-3 persistent-split engine (Step B1) -- DEFAULT OFF, isolated subclass.

Realizes the FLOP win that Step B0's differential shadow proved exact: instead of pulling the full
Clifford frame INTO each rotation generator (-> off-diagonal butterfly c=12, or localizer H+diag c=7),
materialize each magic axis in its BORN basis at promote (a PHYSICAL IDENTITY -- H on the fresh |0>
array bit folded by right_h into the frame), so the triggering generator lands DIAGONAL (mz-only) and
every same-basis rotation on that axis is a c~2-3 diagonal half-array with NO runtime Hadamard and NO
butterfly.  A generator that does NOT land diagonal in the current born basis (a genuine non-Pauli
active-axis basis change -- the Phase-9 case-2 / cross-measurement re-basis) FALLS BACK to the parent's
exact butterfly/localizer.  Measurement / drop / AG / frame kernels are INHERITED UNCHANGED: the born
fold is just another frame entry, so the representation stays globally consistent and every OBSERVABLE
(records / rank / Born p0) is identical to the parent -- verified bit-exact vs a05843e.

This file is the ONLY Policy-3 code; bounded.py / engine.py are untouched.  Selected only when the
backend is built with clifft_axis_policy3=True; otherwise the committed bounded path runs verbatim.
"""
from __future__ import annotations

import numpy as np

from nearclifford_backend.clifft_axis.bounded import CliftAxisBoundedNearClifford


class CliftAxisPolicy3NearClifford(CliftAxisBoundedNearClifford):
    # ---- diagnostics (per shot) ----
    _p3_diag = 0          # rotations dispatched as a diagonal half-array (the win)
    _p3_fallback = 0      # rotations that still needed the butterfly/localizer (non-Pauli re-basis)
    _p3_bornH = 0         # born-basis Hadamards paid at promote (one per born-X axis)

    def __init__(self, n):
        super().__init__(n)
        self._p3_diag = 0
        self._p3_fallback = 0
        self._p3_bornH = 0

    # ---- promote a qubit AND materialize it in the X (born) basis -------------------------------
    def _promote_born_x(self, q):
        """Promote q (parent: fresh MSB axis in |0>) then rotate it to |+> and fold H_q into the
        frame.  `_h_axis(j)` on a |0>-high axis is exactly the |+> fill (a->a/v2, b<-a/v2), and
        right_h(q) folds H into U_C; together they are a PHYSICAL IDENTITY (H on array * H in frame
        = I) that merely re-expresses axis q in the X-eigenbasis.  After this the triggering X_q
        generator pulls back to Z_q (diagonal)."""
        self._promote(q)                       # parent bounded promote: |0> high block
        j = self.M.index(q)
        self._h_axis(j)                        # |0> -> |+>  (born-X)  [the one born Hadamard]
        self.right_h(q)                        # fold H_q into the frame (physical identity)
        self._p3_bornH += 1

    # ---- flush one rotation: diagonal dispatch in the born basis, else exact fallback -----------
    def _flush_one(self, x, z, theta, phase=0):
        xp, zp, pp = self._pullback(x, z)
        # promote every X-support qubit in the BORN-X basis so its X-character becomes diagonal Z
        for qq in range(self.n):
            if (xp >> qq) & 1 and qq not in self.M:
                self._promote_born_x(qq)
        # re-pull the generator through the now-folded frame
        xp, zp, pp = self._pullback(x, z)
        pp = (pp + phase) & 3
        mx, mz = self._masks(xp, zp, promote=True, where="rot")
        if len(self.M) > self.max_M:
            self.max_M = len(self.M)
        if self.cap is not None and len(self.M) > self.cap:
            from nearclifford_backend.backend import MagicCapExceeded
            raise MagicCapExceeded(-1, len(self.M))
        if mx == 0:                            # DIAGONAL in the born basis -> c~2-3 half-array, 0 H, 0 butterfly
            c = np.cos(theta / 2.0); s = np.sin(theta / 2.0)
            self._pauli_lincomb_inplace(0, mz, pp, alpha=c, beta=(-1j * s), where="rot")
            self._p3_diag += 1
            return
        # genuine non-Pauli active-axis basis change -> exact parent path (butterfly / localizer)
        self._p3_fallback += 1
        super()._flush_one(x, z, theta, phase)
