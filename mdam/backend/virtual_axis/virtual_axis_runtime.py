"""Virtual-axis near-Clifford RUNTIME backend.

Holds the magic state as ONE dense register `phi` over the active magic qubits M
(no physical-support blocks), and after every magic measurement runs a FULL-REGISTER
parity reduction: it peels every qubit that is parity-slaved by the rest. Because the
search spans the whole register (no block boundaries), it finds the cross-block
relations the block backend's local search cannot -- so |M| is held at the genuine
independent rank, i.e. clifft's active rank k. The physical NC's transient 2^B (B =
raw support, with parity/dead redundancy) never forms.

Reduction = exact identity insertion: a Clifford W (CNOTs collapsing the Z-string onto
the slaved qubit) is applied to phi AND folded into the Clifford frame, so the physical
state is unchanged; only the lazy frame/RNG ordering differs (distribution-exact, like
decouple_demote -- not necessarily bit-identical).

This trades the block backend's tensor-product factoring (which keeps INDEPENDENT
product pieces in separate small blocks) for full LINEAR-rank reduction: virtual-axis
peak |M| == clifft k always, which beats the block backend where parity redundancy
dominates (cultivation_d5: 14 -> 10) and loses where the state tensor-factors
(distillation: 5 vs the block's 2)."""
from __future__ import annotations

import numpy as np

from mdam.backend.lazy import LazyNearClifford
from mdam.backend.block_magic import (
    _vec_cx, _apply_pauli_local, _gf2_solve)


class VirtualAxisNearClifford(LazyNearClifford):
    def __init__(self, n):
        super().__init__(n)
        self.max_M = 0
        self._reduce_cap = 22          # skip the GF(2) search above this rank (cost guard)

    # ---- full-register Z-parity stabiliser of a single magic qubit q ----
    def _find_z_stab(self, q):
        """Parity mz of the OTHER magic qubits such that Z_q (x) Z^mz stabilises phi
        (numerically verified). None if q is not parity-slaved over the whole register."""
        M = self.M
        k = len(M)
        if k <= 1 or k - 1 > self._reduce_cap:
            return None
        j = M.index(q)
        arr = self.phi.reshape(-1, 2, 1 << j)
        a = arr[:, 0, :].ravel(); b = arr[:, 1, :].ravel()
        sa = np.nonzero(np.abs(a) > 1e-9)[0]
        sb = np.nonzero(np.abs(b) > 1e-9)[0]
        if len(sa) == 0 or len(sb) == 0:
            return None                 # already a product -> _compress_magic handles it
        x0 = int(sa[0])
        rows = [(int(x) ^ x0, 0) for x in sa[1:]] + [(int(y) ^ x0, 1) for y in sb]
        mz = _gf2_solve(rows, k - 1)
        if not mz:
            return None
        zmask = 1 << q
        for t in range(k - 1):
            if (mz >> t) & 1:
                zmask |= 1 << M[t if t < j else t + 1]
        Pv = _apply_pauli_local(M, self.phi, 0, zmask, 0)
        if abs(abs(complex(np.vdot(self.phi, Pv))) - 1.0) > 1e-6:
            return None
        return zmask

    def _reduce_full(self):
        """Peel every parity-slaved magic qubit (CNOT-collapse the Z-string onto it,
        fold W into the frame, compress). Idempotent; bounds |M| to the active rank."""
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
                        self.phi = _vec_cx(self.phi, pos[s], pos[q])  # CNOT(s->q)
                        self.right_cx(s, q)
                self._compress_magic()          # q is now a product Z-eigenstate
                changed = True
                break

    def measure_z(self, q):
        out = super().measure_z(q)              # lazy core flush + collapse + compress
        self._reduce_full()                     # then drop every parity-slaved qubit
        if len(self.M) > self.max_M:
            self.max_M = len(self.M)
        return out
