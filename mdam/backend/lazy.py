"""Lazy near-Clifford simulator: defer non-Clifford rotations as PENDING Pauli
rotations and materialise only the anticommutation-connected core at each
measurement. Realises "push all Cliffords into the frame, store each RZ as a Pauli
rotation, group commuting rotations, materialise only the anticommuting core".

State form (all Cliffords pushed to the right of the rotations):

    |psi> = ( prod_j R_{L_j}(theta_j) )  U_C  ( |0>_{notM} (x) |phi>_M )

* L_j -- the rotation generators in the PHYSICAL (lab) frame: an RZ on qubit q is
  R_{Z_q}, so L_j = Z_q at application time. Subsequent Cliffords G conjugate every
  pending generator (L_j <- G L_j G^dag) -- O(1) per pending Pauli per 1-2q gate.
  Physical-frame storage is the crux: it is INVARIANT under a measurement's tableau
  relabelling (if U_C -> U_C V then U_C P U_C^dag is unchanged), so a pending
  rotation survives across measurements correctly. (Pre-frame storage is NOT
  invariant -- the projection relabels the |0> frame.)
* U_C, |phi>_M -- inherited Clifford tableau + dense magic register (NearClifford).

A measurement of physical Z_q: the generators L_j that commute with Z_q (directly,
and transitively through the anticommutation graph) commute through the projector
and stay pending; only the anticommutation-connected core of Z_q is flushed
(applied to the dense register, pulling each generator back through U_C). A qubit
born |+> with a coherent R_Z read later in Z keeps L = Z_q commuting with the Z_q
measurement -> never flushed (free). An ancilla whose R_Z is turned to X_anc by a
syndrome-extraction H anticommutes with its Z measurement -> flushed, but it is
re-measured each round so the live core stays small.

Verified against dense (scripts/verify_lazy.py) and clifft (scripts/verify_backend.py --lazy).
"""
from __future__ import annotations

import numpy as np

from mdam.backend.simulator import NearClifford


# --- single-Pauli Clifford conjugations P -> G P G^dag, P=(x,z,phase) ---------
def _conj_h(P, q):
    x, z, p = P; bit = 1 << q
    xq = (x >> q) & 1; zq = (z >> q) & 1
    x2 = (x & ~bit) | (zq << q)
    z2 = (z & ~bit) | (xq << q)
    return (x2, z2, (p + 2 * (xq & zq)) & 3)


def _conj_s(P, q, dag):
    x, z, p = P
    xq = (x >> q) & 1
    z2 = z ^ (xq << q)
    return (x, z2, (p + (xq * (3 if dag else 1))) & 3)


def _conj_cx(P, c, t):
    x, z, p = P; bt = 1 << t; bc = 1 << c
    xc = (x >> c) & 1; xt = (x >> t) & 1
    zc = (z >> c) & 1; zt = (z >> t) & 1
    x2 = x | (xc << t) if xc else x
    x2 = (x2 & ~bt) | (((xt ^ xc) & 1) << t)
    z2 = (z & ~bc) | (((zc ^ zt) & 1) << c)
    return (x2, z2, p)


def _commute_xz(ax, az, bx, bz):
    return (((ax & bz).bit_count() + (az & bx).bit_count()) & 1) == 0


class LazyNearClifford(NearClifford):
    def __init__(self, n):
        super().__init__(n)
        # pending is a uid -> [x, z, phase, theta, uid] MAP (insertion-ordered). uids
        # increase monotonically (apply_rotation) and conjugation/flush preserve order,
        # so list(pending.values()) is ALWAYS in increasing-uid order. The map gives the
        # runtime fast path O(1) gather of a precomputed core by uid.
        self.pending = {}
        self.max_M = 0
        self.cap = None            # if set, raise MagicCapExceeded when |M| exceeds
        # resource_only: do NOT materialise the dense register; just record the core
        # SUPPORT size (the |M| a flush would need) and evolve the tableau as a
        # Gottesman-Knill stabilizer sim. Gives exact core sizes with no 2^|M| cost.
        self.resource_only = False
        # --- dead-rotation pruning (structure-once) -------------------------------
        # Every apply_rotation gets a stable creation id (uid). The active-gate stream
        # is outcome-independent (records only steer the Pauli FRAME, never the NC
        # tableau/rotations), so a rotation's uid -> flush behaviour is shot-invariant.
        # A rotation that is NEVER flushed (never enters any measurement's anticommuting
        # core, transitively) commutes with every measured Pauli forever -> it never
        # touches the dense register and never affects a record bit. Dropping it is
        # record-bit-IDENTICAL. _dead_uids holds those uids (filled by a one-off
        # structure pass); _flushed_uids, when a set, records flushes (the structure pass).
        self._rot_uid = 0
        self._dead_uids = None     # set of never-flushed uids to drop; None = keep all
        self._flushed_uids = None  # set; if not None, record every flushed uid
        # --- structure-once runtime fast path -------------------------------------
        # The anticommuting core flushed at each measurement is ALSO outcome-independent.
        # _meas_ctr indexes measurements (one per _flush_core call), aligned across shots.
        # PRE-PASS: _record_cores is a dict {meas_idx -> [core uids in flush order]}.
        # RUNTIME: _fast_cores is that table; at each measurement we look up the core and
        # gather it from the uid map instead of re-scanning all pending with _core_indices.
        # _debug_compare cross-checks the precomputed core against a live _core_indices scan.
        self._meas_ctr = 0
        self._record_cores = None  # dict to FILL with per-measurement core uids (pre-pass)
        self._fast_cores = None    # dict to USE  (runtime fast path); None = dynamic scan
        self._debug_compare = False
        # counters (per shot; reset by the backend's _reset via fresh sim construction)
        self._cnt_dynamic_core_scan = 0   # # of live _core_indices scans
        self._cnt_fastpath_lookup = 0     # # of precomputed-core lookups
        self._cnt_commute_xz = 0          # # of _commute_xz calls (the scan cost)
        self._fast_mismatch_count = 0     # # of debug/structural fallbacks to dynamic

    # ---- Clifford gates: update tableau (super) AND conjugate pending (order kept) ----
    def h(self, q):
        super().h(q)
        self.pending = {u: [*_conj_h((r[0], r[1], r[2]), q), r[3], r[4]]
                        for u, r in self.pending.items()}

    def s(self, q, dag=False):
        super().s(q, dag)
        self.pending = {u: [*_conj_s((r[0], r[1], r[2]), q, dag), r[3], r[4]]
                        for u, r in self.pending.items()}

    def cx(self, c, t):
        super().cx(c, t)
        self.pending = {u: [*_conj_cx((r[0], r[1], r[2]), c, t), r[3], r[4]]
                        for u, r in self.pending.items()}

    def cz(self, a, b):
        # cz = H_b CX(a,b) H_b ; reuse the composed conjugations on pending via super
        super().cz(a, b)
        new = {}
        for u, r in self.pending.items():
            P = (r[0], r[1], r[2])
            P = _conj_h(P, b); P = _conj_cx(P, a, b); P = _conj_h(P, b)
            new[u] = [P[0], P[1], P[2], r[3], r[4]]
        self.pending = new

    # ---- defer a rotation (physical generator (x,z)) instead of applying ----
    def apply_rotation(self, x, z, theta):
        uid = self._rot_uid
        self._rot_uid += 1
        if self._dead_uids is not None and uid in self._dead_uids:
            return                                 # never-flushed: drop (record-exact)
        self.pending[uid] = [x, z, 0, theta, uid]

    # ---- apply an already-PHYSICAL generator to the dense register (flush) ----
    def _flush_one(self, x, z, theta):
        xp, zp, pp = self._pullback(x, z)        # physical -> pre-frame
        for qq in range(self.n):
            if (xp >> qq) & 1 and qq not in self.M:
                self._promote(qq)
        if len(self.M) > self.max_M:
            self.max_M = len(self.M)
        if self.cap is not None and len(self.M) > self.cap:
            from mdam.backend.backend import MagicCapExceeded
            raise MagicCapExceeded(-1, len(self.M))
        Pphi = self._apply_magic_pauli(xp, zp, pp)
        c = np.cos(theta / 2.0); s = np.sin(theta / 2.0)
        self.phi = c * self.phi - 1j * s * Pphi

    # ---- anticommutation-connected core of measured physical Pauli (qx,qz) ----
    # Returns a boolean list aligned to list(self.pending.values()) (increasing-uid order)
    # -- the live O(pending^2) scan. Counts every _commute_xz call into _cnt_commute_xz.
    def _core_indices(self, qx, qz):
        entries = list(self.pending.values())
        n = len(entries)
        in_core = [False] * n
        stack = []
        cc = 0
        for j, r in enumerate(entries):
            cc += 1
            if not _commute_xz(qx, qz, r[0], r[1]):
                in_core[j] = True; stack.append(j)
        while stack:
            j = stack.pop(); rj = entries[j]
            for k in range(n):
                if not in_core[k]:
                    rk = entries[k]
                    cc += 1
                    if not _commute_xz(rj[0], rj[1], rk[0], rk[1]):
                        in_core[k] = True; stack.append(k)
        self._cnt_commute_xz += cc
        self._cnt_dynamic_core_scan += 1
        return in_core

    def _dynamic_core(self, qx, qz):
        """Live scan -> ordered list of pending ENTRIES to flush (increasing-uid order)."""
        entries = list(self.pending.values())
        in_core = self._core_indices(qx, qz)
        return [entries[j] for j in range(len(entries)) if in_core[j]]

    def _flush_core(self, qx, qz):
        meas_idx = self._meas_ctr
        self._meas_ctr += 1
        if self._fast_cores is None:
            # DYNAMIC PATH (pre-pass, structure-once disabled, or feedback circuit).
            flush = self._dynamic_core(qx, qz)
            if self._record_cores is not None:     # pre-pass: store this core's uids
                self._record_cores[meas_idx] = [r[4] for r in flush]
            self._do_flush(qx, qz, flush)
            return
        # FAST PATH: look up the precomputed core and gather it from the uid map.
        core_uids = self._fast_cores.get(meas_idx, ())
        self._cnt_fastpath_lookup += 1
        flush = None
        if self._debug_compare:                    # cross-check vs a live scan
            dyn = self._dynamic_core(qx, qz)
            if [r[4] for r in dyn] != list(core_uids):
                self._fast_mismatch_count += 1
                flush = dyn                        # fall back to the verified live core
        if flush is None:
            flush = []
            for u in core_uids:                    # gather in recorded (uid) order
                e = self.pending.get(u)
                if e is None:                      # structural miss -> safe fallback
                    self._fast_mismatch_count += 1
                    flush = self._dynamic_core(qx, qz)
                    break
                flush.append(e)
        self._do_flush(qx, qz, flush)

    def _do_flush(self, qx, qz, flush):
        """Apply an ordered list of pending core ENTRIES: remove from the uid map, log
        flushed uids, then materialise (or, in resource_only, just size the support)."""
        if not flush:
            return
        for r in flush:
            del self.pending[r[4]]
        if self._flushed_uids is not None:       # structure pass: log which uids flush
            for r in flush:
                self._flushed_uids.add(r[4])
        if self.resource_only:
            # |M| this flush would need = #qubits in the union X-support of the
            # pulled-back core generators + the measured qubit's pulled-back X-support.
            supp = 0
            qxp, qzp, _ = self._pullback(qx, qz)
            supp |= qxp
            for (x, z, p, theta, uid) in flush:
                xp, zp, pp = self._pullback(x, z)
                supp |= xp
            self.max_M = max(self.max_M, supp.bit_count())
            return
        for (x, z, p, theta, uid) in flush:  # increasing-uid order preserved
            self._flush_one(x, z, theta)

    # ---- measurement: flush core for the measured physical Pauli, then measure ----
    def measure_z(self, q):
        self._flush_core(0, 1 << q)            # measured physical Pauli = Z_q
        return super().measure_z(q)

    # ---- verification helper: realise the full state ----
    def statevector(self):
        for (x, z, p, theta, uid) in self.pending.values():
            self._flush_one(x, z, theta)
        self.pending = {}
        return super().statevector()

    def live_magic(self):
        return len(self.M)
