"""Block-factored magic register.

The plain near-Clifford backend stores the whole magic register as ONE dense
vector of dimension 2^|M|. That conflates *entanglement* with mere *count*: a
coherent R_Z on a |+>-born data qubit makes it non-stabilizer (magic) but it is
an unentangled single-qubit (equatorial) state -- it should cost dim 2, not
double the whole register. Holding it in the monolithic vector makes |M|
accumulate (coherent_d5_r5 -> >20) even though the genuinely-entangled core is ~7.

`MagicRegister` instead stores the magic part as a TENSOR PRODUCT of independent
blocks {(qubits, vector)}, and after every operation peels off single-qubit
product factors (`factor()`): an unentangled qubit becomes its own dim-2 block,
and a |0> qubit leaves the register entirely. The cost of the representation is
the LARGEST block (`max_block()`), so equatorial data qubits no longer inflate it.

This is the missing piece that turns the small per-measurement anticommuting core
(measured ~7 and flat in distance) into the actual live resource.
"""
from __future__ import annotations

import numpy as np

_TOL = 1e-9
_INV_SQRT2 = 0.7071067811865476


# --- single-/two-qubit Clifford gates applied to a dense block vector (bit j = LSB
#     of qubit qubits[j]). Used by the measured-magic purge to fold a block-local
#     Clifford W into the vector; W^dag is simultaneously folded into the frame. ---
def _vec_h(vec, j):
    a = vec.reshape(-1, 2, 1 << j)
    x0 = a[:, 0, :].copy(); x1 = a[:, 1, :].copy()
    out = np.empty_like(a)
    out[:, 0, :] = (x0 + x1) * _INV_SQRT2
    out[:, 1, :] = (x0 - x1) * _INV_SQRT2
    return out.reshape(-1)


def _vec_s(vec, j, dag):
    a = vec.reshape(-1, 2, 1 << j).copy()
    a[:, 1, :] *= (-1j if dag else 1j)     # S = diag(1, i); S^dag = diag(1, -i)
    return a.reshape(-1)


def _vec_cx(vec, jc, jt):
    idx = np.arange(vec.size)
    perm = np.where((idx >> jc) & 1, idx ^ (1 << jt), idx)   # flip target where ctrl=1
    return vec[perm]


def _apply_pauli_local(qubits, vec, xmask, zmask, phase):
    """Apply i^phase * X^x Z^z (global masks) restricted to a block's `qubits`
    to its `vec`. qubits[j] is bit j (LSB) of the block vector."""
    k = len(qubits)
    mx = mz = 0
    for j, q in enumerate(qubits):
        if (xmask >> q) & 1:
            mx |= 1 << j
        if (zmask >> q) & 1:
            mz |= 1 << j
    idx = np.arange(1 << k, dtype=np.int64)
    v = idx & mz
    for sh in (32, 16, 8, 4, 2, 1):
        v ^= v >> sh
    sign = (1j ** phase) * (1 - 2 * (v & 1))
    out = np.empty_like(vec)
    out[idx ^ mx] = sign * vec[idx]
    return out


class MagicRegister:
    def __init__(self):
        self.blocks = []           # list of [qubits(list), vec(np.ndarray)]
        self.q2b = {}              # qubit -> block index
        # --- FLOP accounting (purely additive; never affects the trajectory) ---
        # complex-arith convention: mult=6, add=2, |z|^2-accumulate(norm)=4, vdot=8
        self.flop_mm = 0.0         # state-evolution work (rotation apply, kron, measure)
        self.flop_norm = 0.0       # factoring work (the norm/vdot scans in factor())

    # ---- queries ----
    def qubits(self):
        return set(self.q2b.keys())

    def has(self, q):
        return q in self.q2b

    def max_block(self):
        return max((len(b[0]) for b in self.blocks), default=0)

    def total(self):
        return len(self.q2b)

    # ---- structural ops ----
    def promote(self, q):
        if q in self.q2b:
            return
        self.q2b[q] = len(self.blocks)
        self.blocks.append([[q], np.array([1.0 + 0j, 0.0])])

    def _merge(self, support):
        """Merge every block touching any qubit in `support` (promoting missing
        ones) into a single block; return its index."""
        for q in support:
            if q not in self.q2b:
                self.promote(q)
        bidxs = sorted({self.q2b[q] for q in support})
        if len(bidxs) == 1:
            return bidxs[0]
        # combine in increasing index order; vector = kron(higher, lower-as-LSB)
        keep = bidxs[0]
        qubits, vec = self.blocks[keep][0][:], self.blocks[keep][1]
        for bi in bidxs[1:]:
            q2, v2 = self.blocks[bi]
            vec = np.kron(v2, vec)         # existing `vec` qubits stay LSB
            self.flop_mm += 6.0 * vec.size   # complex kron (one mult per output amp)
            qubits = qubits + q2
        self.blocks[keep] = [qubits, vec]
        # blank out merged-away blocks, then rebuild compactly
        for bi in bidxs[1:]:
            self.blocks[bi] = None
        self._rebuild()
        return self.q2b[qubits[0]]

    def _rebuild(self):
        new = [b for b in self.blocks if b is not None]
        self.blocks = new
        self.q2b = {}
        for i, (qs, _) in enumerate(self.blocks):
            for q in qs:
                self.q2b[q] = i

    # ---- physics ----
    def apply_rotation(self, xmask, zmask, phase, theta):
        """exp(-i theta P/2) with P = i^phase X^x Z^z (global masks)."""
        support = _support(xmask, zmask)
        b = self._merge(support)
        qubits, vec = self.blocks[b]
        Pv = _apply_pauli_local(qubits, vec, xmask, zmask, phase)
        c = np.cos(theta / 2.0); s = np.sin(theta / 2.0)
        self.blocks[b][1] = c * vec - 1j * s * Pv
        self.flop_mm += 6.0 * vec.size + 14.0 * vec.size  # Pauli apply + (c*vec - i s*Pv)

    def measure_pauli(self, xmask, zmask, phase, rng):
        """Measure the +-1 Pauli P=i^phase X^x Z^z; sample outcome, collapse,
        return 0 (=+1) / 1 (=-1). Z-support on non-magic qubits acts as +1 (|0>)."""
        support = _support(xmask, zmask)
        if not support:
            return 0
        b = self._merge(support)
        qubits, vec = self.blocks[b]
        Pv = _apply_pauli_local(qubits, vec, xmask, zmask, phase)
        exp = float(np.real(np.vdot(vec, Pv)))
        p0 = min(1.0, max(0.0, 0.5 * (1.0 + exp)))
        out = 0 if float(rng.random()) < p0 else 1
        sign = 1.0 if out == 0 else -1.0
        proj = 0.5 * (vec + sign * Pv)
        nrm = np.linalg.norm(proj)
        if nrm > 1e-12:
            self.blocks[b][1] = proj / nrm
        # Pauli apply(6) + vdot(8) + proj(14) + norm(4) + divide(2) per amplitude.
        # This is core measurement-collapse work (clifft pays it too) -> matmul col.
        self.flop_mm += 34.0 * vec.size
        return out

    # ---- factor out single-qubit product structure ----
    def factor(self, only=None):
        """Peel single-qubit product factors. If `only` is a set of qubit indices,
        only those qubits' positions are probed (the rest of every block is left
        untouched). That is exact after a rotation `exp(-i th P_S/2)`: it is a LOCAL
        UNITARY on its support `S` (`P_S = P'_S (x) I_rest`), so it cannot change the
        factorability of any qubit OUTSIDE `S` -- only `S` qubits can newly factor.
        A measurement is a non-unitary projection and CAN disentangle qubits outside
        its support, so the measurement path passes `only=None` (full scan). This
        turns the per-op cost from O(sum_b |Q_b|*2^|Q_b|) (rescan every block every
        op -- the dominant `flop_norm`) into O(|S|*2^|touched block|)."""
        i = 0
        while i < len(self.blocks):
            changed = self._factor_block(i, only)
            if not changed:
                i += 1
            # if changed, re-examine the (possibly shrunk) block at i
        # drop empty blocks / rebuild indices
        self.blocks = [b for b in self.blocks if b is not None and len(b[0]) > 0]
        self._reindex()

    def _factor_block(self, i, only=None):
        qubits, vec = self.blocks[i]
        k = len(qubits)
        if k <= 1:
            return False
        cand = range(k) if only is None else [j for j in range(k) if qubits[j] in only]
        if not cand:
            return False
        for j in cand:
            # qubit position j == bit j (LSB) of the flat vector. Reshape to
            # (high, 2, low) with the middle axis = bit j (low = 2^j), so [:,0,:]
            # / [:,1,:] are the bit_j=0 / =1 slices -- a strided view, far cheaper
            # than np.take(axis=...). The C-order ravel reproduces the exact LSB
            # repacking (drop bit j, shift higher bits down) the splitter expects.
            arr3 = vec.reshape(-1, 2, 1 << j)
            b0 = arr3[:, 0, :].ravel()
            b1 = arr3[:, 1, :].ravel()
            n0 = np.linalg.norm(b0); n1 = np.linalg.norm(b1)
            self.flop_norm += 4.0 * b0.size + 4.0 * b1.size     # the two probe norms
            if n1 < _TOL * max(n0, 1e-30):           # qubit |0>: drop it
                self._split(i, j, qubit_state=None, rest=b0)
                return True
            if n0 < _TOL * max(n1, 1e-30):           # qubit |1>: own block
                self._split(i, j, qubit_state=np.array([0.0 + 0j, 1.0]), rest=b1)
                return True
            alpha = np.vdot(b0, b1) / np.vdot(b0, b0)
            self.flop_norm += 8.0 * b0.size + 8.0 * b0.size     # two vdots
            self.flop_norm += 12.0 * b0.size                    # alpha*b0 + sub + norm
            if np.linalg.norm(b1 - alpha * b0) < _TOL * n1:   # product (equatorial)
                qs = np.array([1.0 + 0j, alpha]); qs = qs / np.linalg.norm(qs)
                self._split(i, j, qubit_state=qs, rest=b0 / n0)
                return True
        return False

    def _split(self, i, j, qubit_state, rest):
        """Remove qubit at position j from block i. If qubit_state is None the
        qubit is |0> and leaves the register; else it becomes its own block."""
        qubits, _ = self.blocks[i]
        q = qubits[j]
        rest_qubits = qubits[:j] + qubits[j + 1:]
        self.blocks[i] = [rest_qubits, rest]
        if qubit_state is None:
            del self.q2b[q]                # rejoins |0>_{notM}
        else:
            self.blocks.append([[q], qubit_state])
        self._reindex()

    def _reindex(self):
        self.blocks = [b for b in self.blocks if b is not None]
        self.q2b = {}
        for idx, (qs, _) in enumerate(self.blocks):
            for q in qs:
                self.q2b[q] = idx

    def live_stats(self, active):
        """READ-ONLY accounting of the genuine *active* resource: ignore blocks all
        of whose qubits are inactive (measured out -- a dead tensor factor an ideal
        backend would have dropped; the measure path keeps it resident because the
        magic-register membership is coupled to the stabilizer tableau, so it is NOT
        safe to drop mid-run). Returns (max_active_block_qubits, sum 16*2^|block|).
        Does NOT mutate the register, so it never perturbs the simulation."""
        mx = 0
        magic_bytes = 0
        for qs, _ in self.blocks:
            if any(q in active for q in qs):
                mx = max(mx, len(qs))
                magic_bytes += 16 * (1 << len(qs))
        return mx, magic_bytes

    # ---- dense reconstruction over a given qubit ordering (verification) ----
    def amplitude_table(self):
        """Return (qubits_order, vec) for the full magic state as one dense vector
        (only used by statevector() for verification; may be large)."""
        if not self.blocks:
            return [], np.array([1.0 + 0j])
        qubits = []
        vec = np.array([1.0 + 0j])
        for qs, v in self.blocks:
            vec = np.kron(v, vec)
            qubits = qubits + qs
        return qubits, vec


def _support(xmask, zmask):
    m = xmask | zmask
    out = []
    while m:
        low = m & -m
        out.append(low.bit_length() - 1)
        m ^= low
    return out


def _gf2_solve(rows, n):
    """Find any n-bit x with parity(mask & x) == rhs for each (mask, rhs) in rows, or
    None if the system is inconsistent. Reduced Gaussian elimination over GF(2); free
    bits are 0 in the returned solution."""
    piv = []                                   # (mask, rhs), distinct reduced leads
    for m, r in rows:
        for pm, pr in piv:                     # reduce against existing pivots
            if m & (pm & -pm):
                m ^= pm; r ^= pr
        if m == 0:
            if r:
                return None                    # 0 == 1 -> inconsistent
            continue
        lead = m & -m
        piv = [((pm ^ m, pr ^ r) if (pm & lead) else (pm, pr)) for (pm, pr) in piv]
        piv.append((m, r))
    x = 0
    for pm, pr in piv:
        if pr:
            x |= (pm & -pm)
    return x


# ===========================================================================
# Lazy near-Clifford simulator with a BLOCK-FACTORED magic register.
# Reuses LazyNearClifford's tableau + pending-rotation deferral; replaces the
# monolithic dense `phi` with a MagicRegister (tensor product of entangled blocks).
# The live resource is max_block() (the largest entangled block), not the total
# magic-qubit count -- so product / equatorial qubits cost dim 2 each.
# ===========================================================================
from nearclifford_backend.simulator import pauli_commute            # noqa: E402
from nearclifford_backend.lazy import LazyNearClifford              # noqa: E402


class BlockLazyNearClifford(LazyNearClifford):
    def __init__(self, n):
        super().__init__(n)
        self.mag = MagicRegister()
        self.max_M = 0            # peak max-block size (the live resource)
        self.cap = None
        self.step_mem_peak = 0    # peak memory_bytes() since the last per-step record
        self.step_block_peak = 0  # peak max_block() (magic qubits) since last record
        # frame-reduction (opt-in): at a magic measurement, peel the DEMOTED index q
        # itself (W-peel collapses onto r=q when q is in the consumed support) instead
        # of supp[0], so the just-demoted qubit does not linger as dead residue. Exact
        # identity insertion (state-exact, distribution-exact); only the RNG ordering
        # differs, so it is NOT bit-identical to the default -- hence opt-in.
        self.decouple_demote = False

    def memory_bytes(self):
        """Actual resident memory of the near-Clifford representation:
          magic register = sum_B 16 * 2^|block_B| (complex128 vectors, the only
            exponential term -- one dense vector per independent entangled block),
          + Clifford tableau (2n Pauli images, x/z bitmasks -- polynomial, 'free'),
          + pending physical-frame rotations (lazy deferral, polynomial).
        The magic term dominates whenever there is genuine magic; for a pure
        stabilizer circuit it is 0 and only the tiny tableau remains."""
        magic = sum(16 * (1 << len(b[0])) for b in self.mag.blocks)
        return magic + self.overhead_bytes()

    def overhead_bytes(self):
        """Polynomial (non-exponential) part: the Clifford tableau (2n Pauli images,
        x/z bitmasks) plus the pending physical-frame rotations (lazy deferral)."""
        wbytes = (self.n + 7) // 8                 # bytes per n-qubit bitmask
        tableau = 2 * self.n * (2 * wbytes + 1)    # Xc,Zc images: x,z masks + phase
        pending = len(self.pending) * (2 * wbytes + 16)
        return tableau + pending

    def _bump(self):
        mb = self.mag.max_block()
        if mb > self.max_M:
            self.max_M = mb
        if mb > self.step_block_peak:
            self.step_block_peak = mb
        m = self.memory_bytes()
        if m > self.step_mem_peak:
            self.step_mem_peak = m
        if self.cap is not None and mb > self.cap:
            from nearclifford_backend.backend import MagicCapExceeded
            raise MagicCapExceeded(-1, mb)

    def take_step_peak(self):
        """Return the (max_block, memory_bytes) INTRA-step high-water mark reached
        since the last call, then rearm the accumulators to the current settled
        state. A per-step recorder calls this to capture the transient memory peak
        -- the genuine high-water mark a measurement's anticommutation-core flush
        briefly forms (all pending rotations applied + factored) just BEFORE the
        measurement projector collapses the block -- instead of only the settled
        step-boundary value. The settled value under-reports the true peak (e.g.
        coherent_d5_r5: transient max_block 13 / ~133 KB vs settled resident 12 /
        ~72 KB); for a memory-feasibility figure the transient is the honest,
        conservative number. Rearming to the settled state (not 0) keeps the
        carried resident as each interval's baseline, so steps with no magic op
        (pure Clifford) still report the resident they hold."""
        blk, mem = self.step_block_peak, self.step_mem_peak
        settled_blk = self.mag.max_block()
        settled_mem = self.memory_bytes()
        self.step_block_peak = settled_blk
        self.step_mem_peak = settled_mem
        return max(blk, settled_blk), max(mem, settled_mem)

    # ---- flush one pending PHYSICAL generator into the block register ----
    def _flush_one(self, x, z, theta):
        xp, zp, pp = self._pullback(x, z)         # physical -> pre-frame
        self.mag.apply_rotation(xp, zp, pp, theta)
        # a rotation is a local unitary on its support -> only support qubits can
        # newly factor; restrict the scan to them (Problem 1: cut the factor FLOP).
        self.mag.factor(only=set(_support(xp, zp)))
        self._bump()

    # ---- measurement: lazy core flush, then stabilizer- or block-measure ----
    def measure_z(self, q):
        self._flush_core(0, 1 << q)               # flush anticommuting core
        Pm = (0, 1 << q, 0)
        magset = self.mag.qubits()
        anti_s = [i for i in range(self.n)
                  if i not in magset and not pauli_commute(self.Zc[i], Pm)]
        if anti_s:
            return self._ag_measure(Pm, anti_s)   # pure stabilizer measurement
        xp, zp, pp = self._pullback(0, 1 << q)
        out = self.mag.measure_pauli(xp, zp, pp, self.rng)
        # absorb the consumed dof; frame-reduction peels the demoted index q itself
        # when q carries the consumed dof (q in the pulled-back support).
        self._purge_redundant(xp, zp, prefer=(q if self.decouple_demote else None))
        self._bump()
        return out

    # ---- purge the dof a magic measurement consumed (Problem 2: no dead resident) --
    def _purge_redundant(self, xp, zp, prefer=None):
        """A magic-path projection on P' = i^pp X^xp Z^zp leaves the touched block in
        a +-1 eigenspace of P' -- one qubit is now redundant (the measurement consumed
        a dof). Reduce P' to a single-qubit Z_r with a block-local Clifford W (turn
        each support qubit's Pauli into Z via H / S^dag,H, then CNOT-collapse the
        Z-string onto r=supp[0]), apply W to the block vector and fold W^dag into the
        frame (U_C <- U_C W^dag -- an EXACT identity insertion, |psi> unchanged), so r
        becomes a product Z-eigenstate that factor() peels. Without this the measured
        (now dead) magic qubit stays entangled-resident and inflates the block
        (e.g. cultivation_d3's 1-live+3-dead block). Updating BOTH the frame and the
        register membership keeps the stabilizer/magic measurement-path decision
        consistent, so the trajectory distribution is preserved (a naive drop that
        touched only membership would corrupt it)."""
        supp = [s for s in _support(xp, zp) if self.mag.has(s)]
        if not supp:
            self.mag.factor()
            return
        # collapse target r: prefer the demoted index q (frame-reduction) so the
        # just-demoted qubit is the one peeled; else supp[0]. Exact for any r in supp.
        r = prefer if (prefer is not None and prefer in supp) else supp[0]
        b = self.mag.q2b[r]
        qubits = self.mag.blocks[b][0]
        pos = {s: qubits.index(s) for s in supp}
        W = []                                    # gates applied to the ket, in order
        for s in supp:
            xb = (xp >> s) & 1; zb = (zp >> s) & 1
            if xb and zb:                         # local Y (=XZ) -> S^dag then H -> Z
                W.append(('s', s, True)); W.append(('h', s))
            elif xb:                              # local X -> H -> Z
                W.append(('h', s))
            # local Z: already Z
        for s in supp:
            if s != r:
                W.append(('cx', s, r))            # CNOT(ctrl=s,tgt=r): Z_s Z_r -> Z_r
        vec = self.mag.blocks[b][1]
        for g in W:                               # apply W to the block vector
            if g[0] == 'h':   vec = _vec_h(vec, pos[g[1]])
            elif g[0] == 's': vec = _vec_s(vec, pos[g[1]], g[2])
            else:             vec = _vec_cx(vec, pos[g[1]], pos[g[2]])
        self.mag.blocks[b][1] = vec
        for g in W:                               # fold W^dag into the frame
            if g[0] == 'h':   self.right_h(g[1])
            elif g[0] == 's': self.right_s(g[1], dag=(not g[2]))   # (S^dag)^dag = S
            else:             self.right_cx(g[1], g[2])
        self.mag.factor()                         # r is now a product Z-eigenstate -> peel

    # ---- full frame reduction: peel dead (inactive) qubits left in live blocks ----
    def _reduce_dead(self, active):
        """Peel dead (inactive) qubits still entangled in a live block. Handles the
        residue the r=q retarget cannot reach (q not in its own measurement's support,
        e.g. cultivation_d5): a measured-out qubit is typically PARITY-SLAVED -- some
        parity mz of the rest gives a stabiliser Z_q (x) Z^mz -- so a block-local Clifford
        (CNOTs) reduces it to Z_q and factor() peels it. Reuses _purge_redundant with the
        found stabiliser. Exact identity insertion (state-exact -> distribution-exact),
        consumes no rng; this is the near-Clifford analogue of clifft's active-rank
        reduction at demotion (bounds max_block to the active rank)."""
        changed = True
        while changed:
            changed = False
            for qubits, vec in self.mag.blocks:
                if len(qubits) <= 1 or all(qq in active for qq in qubits):
                    continue
                for q in qubits:
                    if q in active:
                        continue
                    zmask = self._find_z_stabilizer(qubits, vec, q)
                    if zmask is not None:
                        self._purge_redundant(0, zmask, prefer=q)
                        changed = True
                        break
                if changed:
                    break

    def _find_z_stabilizer(self, qubits, vec, q):
        """If dead qubit q is parity-slaved (some parity mz of the rest gives a stabiliser
        Z_q (x) Z^mz), return the global register Z-mask (1<<q) | <mz qubits>, else None.
        mz is found by GF(2) elimination and then NUMERICALLY VERIFIED to stabilise the
        block, so _purge_redundant only ever receives a genuine stabiliser (exactness)."""
        k = len(qubits)
        if k - 1 > 20:
            return None
        j = qubits.index(q)
        arr = vec.reshape(-1, 2, 1 << j)
        a = arr[:, 0, :].ravel(); b = arr[:, 1, :].ravel()
        sa = np.nonzero(np.abs(a) > 1e-9)[0]
        sb = np.nonzero(np.abs(b) > 1e-9)[0]
        if len(sa) == 0 or len(sb) == 0:
            return None                       # q already a product -> factor() handles it
        x0 = int(sa[0])
        rows = [(int(x) ^ x0, 0) for x in sa[1:]] + [(int(y) ^ x0, 1) for y in sb]
        mz = _gf2_solve(rows, k - 1)
        if not mz:
            return None
        zmask = 1 << q                        # rest-bit t -> block pos (skip j) -> qubit
        for t in range(k - 1):
            if (mz >> t) & 1:
                zmask |= 1 << qubits[t if t < j else t + 1]
        Pv = _apply_pauli_local(qubits, vec, 0, zmask, 0)   # verify it stabilises
        if abs(abs(complex(np.vdot(vec, Pv))) - 1.0) > 1e-6:
            return None
        return zmask

    # ---- dense reconstruction (verification only) ----
    def statevector(self):
        for (x, z, p, theta) in self.pending:
            self._flush_one(x, z, theta)
        self.pending = []
        n = self.n
        qubits, mvec = self.mag.amplitude_table()
        psi = np.zeros(1 << n, dtype=complex)
        for idx in range(len(mvec)):
            full = 0
            for j, qq in enumerate(qubits):
                if (idx >> j) & 1:
                    full |= (1 << qq)
            psi[full] = mvec[idx]
        return self._clifford_matrix() @ psi

