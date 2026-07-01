"""Integration of the FUSED measurement-core map into the streaming TableauEngine.

`flush_core_virtual(eng, rots, Pm, forced)` advances `eng` by one measurement core
WITHOUT ever materialising the streaming `peak = W = r_out+1` transient. It computes the
whole core as ONE contraction

    |phi_out> = <b|_a ( prod_i R_{P_i} ) ( |phi_in> (x) |0>_new )

over the W-axis work basis, dropping the measured axis `a` analytically so the workspace
never exceeds 2^(W-1) = 2^r_out. No streaming `apply_rotation` promote, no `_flush_one`,
no `reduce_full`.

Sub-step A (this file): the SINGLE-AXIS magic measurement -- the case that drives the
binding peak (cultivation_d5's W=11 single-Z cores, both fresh and pre-existing measured
axes; coherent_d3_r3's fresh single-axis X). Multi-axis / pure-Z-parity / |0>-direction
(antis) / trivial measurements fall back to the streaming step for now (Sub-step B fuses
them). Every fallback is COUNTED so the "0 ephemeral materialisation" invariant is auditable.
"""
from __future__ import annotations

import copy

import numpy as np

from mdam.backend.simulator import pauli_mul
from mdam.backend.block_magic import _apply_pauli_local, _vec_h, _vec_s
from mdam.backend.virtual_axis.virtual_axis import _herm
from mdam.backend.virtual_axis.virtual_engine import _symp
from mdam.backend.virtual_axis.clifford_synth import conj_h, conj_s
from mdam.backend.virtual_axis import flop_meter as _fm


# ---- C-7.1: compute (Pauli-sum) blow-up guard -----------------------------------------------
# The fused contraction expands prod_i (cI + dP_i) to a Pauli dict of up to min(2^L, 4^W) terms
# (2^L from L rotations; capped by 4^W = the number of distinct Paulis on the W work axes -- terms
# with equal (x,z) MERGE, which is why cultivation_d5's L=38 core stays at ~2^20, feasible).  When
# the estimate exceeds TERM_CAP the COMPUTE (not the memory: workspace is still 2^(W-1)) blows up
# -- off-axis R_Y bunches L~48 rotations into a core with 4^W ~ 2^34 terms.  We RAISE (never hang,
# never sequential/Clifft fallback) so the offending core is reported, and a projected-TN / sliced
# executor can take over (C-7.3, not yet built).
TERM_CAP = 1 << 24          # ~16.8M Pauli-sum terms; allows cult_d5 (~2^20), rejects R_Y (~2^34)


class LargeCoreNeedsProjectedTN(Exception):
    """A measurement core's fused Pauli-sum would exceed TERM_CAP terms.  This is a COMPUTE wall
    (workspace 2^(W-1) is memory-fine); it MUST be handled by a projected-TN/sliced contraction,
    NOT by sequential rotation application or a Clifft fallback (either would re-open a measured
    axis as state and break the survivor-only invariant)."""
    def __init__(self, *, meas_id, kind, L, estimated_terms, W, k_clifft):
        self.meas_id, self.kind, self.L = meas_id, kind, L
        self.estimated_terms, self.W, self.k_clifft = estimated_terms, W, k_clifft
        et = estimated_terms.bit_length() - 1
        super().__init__(
            f"meas {meas_id}: core kind={kind} L={L} rotations -> fused Pauli-sum ~2^{et} terms "
            f"(> TERM_CAP 2^{TERM_CAP.bit_length()-1}); work W={W}, survivor<=2^{W-1} is memory-OK, "
            f"COMPUTE blows up -> projected-TN executor required (no sequential/Clifft fallback)")


def _term_guard(eng, kind, info, rots):
    """Raise LargeCoreNeedsProjectedTN if this core's Pauli-sum is estimated above TERM_CAP.
    Cheap (tableau-only W from classify_core); runs BEFORE any _pauli_sum so R_Y fails loudly
    instead of hanging on the 2^L expansion."""
    L = len(rots)
    if L == 0:
        return
    W = info[1] if info is not None else (len(eng.magic) + L + 1)   # real W for single/multi
    est = min(1 << L, 1 << (2 * W))                                  # 4^W bound: equal (x,z) merge
    if est > TERM_CAP:
        raise LargeCoreNeedsProjectedTN(
            meas_id=len(getattr(eng, 'core_log', [])), kind=kind, L=L,
            estimated_terms=est, W=W, k_clifft=getattr(eng, 'k_clifft', None))


def _commit_alloc(eng, vec, tag):
    """Record the ACTUAL dense-state allocation rank (log2 of the largest materialised magic
    vector), NOT the `W-1` work-basis heuristic.  This is the peak the user's invariant cares
    about: the real survivor/workspace exponent.  If `eng.k_clifft` is set, ENFORCE that no
    fused path ever allocates beyond clifft's active rank -- a failure here means a non-fused
    `2^W`-style path leaked through (e.g. a large antis/trivial fallback).  Returns the rank."""
    sz = int(vec.size)
    rank = (sz.bit_length() - 1) if sz > 1 else 0
    eng.fused_peak = max(getattr(eng, "fused_peak", 0), rank)
    eng.max_alloc_rank = max(getattr(eng, "max_alloc_rank", 0), rank)
    kc = getattr(eng, "k_clifft", None)
    if kc is not None:
        assert rank <= kc, (f"non-fused alloc in {tag}: built 2^{rank} > clifft active 2^{kc} "
                            f"(a measured/closure axis was materialised as state)")
    return rank


def _remove_bit(mask, a):
    return (mask & ((1 << a) - 1)) | ((mask >> (a + 1)) << a)


def _pauli_sum(masks):
    """prod_i (cos(th/2) I - i sin(th/2) P_i) as {(x,z): complex coeff}, P on the LEFT."""
    s = {(0, 0): 1.0 + 0j}
    for (mx, mz, mph, th) in masks:
        c = np.cos(th / 2.0)
        d = -1j * np.sin(th / 2.0) * (1j ** mph)
        new = {}
        for (x, z), co in s.items():
            new[(x, z)] = new.get((x, z), 0j) + c * co
            x2, z2, ph2 = pauli_mul((mx, mz, 0), (x, z, 0))
            new[(x2, z2)] = new.get((x2, z2), 0j) + co * d * (1j ** ph2)
        s = new
    return s


def _basis_of(mx, mz, a):
    xb = (mx >> a) & 1
    zb = (mz >> a) & 1
    return 'X' if (xb and not zb) else 'Y' if (xb and zb) else 'Z'


def _meas_sign(basis, mph):
    """The +-1 sign s such that i^mph X^mx Z^mz = s * canonical(basis) on the pivot axis.
    Z/X: canonical Hermitian is Z/X, mph in {0,2} -> s = i^mph.  Y: X Z = -i Y so
    i^mph X Z = i^(mph-1) Y, mph in {1,3} -> s = i^(mph-1)."""
    p = mph if basis != 'Y' else (mph - 1) & 3
    return 1 if (p & 3) == 0 else -1


_ISQRT2 = 1.0 / np.sqrt(2.0)


def _fresh_factor(basis, xa, b):
    """<b_basis|_a X^xa Z^za |0>_a  (Z^za acts trivially on |0>, so only xa matters).  The
    1/sqrt(2) is kept for X/Y so ||phi_out||^2 = P(outcome) exactly (Z is a delta -> no norm
    loss).  Z is a delta so its norm is already exact."""
    if basis == 'Z':
        return 1.0 + 0j if xa == b else 0.0 + 0j
    if basis == 'X':
        return _ISQRT2 if b == 0 else (_ISQRT2 if xa == 0 else -_ISQRT2)
    # Y:  b=0 -> (-i)^xa,  b=1 -> (i)^xa
    base = (-1j) if b == 0 else (1j)
    return _ISQRT2 * (base ** xa)


def _demote(eng, a, basis, b):
    """Drop axis a (magic index): set its stabiliser row to the measured single-qubit Pauli
    with the outcome sign (-1)^b, keep an anticommuting destabiliser, remove it from magic."""
    row = eng.magic[a]
    s = eng.stab[row]
    d = eng.destab[row]
    if basis == 'Z':
        op, partner = s, d
    elif basis == 'X':
        op, partner = d, s
    else:                                   # Y = i AX AZ (Hermitian normalisation)
        op, partner = _herm(pauli_mul(d, s)), d
    eng.stab[row] = (op[0], op[1], (op[2] + 2 * b) & 3)
    eng.destab[row] = partner
    eng.magic = [m for i, m in enumerate(eng.magic) if i != a]


def _conj_sum(s, gate):
    """Conjugate every Pauli term (x,z)->co of the sum by a single-axis Clifford gate,
    folding the resulting i-phase into the complex coefficient.  gate in
    {('h',a), ('sdg',a)}."""
    out = {}
    for (x, z), co in s.items():
        if gate[0] == 'h':
            nx, nz, dph = conj_h((x, z, 0), gate[1])
        else:                                       # sdg
            nx, nz, dph = conj_s((x, z, 0), gate[1], True)
        key = (nx, nz)
        out[key] = out.get(key, 0j) + co * (1j ** dph)
    return out


def _contract_single(eng, phi_in, r_in, masks, a, basis, b, W):
    """Fused contraction for a single-axis measurement collapsing axis a, outcome b (the
    CANONICAL-basis outcome).  Returns (phi_out, demote_basis): phi_out is the unnormalised
    post-measurement vector over r_out = W-1 axes (workspace 2^r_out, never 2^W), demote_basis
    is the Pauli basis to stabilise the dropped axis with.  `eng.magic` is the W-axis work
    basis; phi_in is over the first r_in axes."""
    r_out = W - 1
    s = _pauli_sum(masks)

    if a >= r_in:
        # ---- FRESH axis (started |0>): per-term scalar f(xa,b), Z^za trivial on |0>.
        # Surviving axes keep their frame; the dropped axis is stabilised in `basis`. ----
        n_newpers = r_out - r_in
        phi_sys = phi_in
        for _ in range(n_newpers):
            phi_sys = np.kron(np.array([1.0 + 0j, 0.0]), phi_sys)   # new |0> as a HIGH bit
        out = np.zeros(1 << r_out, dtype=complex)
        for (x, z), co in s.items():
            f = _fresh_factor(basis, (x >> a) & 1, b)
            if f == 0:
                continue
            out += (co * f) * _apply_pauli_local(
                list(range(r_out)), phi_sys, _remove_bit(x, a), _remove_bit(z, a), 0)
            _fm.el(1 << r_out, 8.0)            # (co*f)*vec scale (6) + out += (2)
        return out, basis

    # ---- PRE-EXISTING axis (holds part of phi_in): local-reduce basis->Z, then slice a=b.
    # The local gate on axis a is applied to phi_in (real), to the tableau row (frame fold),
    # and to the Pauli sum (conjugation) so all three stay consistent; axis a becomes Z and
    # is demoted as Z (the rotated row already equals the measured Pauli). ----
    phi = phi_in
    if basis == 'X':
        gseq = [('h', a)]
    elif basis == 'Y':
        gseq = [('sdg', a), ('h', a)]               # Y -> X -> Z
    else:
        gseq = []
    for g in gseq:
        if g[0] == 'h':
            phi = _vec_h(phi, a); eng._right_h(a)
        else:
            phi = _vec_s(phi, a, True); eng._right_s(a, dag=False)
        s = _conj_sum(s, g)

    half = 1 << (r_in - 1)
    out = np.zeros(1 << r_out, dtype=complex)
    old_mask = (1 << r_in) - 1
    for (x, z), co in s.items():
        x_O = x & old_mask
        z_O = z & old_mask
        x_N = x >> r_in
        w = _apply_pauli_local(list(range(r_in)), phi, x_O, z_O, 0)
        w_b = w.reshape(-1, 2, 1 << a)[:, b, :].ravel()    # slice axis a == b, drop it
        base = x_N << (r_in - 1)
        out[base:base + half] += co * w_b
        _fm.el(half, 8.0)                                  # co*w_b scale (6) + += (2)
    return out, 'Z'


def _slice_Z(s, phi, r_in, a, c, W):
    """Computational Z-slice: <c|_a (s_op psi0) over r_out = W-1 axes (drop a), where psi0 =
    phi (over OLD axes 0..r_in-1) (x) |0>_new.  Fresh a (>= r_in): delta(x_a, c) on the |0>
    axis.  Pre-existing a (< r_in): apply the OLD-axis Pauli to phi then slice axis a == c.
    Workspace 2^r_out -- the W-axis state is never built."""
    r_out = W - 1
    if a >= r_in:
        phi_sys = phi
        for _ in range(r_out - r_in):
            phi_sys = np.kron(np.array([1.0 + 0j, 0.0]), phi_sys)       # new |0> as HIGH bit
        out = np.zeros(1 << r_out, dtype=complex)
        for (x, z), co in s.items():
            if ((x >> a) & 1) != c:
                continue
            out += co * _apply_pauli_local(
                list(range(r_out)), phi_sys, _remove_bit(x, a), _remove_bit(z, a), 0)
            _fm.el(1 << r_out, 8.0)                         # co*vec scale (6) + += (2)
        return out
    half = 1 << (r_in - 1)
    out = np.zeros(1 << r_out, dtype=complex)
    old_mask = (1 << r_in) - 1
    for (x, z), co in s.items():
        w = _apply_pauli_local(list(range(r_in)), phi, x & old_mask, z & old_mask, 0)
        w_b = w.reshape(-1, 2, 1 << a)[:, c, :].ravel()
        base = (x >> r_in) << (r_in - 1)
        out[base:base + half] += co * w_b
        _fm.el(half, 8.0)                                  # co*w_b scale (6) + += (2)
    return out


def _reduce_to_Z(eng, s, phi, r_in, mmx, mmz, mmph, W):
    """Local-Clifford reduce the OLD (< r_in) X/Y support axes of Pm to Z: apply H (X->Z) or
    Sdg,H (Y->Z) to the Pauli sum (conjugate), the OLD-axis state phi (real gate), and the
    tableau ROWS (frame fold).  FRESH (|0>) support axes are left untouched -- conjugating
    their row would desync the |0> initialisation; they are handled by the delta-contraction.
    Returns (s, phi, Pm=(mmx',mmz',php)).  Cheap -- no 2^W vector."""
    Pm = (mmx, mmz, mmph)
    for axis in range(r_in):                       # OLD axes only
        xb = (Pm[0] >> axis) & 1
        zb = (Pm[1] >> axis) & 1
        if not xb:
            continue
        gseq = [('h', axis)] if not zb else [('sdg', axis), ('h', axis)]
        for g in gseq:
            if g[0] == 'h':
                s = _conj_sum(s, ('h', axis))
                phi = _vec_h(phi, axis)
                eng._right_h(axis)
                Pm = conj_h(Pm, axis)
            else:
                s = _conj_sum(s, ('sdg', axis))
                phi = _vec_s(phi, axis, True)
                eng._right_s(axis, dag=False)
                Pm = conj_s(Pm, axis, True)
    return s, phi, Pm


def _contract_parity(s, phi, r_in, surv_mz, beff, W, t):
    """Fused pure-Z PARITY drop of pivot axis t: phi_out[g'] = where(parity(surv_mz.g')==beff,
    phi0, phi1) with phi0/phi1 the computational t=0/1 slices.  Workspace 2^(W-1)."""
    phi0 = _slice_Z(s, phi, r_in, t, 0, W)
    phi1 = _slice_Z(s, phi, r_in, t, 1, W)
    idx = np.arange(1 << (W - 1))
    par = np.array([(int(i) & surv_mz).bit_count() & 1 for i in idx], dtype=np.int8)
    return np.where(par == beff, phi0, phi1)


def classify_core(eng, rots, Pm):
    """Tableau-only probe (no phi) on a COPY: returns (kind, info).  kind in
    {'antis','trivial','single','multi'}."""
    pe = copy.deepcopy(eng)
    pe.phi = None
    r_in = len(eng.magic)
    for (P, th) in rots:
        pe._mask_for(P)
    W = len(pe.magic)
    _, _, _, _, R = pe._express(Pm)
    ms = pe._magicset()
    antis = [row for row in range(pe.n)
             if row not in ms and _symp(R, pe.stab[row])]
    if antis:
        return 'antis', None
    mmx, mmz, mmph = pe._mask_for(Pm)
    supp = [t for t in range(len(pe.magic)) if ((mmx >> t) & 1) or ((mmz >> t) & 1)]
    if not supp:
        return 'trivial', None
    if len(supp) == 1:
        a = supp[0]
        basis = _basis_of(mmx, mmz, a)
        return 'single', (r_in, W, a, basis, mmph)
    return 'multi', (r_in, W, supp)


def flush_core_virtual(eng, rots, Pm, forced=None, rng=None):
    """Advance `eng` by one measurement core via the fused contraction (single-axis), or
    fall back to the streaming step (multi-axis / antis / trivial).  Returns (out, p0)."""
    if not hasattr(eng, 'fused_peak'):
        eng.fused_peak = 0
        eng.fused_ephemeral = 0          # cores that materialised a transient ABOVE resident
        eng.fallback_cores = []
        eng.fused_cores = 0
        eng.core_log = []                # (kind, workspace_exp, r_out) per core

    kind, info = classify_core(eng, rots, Pm)
    _term_guard(eng, kind, info, rots)        # C-7.1: loud-fail large-L cores (R_Y) before _pauli_sum

    if kind != 'single':
        phi_in = eng.phi
        r_in = len(eng.magic)
        eng.phi = None
        masks = []
        for (P, th) in rots:
            mx, mz, mph = eng._mask_for(P)
            masks.append((mx, mz, mph, th))
        mmx, mmz, mmph = eng._mask_for(Pm)             # may promote -> compute W AFTER
        W = len(eng.magic)
        supp = [t for t in range(W) if ((mmx >> t) & 1) or ((mmz >> t) & 1)]

        if kind == 'multi':
            # ---- multi-axis magic measurement, fused to 2^(W-1) (NO W transient).  Reduce the
            # OLD X/Y support to Z (local gates on the sum + phi + ROWS).  What remains is a
            # Z-string plus X on <=1 FRESH axes (|0>-initialised, handled by delta-contraction).
            # 0 fresh-X -> pure-Z PARITY drop;  1 fresh-X -> delta-pivot that axis and fold the
            # Z-rest via a CZ-collapse.  (>=2 fresh-X is rare -> the 2^W path below.) ----
            s = _pauli_sum(masks)
            s, phi_r, (mmx2, mmz2, php) = _reduce_to_Z(eng, s, phi_in, r_in, mmx, mmz, mmph, W)
            xfresh = [sx for sx in range(W) if (mmx2 >> sx) & 1]
            if True:
                # Each branch yields c0/c1 = the two contractions projecting the measured
                # observable O onto +1 / -1 (PHYSICAL outcomes 0 / 1), so p0 = ||c0||^2 /
                # (||c0||^2+||c1||^2) is the Born prob directly; sign only maps the physical
                # outcome to the dropped axis's canonical (X/Y/Z) eigenvalue for the demote.
                if not xfresh:
                    # pure-Z PARITY: pivot any Z axis; outcome via parity(surv_z.g')==canonical
                    t = next(sx for sx in range(W) if (mmz2 >> sx) & 1)
                    sign = 1 if (php & 3) == 0 else -1
                    surv_z = _remove_bit(mmz2, t)
                    c0 = _contract_parity(s, phi_r, r_in, surv_z, 0 if sign == 1 else 1, W, t)
                    c1 = _contract_parity(s, phi_r, r_in, surv_z, 1 if sign == 1 else 0, W, t)
                    dbasis = 'Z'
                    collapse = [('cx', sx) for sx in range(W)
                                if sx != t and (mmz2 >> sx) & 1]
                else:
                    # delta-pivot a fresh-X axis t.  O = i^php X_t Z_t^zt (x) P_rest, with
                    # <0|O_t|1> = beta = i^php (-1)^zt, so the a=0 slice of the O=(-1)^b
                    # projection is chi_b = 1/2(phi0 + (-1)^b beta P_rest phi1).  P_rest is the
                    # survivor Pauli (Z on OLD-reduced axes + X/Y/Z on the OTHER fresh axes), so
                    # several fresh-X axes are handled at once.  A controlled-P_rest collapse
                    # from t disentangles it (O X_t P_rest O = X_t since P_rest^2=I), and t is
                    # demoted in X / Y.
                    t = xfresh[0]
                    zt = (mmz2 >> t) & 1
                    dbasis = 'Y' if zt else 'X'
                    sign = _meas_sign(dbasis, php)
                    prx = _remove_bit(mmx2 & ~(1 << t), t)
                    prz = _remove_bit(mmz2 & ~(1 << t), t)
                    beta = (1j ** php) * ((-1) ** zt)
                    phi0 = _slice_Z(s, phi_r, r_in, t, 0, W)
                    phi1 = _slice_Z(s, phi_r, r_in, t, 1, W)
                    Pp1 = beta * _apply_pauli_local(list(range(W - 1)), phi1, prx, prz, 0)
                    c0 = 0.5 * (phi0 + Pp1)
                    c1 = 0.5 * (phi0 - Pp1)
                    _fm.el(phi0.size, 14.0)       # beta*Pp1 (6) + c0 0.5(phi0+Pp1) (4) + c1 (4)
                    collapse = ([('cz', sx) for sx in range(W)            # controlled-P_rest
                                 if sx != t and (prz >> (sx if sx < t else sx - 1)) & 1]
                                + [('cx2', sx) for sx in range(W)
                                   if sx != t and (prx >> (sx if sx < t else sx - 1)) & 1])
                n0 = float(np.vdot(c0, c0).real)
                n1 = float(np.vdot(c1, c1).real)
                p0 = min(1.0, max(0.0, n0 / (n0 + n1) if (n0 + n1) > 1e-15 else 1.0))
                if forced is not None:
                    out_bit = int(forced)
                elif rng is not None:
                    out_bit = 0 if float(rng.random()) < p0 else 1
                else:
                    out_bit = 0 if p0 >= 0.5 else 1
                out = c0 if out_bit == 0 else c1
                for (g, sx) in collapse:
                    if g == 'cx':
                        eng._right_cx(sx, t)                  # parity: collapse Z-string onto t
                    elif g == 'cx2':
                        eng._right_cx(t, sx)                  # controlled-X_sx from t
                    else:
                        eng._right_h(sx); eng._right_cx(t, sx); eng._right_h(sx)   # CZ(t,sx)
                nrm = float(np.linalg.norm(out))
                if nrm > 1e-12:
                    out = out / nrm
                    _fm.el(out.size, 2.0)                     # out / nrm (real scale)
                eng.phi = out
                demote_bit = out_bit if sign == 1 else 1 - out_bit
                _demote(eng, t, dbasis, demote_bit)
                eng._compress()
                if getattr(eng, 'reduce_parities', False):
                    eng._reduce_parities()
                r_out = len(eng.magic)
                rank = _commit_alloc(eng, out, 'multi')      # ACTUAL output rank (= W-1)
                eng.fused_cores += 1
                eng.core_log.append(('multi', rank, r_out))
                eng.max_k_res = max(getattr(eng, 'max_k_res', 0), r_out)
                return out_bit, p0

        # ---- antis / trivial / multi with >=2 fresh-X (rare): the |0>-direction & deterministic
        # measurements DROP NO magic axis (resident rank IS W, no transient); the >=2 fresh-X
        # case is the only one that still materialises the W work basis before a drop.
        # the core's resident rank IS W (the rotations' work basis) -- no transient to avoid.
        # Build that W-axis resident by the FUSED Pauli sum (no streaming apply_rotation) and
        # let the engine apply the |0>-direction / deterministic measurement. ----
        # INVARIANT: this path builds the W-axis work basis (2^W).  It is justified ONLY when the
        # measurement DROPS NO magic axis -- i.e. antis (|0>-direction) / trivial (+-I), where the
        # core's rotations genuinely open a W-axis RESIDENT (no measured/closure axis to contract
        # away, so 2^W is the resident, not a transient).  `multi` (which DOES drop an axis) must
        # NEVER reach here -- the multi block above returns for every fresh-X count via P_rest.
        assert kind in ('antis', 'trivial'), (
            f"non-fused 2^W fallback reached for kind={kind!r}: a magic-dropping core leaked past "
            f"the projected contraction (this is the path the survivor-only invariant forbids)")
        s = _pauli_sum(masks)
        psi0 = phi_in
        for _ in range(W - r_in):
            psi0 = np.kron(np.array([1.0 + 0j, 0.0]), psi0)        # |0> as HIGH bit
        phi_full = np.zeros(1 << W, dtype=complex)
        for (x, z), co in s.items():
            phi_full += co * _apply_pauli_local(list(range(W)), psi0, x, z, 0)
            _fm.el(1 << W, 8.0)                                # co*vec scale (6) + += (2)
        eng.phi = phi_full
        eng.max_k = max(eng.max_k, W)
        rank = _commit_alloc(eng, phi_full, kind)          # ACTUAL alloc (= W); antis/trivial: =resident
        out, p0 = eng.measure_drop(Pm, forced=forced, rng=rng)
        r_out = len(eng.magic)
        if rank > r_out:
            eng.fused_ephemeral += 1
        eng.core_log.append((kind, rank, r_out))
        eng.max_k_res = max(getattr(eng, 'max_k_res', 0), r_out)
        return out, p0

    r_in, W, a, basis, mmph = info
    phi_in = eng.phi

    # re-run the tableau-only structure on eng ITSELF (mutates the frame to the W work basis)
    eng.phi = None
    masks = []
    for (P, th) in rots:
        mx, mz, mph = eng._mask_for(P)
        masks.append((mx, mz, mph, th))
    mmx, mmz, mmph2 = eng._mask_for(Pm)
    assert len(eng.magic) == W

    # Born outcome (need it before contraction): exp value of the single-axis observable.
    # Cheap to get from the streaming p0 only when forced; otherwise compute from |phi_out|.
    sign = _meas_sign(basis, mmph2)
    # p0 = P(physical outcome 0) = P(O=+1).  Contract the canonical b=0 branch on a COPY to
    # read its Born weight; the realised branch is then contracted on eng itself.
    w0, _ = _contract_single(copy.deepcopy(eng), phi_in, r_in, masks, a, basis, 0, W)
    p_can0 = float(np.vdot(w0, w0).real)
    p0 = p_can0 if sign == 1 else 1.0 - p_can0
    p0 = min(1.0, max(0.0, p0))
    if forced is not None:
        out_bit = int(forced)
    elif rng is not None:
        out_bit = 0 if float(rng.random()) < p0 else 1
    else:
        out_bit = 0 if p0 >= 0.5 else 1

    beff = out_bit if sign == 1 else 1 - out_bit
    out, dbasis = _contract_single(eng, phi_in, r_in, masks, a, basis, beff, W)
    rank = _commit_alloc(eng, out, 'single')           # ACTUAL output rank (= W-1, +1 contracted away)
    eng.fused_cores += 1

    nrm = float(np.linalg.norm(out))
    if nrm > 1e-12:
        out = out / nrm
        _fm.el(out.size, 2.0)                           # out / nrm (real scale)
    eng.phi = out
    _demote(eng, a, dbasis, beff)
    eng._compress()
    if getattr(eng, 'reduce_parities', False):
        eng._reduce_parities()
    r_out = len(eng.magic)
    eng.core_log.append(('single', rank, r_out))       # ACTUAL alloc rank, never the streaming W
    if rank > r_out:
        eng.fused_ephemeral += 1
    eng.max_k_res = max(getattr(eng, 'max_k_res', 0), r_out)
    return out_bit, p0
