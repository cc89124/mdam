"""Standalone measurement-core FUSED apply (the clifft-bounded path).

Instead of applying the core rotations one-by-one (streaming, which materialises the
`peak = r_out + 1` intermediate), compute the whole core as ONE map

    |phi_out> = <b|_a ( prod_i R_{P_i}(theta_i) ) ( |phi_in> (x) |0>_a )

via a Pauli sum contracted on the ephemeral measured axis `a`. The (r_out+1)-axis
intermediate is NEVER built -- the workspace stays 2^r_out.

Structure extraction is TABLEAU-ONLY (promote bookkeeping, no phi, no 2^W vector): it
yields the rotation masks over the W-axis work basis and the measured axis `a`. For the
cultivation cores `a` is the newly-opened |0> axis and P_meas = Z_a (single axis), so the
verified ancilla-contraction kernel applies directly.
"""
from __future__ import annotations

import copy

import numpy as np

from nearclifford_backend.simulator import pauli_mul
from nearclifford_backend.block_magic import _apply_pauli_local
from nearclifford_backend.virtual_axis.virtual_engine import TableauEngine


def _remove_bit(mask, a):
    """Drop bit `a`, shifting higher bits down (W-axis index -> (W-1)-axis index)."""
    return (mask & ((1 << a) - 1)) | ((mask >> (a + 1)) << a)


def fused_core_apply(eng0, rots, Pm, b):
    """Fused apply of one measurement core. `eng0` is a TableauEngine at the core start
    (phi over r_in axes). Returns (phi_out, born_weight, max_exp) where phi_out is the
    UNNORMALISED post-measurement magic vector over r_out axes (||phi_out||^2 = P(outcome
    b)), and max_exp is the largest workspace exponent the fused path materialised."""
    eng = copy.deepcopy(eng0)                      # promote-only structure (tableau mutates)
    r_in = len(eng.magic)
    phi_in = eng0.phi
    eng.phi = None                                 # TABLEAU-ONLY: no 2^W vector is ever built

    masks = []
    for (P, th) in rots:
        mx, mz, mph = eng._mask_for(P)             # promotes the tableau; NO phi, NO compress
        masks.append((mx, mz, mph, th))
    W = len(eng.magic)
    mmx, mmz, mmph = eng._mask_for(Pm)
    supp = [s for s in range(W) if ((mmx >> s) & 1) or ((mmz >> s) & 1)]
    assert mmx == 0 and len(supp) == 1, f"P_meas not single-axis Z over the work basis: {supp}"
    a = supp[0]                                     # ephemeral measured axis (P_meas = Z_a)
    assert a >= r_in, f"measured axis {a} is not a newly-opened |0> axis"

    r_out = W - 1
    # system axes = all but `a`. New PERSISTENT axes (system index >= r_in) start |0>:
    # pad phi_in with |0> for each (cultivation: none -- W = r_in+1, a = r_in).
    n_newpers = r_out - r_in
    phi_sys = phi_in
    for _ in range(n_newpers):
        phi_sys = np.kron(np.array([1.0 + 0j, 0.0]), phi_sys)   # |0> as a HIGH (higher-index) axis

    # prod_i (cos I + d_i X^mx Z^mz), built incrementally, P on the LEFT (R_n..R_1 order)
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

    # contract the ancilla: <b|_a X^xa Z^za |0> = delta(b, x_a); drop axis a from each term
    out = np.zeros(1 << r_out, dtype=complex)
    for (x, z), co in s.items():
        if ((x >> a) & 1) != b:
            continue
        out += co * _apply_pauli_local(list(range(r_out)), phi_sys,
                                       _remove_bit(x, a), _remove_bit(z, a), 0)

    # output engine: the measured axis a is now a |b> stabiliser -- demote its row with the
    # outcome sign (stab <- (-1)^b AZ_a) and drop it from the magic list. WITHOUT this the
    # physical reconstruction is wrong for b=1 (the measured qubit's -1 eigenstate is lost).
    row_a = eng.magic[a]
    if b:
        sx, sz, sp = eng.stab[row_a]
        eng.stab[row_a] = (sx, sz, (sp + 2) & 3)
    eng.magic = [m for i, m in enumerate(eng.magic) if i != a]
    eng.phi = out
    return out, float(np.vdot(out, out).real), r_out, eng
