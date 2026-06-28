"""Opt-in FLOP meter for the fused virtual-axis contraction.

Counts the floating-point work of the runtime measurement-core contraction in three buckets,
all in the repo convention (complex mult/scale=6 or 2, add/sub=2, vdot=8, norm=4 per element):

  * apply_kron  -- Pauli applies (`_apply_pauli_local`, 6N) and `kron` (6N)
  * vdot_norm   -- the Born/normalisation scans (`vdot` 8N, `norm` 4N)
  * elementwise -- the axpy / scale / combine arithmetic INSIDE the contraction kernels
                   (c0=0.5(phi0+-Pp1), beta*vec, out/nrm, the Pauli-sum accumulations, the
                   _vec_h/_vec_s basis combines) -- the term the earlier "floor" missed.

DISABLED by default (`_on=False`) -> zero overhead in normal runs. The fused backend never
imports this on the hot path unless a measurement harness calls `enable()`.
"""

_F = {"apply_kron": 0.0, "vdot_norm": 0.0, "elementwise": 0.0}
_on = [False]


def enable():
    _on[0] = True


def disable():
    _on[0] = False


def reset():
    _F["apply_kron"] = 0.0
    _F["vdot_norm"] = 0.0
    _F["elementwise"] = 0.0


def snapshot():
    return dict(_F)


def el(n, coeff=1.0):
    """elementwise (axpy/scale/combine): coeff FLOP per element over n elements."""
    if _on[0]:
        _F["elementwise"] += coeff * float(n)


def ak(n, coeff=6.0):
    """Pauli-apply / kron: coeff (default 6 = complex mult) FLOP per element."""
    if _on[0]:
        _F["apply_kron"] += coeff * float(n)


def vn(n, coeff):
    """vdot (8) / norm (4) scan: coeff FLOP per element."""
    if _on[0]:
        _F["vdot_norm"] += coeff * float(n)
