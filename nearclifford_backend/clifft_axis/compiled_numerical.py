"""Stage B numerical kernel binding (cpp/mdm_lincomb_kernel.cpp).

Runs the in-place dense Pauli linear combination  phi <- alpha*phi + bph*(P phi)  in C++,
BIT-IDENTICAL to engine._pauli_lincomb_inplace's full-formula branches (scalar/vectorized
diagonal + off-diagonal).  It does NOT replace the Step-1 "diaghalf" global-phase fast path
(kept in Python).  Selected by engine `_compiled_numerical=True` (default OFF).
"""
from __future__ import annotations
import ctypes, os
import numpy as np

_CPP_DIR = os.path.join(os.path.dirname(__file__), "cpp")
_LIB = None


def _lib():
    global _LIB
    if _LIB is None:
        lib = ctypes.CDLL(os.path.join(_CPP_DIR, "mdm_lincomb_kernel.so"))
        D = ctypes.c_double; P = ctypes.c_void_p
        lib.lincomb_offdiag.restype = None
        lib.lincomb_offdiag.argtypes = [P, ctypes.c_int64, ctypes.c_uint64, ctypes.c_uint64,
                                        D, D, D, D]
        lib.lincomb_diag.restype = None
        lib.lincomb_diag.argtypes = [P, ctypes.c_int64, ctypes.c_uint64, D, D, D, D]
        _LIB = lib
    return _LIB


def lincomb(phi, mx, mz, alpha, bph):
    """phi <- alpha*phi + bph*(X^mx Z^mz phi)  IN PLACE.  phi is a C-contiguous complex128 array
    (its buffer is interleaved float64).  mx,mz are magic-register bit masks (fit uint64)."""
    lib = _lib()
    N = phi.size
    p = phi.ctypes.data
    ar, ai = alpha.real, alpha.imag
    br, bi = bph.real, bph.imag
    if mx != 0:
        lib.lincomb_offdiag(p, N, mx, mz, ar, ai, br, bi)
    else:
        lib.lincomb_diag(p, N, mz, ar, ai, br, bi)
