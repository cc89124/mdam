"""S2 frame/ledger C++ kernel binding (cpp/mdm_frame_kernel.cpp).

Conjugates a set of Paulis (i^p X^x Z^z, stored as parallel uint64 X/Z + int32 P arrays) by a
SEQUENCE of Clifford gates in ONE C++ call -- used to batch a measurement-segment's deferred
tableau conjugations (the dominant `_apply_clifford_to_all`+fn Python cost at high rank).  The
per-gate rule is bit-identical to simulator.NearClifford.{h,s,cx} / lazy._conj_{h,s,cx}.

This is a SEPARATE, feature-flagged path; the authoritative tuple-based tableau update in
simulator/lazy is unchanged.  Selected by engine `_compiled_frame=True` (default OFF).
"""
from __future__ import annotations
import ctypes, os
import numpy as np

_CPP_DIR = os.path.join(os.path.dirname(__file__), "cpp")
_LIB = None

# gate ids (match the C++ kernel)
G_H, G_S, G_SDAG, G_CX = 0, 1, 2, 3


def _lib():
    global _LIB
    if _LIB is None:
        so = os.path.join(_CPP_DIR, "mdm_frame_kernel.so")
        lib = ctypes.CDLL(so)
        P = ctypes.c_void_p
        lib.clifford_conj.restype = None
        lib.clifford_conj.argtypes = [P, P, P, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        lib.clifford_conj_seq.restype = None
        lib.clifford_conj_seq.argtypes = [P, P, P, ctypes.c_int, ctypes.c_int, P, P, P, ctypes.c_int]
        _LIB = lib
    return _LIB


def conj_seq(X, Z, P, W, gate, q1, q2):
    """Apply the gate sequence (gate[],q1[],q2[]) to the m Paulis (X,Z,P) IN PLACE.  Each Pauli's
    mask occupies W uint64 WORDS (row-major: Pauli i at X[i*W:(i+1)*W]); P is int32 (one per
    Pauli); gate/q1/q2 are int32 arrays.  All contiguous."""
    lib = _lib()
    m = P.size; ng = gate.size
    lib.clifford_conj_seq(X.ctypes.data, Z.ctypes.data, P.ctypes.data, m, W,
                          gate.ctypes.data, q1.ctypes.data, q2.ctypes.data, ng)
