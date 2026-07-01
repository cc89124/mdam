"""Python binding for the compiled MDAM measurement-core executor (cpp/mdm_core_executor.cpp).

The numerical hot path of ONE measurement core runs in a SINGLE C++ call (mdm_execute_core):
build the branch-pair joint from phi_in -> ordered core rotations (direct general-Pauli, NO
per-rotation localization) -> measurement-axis localization (L-M) -> Born -> outcome -> normalize
-> drop -> survivor.  No symbolic terms, no per-rotation allocation, one dense apply per rotation.

This module is a SEPARATE, feature-flagged path.  The authoritative Python oracle in bounded.py is
unchanged; `compiled_core=True` selects this path.  Default is OFF until full verification.
"""
from __future__ import annotations
import ctypes, os
import numpy as np

_CPP_DIR = os.path.join(os.path.dirname(__file__), "cpp")

class CostCounters(ctypes.Structure):
    _fields_ = [(n, ctypes.c_uint64) for n in (
        "rotation_count", "diagonal_rotation_calls", "butterfly_rotation_calls",
        "complex_adds", "complex_multiplies", "real_adds", "real_multiplies",
        "amplitude_reads", "amplitude_writes", "amplitude_pairs_updated",
        "bytes_read", "bytes_written", "h_passes", "cnot_passes", "measurement_passes",
        "norm_passes", "normalization_passes", "survivor_writes", "allocations_in_hot_loop")]
    def as_dict(self):
        return {n: getattr(self, n) for n, _ in self._fields_}

def load(profile=False):
    so = os.path.join(_CPP_DIR, "mdm_core_profile.so" if profile else "mdm_core_release.so")
    lib = ctypes.CDLL(so)
    lib.mdm_execute_core.restype = ctypes.c_int
    P = ctypes.c_void_p
    lib.mdm_execute_core.argtypes = [
        P, P, P, ctypes.c_int, ctypes.c_int,            # phi_in, joint, survivor, r_in, r_mat
        P, P, P, P, P, ctypes.c_int,                     # rot_x,z,pp,c,s,nrot
        P, P, P, ctypes.c_int,                           # lm_type,a,b,nlm
        ctypes.c_int, ctypes.c_double, ctypes.c_int, ctypes.c_double,  # m_bit,sign,mode,rand
        P, P, P, P]                                      # p0,p1,norm,survivor_rank (out)
    lib.mdm_reset_counters.restype = None
    lib.mdm_get_counters.argtypes = [ctypes.POINTER(CostCounters)]; lib.mdm_get_counters.restype = None
    lib.mdm_is_profile_build.restype = ctypes.c_int
    return lib

class CompiledCoreExecutor:
    """Thin wrapper.  Caller passes a CorePlan (numpy arrays); one C++ call per core."""
    FORCE_ZERO, FORCE_ONE, USE_RANDOM = 0, 1, 2

    def __init__(self, profile=False):
        self.lib = load(profile); self.profile = profile
        self._joint = None; self._surv = None     # scratch buffers, reused across cores

    def _scratch(self, r_mat):
        need_j = 1 << r_mat; need_s = 1 << (r_mat - 1)
        if self._joint is None or self._joint.size < need_j:
            self._joint = np.empty(need_j, dtype=np.complex128)
        if self._surv is None or self._surv.size < need_s:
            self._surv = np.empty(need_s, dtype=np.complex128)

    def execute(self, phi_in, r_in, r_mat, rots, lm_ops, m_bit, sign,
                mode=USE_RANDOM, rand_val=0.0, reset_counters=False):
        """rots: list of (x,z,pp,theta) over the r_mat bit layout.  lm_ops: list of (type,a,b)
        with type 0=H 1=S 2=Sdag 3=CNOT.  Returns (outcome,p0,survivor[2^r_out],counters|None)."""
        self._scratch(r_mat)
        nrot = len(rots)
        rx = np.array([r[0] for r in rots], dtype=np.uint64)
        rz = np.array([r[1] for r in rots], dtype=np.uint64)
        rpp = np.array([r[2] for r in rots], dtype=np.int32)
        rc = np.array([np.cos(r[3] / 2.0) for r in rots], dtype=np.float64)
        rs = np.array([np.sin(r[3] / 2.0) for r in rots], dtype=np.float64)
        nlm = len(lm_ops)
        lt = np.array([o[0] for o in lm_ops], dtype=np.int32)
        la = np.array([o[1] for o in lm_ops], dtype=np.int32)
        lb = np.array([o[2] for o in lm_ops], dtype=np.int32)
        pin = np.ascontiguousarray(phi_in, dtype=np.complex128)
        p0 = ctypes.c_double(); p1 = ctypes.c_double(); nrm = ctypes.c_double(); srk = ctypes.c_int()
        if reset_counters and self.profile:
            self.lib.mdm_reset_counters()
        def vp(a): return a.ctypes.data if a.size else 0
        outcome = self.lib.mdm_execute_core(
            pin.ctypes.data, self._joint.ctypes.data, self._surv.ctypes.data, r_in, r_mat,
            vp(rx), vp(rz), vp(rpp), vp(rc), vp(rs), nrot,
            vp(lt), vp(la), vp(lb), nlm,
            m_bit, float(sign), mode, float(rand_val),
            ctypes.byref(p0), ctypes.byref(p1), ctypes.byref(nrm), ctypes.byref(srk))
        surv = self._surv[:1 << srk.value].copy()
        counters = None
        if self.profile:
            cc = CostCounters(); self.lib.mdm_get_counters(ctypes.byref(cc)); counters = cc.as_dict()
        return outcome, p0.value, surv, counters


# ===================================================================== #
#  FULL-SHOT INTEGRATION: measure_z fast path using the compiled core    #
#  executor.  Selected by `engine._compiled_core = True` (default OFF).  #
#  Mirrors the oracle's localize-drop physics but runs the numerical     #
#  core in ONE C++ call (direct general-Pauli, no L-R).  Control plane    #
#  (core selection, L-M plan, frame/ledger folds) stays in Python.       #
# ===================================================================== #
_SHARED_EXEC = None
def _get_executor():
    global _SHARED_EXEC
    if _SHARED_EXEC is None:
        _SHARED_EXEC = CompiledCoreExecutor(profile=False)
    return _SHARED_EXEC


def try_compiled_measure(eng, q):
    """Attempt the compiled measurement of Z_q.  Returns the outcome, or None to fall back to the
    oracle (stabilizer / deterministic / X-residual-on-non-magic cases).  NO RNG is consumed on the
    None paths (all guards precede the single rng.random() draw), so fallback is bit-exact."""
    from mdam.backend.simulator import pauli_commute
    from mdam.backend.lazy import _conj_h, _conj_s, _conj_cx
    M_in = list(eng.M)
    # tail deferral (default OFF): execute only the PREFIX; the maximal trailing measurement-commuting
    # suffix stays UNTOUCHED in pending and is consumed at a future measurement.
    plan = eng._measurement_execution_plan(0, 1 << q, eng._fused_core_entries(q))
    core = list(plan.execute_entries)                      # anticommuting core PREFIX (no _meas_ctr advance)
    M_mat = list(M_in); pulled = []
    for (x, z, p, theta, uid) in core:
        xp, zp, pp0 = eng._pullback(x, z); pp = (pp0 + p) & 3
        pulled.append((xp, zp, pp, theta))
        for qq in range(eng.n):
            if (xp >> qq) & 1 and qq not in M_mat:
                M_mat.append(qq)
    Pm = (0, 1 << q, 0); magset = set(M_mat)
    if any(i not in magset and not pauli_commute(eng.Zc[i], Pm) for i in range(eng.n)):
        return None                                        # genuine stabilizer measurement -> oracle
    xpq, zpq, ppq = eng._pullback(0, 1 << q)

    def tb(xp, zp):
        xb = zb = 0
        for l, qq in enumerate(M_mat):
            if (xp >> qq) & 1: xb |= 1 << l
            if (zp >> qq) & 1: zb |= 1 << l
        return xb, zb
    Mx, Mz = tb(xpq, zpq)
    mmask = 0
    for qq in M_mat: mmask |= 1 << qq
    if (xpq & ~mmask) != 0 or (Mx == 0 and Mz == 0):
        return None                                        # X on non-magic / deterministic -> oracle
    supp = sorted(qq for qq in M_mat if ((xpq | zpq) >> qq) & 1)
    r = q if q in supp else supp[0]; m = M_mat.index(r)
    # localizer W (physical qubits) + bit-encoded lm + sign, EXACTLY as _localize_to_Z(prefer=q)
    P = (xpq, zpq, ppq); seq = []
    for s in supp:
        xs = (xpq >> s) & 1; zs = (zpq >> s) & 1
        if xs and zs: seq += [("s", s, True), ("h", s)]
        elif xs: seq += [("h", s)]
    for s in supp:
        if s != r: seq += [("cx", s, r)]
    W = []; lm = []
    for g in seq:
        if g[0] == "h": P = _conj_h(P, g[1]); W.append(g); lm.append((0, M_mat.index(g[1]), 0))
        elif g[0] == "s": P = _conj_s(P, g[1], g[2]); W.append(g); lm.append((2 if g[2] else 1, M_mat.index(g[1]), 0))
        else: P = _conj_cx(P, g[1], g[2]); W.append(g); lm.append((3, M_mat.index(g[1]), M_mat.index(g[2])))
    if P[0] != 0 or P[1] != (1 << r):
        return None                                        # localizer did not reach +-Z_r -> oracle
    sign = 1.0 if (P[2] & 3) == 0 else -1.0
    rots = [(*tb(xp, zp), pp, th) for (xp, zp, pp, th) in pulled]
    rin = len(M_in); rmat = len(M_mat); rout = rmat - 1
    # ---- COMMITTED past here: one rng draw, one C++ call ----
    rand_val = float(eng.rng.random())
    ex = eng._compiled_executor if getattr(eng, "_compiled_executor", None) is not None else _get_executor()
    outcome, p0, surv, _ = ex.execute(eng.phi, rin, rmat, rots, lm, m, sign,
                                      mode=ex.USE_RANDOM, rand_val=rand_val)
    eng._meas_ctr += 1
    for ce in core: del eng.pending[ce[4]]
    plus_bit = 0 if sign > 0 else 1
    keepbit = plus_bit if outcome == 0 else (1 - plus_bit)
    M_A = [M_mat[i] for i in range(rmat) if i != m]        # survivor layout-A (m removed, ascending)
    eng._ensure_inited()
    eng.budget.charge(1 << rout, 0, "compiled:survivor")
    eng._grow_capacity(1 << rout)
    eng._storage[:1 << rout] = surv
    eng._sz = 1 << rout
    eng.M = M_A
    eng.phi = eng._storage[:eng._sz]
    for g in W:                                            # fold the L-M localizer into the frame
        if g[0] == "h": eng.right_h(g[1])
        elif g[0] == "s": eng.right_s(g[1], dag=(not g[2]))
        else: eng.right_cx(g[1], g[2])
    if keepbit == 1:                                       # |1> product -> fold X_r into frame
        zr = eng.Zc[r]; eng.Zc[r] = (zr[0], zr[1], (zr[2] + 2) & 3); eng._frame_ver += 1
        if eng._inv_enabled: eng._inv_fold_x(r)
    eng._drop_residual_products()
    if len(eng.M) > eng.max_M: eng.max_M = len(eng.M)
    eng.budget.note_resident(eng.phi.size, "compiled:post")
    if eng.log_cores:
        eng.core_log.append(dict(meas=eng._meas_log_ctr, branch="magic-compiled",
                                 M_before=rin, M_after=len(eng.M), p0=p0,
                                 peak_live_words=eng.budget.peak))
    eng._meas_log_ctr += 1
    return outcome
