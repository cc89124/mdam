"""Single-frame, dense-free live fused virtual-axis backend.

The fused engine's own n-row CHP/tableau frame (TableauEngine.stab/destab) is the SOLE source
of truth.  There is NO separate lazy Clifford frame:

  * `stab[row] = U_C Z_row U_C^dag`, `destab[row] = U_C X_row U_C^dag` -- byte-for-byte the same
    convention as LazyNearClifford's `Zc/Xc`, so circuit Clifford gates LEFT-conjugate it exactly
    as the lazy frame does (and also conjugate the pending physical generators);
  * non-Clifford rotations are DEFERRED in a pending ledger as PHYSICAL Pauli generators
    (`|psi> = (prod_j R_{L_j}) U_C (|0> (x) |phi>)`), conjugated by every later Clifford gate;
  * a measurement of physical Z_q flushes the anticommutation-connected core (graph reachability
    in the physical basis -- no frame needed) and hands the PHYSICAL core rotations + Z_q to
    `flush_core_virtual`, whose `_mask_for` expresses them over THIS frame (= the pullback) and
    contracts the core to 2^(W-1), classifies the measurement, projects (magic Born) or collapses
    (stabiliser), drops/compresses axes, and returns the outcome.

Because everything lives on one frame, the classification + collapse can never disagree with a
second frame (the bug that broke the two-frame `fused_backend`).  No dense 2^k state, no forced
outcomes, no clifft -- the engine samples its own Born outcomes from its compact magic register.

Folding U_C and the magic-internal Clifford V into ONE frame is mathematically identical to the
validated `capture_stream` -> `flush_core_virtual` pipeline (which keeps U_C in the pullback and V
in a fresh engine frame): `_mask_for(P)|_{U_C·V} == _mask_for(U_C^dag P U_C)|_{V}`.
"""
from __future__ import annotations

import os
import sys

os.chdir("/home/jung/clifft-paper")
sys.path.insert(0, "/home/jung/clifft-paper")
sys.setrecursionlimit(400000)

import numpy as np

from mdam.backend.lazy import _conj_h, _conj_s, _conj_cx, _commute_xz
from mdam.backend.virtual_axis.virtual_engine import TableauEngine
from mdam.backend.virtual_axis.virtual_axis import _herm
from mdam.backend.virtual_axis.fused_integrate import flush_core_virtual


class FusedSingleFrame(TableauEngine):
    def __init__(self, n):
        super().__init__(n)
        self.pending = {}                 # uid -> [x, z, phase, theta, uid] (physical generator)
        self._rot_uid = 0
        self.rng = np.random.default_rng(0)
        self.max_fused_ws = 0
        self.max_M = 0
        self.max_alloc_rank = 0           # ACTUAL peak dense-state allocation (honest metric)
        self.k_clifft = None              # if set, every fused alloc is asserted <= this
        self._forced = None               # optional outcome list (validation only)
        self._forced_i = 0
        # backend-interface stubs (only touched in structure/drop_dead passes, which we disable)
        self.cap = None
        self.resource_only = False
        self._dead_uids = None
        self._flushed_uids = None
        self._record_cores = None
        self._fast_cores = None
        self._meas_ctr = 0
        self._debug_compare = False

    # M (magic qubit membership) is the engine's `magic` row list -- one frame, one membership
    @property
    def M(self):
        return self.magic

    # ---- circuit Clifford gates: LEFT-conjugate the frame AND the pending generators ----
    def _gate(self, conj, *a):
        for row in range(self.n):
            self.stab[row] = conj(self.stab[row], *a)
            self.destab[row] = conj(self.destab[row], *a)
        self.pending = {u: [*conj((r[0], r[1], r[2]), *a), r[3], r[4]]
                        for u, r in self.pending.items()}

    def h(self, q):
        self._gate(_conj_h, q)

    def s(self, q, dag=False):
        self._gate(_conj_s, q, dag)

    def cx(self, c, t):
        self._gate(_conj_cx, c, t)

    def cz(self, a, b):
        self.h(b); self.cx(a, b); self.h(b)

    # ---- defer a rotation as a PHYSICAL generator (x, z) ----
    def apply_rotation(self, x, z, theta):
        # Hermitian-normalise the generator's phase: exp(-i theta P / 2) needs P = P^dag.
        # X/Z (x & z == 0) -> ph 0 (unchanged); a Y-axis rotation (x & z != 0, e.g. R_Y =
        # exp(-i theta Y/2), Y = i XZ) -> ph = popcount(x & z) mod 2 so the literal mask
        # i^ph X^x Z^z is the Hermitian Pauli, NOT the non-Hermitian XZ.
        _, _, ph = _herm((x, z, 0))
        uid = self._rot_uid
        self._rot_uid += 1
        self.pending[uid] = [x, z, ph, theta, uid]

    # ---- anticommutation-connected core of measured physical Pauli (qx, qz) ----
    def _core_entries(self, qx, qz):
        entries = list(self.pending.values())
        nE = len(entries)
        in_core = [False] * nE
        stack = []
        for j, r in enumerate(entries):
            if not _commute_xz(qx, qz, r[0], r[1]):
                in_core[j] = True; stack.append(j)
        while stack:
            j = stack.pop(); rj = entries[j]
            for k in range(nE):
                if not in_core[k]:
                    rk = entries[k]
                    if not _commute_xz(rj[0], rj[1], rk[0], rk[1]):
                        in_core[k] = True; stack.append(k)
        return [entries[j] for j in range(nE) if in_core[j]]   # increasing-uid order

    # ---- measurement: flush the core via the fused map on THIS frame, return outcome ----
    def measure_z(self, q):
        core = self._core_entries(0, 1 << q)
        for r in core:
            del self.pending[r[4]]
        # PHYSICAL generators, carrying their Hermitian phase r[2] (0 for X/Z; non-zero once a
        # Clifford has conjugated a generator onto a Y component -- e.g. an R_Y rotation, or X/Z
        # rotated by S/H/CX).  _mask_for folds this into the exact rotation-mask phase; dropping
        # it (passing 0) would mis-Hermitise any x&z!=0 generator and break R_Y outcomes.
        rots = [((r[0], r[1], r[2]), r[3]) for r in core]
        Pm = (0, 1 << q, 0)
        if self._forced is not None:
            f = self._forced[self._forced_i]; self._forced_i += 1
            out, _ = flush_core_virtual(self, rots, Pm, forced=f)
        else:
            out, _ = flush_core_virtual(self, rots, Pm, rng=self.rng)
        self.max_fused_ws = max(self.max_fused_ws,
                                getattr(self, "fused_peak", 0), len(self.magic))
        self.max_M = max(self.max_M, len(self.magic))
        return out


def compile_circuit(stim_text, **kw):
    """Compile a circuit for the near-Clifford backend, routing off-axis `R_Y` coherent
    rotations as DIRECT Pauli rotations rather than dense 2-qubit unitaries.

    `R_Y(theta) = exp(-i theta Y/2)` is a single-qubit Pauli rotation (`Y = i XZ`), NOT a
    general unitary.  clifft's default bytecode fusion, however, folds an off-axis `R_Y` into
    dense `U2`/`U4` nodes (it only keeps the axis-diagonal `R_Z`/`R_X` as `OP_ARRAY_ROT`); a
    fused `R_Y . CNOT` `U4` is an entangling block that `_u4_decompose` cannot express as a
    Clifford + 1q-rotation, so the backend raised `NotImplementedError`.  Skipping the bytecode
    fusion pass keeps `R_Y` lowered as `OP_ARRAY_ROT` + basis-change Cliffords (`OP_ARRAY_H/S`),
    so it flows through the deferred Pauli-rotation ledger and the generator becomes the Y mask
    `(x=1<<q, z=1<<q)` under Clifford conjugation -- exactly the direct-Pauli-rotation path the
    engine handles (Hermitian phase via `_herm`).  This is bit-faithful to the default lowering
    (verified: identical `record_probabilities` vs clifft's exact reference, identical
    `peak_rank`); it is only triggered when `R_Y` is present, so the fused fast path is
    unchanged for every other circuit.
    """
    import clifft
    if "bytecode_passes" not in kw and "R_Y" in stim_text:
        kw["bytecode_passes"] = None          # keep R_Y as a Pauli rotation, not a fused U4
    return clifft.compile(stim_text, **kw)


def fused_ws_single(circ, seed=1, guard=True):
    """Peak fused workspace exponent for `circ`, run dense-free on the single frame.  With
    `guard=True` the engine asserts EVERY dense allocation stays <= clifft's active rank
    (`prog.peak_rank`) -- so any residual non-fused 2^W path fails loudly."""
    import mdam.backend.backend as bk
    prog = compile_circuit(open(f"qec_bench/circuits/{circ}.stim").read())
    orig = bk.LazyNearClifford
    bk.LazyNearClifford = FusedSingleFrame              # _reset picks this for lazy=True

    def _arm(step, be):                                 # nc exists once run_shot has _reset'd
        if guard and be.nc.k_clifft is None:
            be.nc.k_clifft = prog.peak_rank

    try:
        be = bk.NearCliffordBackend(lazy=True, drop_dead=False, structure_once=False)
        be.run_shot(prog, seed, step_recorder=_arm)
        return be.nc.max_fused_ws
    finally:
        bk.LazyNearClifford = orig


if __name__ == "__main__":
    import time
    ref = {"coherent_d3_r1": 1, "distillation": 4, "cultivation_d3": 4,
           "cultivation_d5": 10, "coherent_d3_r3": 4}
    circs = sys.argv[1:] or list(ref)
    allok = True
    for c in circs:
        t = time.time()
        try:
            ws = fused_ws_single(c)
            tag = ("OK" if ws == ref[c] else f"MISMATCH(exp {ref[c]})") if c in ref else ""
            if c in ref and ws != ref[c]:
                allok = False
            print(f"{c:16} fused_ws={ws}  {tag}  ({time.time()-t:.1f}s)", flush=True)
        except Exception:
            import traceback
            traceback.print_exc()
            allok = False
    print("REGRESSION", "PASS" if allok else "FAIL", flush=True)
