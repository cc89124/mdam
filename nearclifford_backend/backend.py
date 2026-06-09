"""Complete near-Clifford backend for clifft bytecode.

This is a full, standalone simulation backend -- a sibling of (not a part of) the
tensor TTN backend. It produces the SAME measurement-record distribution as
``clifft.sample`` (the authoritative ground truth), but represents the active
state in the near-Clifford form

    |psi>_active = U_C ( (x)_{i notin M} |0>_i  (x)  |phi>_M )

(see ``nearclifford_backend.simulator.NearClifford``): a Clifford tableau U_C
that absorbs ALL Clifford structure for free, plus a dense magic register over
only the |M| qubits that a non-Clifford rotation has genuinely promoted. For the
coherent-error QEC circuits |M| stays at the anticommutation rank k (k=0 for the
coherent families -- so chi=2048 in the TTN is pure stabilizer entanglement),
which is why this backend sidesteps the bond-dimension wall entirely.

Architecture (mirrors clifft / the tensor backend's run_shot EXACTLY so the
record distribution matches):

  * frame layer (PauliFrame): tracks the deferred Pauli correction. Every active
    Clifford G applied to the active state ALSO conjugates the frame (G F G^dag);
    OP_FRAME_* ops act on the frame ONLY (dormant/stabilizer qubits). Noise,
    conditional Paulis, readout flips and dormant measurements are handled by the
    SAME shared helpers (ttn_backend.frame_layer) used by clifft validation and
    the tensor backend.
  * active state (NearClifford): plain H / S / CNOT / CZ / SWAP go into the
    Clifford tableau (free); RZ/T/ROT become Pauli rotations exp(-i theta P/2) on
    the pulled-back P = U_C^dag Z_q U_C (promote a qubit into M only when P has
    X-support there); Z measurements collapse and de-promote.

  * rotations are frame-conjugated (X-frame on the axis flips theta -> -theta),
    measurements are combined with the frame parity (Z-meas ^ frame.xb,
    X-meas ^ frame.zb) -- byte-for-byte the same assembly as core.run_shot.

Because the active state is applied directly (no carving / op-class / path
plan), this backend needs NO backend spec -- only the compiled program.

IMPORTANT: compile with ``bytecode_passes=None`` so the active stream is the
UNFUSED plain-gate form (H / ROT / CNOT). The default (fused) compile folds gates
into frame-dependent U2/U4 nodes; this backend de-fuses those too (see
``_apply_u2`` / ``_apply_u4``) but the unfused form is the canonical, simplest
input and is what the verify scripts use.
"""
from __future__ import annotations

import cmath
import math

import numpy as np

from ttn_backend import treewidth as T_mod
from ttn_backend import frame_layer as ds_mod
from nearclifford_backend.simulator import NearClifford
from nearclifford_backend.lazy import LazyNearClifford
from nearclifford_backend.block_magic import BlockLazyNearClifford

_FLAG_SIGN = ds_mod.FLAG_SIGN
_T_ANGLE = math.pi / 4.0


def _opname(opcode):
    return T_mod._opname(opcode)


def count_idents(prog):
    """Number of active qubits ever created = number of OP_EXPAND* ops."""
    return sum(1 for k in range(len(prog))
               if _opname(prog[k].opcode).startswith("OP_EXPAND"))


class MagicCapExceeded(Exception):
    """Raised mid-shot when the live magic register exceeds the configured cap."""
    def __init__(self, step, M):
        super().__init__(f"|M|={M} exceeded cap at step {step}")
        self.step = step; self.M = M


class NearCliffordBackend:
    def __init__(self, prog=None, magic_cap=None, lazy=False, resource_only=False,
                 block=False, decouple_demote=True):
        # Stateless across progs; run_shot/sample size the simulator per prog.
        self.last_max_M = 0
        self.magic_cap = magic_cap     # if set, abort a shot when |M| exceeds it
        self._cur_step = -1
        self.lazy = lazy               # defer rotations, materialise only the core
        self.resource_only = resource_only  # measure core sizes without dense cost (lazy)
        self.block = block             # block-factored magic register (implies lazy)
        # frame-reduction (block only): peel the demoted index at each magic measurement
        # so dead residue does not linger -> removes the per-measurement memory loss
        # (distillation/cultivation_d3 fully; cultivation_d5 partially). State-exact
        # (distribution-exact), NOT bit-identical. DEFAULT ON. Pass decouple_demote=False
        # for the legacy bit-identical path (used by the bit-identical regression check).
        self.decouple_demote = decouple_demote
        if block:
            self.lazy = True

    # ------------------------------------------------------------------ reset
    def _reset(self, prog):
        self.frame = ds_mod.PauliFrame()
        self.record = {}
        self.slot2id = {}                  # active slot -> NearClifford qubit index
        self._next_q = 0
        if self.block:
            sim_cls = BlockLazyNearClifford
        elif self.lazy:
            sim_cls = LazyNearClifford
        else:
            sim_cls = NearClifford
        self.nc = sim_cls(count_idents(prog))
        if self.lazy and self.magic_cap is not None:
            self.nc.cap = self.magic_cap
        if self.lazy and self.resource_only and not self.block:
            self.nc.resource_only = True
        if self.block and self.decouple_demote:
            self.nc.decouple_demote = True
        self.max_M = 0

    def _new_q(self, slot):
        q = self._next_q
        self._next_q += 1
        self.slot2id[slot] = q
        return q

    def _birth(self, slot):
        """clifft EXPAND creates the active leg in |+> = (|0>+|1>)/sqrt2 (see
        core._expand_method: tensor = [INV_SQRT2, INV_SQRT2]). We realise |+> by an
        H on the freshly-allocated |0> qubit -- a Clifford (free, into the tableau).
        This is state-prep, NOT a circuit gate, so the Pauli frame is NOT updated."""
        q = self._new_q(slot)
        self.nc.h(q)
        return q

    def _track_M(self):
        # for lazy, nc.max_M captures the transient flush peak (compressed away after)
        m = max(len(self.nc.M), getattr(self.nc, "max_M", 0))
        if m > self.max_M:
            self.max_M = m
        if self.magic_cap is not None and m > self.magic_cap:
            raise MagicCapExceeded(self._cur_step, m)

    def _reduce_dead(self):
        """After a demotion, peel any dead qubit still entangled in a live block (full
        frame reduction; only when block + decouple_demote). Exact/distribution-safe."""
        if self.block and getattr(self.nc, "decouple_demote", False):
            self.nc._reduce_dead(set(self.slot2id.values()))

    # ------------------------------------------------------------- primitives
    def _rot(self, slot, angle):
        """Active RZ(angle) on the qubit at `slot`, frame-conjugated. angle is the
        phase put on |1> (diag(1, e^{i angle})); an X-frame on the axis flips it."""
        q = self.slot2id.get(slot)
        if q is None:
            return
        theta = -angle if self.frame.xb(slot) else angle
        self.nc.apply_rotation(0, 1 << q, theta)
        self._track_M()

    def _apply_u2(self, prog, inst, a1):
        """De-fuse a frame-dependent fused U2 node and apply it exactly: select the
        2x2 by the incoming frame (as core.py does), decompose into Pauli rotations
        (ZYZ), apply to the active state, then reset the frame to the node's out."""
        from ttn_backend.core import _u2_node_matrix_and_frame
        d = ds_mod._d(inst)
        U, out = _u2_node_matrix_and_frame(prog, d["cp_idx"], self.frame, a1)
        q = self.slot2id.get(a1)
        if q is not None:
            self._apply_1q_unitary(q, np.asarray(U, dtype=complex))
        self.frame.set_xz(a1, out & 1, (out >> 1) & 1)

    def _apply_u4(self, prog, inst, a1, a2):
        """De-fuse a fused U4 node = (single-qubit unitary on lo) . CNOT(lo->hi),
        frame-selected. We recover it by applying the exact 4x4 via: CNOT (Clifford)
        then the residual single-qubit unitary on lo. The residual is read off the
        matrix; if the node is not of that structure we fall back to a general 2q
        decomposition."""
        from ttn_backend.core import _u4_node_matrix_and_frame
        d = ds_mod._d(inst)
        U, out = _u4_node_matrix_and_frame(prog, d["cp_idx"], self.frame, a1, a2)
        lo = self.slot2id.get(a1)
        hi = self.slot2id.get(a2)
        if lo is not None and hi is not None:
            self._apply_2q_unitary(lo, hi, np.asarray(U, dtype=complex))
        self.frame.set_xz(a1, out & 1, (out >> 1) & 1)
        self.frame.set_xz(a2, (out >> 2) & 1, (out >> 3) & 1)

    # ---- general unitary application via Pauli-rotation decomposition ----
    def _apply_1q_unitary(self, q, U):
        """Apply an arbitrary 2x2 unitary to active qubit q by ZXZ decomposition
        U = e^{i a} Rz(b) Rx(c) Rz(d) -> 3 Pauli rotations (global phase dropped).
        ZXZ (not ZYZ): the simulator's apply_rotation uses the literal Pauli string
        X^x Z^z (phase 0), so the Y generator (x=1,z=1 -> XZ) would be non-Hermitian
        and exp(-i th XZ/2) non-unitary. X and Z are Hermitian -> valid rotations."""
        b, c, d = _zxz_angles(U)
        # Rz(d) first (right-most acts first on the ket)
        if abs(d) > 1e-12:
            self.nc.apply_rotation(0, 1 << q, d)
        if abs(c) > 1e-12:
            self.nc.apply_rotation(1 << q, 0, c)        # Rx: P = X = (x=1,z=0)
        if abs(b) > 1e-12:
            self.nc.apply_rotation(0, 1 << q, b)
        self._track_M()

    def _apply_2q_unitary(self, lo, hi, U):
        """Apply a fused U4 (basis |hi,lo>, lo = LSB). The clifft U4 is
        (single-qubit M on lo) . CNOT(lo->hi) up to frame Paulis already folded in.
        Recover M and the 2q Clifford by matching; general KAK fallback if needed."""
        decomp = _u4_decompose(U)
        for (kind, args) in decomp:
            if kind == "cx":
                self.nc.cx(lo if args[0] == 0 else hi, lo if args[1] == 0 else hi)
            elif kind == "cz":
                self.nc.cz(lo, hi)
            elif kind == "h":
                self.nc.h(lo if args[0] == 0 else hi)
            elif kind == "s":
                self.nc.s(lo if args[0] == 0 else hi, dag=args[1])
            elif kind == "rot1":  # (which, x, z, theta)
                which, x, z, theta = args
                qq = lo if which == 0 else hi
                self.nc.apply_rotation(x << qq, z << qq, theta)
            elif kind == "rot2":  # two-qubit Pauli rotation (xl,zl,xh,zh,theta)
                xl, zl, xh, zh, theta = args
                self.nc.apply_rotation((xl << lo) | (xh << hi),
                                       (zl << lo) | (zh << hi), theta)
        self._track_M()

    # --------------------------------------------------------------- run_shot
    def run_shot(self, prog, seed, max_steps=None, step_recorder=None):
        """step_recorder(step, self) -- optional per-step hook, called at the top of
        each step (state as of after steps 0..step-1) and once after the last step.
        Used by the per-step memory comparison to sample the live representation."""
        rng = np.random.default_rng(seed)
        self._reset(prog)
        self.nc.rng = rng
        total = len(prog)
        run_steps = total if max_steps is None else min(total, int(max_steps))
        noise_sampler = ds_mod.ClifftNoiseSampler(prog, rng)

        for step in range(run_steps):
            self._cur_step = step
            if step_recorder is not None:
                step_recorder(step, self)
            inst = prog[step]
            name = _opname(inst.opcode)
            if name in ds_mod.IGNORE_OPS:
                continue
            a1 = int(inst.axis_1); a2 = int(inst.axis_2)
            flags = int(getattr(inst, "flags", 0))
            sign = 1 if (flags & _FLAG_SIGN) else 0

            # ---- frame-only Clifford (dormant/stabilizer qubits) ----
            if name == "OP_FRAME_H":      self.frame.h(a1); continue
            if name in ("OP_FRAME_S", "OP_FRAME_S_DAG"): self.frame.s_gate(a1); continue
            if name == "OP_FRAME_CNOT":   self.frame.cnot(a1, a2); continue
            if name == "OP_FRAME_CZ":     self.frame.cz(a1, a2); continue
            if name == "OP_FRAME_SWAP":   self.frame.swap(a1, a2); continue

            # ---- noise / conditional Pauli (shared helpers, identical) ----
            if name == "OP_APPLY_PAULI":
                d = ds_mod._d(inst); cond = d.get("condition_idx"); mask = d.get("cp_mask_idx")
                if cond is not None and mask is not None and int(self.record.get(int(cond), 0)) == 1:
                    ds_mod._apply_cp_mask(prog, int(mask), self.frame, rng)
                continue
            if name == "OP_NOISE":
                d = ds_mod._d(inst); site = d.get("noise_site_idx")
                if site is not None:
                    ds_mod._apply_noise_site(prog, int(site), self.frame, rng, noise_sampler)
                continue
            if name == "OP_NOISE_BLOCK":
                d = ds_mod._d(inst)
                start = d.get("start_site", d.get("noise_site_idx", d.get("block_idx")))
                count = d.get("count", 1)
                if start is not None:
                    for s in range(int(start), int(start) + int(count)):
                        ds_mod._apply_noise_site(prog, s, self.frame, rng, noise_sampler)
                continue
            if name == "OP_READOUT_NOISE":
                d = ds_mod._d(inst); entry_idx = d.get("readout_noise_idx")
                entries = getattr(prog, "readout_noise", None)
                if entry_idx is not None and entries is not None:
                    entry = entries[int(entry_idx)]
                    meas_idx = int(entry["meas_idx"])
                    if float(rng.random()) < float(entry["prob"]):
                        self.record[meas_idx] = int(self.record.get(meas_idx, 0)) ^ 1
                continue

            # ---- dormant measurements (frame-only, identical) ----
            if name in ("OP_MEAS_DORMANT_STATIC", "OP_MEAS_DORMANT_STATIC_FORCED"):
                d = ds_mod._d(inst); cidx = int(d.get("classical_idx", 0))
                self.record[cidx] = self.frame.xb(a1) ^ sign
                continue
            if name in ("OP_MEAS_DORMANT_RANDOM", "OP_MEAS_DORMANT_RANDOM_FORCED"):
                d = ds_mod._d(inst); cidx = int(d.get("classical_idx", 0))
                m_abs = int(rng.integers(0, 2))
                self.record[cidx] = m_abs ^ sign
                self.frame.set_xz(a1, m_abs, 0)
                continue

            # ---- EXPAND family: birth an active qubit in |+>, then apply expand rot ----
            if name == "OP_EXPAND":
                self._birth(a1); continue
            if name in ("OP_EXPAND_T", "OP_EXPAND_T_DAG"):
                self._birth(a1)
                self._rot(a1, _T_ANGLE if name == "OP_EXPAND_T" else -_T_ANGLE)
                continue
            if name == "OP_EXPAND_ROT":
                d = ds_mod._d(inst); self._birth(a1)
                self._rot(a1, cmath.phase(complex(d["weight_re"], d["weight_im"])))
                continue

            # ---- diagonal phase ops on active idents ----
            if name == "OP_PHASE_T":
                q = self.slot2id.get(a1)
                if q is not None: self.nc.apply_rotation(0, 1 << q, _T_ANGLE)
                self._track_M(); continue
            if name == "OP_PHASE_T_DAG":
                q = self.slot2id.get(a1)
                if q is not None: self.nc.apply_rotation(0, 1 << q, -_T_ANGLE)
                self._track_M(); continue
            if name == "OP_PHASE_ROT":
                d = ds_mod._d(inst); q = self.slot2id.get(a1)
                if q is not None:
                    self.nc.apply_rotation(0, 1 << q,
                                           cmath.phase(complex(d["weight_re"], d["weight_im"])))
                self._track_M(); continue

            # ---- active single-axis gates ----
            if name == "OP_ARRAY_H":
                q = self.slot2id.get(a1)
                if q is not None: self.nc.h(q)
                self.frame.h(a1)
                continue
            if name == "OP_ARRAY_S":
                q = self.slot2id.get(a1)
                if q is not None: self.nc.s(q, dag=False)
                self.frame.s_gate(a1)
                continue
            if name == "OP_ARRAY_S_DAG":
                q = self.slot2id.get(a1)
                if q is not None: self.nc.s(q, dag=True)
                self.frame.s_gate(a1)
                continue
            if name == "OP_ARRAY_T":
                self._rot(a1, _T_ANGLE); continue
            if name == "OP_ARRAY_T_DAG":
                self._rot(a1, -_T_ANGLE); continue
            if name == "OP_ARRAY_ROT":
                d = ds_mod._d(inst)
                self._rot(a1, cmath.phase(complex(d["weight_re"], d["weight_im"])))
                continue
            if name == "OP_ARRAY_U2":
                self._apply_u2(prog, inst, a1); continue

            # ---- active two-axis gates ----
            if name == "OP_ARRAY_CNOT":
                u = self.slot2id.get(a1); v = self.slot2id.get(a2)
                if u is not None and v is not None:
                    self.nc.cx(u, v)
                self.frame.cnot(a1, a2)
                continue
            if name == "OP_ARRAY_CZ":
                u = self.slot2id.get(a1); v = self.slot2id.get(a2)
                if u is not None and v is not None:
                    self.nc.cz(u, v)
                self.frame.cz(a1, a2)
                continue
            if name == "OP_ARRAY_MULTI_CNOT":
                d = ds_mod._d(inst); tgt_slot = a1; tgt = self.slot2id.get(tgt_slot)
                for ctrl_slot in ds_mod._bits(int(d["mask"])):
                    if ctrl_slot == tgt_slot:
                        continue
                    c = self.slot2id.get(ctrl_slot)
                    if tgt is not None and c is not None:
                        self.nc.cx(c, tgt)
                    self.frame.cnot(ctrl_slot, tgt_slot)
                self._track_M()
                continue
            if name == "OP_ARRAY_MULTI_CZ":
                d = ds_mod._d(inst)
                for tgt_slot in ds_mod._bits(int(d["mask"])):
                    if tgt_slot == a1:
                        continue
                    u = self.slot2id.get(a1); v = self.slot2id.get(tgt_slot)
                    if u is not None and v is not None:
                        self.nc.cz(u, v)
                    self.frame.cz(a1, tgt_slot)
                continue
            if name == "OP_ARRAY_U4":
                self._apply_u4(prog, inst, a1, a2); continue

            # ---- active measurements ----
            if name in ("OP_MEAS_ACTIVE_DIAGONAL", "OP_MEAS_ACTIVE_DIAGONAL_FORCED"):
                d = ds_mod._d(inst); cidx = int(d.get("classical_idx", 0))
                q = self.slot2id.get(a1)
                if q is None: continue
                b = self.nc.measure_z(q)
                del self.slot2id[a1]
                self._reduce_dead()
                m_abs = b ^ self.frame.xb(a1)
                self.record[cidx] = m_abs ^ sign
                self.frame.set_xz(a1, m_abs, 0)
                self._track_M()
                continue
            if name in ("OP_MEAS_ACTIVE_INTERFERE", "OP_MEAS_ACTIVE_INTERFERE_FORCED"):
                d = ds_mod._d(inst); cidx = int(d.get("classical_idx", 0))
                q = self.slot2id.get(a1)
                if q is None: continue
                self.nc.h(q)
                b_x = self.nc.measure_z(q)
                del self.slot2id[a1]
                self._reduce_dead()
                m_abs = b_x ^ self.frame.zb(a1)
                self.record[cidx] = m_abs ^ sign
                self.frame.set_xz(a1, m_abs, 0)
                self._track_M()
                continue

            # ---- SWAP (relabel + frame) ----
            if name == "OP_ARRAY_SWAP":
                self._swap_slots(a1, a2)
                self.frame.swap(a1, a2)
                continue
            if name in ("OP_SWAP_MEAS_INTERFERE", "OP_SWAP_MEAS_INTERFERE_FORCED"):
                d = ds_mod._d(inst); cidx = int(d.get("classical_idx", 0))
                self._swap_slots(a1, a2)
                self.frame.swap(a1, a2)
                q = self.slot2id.get(a2)
                if q is None: continue
                self.nc.h(q)
                b_x = self.nc.measure_z(q)
                del self.slot2id[a2]
                self._reduce_dead()
                m_abs = b_x ^ self.frame.zb(a2)
                self.record[cidx] = m_abs ^ sign
                self.frame.set_xz(a2, m_abs, 0)
                self._track_M()
                continue
            # unknown opcode: ignore (matches tensor backend fall-through)

        if step_recorder is not None:
            step_recorder(run_steps, self)
        self.last_max_M = self.max_M
        return self.record

    def _swap_slots(self, a1, a2):
        i1 = self.slot2id.get(a1); i2 = self.slot2id.get(a2)
        if i1 is not None: del self.slot2id[a1]
        if i2 is not None: del self.slot2id[a2]
        if i1 is not None: self.slot2id[a2] = i1
        if i2 is not None: self.slot2id[a1] = i2

    # ----------------------------------------------------------------- sample
    def sample(self, prog, shots, seed=None, num_measurements=None):
        master = np.random.default_rng(seed)
        if num_measurements is None:
            num_measurements = prog.num_measurements
        out = np.zeros((shots, num_measurements), dtype=np.uint8)
        peak_M = 0
        for sh in range(shots):
            sd = int(master.integers(0, 2**63 - 1))
            rec = self.run_shot(prog, sd)
            peak_M = max(peak_M, self.last_max_M)
            for cidx, bit in rec.items():
                if 0 <= cidx < num_measurements:
                    out[sh, cidx] = bit
        self.last_max_M = peak_M
        return out


# ===========================================================================
# Single-qubit ZXZ decomposition: U = e^{i a} Rz(b) Rx(c) Rz(d),
# Rz(t) = diag(e^{-i t/2}, e^{i t/2}),  Rx(t) = cos(t/2) I - i sin(t/2) X.
# (Global phase a dropped: it is a true global phase when U acts on one qubit.)
# ZXZ rather than ZYZ so the middle rotation generator is X (Hermitian); the
# simulator's apply_rotation cannot represent a Hermitian Y (it would use the
# non-Hermitian XZ).  Explicit form:
#   Rz(b)Rx(c)Rz(d) =
#     [[ e^{-i(b+d)/2} cos(c/2),  -i e^{-i(b-d)/2} sin(c/2)],
#      [-i e^{ i(b-d)/2} sin(c/2),     e^{ i(b+d)/2} cos(c/2)]]
# ===========================================================================
def _zxz_angles(U):
    U = np.asarray(U, dtype=complex)
    det = U[0, 0] * U[1, 1] - U[0, 1] * U[1, 0]
    if abs(det) < 1e-15:
        raise ValueError("singular U2")
    U = U / cmath.sqrt(det)              # normalise to SU(2)
    c = 2.0 * math.atan2(abs(U[1, 0]), abs(U[0, 0]))
    if abs(U[0, 0]) < 1e-12:             # cos(c/2) ~ 0: only b-d defined; pick d=0
        bmd = 2.0 * (cmath.phase(U[1, 0]) + math.pi / 2.0)
        bpd = 0.0
    elif abs(U[1, 0]) < 1e-12:           # sin(c/2) ~ 0: only b+d defined; pick d=0
        bpd = -2.0 * cmath.phase(U[0, 0])
        bmd = 0.0
    else:
        bpd = -2.0 * cmath.phase(U[0, 0])                       # b + d
        bmd = 2.0 * (cmath.phase(U[1, 0]) + math.pi / 2.0)      # b - d  (U10 carries -i)
    b = 0.5 * (bpd + bmd)
    d = 0.5 * (bpd - bmd)
    return b, c, d


# ===========================================================================
# Fused-U4 de-fusion.  The clifft U4 (basis |hi,lo>, lo = LSB) is structurally
# (single-qubit unitary M on lo) . CNOT(lo->hi), with the incoming frame Pauli
# already folded in by node selection.  We recover M . CNOT exactly by:
#   1. right-multiplying U by CNOT(lo->hi) to strip the CNOT  ->  V = U . CNOT,
#   2. V must be block-diagonal = M (x) I_hi-controlled?  In fact for
#      U = (M_lo (x) I) . CNOT(lo->hi),  U . CNOT = M_lo (x) I, a 1q gate on lo.
# If that holds we emit [CNOT(lo->hi), ZYZ(M_lo) on lo].  Otherwise we fall back
# to a full KAK-free generic decomposition into <=3 CNOTs (rare / defensive).
# ===========================================================================
_CNOT_lohi = np.array([[1, 0, 0, 0],   # basis |hi,lo>, control=lo(LSB), target=hi
                       [0, 0, 0, 1],
                       [0, 0, 1, 0],
                       [0, 1, 0, 0]], dtype=complex)


def _u4_decompose(U):
    """Return a list of NearClifford-applicable ops reproducing U (up to global
    phase). Ops: ('cx',(c,t)), ('cz',()), ('h',(w,)), ('s',(w,dag)),
    ('rot1',(which,x,z,theta)), ('rot2',(xl,zl,xh,zh,theta))."""
    U = np.asarray(U, dtype=complex)
    # Try the structural form  U = (M_lo (x) I) . CNOT(lo->hi).
    V = U @ _CNOT_lohi               # = M_lo (x) I  if structure holds
    M_lo = _extract_1q_on_lo(V)
    if M_lo is not None:
        b, c, d = _zxz_angles(M_lo)
        ops = [("cx", (0, 1))]       # CNOT control=lo(0) target=hi(1)
        if abs(d) > 1e-12: ops.append(("rot1", (0, 0, 1, d)))   # Rz
        if abs(c) > 1e-12: ops.append(("rot1", (0, 1, 0, c)))   # Rx
        if abs(b) > 1e-12: ops.append(("rot1", (0, 0, 1, b)))   # Rz
        if _check_u4(ops, U):
            return ops
    # Try  U = (M_lo (x) I) . CZ.
    V = U @ _gate_cz4()
    M_lo = _extract_1q_on_lo(V)
    if M_lo is not None:
        b, c, d = _zxz_angles(M_lo)
        ops = [("cz", ())]
        if abs(d) > 1e-12: ops.append(("rot1", (0, 0, 1, d)))   # Rz
        if abs(c) > 1e-12: ops.append(("rot1", (0, 1, 0, c)))   # Rx
        if abs(b) > 1e-12: ops.append(("rot1", (0, 0, 1, b)))   # Rz
        if _check_u4(ops, U):
            return ops
    raise NotImplementedError(
        "U4 node is not (1q-on-lo).(CNOT|CZ); generic 2q decomposition not "
        "implemented (no such node occurs in the QEC benchmark circuits).")


def _extract_1q_on_lo(V):
    """If V == M (x) I_hi (i.e. acts only on lo), return the 2x2 M else None.
    Basis |hi,lo>: index = 2*hi + lo. V[2*h+l, 2*h'+l'] must be M[l,l'] * delta_hh'."""
    M = np.zeros((2, 2), dtype=complex)
    for l in range(2):
        for lp in range(2):
            vals = [V[2 * h + l, 2 * h + lp] for h in range(2)]
            if abs(vals[0] - vals[1]) > 1e-9:
                return None
            M[l, lp] = vals[0]
    # off-diagonal (hi-changing) blocks must vanish
    for h in range(2):
        for hp in range(2):
            if h == hp:
                continue
            for l in range(2):
                for lp in range(2):
                    if abs(V[2 * h + l, 2 * hp + lp]) > 1e-9:
                        return None
    # M must be unitary
    if np.linalg.norm(M.conj().T @ M - np.eye(2)) > 1e-7:
        return None
    return M


def _gate_cz4():
    U = np.eye(4, dtype=complex); U[3, 3] = -1.0
    return U


def _check_u4(ops, U):
    """Verify the op list reproduces U (up to global phase) as a 4x4 on |hi,lo>."""
    M = _ops_to_matrix(ops)
    # match up to global phase
    i, j = np.unravel_index(np.argmax(np.abs(U)), U.shape)
    if abs(M[i, j]) < 1e-12:
        return False
    ph = U[i, j] / M[i, j]
    return np.linalg.norm(U - ph * M) < 1e-6


def _ops_to_matrix(ops):
    """Build the 4x4 matrix (basis |hi,lo>, lo=LSB) from the op list, for checking."""
    I2 = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)

    def onq(m, which):   # which: 0=lo (LSB), 1=hi (MSB) in |hi,lo>
        return np.kron(m, I2) if which == 1 else np.kron(I2, m)

    M = np.eye(4, dtype=complex)
    for kind, args in ops:
        if kind == "cx":
            M = _CNOT_lohi @ M
        elif kind == "cz":
            M = _gate_cz4() @ M
        elif kind == "h":
            H = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)
            M = onq(H, args[0]) @ M
        elif kind == "s":
            S = np.array([[1, 0], [0, -1j if args[1] else 1j]], dtype=complex)
            M = onq(S, args[0]) @ M
        elif kind == "rot1":
            which, x, z, theta = args
            P = (X @ Z) if (x and z) else (X if x else Z)
            R = math.cos(theta / 2) * I2 - 1j * math.sin(theta / 2) * P
            M = onq(R, which) @ M
    return M
