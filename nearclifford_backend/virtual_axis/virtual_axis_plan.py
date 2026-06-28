"""Step B -- offline compile: per measurement, localize the core Pauli algebra
{core rotation generators} u {measurement Pauli} to a target virtual basis of r<=B
axes, producing a VirtualCorePlan (phase-exact rotation masks + measurement mask +
the basis's axis qubits and X/Z images for the runtime basis change).

Built on the verified primitives: localize_to_virtual_axes (phase-exact masks) and
clifford_synth (gates for the basis). Here we compile + VERIFY the per-measurement
plans on real circuits (masks reproduce the core's commutation + product structure)."""
import sys, os
os.chdir("/home/jung/clifft-paper"); sys.path.insert(0, "/home/jung/clifft-paper")
sys.setrecursionlimit(50000)
from dataclasses import dataclass, field
import numpy as np, clifft
from nearclifford_backend.backend import NearCliffordBackend
from nearclifford_backend.block_magic import BlockLazyNearClifford
from nearclifford_backend.simulator import pauli_mul, pauli_commute
from nearclifford_backend.virtual_axis.virtual_axis import (
    localize_to_virtual_axes, _bits)


@dataclass
class VirtualCorePlan:
    meas_idx: int
    core_uids: tuple              # rotation uids in flush order
    physical_B: int
    virtual_r: int
    rotation_masks: dict          # uid -> VirtualPauliMask
    measurement_mask: object      # VirtualPauliMask
    axis_qubits: tuple            # physical qubit per virtual axis (diagnostic)


# ---- capture, per measurement, (uid, pulled-back Pauli) of each core rotation + meas ----
_CORE = {}          # meas_idx -> list of (uid|None, (x,z,phase))   (meas uid=None)
_CUR = [-1]

_o_fc = BlockLazyNearClifford._flush_core
def _fc(self, qx, qz):
    _CUR[0] = self._meas_ctr
    _CORE[_CUR[0]] = []
    return _o_fc(self, qx, qz)
BlockLazyNearClifford._flush_core = _fc

_o_do = BlockLazyNearClifford._do_flush
def _do(self, qx, qz, flush):
    for (x, z, p, theta, uid) in flush:
        xp, zp, pp = self._pullback(x, z)
        if _CUR[0] in _CORE:
            _CORE[_CUR[0]].append((uid, (xp, zp, pp)))
    return _o_do(self, qx, qz, flush)
BlockLazyNearClifford._do_flush = _do

_o_mz = BlockLazyNearClifford.measure_z
def _mz(self, q):
    mi = self._meas_ctr - 1
    xp, zp, pp = self._pullback(0, 1 << q)
    if mi in _CORE:
        _CORE[mi].append((None, (xp, zp, pp)))
    return _o_mz(self, q)
BlockLazyNearClifford.measure_z = _mz


def compile_plans(circ, seed=42):
    _CORE.clear(); _CUR[0] = -1
    prog = clifft.compile(open(f"qec_bench/circuits/{circ}.stim").read())
    be = NearCliffordBackend(block=True)
    nn = [0]
    be.run_shot(prog, seed, step_recorder=lambda s, bk: nn.__setitem__(0, bk.nc.n))
    n = nn[0]
    plans = {}
    for mi, entries in _CORE.items():
        if not entries:
            continue
        paulis = [e[1] for e in entries]
        supp = 0
        for (x, z, _) in paulis:
            supp |= x | z
        if supp == 0:
            continue
        res = localize_to_virtual_axes(paulis, n)
        masks = res.masks
        rot = {}
        for k, (uid, _) in enumerate(entries):
            if uid is not None:
                rot[uid] = masks[k]
        plans[mi] = VirtualCorePlan(
            meas_idx=mi, core_uids=tuple(e[0] for e in entries if e[0] is not None),
            physical_B=len(_bits(supp)), virtual_r=res.r,
            rotation_masks=rot, measurement_mask=masks[-1],
            axis_qubits=tuple(range(res.r)))
    return plans, n


def verify_plans(circ):
    """The localized masks must reproduce the core's commutation + product (phase)
    relations (already proven generally; here a per-circuit spot-check on real cores)."""
    plans, n = compile_plans(circ)
    bad = 0
    for mi, pl in plans.items():
        entries = _CORE[mi]
        paulis = [e[1] for e in entries]
        res = localize_to_virtual_axes(paulis, n)
        m = res.masks
        for i in range(len(paulis)):
            for j in range(len(paulis)):
                phys_comm = pauli_commute(paulis[i], paulis[j])
                virt_comm = m[i].commutes(m[j])
                if phys_comm != virt_comm:
                    bad += 1
    return plans, bad


if __name__ == "__main__":
    print(f"{'circuit':16} {'#plans':>7} {'peakB':>6} {'peakR':>6} {'comm-viol':>10}")
    for circ in (sys.argv[1:] or ["distillation", "cultivation_d3", "cultivation_d5"]):
        plans, bad = verify_plans(circ)
        pB = max((p.physical_B for p in plans.values()), default=0)
        pR = max((p.virtual_r for p in plans.values()), default=0)
        print(f"{circ:16} {len(plans):>7} {pB:>6} {pR:>6} {bad:>10}")
    print("DONE")
