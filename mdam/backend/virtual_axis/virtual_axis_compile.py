"""Offline compile-pass probe: capture, per measurement, the pulled-back core Pauli
algebra the physical NC backend materialises, and localize it to its virtual rank r.
Demonstrates the reduction on REAL circuit cores (physical support B vs virtual r).

This is the per-flush-core view (validates the localization mechanism on real Paulis).
The full resident-state comparison (continuous localization reaching clifft_k, i.e.
cultivation 14->10) needs the runtime backend -- the next increment."""
import sys, os
os.chdir("/home/jung/clifft-paper"); sys.path.insert(0, "/home/jung/clifft-paper")
sys.setrecursionlimit(50000)
import numpy as np, clifft
from mdam.backend.backend import NearCliffordBackend
from mdam.backend.block_magic import BlockLazyNearClifford
from mdam.backend.virtual_axis.virtual_axis import (
    localize_to_virtual_axes, _bits)

# capture per-measurement pulled-back core Paulis
BUCKET = {}          # meas_idx -> list of (x,z,phase) pullbacks
CUR = [-1]

_orig_flush_core = BlockLazyNearClifford._flush_core
def flush_core_w(self, qx, qz):
    CUR[0] = self._meas_ctr           # this flush's meas index (pre-increment)
    BUCKET[CUR[0]] = []
    return _orig_flush_core(self, qx, qz)
BlockLazyNearClifford._flush_core = flush_core_w

_orig_flush_one = BlockLazyNearClifford._flush_one
def flush_one_w(self, x, z, theta):
    xp, zp, pp = self._pullback(x, z)
    if CUR[0] in BUCKET:
        BUCKET[CUR[0]].append((xp, zp, pp))
    return _orig_flush_one(self, x, z, theta)
BlockLazyNearClifford._flush_one = flush_one_w

_orig_measure_z = BlockLazyNearClifford.measure_z
def measure_z_w(self, q):
    mi = self._meas_ctr - 1           # _flush_core already incremented
    xp, zp, pp = self._pullback(0, 1 << q)
    if mi in BUCKET:
        BUCKET[mi].append((xp, zp, pp))
    return _orig_measure_z(self, q)
BlockLazyNearClifford.measure_z = measure_z_w


def analyze(circ, seed=42):
    BUCKET.clear(); CUR[0] = -1
    prog = clifft.compile(open(f"qec_bench/circuits/{circ}.stim").read())
    be = NearCliffordBackend(block=True)
    n = [0]
    def rec(s, bk): n[0] = bk.nc.n
    be.run_shot(prog, seed, step_recorder=rec)
    rows = []
    for mi, paulis in BUCKET.items():
        if not paulis:
            continue
        supp = 0
        for (x, z, _) in paulis:
            supp |= x | z
        B = len(_bits(supp))
        if B == 0:
            continue
        res = localize_to_virtual_axes(paulis, n[0])
        rows.append((B, res.r))
    return rows


print(f"{'circuit':16} {'#meas':>6} {'peakB':>6} {'peakR':>6} {'r<B':>6} {'sumB':>7} {'sumR':>7} {'save%':>6}")
for circ in ["distillation", "cultivation_d3", "cultivation_d5"]:
    rows = analyze(circ)
    if not rows:
        print(f"{circ:16}  (no magic flushes)"); continue
    peakB = max(b for b, r in rows); peakR = max(r for b, r in rows)
    nred = sum(1 for b, r in rows if r < b)
    sumB = sum(b for b, r in rows); sumR = sum(r for b, r in rows)
    save = 100 * (1 - sumR / max(sumB, 1))
    print(f"{circ:16} {len(rows):>6} {peakB:>6} {peakR:>6} {nred:>6} {sumB:>7} {sumR:>7} {save:>5.1f}%")
print("DONE")
