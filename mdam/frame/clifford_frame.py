"""Region-local Clifford frame helpers.

This module implements the linear computational-basis part of Clifford windows:
CNOT, MULTI_CNOT, and SWAP. It is intentionally independent from TTN runtime
state so it can be unit-tested before being used by an executor policy.
"""

from __future__ import annotations

import numpy as np


class RegionLinearFrame:
    """Affine GF(2) map over a fixed list of region idents.

    The map acts on computational-basis bit vectors in `support_idents` order:

        x_out = A x_in xor b

    The current v1 only composes linear Clifford gates, so b remains zero unless
    future extensions add affine bit flips.
    """

    def __init__(self, support_idents):
        self.support_idents = [int(x) for x in support_idents]
        self.pos = {ident: i for i, ident in enumerate(self.support_idents)}
        n = len(self.support_idents)
        self.A = np.eye(n, dtype=np.uint8)
        self.b = np.zeros(n, dtype=np.uint8)

    def copy(self):
        other = RegionLinearFrame(self.support_idents)
        other.A = self.A.copy()
        other.b = self.b.copy()
        return other

    def _p(self, ident):
        ident = int(ident)
        if ident not in self.pos:
            raise KeyError(f"ident {ident} not in RegionLinearFrame support")
        return self.pos[ident]

    def compose_cnot(self, control, target):
        """Compose CNOT(control -> target) after the current map."""
        c = self._p(control)
        t = self._p(target)
        if c == t:
            return
        self.A[t, :] ^= self.A[c, :]
        self.b[t] ^= self.b[c]

    def compose_multicnot(self, target, controls):
        for control in controls:
            if int(control) != int(target):
                self.compose_cnot(control, target)

    def compose_swap(self, a, b):
        pa = self._p(a)
        pb = self._p(b)
        if pa == pb:
            return
        self.A[[pa, pb], :] = self.A[[pb, pa], :]
        self.b[[pa, pb]] = self.b[[pb, pa]]

    def apply_bits(self, bits):
        x = np.asarray(bits, dtype=np.uint8)
        if x.shape != (len(self.support_idents),):
            raise ValueError(f"expected {len(self.support_idents)} bits, got {x.shape}")
        return ((self.A @ x) & 1) ^ self.b

    def apply_to_bit_index(self, idx):
        idx = int(idx)
        bits = np.array([(idx >> i) & 1 for i in range(len(self.support_idents))],
                        dtype=np.uint8)
        out = self.apply_bits(bits)
        y = 0
        for i, bit in enumerate(out):
            y |= int(bit) << i
        return int(y)

    def materialize_to_tensor(self, tensor, axis_map):
        """Return tensor with the pending frame applied.

        `axis_map` maps each ident in the frame support to an axis of `tensor`.
        All frame axes must be dimension-2 axes.
        """
        axes = [int(axis_map[i]) for i in self.support_idents]
        for ax in axes:
            if tensor.shape[ax] != 2:
                raise ValueError("RegionLinearFrame only supports dimension-2 axes")
        T = np.moveaxis(tensor, axes, list(range(len(axes))))
        leading = T.shape[:len(axes)]
        rest = T.shape[len(axes):]
        out = np.empty_like(T)
        for src_bits in np.ndindex(*leading):
            dst_bits = tuple(int(x) for x in self.apply_bits(np.array(src_bits, dtype=np.uint8)))
            out[dst_bits] = T[src_bits]
        return np.moveaxis(out, list(range(len(axes))), axes)


def apply_cnot_tensor(tensor, axis_control, axis_target):
    axes = [int(axis_control), int(axis_target)]
    T = np.moveaxis(tensor, axes, [0, 1])
    out = T.copy()
    out[1] = np.flip(out[1], axis=0)
    return np.moveaxis(out, [0, 1], axes)


def apply_swap_tensor(tensor, axis_a, axis_b):
    return np.swapaxes(tensor, axis_a, axis_b)
