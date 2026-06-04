"""Out-of-core blocked tensor storage for resident-streaming TTN execution.

The staged transport pass streams the *computation* tensor (theta) so it is
never materialized. That removes the workspace term from the peak, but the
resident bag tensors (e.g. the two ~67 MB internal bags joined by a saturated
2^11 Schmidt bond in coherent_d5_r5) still sit in RAM between operations.

`BlockTensorStore` keeps such a bag blocked along one axis and backed
out-of-core (a single memmap file on disk). Only the block(s) currently needed
are copied into the Python heap; the full tensor is never RAM-resident. This is
exact (no information is discarded) — it only changes *where* the tensor lives.

RAM accounting convention (matches the existing TTN methodology, which measures
peak from tensor shapes, not process RSS): the *resident* size of a block-store
is one block (the cache), and the full size is tracked separately as
out-of-core bytes.

Operations divide into three kinds w.r.t. the block axis `a`:
  - act on a non-block axis (diagonal/local single-axis, measurement on b != a):
    apply per block, write back. RAM = one block.
  - contract over the block axis (e.g. an adjacent transport across that bond):
    stream blocks and accumulate (the staged-QR pattern). RAM = one block.
  - reshape the block axis together with another axis into a matrix partition:
    cannot stay blocked along `a`; must re-block or materialize. (Not streamed.)
"""
from __future__ import annotations

import os
import numpy as np

_ITEM = np.dtype(np.complex128).itemsize


class BlockTensorStore:
    def __init__(self, shape, block_axis, block_size, path,
                 dtype=np.complex128, _create=True):
        self.shape = tuple(int(s) for s in shape)
        self.ndim = len(self.shape)
        self.block_axis = int(block_axis)
        self.block_size = max(1, int(block_size))
        self.dtype = np.dtype(dtype)
        self.path = str(path)
        axlen = self.shape[self.block_axis]
        self.n_blocks = (axlen + self.block_size - 1) // self.block_size
        mode = "w+" if _create else "r+"
        self._mm = np.memmap(self.path, dtype=self.dtype, mode=mode, shape=self.shape)

    # ---- construction -----------------------------------------------------
    @classmethod
    def from_dense(cls, arr, block_axis, block_size, path):
        arr = np.ascontiguousarray(arr, dtype=np.complex128)
        s = cls(arr.shape, block_axis, block_size, path, arr.dtype)
        s._mm[...] = arr
        s._mm.flush()
        return s

    # ---- block access -----------------------------------------------------
    def _slice(self, i):
        a = self.block_axis
        lo = i * self.block_size
        hi = min(lo + self.block_size, self.shape[a])
        sl = [slice(None)] * self.ndim
        sl[a] = slice(lo, hi)
        return tuple(sl)

    def get_block(self, i):
        """Return block i as a fresh heap array (bounded to block size)."""
        return np.array(self._mm[self._slice(i)])

    def set_block(self, i, data):
        self._mm[self._slice(i)] = data
        self._mm.flush()

    def iter_blocks(self):
        for i in range(self.n_blocks):
            yield i, self.get_block(i)

    # ---- sizes ------------------------------------------------------------
    def _block_numel(self):
        per = 1
        for k, s in enumerate(self.shape):
            per *= (min(self.block_size, s) if k == self.block_axis else s)
        return int(per)

    @property
    def ram_bytes(self):
        """Logical resident size: one block cache."""
        return int(self._block_numel() * self.dtype.itemsize)

    @property
    def ooc_bytes(self):
        """Full tensor size, held out-of-core (on disk)."""
        return int(np.prod(self.shape) * self.dtype.itemsize)

    @property
    def nbytes(self):
        # Resident footprint for peak accounting = the block cache only.
        return self.ram_bytes

    # ---- exact block-wise operations (no block-axis mixing) ---------------
    def squared_norm(self):
        """sum |x|^2 over the whole tensor, streamed one block at a time."""
        total = 0.0
        for _, blk in self.iter_blocks():
            total += float(np.vdot(blk.ravel(), blk.ravel()).real)
        return total

    def apply_diagonal_on_axis(self, axis, factors):
        """Multiply slabs along a NON-block `axis` by `factors[k]`, in place."""
        if axis == self.block_axis:
            raise ValueError("apply_diagonal_on_axis: axis must differ from block axis")
        factors = np.asarray(factors, dtype=np.complex128)
        for i in range(self.n_blocks):
            blk = self.get_block(i)
            shape = [1] * blk.ndim
            shape[axis] = blk.shape[axis]
            blk = blk * factors.reshape(shape)
            self.set_block(i, blk)

    def apply_matrix_on_axis(self, axis, M):
        """Apply 2D matrix `M` (d x d) to a NON-block `axis`, per block."""
        if axis == self.block_axis:
            raise ValueError("apply_matrix_on_axis: axis must differ from block axis")
        M = np.asarray(M, dtype=np.complex128)
        for i in range(self.n_blocks):
            blk = self.get_block(i)
            moved = np.moveaxis(blk, axis, -1)
            sh = moved.shape
            out = (moved.reshape(-1, sh[-1]) @ M.T).reshape(sh)
            self.set_block(i, np.moveaxis(out, -1, axis))

    def axis1_squared_norm(self, axis):
        """sum |x|^2 over the slab where NON-block `axis` index == 1."""
        if axis == self.block_axis:
            raise ValueError("axis1_squared_norm: axis must differ from block axis")
        total = 0.0
        for i in range(self.n_blocks):
            blk = self.get_block(i)
            sl = [slice(None)] * blk.ndim
            sl[axis] = 1
            sub = blk[tuple(sl)]
            total += float(np.vdot(sub.ravel(), sub.ravel()).real)
        return total

    def gram_over_block_axis(self):
        """Streamed Gram on the *non-block* sides for contraction over the block
        axis. Returns sum_i B_i^H B_i where B_i flattens block i with the block
        axis last as columns — i.e. the (rest x blockaxis) reshape. Used when an
        adjacent transport contracts this bond. (Accumulator pattern.)"""
        a = self.block_axis
        rest = int(np.prod([s for k, s in enumerate(self.shape) if k != a]))
        G = None
        for i in range(self.n_blocks):
            blk = self.get_block(i)
            mat = np.moveaxis(blk, a, -1).reshape(rest, blk.shape[a])
            g = mat.conj().T @ mat
            G = g if G is None else _pad_add(G, g)
        return G

    # ---- materialization (fallback / verification) ------------------------
    def to_dense(self):
        return np.array(self._mm)

    def close(self, unlink=True):
        try:
            self._mm.flush()
        except Exception:
            pass
        self._mm = None
        if unlink and os.path.exists(self.path):
            try:
                os.remove(self.path)
            except OSError:
                pass


def _pad_add(A, B):
    n = max(A.shape[0], B.shape[0])
    out = np.zeros((n, n), dtype=np.complex128)
    out[:A.shape[0], :A.shape[1]] += A
    out[:B.shape[0], :B.shape[1]] += B
    return out


def choose_block_axis(shape):
    """Pick the largest axis as the block axis (the dominant bond)."""
    shape = [int(s) for s in shape]
    return int(np.argmax(shape))


def block_size_for_cap(shape, block_axis, cap_bytes, item=_ITEM):
    """Largest block length along block_axis whose block fits in cap_bytes."""
    rest = 1
    for k, s in enumerate(shape):
        if k != block_axis:
            rest *= int(s)
    per_index = max(rest * item, 1)
    bs = max(1, int(cap_bytes) // per_index)
    return min(bs, int(shape[block_axis]))
