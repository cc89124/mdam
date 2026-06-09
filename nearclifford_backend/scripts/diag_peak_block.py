"""Diagnose the genuine entanglement structure of the PEAK magic block.

End-to-end block factoring reports d5_r5 peak max-block = 13, while the
resource-only per-flush core estimate was ~7. The gap means one of:

  (A) the 13-qubit block is GENUINELY 13-way entangled (the magic really does
      grow with distance -- the optimistic 7 was a per-measurement under-count), or
  (B) the block is a TENSOR PRODUCT of smaller irreducible clusters that the
      current `factor()` cannot see (it only peels SINGLE-qubit product factors),
      so a multi-qubit / connected-component split would recover ~7.

This script captures the largest block at the moment a new peak max-block is hit
during real backend shots, then computes its FINEST tensor factorization (the
partition of qubits into independent factors) via product-cut bipartitions:

  qubits i,j are in DIFFERENT factors  <=>  some bipartition (A|B) separating them
  has Schmidt rank 1 (a product cut). The finest factorization = connected
  components of the graph whose edges join qubits no product cut separates.

The largest component size is the TRUE irreducible block size. If it is ~7 while
max_block is 13, hypothesis (B) holds and a stronger factor() wins.
"""
from __future__ import annotations
import sys
import numpy as np

import clifft
from nearclifford_backend.backend import NearCliffordBackend, count_idents
from nearclifford_backend.block_magic import BlockLazyNearClifford

_TOL = 1e-8


def pairwise_components(qubits, vec):
    """Cheap O(k^2) connectivity: edge(i,j) iff mutual information I(i:j)>0
    (rho_ij != rho_i (x) rho_j). For a pure state this MERGES correlated qubits;
    GHZ-type pairs with zero pairwise MI may be split, so the largest component is
    a LOWER BOUND on the true irreducible factor size. Tractable for any k."""
    k = len(qubits)
    arr = vec.reshape([2] * k)
    psi = arr / np.linalg.norm(arr.ravel())

    def rdm(axes_keep):
        axes_tr = [a for a in range(k) if a not in axes_keep]
        t = np.transpose(psi, list(axes_keep) + axes_tr)
        d = 1 << len(axes_keep)
        t = t.reshape(d, -1)
        return t @ t.conj().T

    sep = [[False] * k for _ in range(k)]
    for i in range(k):
        ai = k - 1 - i
        ri = rdm([ai])
        for j in range(i + 1, k):
            aj = k - 1 - j
            rj = rdm([aj])
            rij = rdm(sorted([ai, aj]))
            prod = np.kron(ri, rj) if ai < aj else np.kron(rj, ri)
            if np.linalg.norm(rij - prod) < 1e-7:
                sep[i][j] = sep[j][i] = True
    return _components(qubits, sep, k)


def _components(qubits, sep, k):
    seen = [False] * k
    comps = []
    for start in range(k):
        if seen[start]:
            continue
        stack = [start]; comp = []; seen[start] = True
        while stack:
            x = stack.pop(); comp.append(x)
            for y in range(k):
                if not seen[y] and not sep[x][y]:
                    seen[y] = True; stack.append(y)
        comps.append(sorted(qubits[j] for j in comp))
    return comps


def finest_factorization(qubits, vec):
    """Return list of factor qubit-groups (finest tensor product partition).
    EXACT via product-cut bipartitions for k<=15; else falls back to the cheap
    pairwise lower-bound (full 2^k enumeration is intractable)."""
    k = len(qubits)
    if k <= 1:
        return [list(qubits)]
    if k > 20:                       # 2^k too large to analyse safely; report size only
        return [list(qubits)]
    if k > 15:
        return pairwise_components(qubits, vec)
    arr = vec.reshape([2] * k)  # numpy axis a <-> qubit position (k-1-a) ... LSB=qubits[0]
    # axis for qubit-position j (qubits[j], LSB=j) is a = k-1-j
    sep = [[False] * k for _ in range(k)]   # sep[i][j]: i,j proven separable
    # enumerate proper non-empty subsets A of positions; test product cut A|B.
    full = (1 << k) - 1
    for mask in range(1, 1 << k):
        if mask == full:
            continue
        A = [j for j in range(k) if (mask >> j) & 1]
        if len(A) > k - len(A):       # test each unordered cut once
            continue
        B = [j for j in range(k) if not (mask >> j) & 1]
        axesA = [k - 1 - j for j in A]
        axesB = [k - 1 - j for j in B]
        M = np.transpose(arr, axesA + axesB).reshape(1 << len(A), 1 << len(B))
        s = np.linalg.svd(M, compute_uv=False)
        rank = int(np.sum(s > _TOL * s[0]))
        if rank == 1:                 # product cut: every i in A separable from j in B
            for i in A:
                for j in B:
                    sep[i][j] = sep[j][i] = True
    return _components(qubits, sep, k)


class CapturingBlock(BlockLazyNearClifford):
    def __init__(self, n):
        super().__init__(n)
        self.peak_snapshot = None   # (qubits, vec copy) at the highest peak
        self.peak_all_blocks = None

    def _bump(self):
        mb = self.mag.max_block()
        if mb > self.max_M:
            self.max_M = mb
            big = max(self.mag.blocks, key=lambda b: len(b[0]))
            self.peak_snapshot = (list(big[0]), big[1].copy())
            self.peak_all_blocks = [len(b[0]) for b in self.mag.blocks]
        if self.cap is not None and mb > self.cap:
            from nearclifford_backend.backend import MagicCapExceeded
            raise MagicCapExceeded(-1, mb)


def main():
    circ = sys.argv[1] if len(sys.argv) > 1 else "coherent_d5_r5"
    nshots = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    src = open(f"qec_bench/circuits/{circ}.stim").read()
    prog = clifft.compile(src)
    print(f"circuit={circ} idents={count_idents(prog)} shots={nshots}", flush=True)

    be = NearCliffordBackend(block=True)
    # swap in the capturing simulator class
    import nearclifford_backend.backend as B
    B.BlockLazyNearClifford = CapturingBlock

    master = np.random.default_rng(7)
    best = None
    for sh in range(nshots):
        sd = int(master.integers(0, 2**63 - 1))
        be.run_shot(prog, sd)
        sim = be.nc
        snap = sim.peak_snapshot
        mb = sim.max_M
        blocks = sim.peak_all_blocks
        print(f"  shot {sh}: max_block={mb}  block-sizes-at-peak={blocks}", flush=True)
        if snap is not None and (best is None or len(snap[0]) > len(best[0])):
            best = snap
    if best is None:
        print("no magic block ever formed (max_block stayed 0)"); return
    qubits, vec = best
    print(f"\nLargest captured peak block: {len(qubits)} qubits", flush=True)
    comps = finest_factorization(qubits, vec)
    sizes = sorted((len(c) for c in comps), reverse=True)
    print(f"finest tensor factorization -> {len(comps)} independent factors, sizes={sizes}")
    print(f"TRUE irreducible max-block = {sizes[0]}  (reported max_block = {len(qubits)})")
    if sizes[0] < len(qubits):
        print(">>> hypothesis (B): factor() leaves tensor-separable structure on the "
              "table; a connected-component split recovers the smaller true size.")
    else:
        print(">>> hypothesis (A): the block is genuinely irreducible at this size.")


if __name__ == "__main__":
    main()
