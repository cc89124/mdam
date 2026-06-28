"""Strict live-memory invariant for clifft_axis_bounded: NO Theta(2^r) working buffer.

The invariant is

    peak amplitude (complex128) words  <=  2^k_clifft

i.e. at r=k the ONLY amplitude-sized object is the resident register; every kernel's
working set is O(1)/O(r), never a second buffer that scales with 2^r.

A single-point tracemalloc reading cannot prove this: every numpy call (even a pure
in-place `phi *= c`) shows a fixed ~700-word Python/tracemalloc baseline, so an absolute
threshold spuriously "fails".  The rigorous test is SCALING: run each kernel at r =
14,16,18,20 with cap = 2^r (slack 0 = the worst case r=k) and assert the tracemalloc peak
does NOT grow proportionally to 2^r.  A genuine half/full copy (np.linalg.norm on a strided
view, ndarray.resize, an unbounded chunk gather) would grow 4x per +2 in r (64x over the
sweep); a constant reading proves there is no such amplitude temporary.

History: the earlier "peak ~1.5*2^k due to a numpy nditer buffer" reading was WRONG.  The
real Theta(2^(r-1)) temporaries were (A) np.linalg.norm on a non-contiguous half-view (a
ravel/ascontiguousarray copy) and (B) ndarray.resize realloc; both are removed -- norms go
through einsum on the .real/.imag VIEWS (_branch_sqnorm), and the register lives in a
capacity buffer (storage[:sz], grown/shrunk by size+memmove, never resize).  The chunked
rotation kernel is slack-aware and drops to a scalar no-array loop at slack 0.
"""
from __future__ import annotations

import sys
import tracemalloc

import numpy as np

sys.path.insert(0, "/home/jung/clifft-paper")
from nearclifford_backend.clifft_axis.bounded import CliftAxisBoundedNearClifford as B


def _engine(r, cap_rank, seed=0):
    """A bounded engine with |M|=r, a capacity buffer of exactly 2^r, and budget cap
    2^cap_rank (cap_rank=r => slack 0, the strict r=k worst case)."""
    e = B(r + 2)
    e.set_clifft_budget(cap_rank, enforce=False)
    rng = np.random.default_rng(seed)
    e.M = list(range(r))
    st = (rng.standard_normal(1 << r) + 1j * rng.standard_normal(1 << r)).astype(complex)
    st /= np.linalg.norm(st)
    e._storage = st
    e._sz = 1 << r
    e.phi = st[: 1 << r]
    return e


def _peak_words(fn):
    import gc
    gc.collect()
    tracemalloc.start()
    tracemalloc.reset_peak()
    fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak // 16


def _kernel(name, ranks, build):
    """Measure the kernel at each rank (slack 0) and report the growth factor vs 2^r."""
    rows = []
    for r in ranks:
        rows.append((r, _peak_words(build(r))))
    base = rows[0][1] or 1
    top = rows[-1][1]
    span = (1 << ranks[-1]) / (1 << ranks[0])          # 2^r grew by this factor
    grow = top / base
    # PASS if the peak grew far slower than 2^r (a Theta(2^r) temp would grow ~= span)
    ok = grow < max(4.0, 0.05 * span)
    pts = "  ".join(f"r{r}:{w}w" for r, w in rows)
    print(f"  {name:<26} {pts}    grow={grow:.1f}x  (2^r grew {span:.0f}x)  "
          f"{'PASS' if ok else 'FAIL: scales with 2^r'}")
    return ok


def main():
    print("STRICT MEMORY SCALING (slack 0, cap=2^r): amplitude temp must be r-INDEPENDENT\n")
    big = [14, 16, 18, 20]
    ok = True

    # norm fix: squared branch norm via einsum on strided .real/.imag VIEWS (was a 2^(r-1)
    # contiguity copy from np.linalg.norm on the non-contiguous half-view).
    ok &= _kernel("branch_sqnorm (norm)", big,
                  lambda r: (lambda e=_engine(r, r): (lambda: e._branch_sqnorm(0, 0)))())

    # resize fix: drop via in-place axis-swap-to-MSB + size/memmove (was ndarray.resize,
    # which reallocs a fresh 2^(r-1)/2^(r+1) buffer).  Force branch 1 (high-half memmove).
    def build_drop(r):
        e = _engine(r, r)
        e.phi.reshape(-1, 2, 1)[:, 0, :] = 0.0         # low half = 0 -> keep the high half
        return lambda: e._drop_axis_inplace(0)
    ok &= _kernel("drop (resize->memmove)", big, build_drop)

    # in-place bit-swap used by drop (move dropped axis to the MSB)
    ok &= _kernel("swap_axes", big,
                  lambda r: (lambda e=_engine(r, r): (lambda: e._swap_axes(0, r - 1)))())

    # promote (grow) at r-1 -> r within the capacity buffer: zero the new MSB block in
    # place (was ndarray.resize grow -> a fresh 2^r buffer).  cap=2^r so the grow fits.
    def build_promote(r):
        e = _engine(r - 1, r)                          # |M|=r-1, storage 2^(r-1), cap 2^r
        e._storage = np.resize(e._storage, 1 << r)     # capacity room (setup only, not timed)
        e._storage[: 1 << (r - 1)] = e.phi
        e.phi = e._storage[: 1 << (r - 1)]
        return lambda: e._promote(r - 1)
    ok &= _kernel("promote (resize->in-place)", big, build_promote)

    # slack-aware rotation kernel at slack 0 -> SCALAR no-array path (offdiag + diag).
    # Python loop is O(2^r) in time, so sweep only the small ranks; memory is structurally
    # O(1) (numpy 0-d scalars, no work array).
    small = [10, 12, 14]
    ok &= _kernel("lincomb scalar (offdiag)", small,
                  lambda r: (lambda e=_engine(r, r):
                             (lambda: e._pauli_lincomb_inplace(0b101, 0b011, 1, 0.7, -0.5j)))())
    ok &= _kernel("lincomb scalar (diag)", small,
                  lambda r: (lambda e=_engine(r, r):
                             (lambda: e._pauli_lincomb_inplace(0, 0b011, 0, 0.7, -0.5j)))())

    print(f"\nSTRICT MEMORY: {'PASS' if ok else 'FAIL'}  -- at r=k every kernel's amplitude "
          f"workspace is r-independent (no second 2^r buffer); peak amplitude words = 2^k.")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
