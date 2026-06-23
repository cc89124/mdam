"""Phase 2 runtime reality check: does Design A (localize -> diagonal) actually win in
WALL-CLOCK, or only in FLOP?  CNOT/SWAP are 0 FLOP but each is a full array sweep (memory
traffic).  So a weight-w rotation costs:
   current  : 1 off-diagonal butterfly sweep        (12*2^r FLOP, 1 sweep)
   design A : (w-1) CNOT + 1 H + 1 diag = (w+1) sweeps ( ~7*2^r FLOP, w+1 sweeps)
In a memory-bound regime more sweeps can be SLOWER despite fewer FLOP.  Time both with the
ACTUAL bounded numpy kernels at fixed rank, warmup + median.  Reports per-op wall-clock,
sweep count, and bytes moved.
"""
import sys, time, statistics
sys.path.insert(0, "/home/jung/clifft-paper")
import numpy as np
from nearclifford_backend.clifft_axis.bounded import CliftAxisBoundedNearClifford


def make_engine(r):
    """A bounded engine with an active register of rank r (M=[0..r-1]), random unit state."""
    e = CliftAxisBoundedNearClifford(r)
    e.set_clifft_budget(r, enforce=False)
    e.M = list(range(r))
    rng = np.random.default_rng(0)
    v = (rng.standard_normal(1 << r) + 1j * rng.standard_normal(1 << r)).astype(np.complex128)
    v /= np.linalg.norm(v)
    e._storage = v.copy()
    e._sz = 1 << r
    e.phi = e._storage[: e._sz]
    return e


def med_time(fn, iters, warmup=3):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def bench_butterfly(r, mx, mz, iters):
    """time one off-diagonal butterfly R_P (the CURRENT path)."""
    e = make_engine(r)
    base = e.phi.copy()
    c, s = np.cos(0.3), np.sin(0.3)

    def op():
        e.phi[:] = base
        e._pauli_lincomb_inplace(mx, mz, 0, alpha=c, beta=(-1j * s), where="rot")
    return med_time(op, iters)


def bench_localize(r, w, iters):
    """time the DESIGN-A sequence for a weight-w X-string: (w-1) CNOT + 1 H + 1 diagonal R_Z."""
    e = make_engine(r)
    base = e.phi.copy()
    c, s = np.cos(0.3), np.sin(0.3)
    ctrls = list(range(1, w))          # collapse axes 1..w-1 onto axis 0
    m_odd_over_even = np.exp(1j * 0.3)  # diagonal half-array phase placeholder

    def op():
        e.phi[:] = base
        for jc in ctrls:               # (w-1) CNOT permutations (0 FLOP, full sweep each)
            e._cnot_axes(jc, 0)
        e._h_axis(0)                   # 1 H butterfly (4*2^r)
        # diagonal R_Z on axis 0, HALF-array (apply phase to bit0=1 half only)
        v = e.phi.reshape(-1, 2, 1)
        v[:, 1, :] *= m_odd_over_even
    return med_time(op, iters)


def bench_diag_half(r, iters):
    """time a single half-array diagonal R_Z (the Step-1 target kernel)."""
    e = make_engine(r)
    base = e.phi.copy()
    ph = np.exp(1j * 0.3)

    def op():
        e.phi[:] = base
        v = e.phi.reshape(-1, 2, 1)
        v[:, 1, :] *= ph
    return med_time(op, iters)


def bench_diag_full(r, iters):
    """time the CURRENT full-array diagonal (both halves multiplied)."""
    e = make_engine(r)
    base = e.phi.copy()

    def op():
        e.phi[:] = base
        e._pauli_lincomb_inplace(0, 1, 0, alpha=np.cos(0.3), beta=(-1j * np.sin(0.3)), where="rot")
    return med_time(op, iters)


def us(t):
    return f"{t*1e6:8.1f}us"


print(f"{'rank':>4} {'w':>2} | {'butterfly(1sw)':>16} {'localize(w+1 sw)':>18} "
      f"{'ratio A/cur':>12} | {'diag_full':>11} {'diag_half':>11}")
print("-" * 92)
for r in (10, 13, 16, 18):
    iters = max(5, min(200, 2_000_000 // (1 << r)))
    for w in (1, 2, 4):
        mx = 0
        for b in range(w):
            mx |= (1 << b)              # weight-w X-string on axes 0..w-1
        t_bf = bench_butterfly(r, mx, 0, iters)
        t_loc = bench_localize(r, w, iters)
        ratio = t_loc / t_bf if t_bf > 0 else 0
        if w == 1:
            t_df = bench_diag_full(r, iters)
            t_dh = bench_diag_half(r, iters)
            extra = f"{us(t_df):>11} {us(t_dh):>11}"
        else:
            extra = f"{'':>11} {'':>11}"
        print(f"{r:>4} {w:>2} | {us(t_bf):>16} {us(t_loc):>18} {ratio:>11.2f}x | {extra}",
              flush=True)
