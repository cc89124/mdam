"""Verify + benchmark the deferred-affine backend.

Correctness: on boundary-free circuits the affine backend must reproduce the
measurement-record DISTRIBUTION of clifft.sample (ground truth). We compare
per-measurement marginals P(bit=1) over many shots (RNG streams differ, so this
is a distribution check, not bit-identical). Also confirms it REFUSES circuits
with non-diagonal boundaries (d5_r5, distillation).

Performance: per-shot wall-clock of affine vs the tensor TTN backend vs clifft.
"""
import os
import time
import numpy as np
import clifft

from ttn_backend.backend_spec import export_backend_spec, assign_homes_and_classify
from ttn_backend import TTNBackend
from ttn_backend.affine_backend import AffineActiveBackend, BoundaryEncountered

# Truly boundary-free (audited): only diagonal/CNOT/Z-meas in the active stream.
# NOTE: cultivation_d3 is NOT boundary-free -- it has OP_SWAP_MEAS_INTERFERE
# (X-basis measurement); the earlier regime analysis missed that opcode.
BOUNDARY_FREE = ["coherent_d3_r1", "coherent_d5_r1", "coherent_d7_r1"]
HAS_BOUNDARY = ["cultivation_d3", "coherent_d5_r5", "distillation"]
N_BIG = 8000        # clifft + affine (both fast)
N_TTN = 80          # tensor backend (slow) -- sanity column, looser marginal tol
N_TTN_PERF = 25     # tensor backend per-shot timing (slow; small sample is enough)


def marg(arr):
    return arr.mean(axis=0)


def load(circ):
    return clifft.compile(open(f"qec_bench/circuits/{circ}.stim").read())


def time_ttn(prog, budget_s=20.0, max_shots=None):
    """Per-shot TTN time, adaptive: stop after max_shots OR budget_s wall-clock
    (whichever first), so a slow circuit can't hang. Returns (per_shot_s, n)."""
    max_shots = max_shots or N_TTN_PERF
    spec = export_backend_spec(prog, strict=False)
    bk = TTNBackend(spec, assign_homes_and_classify(spec))
    t0 = time.perf_counter()
    n = 0
    while n < max_shots:
        bk.run_shot(prog, n)
        n += 1
        if time.perf_counter() - t0 >= budget_s:
            break
    return (time.perf_counter() - t0) / n, n


def clifft_marg_budget(prog, budget_s=120.0, chunk=10, seed=1):
    """clifft marginal P(bit=1), sampled in small chunks until N_BIG shots OR
    budget_s wall-clock. clifft dense sim costs ~seconds/shot at large k, so the
    small chunk lets the budget actually bound d7 (a single 200-shot chunk would
    blow past any budget). Returns (marginal, n_used)."""
    nm = prog.num_measurements
    tot = np.zeros(nm, dtype=np.float64)
    n = 0
    s = seed
    t0 = time.perf_counter()
    while n < N_BIG:
        m = clifft.sample(prog, chunk, seed=s).measurements
        tot += m.sum(axis=0)
        n += chunk
        s += 1
        if time.perf_counter() - t0 >= budget_s:
            break
    return tot / n, n


def main():
    # --- correctness: affine vs clifft (ground truth). affine is instant; clifft
    #     (dense sim) is time-budgeted per circuit -- d7_r1 at 8000 shots would
    #     take >1h, which is itself the point. tol scales with the clifft n used.
    print("=== correctness: per-measurement marginal P(bit=1), affine vs clifft ===")
    print(f"{'circuit':16s} {'meas':>5s} {'clifftN':>8s} {'max|aff-clifft|':>15s} "
          f"{'tol(3sig)':>10s} {'verdict':>8s}", flush=True)
    for circ in BOUNDARY_FREE:
        prog = load(circ)
        nm = prog.num_measurements
        try:
            aff_arr = AffineActiveBackend(export_backend_spec(prog, strict=False)).sample(
                prog, N_BIG, seed=2)
        except BoundaryEncountered as e:
            print(f"{circ:16s} {nm:5d} {'-':>8s} {'REFUSED: ' + str(e):>15s}", flush=True)
            continue
        cl_marg, n_cl = clifft_marg_budget(prog, budget_s=120.0)
        d_ac = float(np.max(np.abs(marg(aff_arr) - cl_marg)))
        tol = 3 * (0.5 / np.sqrt(n_cl)) + 0.01   # 3 sigma at the clifft n actually used
        ok = d_ac <= tol
        print(f"{circ:16s} {nm:5d} {n_cl:8d} {d_ac:15.4f} {tol:10.4f} "
              f"{'PASS' if ok else 'FAIL':>8s}", flush=True)

    print("\n=== boundary circuits must be REFUSED by the affine backend ===", flush=True)
    for circ in HAS_BOUNDARY:
        prog = load(circ)
        be = AffineActiveBackend(export_backend_spec(prog, strict=False))
        try:
            be.run_shot(prog, 0)
            print(f"{circ:16s} NOT refused (unexpected)", flush=True)
        except BoundaryEncountered as e:
            print(f"{circ:16s} refused at: {e}", flush=True)

    print("\n=== performance: per-shot wall-clock (affine = O(k) integer ops) ===", flush=True)
    print(f"{'circuit':16s} {'k':>3s} {'clifft':>11s} {'affine':>11s} {'TTN':>12s} "
          f"{'TTN/affine':>11s} {'(ttn n)':>8s}", flush=True)
    for circ in BOUNDARY_FREE:
        prog = load(circ)
        k = int(prog.peak_rank)
        # clifft per-shot: time-budgeted, small chunk (dense sim ~seconds/shot at large k)
        t0 = time.perf_counter(); ncl = 0
        while ncl < 2000:
            clifft.sample(prog, 10, seed=ncl); ncl += 10
            if time.perf_counter() - t0 >= 15.0:
                break
        t_cl = (time.perf_counter() - t0) / ncl
        be = AffineActiveBackend(export_backend_spec(prog, strict=False))
        t0 = time.perf_counter()
        for s in range(2000):
            be.run_shot(prog, s)
        t_aff = (time.perf_counter() - t0) / 2000
        t_ttn, n_ttn = time_ttn(prog, budget_s=20.0)
        print(f"{circ:16s} {k:3d} {t_cl*1e3:9.3f}ms {t_aff*1e3:9.3f}ms "
              f"{t_ttn*1e3:10.3f}ms {t_ttn/t_aff:10.0f}x {n_ttn:8d}", flush=True)


if __name__ == "__main__":
    main()
