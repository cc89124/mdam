"""END-TO-END validation of the complete near-Clifford backend against clifft.

clifft.sample is the authoritative ground truth. On each QEC circuit we compare
the per-measurement marginal P(bit=1) of NearCliffordBackend.sample to clifft's,
over many shots (independent RNG streams -> a DISTRIBUTION check, not bit-exact),
and report the peak magic-register size |M| the backend actually needed.

This is the real test the simulator-core verification could not give: it runs the
ACTUAL clifft bytecode (noise, frame, dormant, readout, detectors and all),
not hand-built circuits.

Usage:  python -m nearclifford_backend.scripts.verify_backend [circ ...]
"""
from __future__ import annotations
import sys
import time
import numpy as np
import clifft

from nearclifford_backend.backend import NearCliffordBackend, count_idents

DEFAULT = ["cultivation_d3", "coherent_d5_r1", "coherent_d7_r1",
           "distillation", "coherent_d5_r5"]

NC_BUDGET_S = 90.0
CL_BUDGET_S = 120.0
TARGET = 6000


def load(circ, fused=True):
    """Default = fused (the canonical clifft compile, same noise/frame/dormant
    representation the shared helpers and the tensor backend are tuned for). The
    boundary gates U2/U4 are de-fused inside the backend. (bytecode_passes=None
    yields plain H/ROT/CNOT but renumbers the noise-site pool, which the shared
    noise helpers do not track -- so fused is the validated input.)"""
    src = open(f"qec_bench/circuits/{circ}.stim").read()
    return clifft.compile(src) if fused else clifft.compile(src, bytecode_passes=None)


def nc_marg(prog, budget_s, target, seed=2, lazy=False, block=False):
    """Backend marginal P(bit=1), time-budgeted. Returns (marg, n, peak_M, per_shot_ms)."""
    be = NearCliffordBackend(lazy=lazy, block=block)
    nm = prog.num_measurements
    tot = np.zeros(nm, dtype=np.float64)
    n = 0; peak_M = 0
    master = np.random.default_rng(seed)
    t0 = time.perf_counter()
    while n < target:
        sd = int(master.integers(0, 2**63 - 1))
        rec = be.run_shot(prog, sd)
        peak_M = max(peak_M, be.last_max_M)
        for cidx, bit in rec.items():
            if 0 <= cidx < nm:
                tot[cidx] += bit
        n += 1
        if time.perf_counter() - t0 >= budget_s:
            break
    return tot / n, n, peak_M, (time.perf_counter() - t0) / n * 1e3


def cl_marg(prog, budget_s, target, chunk=20, seed=1):
    """clifft marginal P(bit=1), time-budgeted in small chunks."""
    nm = prog.num_measurements
    tot = np.zeros(nm, dtype=np.float64); n = 0; s = seed
    t0 = time.perf_counter()
    while n < target:
        m = clifft.sample(prog, chunk, seed=s).measurements
        tot += m.sum(axis=0); n += chunk; s += 1
        if time.perf_counter() - t0 >= budget_s:
            break
    return tot / n, n


def main():
    argv = sys.argv[1:]
    block = "--block" in argv
    lazy = "--lazy" in argv or block
    circs = [a for a in argv if not a.startswith("-")] or DEFAULT
    mode = ("BLOCK (defer + anticommuting-core flush + block-factored magic)" if block
            else "LAZY (defer + anticommuting-core flush)" if lazy else "EAGER")
    print(f"mode: {mode}")
    print(f"{'circuit':16s} {'idents':>6s} {'pk maxblk' if block else 'peak|M|':>9s} "
          f"{'ncN':>6s} {'clN':>6s} {'max|nc-cl|':>10s} {'tol':>7s} {'ms/shot':>8s} "
          f"{'verdict':>8s}", flush=True)
    allok = True
    for circ in circs:
        prog = load(circ, fused=True)
        nid = count_idents(prog)
        m_nc, n_nc, peakM, ms = nc_marg(prog, NC_BUDGET_S, TARGET, lazy=lazy, block=block)
        m_cl, n_cl = cl_marg(prog, CL_BUDGET_S, TARGET)
        dmax = float(np.max(np.abs(m_nc - m_cl)))
        tol = 3.0 * np.sqrt(0.25 / min(n_nc, n_cl)) + 0.01
        ok = dmax <= tol
        allok &= ok
        print(f"{circ:16s} {nid:6d} {peakM:7d} {n_nc:6d} {n_cl:6d} "
              f"{dmax:10.4f} {tol:7.4f} {ms:8.2f} {'PASS' if ok else 'FAIL':>8s}",
              flush=True)
    print("\nALL", "PASS" if allok else "FAIL")
    return allok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
