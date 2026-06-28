"""State-size of the Clifft-axis engine across the QEC benchmark suite (coherent_d7
excluded). Reports, per circuit, the near-Clifford dense magic-register size -- the only
exponential object -- as peak |M| (the transient core-flush high-water = the honest
memory-feasibility figure) and its words/bytes, beside clifft's own active rank
2^peak_rank for comparison."""
from __future__ import annotations

import sys
import time
import tracemalloc

sys.path.insert(0, "/home/jung/clifft-paper")
import clifft

from nearclifford_backend.backend import NearCliffordBackend

CIRCS = ["coherent_d3_r1", "coherent_d3_r3", "coherent_d5_r1", "coherent_d5_r5",
         "cultivation_d3", "cultivation_d5", "distillation", "surface_d7_r7"]
import os
SEEDS = [int(s) for s in os.environ.get("BENCH_SEEDS", "7").split(",")]


def measure(circ):
    src = open(f"/home/jung/clifft-paper/qec_bench/circuits/{circ}.stim").read()
    prog = clifft.compile(src)
    k = int(prog.peak_rank)
    peakM = 0
    res_w = 0
    live_w = 0
    tm_w = 0
    dts = []
    for sd in SEEDS:
        be = NearCliffordBackend(clifft_axis=True, structure_once=False,
                                 drop_dead=False, clifft_axis_enforce=False)
        tracemalloc.start()
        t0 = time.perf_counter()
        be.run_shot(prog, sd)
        dt = time.perf_counter() - t0
        _, pk = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        dts.append(dt)
        b = be.nc.budget.summary()
        peakM = max(peakM, be.last_max_M)
        res_w = max(res_w, b["peak_resident_words"])
        live_w = max(live_w, b["peak_live_words"])
        tm_w = max(tm_w, pk // 16)
    return dict(circ=circ, k=k, cap_w=(1 << k), peakM=peakM, res_w=res_w,
                res_bytes=16 * res_w, live_w=live_w, tm_w=tm_w,
                ms=sum(dts) / len(dts) * 1e3)


def main():
    print(f"Clifft-axis engine -- magic-register STATE SIZE (max over seeds {SEEDS}), "
          f"coherent_d7 excluded\n", flush=True)
    hdr = (f"{'circuit':<16} {'clifft k':>8} {'2^k(words)':>11} | "
           f"{'NC peak|M|':>10} {'2^|M|(words)':>12} {'bytes':>9} | "
           f"{'vs clifft':>9} {'tracemalloc':>11} {'ms/shot':>8}")
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    for circ in CIRCS:
        try:
            r = measure(circ)
        except Exception as e:
            print(f"{circ:<16}  ERROR: {repr(e)[:70]}", flush=True)
            continue
        ratio = r["res_w"] / r["cap_w"]
        nb = r["res_bytes"]
        bs = f"{nb}B" if nb < 1024 else (f"{nb/1024:.1f}KB" if nb < 1024**2 else f"{nb/1024**2:.1f}MB")
        print(f"{r['circ']:<16} {r['k']:>8} {r['cap_w']:>11} | "
              f"{r['peakM']:>10} {1<<r['peakM']:>12} {bs:>9} | "
              f"{ratio:>8.2f}x {r['tm_w']:>11} {r['ms']:>7.0f}", flush=True)
    print("\nNC peak|M| = transient core-flush high-water (honest memory-feasibility "
          "figure); resident after reduction settles <= clifft k. 2^|M| words x16B = "
          "complex128 register bytes.", flush=True)


if __name__ == "__main__":
    main()
