#!/usr/bin/env python
"""Wall-time @ 1M shots: Clifft (all benchmarks) vs MDAM native-VM (cultivation only).

The MDAM native VM (Gate K FAST, cmode5) only supports the cultivation family (translate handles its discrete
T/EXPAND_T opcodes; the coherent_* circuits use rotation opcodes the VM does not implement).  So MDAM-native
wall is reported only where the FAST path is bit-exact-verified here; everything else is Clifft-only.

Clifft shots are adaptive: measure a small batch, run the full 1M if it would finish in a reasonable time,
else time a feasible count and report ns/shot (total@1M extrapolated, flagged).  taskset + single-thread.
"""
import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ.setdefault(v,"1")
import sys, ctypes, time, statistics, csv
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "native_vm")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from verify_mdam_oneshot import translate, make_prog, pcg
from gate_k_fast import bind
import clifft

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SHOTS = 1_000_000
PROBE_BUDGET_S = 8.0          # Clifft probe: size the probe to ~this many seconds (Clifft is shot-independent)
CLIFFT_PROBE_MAX = 100_000    # never probe Clifft for more than this (no reason to — per-shot is constant)
BENCHES = ["coherent_d3_r1","coherent_d3_r3","coherent_d5_r1","coherent_d5_r5","surface_d7_r7",
           "coherent_rx_d3_r1","coherent_rx_d3_r3","coherent_rx_d5_r1","coherent_rx_d5_r5",
           "coherent_ry_d3_r1","coherent_ry_d3_r3","coherent_ry_d5_r1","coherent_ry_d5_r5",
           "cultivation_d3","cultivation_d5","distillation"]
NATIVE_TRY = {"cultivation_d3", "cultivation_d5"}   # FAST path candidates (others: translate-unsupported)

def clifft_progD(text):
    return clifft.compile(text, hir_passes=clifft.default_hir_pass_manager(),
                          bytecode_passes=clifft.default_bytecode_pass_manager())

def time_clifft(progD):
    """Return (ns_per_shot, shots_probed).

    Clifft has NO cross-shot reuse: its per-shot time is shot-count-independent (verified flat across
    1k..1M in the shot-sweep audit), so there is NO reason to run the full 1M.  We PROBE a small count
    and report ns/shot; the exact 1M total = ns/shot * 1M.  (Only MDAM-native must run a real 1M, because
    its Gate-K VM amortizes repeated transitions across shots -> per-shot cost drops with shot count.)"""
    clifft.sample(progD, 5)                                     # micro warmup (cheap even at k=24)
    t0 = time.perf_counter(); clifft.sample(progD, 20); est = (time.perf_counter()-t0)/20
    n = max(20, min(CLIFFT_PROBE_MAX, int(PROBE_BUDGET_S/est))) # probe sized to the time budget, capped
    clifft.sample(progD, min(n, 2000))                          # warmup proportional to probe
    t0 = time.perf_counter(); clifft.sample(progD, n); dt = time.perf_counter()-t0
    return dt/n*1e9, n

def setup_native(lib, t, seed=12345):
    ph = make_prog(lib, t); eb = ctypes.create_string_buffer(256)
    info = (ctypes.c_int*5)(); cp = lib.nvm_jcompile(ph); lib.nvm_jcompile_info(cp, info)
    vm = lib.nvm_mdam_vm_create(ph); jp = lib.nvm_jphase_compile(ph, vm, *pcg(seed))
    lib.nvm_mdam_vm_set_imem(vm, 2); lib.nvm_mdam_vm_set_fb(vm, 1)
    sb = np.zeros((1, t["num_meas"]), np.uint8)
    lib.nvm_mdam_sample_batch(ph, vm, 1, *pcg(seed), sb.ctypes.data, None, eb, 256)
    return ph, cp, vm, jp, eb

def native_bitexact(lib, ph, cp, vm, jp, nm, ws, eb, n=2000):
    """cmode5 FAST == authoritative on n shots?"""
    a = np.zeros((n, nm), np.uint8); k = np.zeros((n, nm), np.uint8)
    va = lib.nvm_mdam_vm_create(ph)
    lib.nvm_mdam_sample_batch(ph, va, n, *pcg(4242), a.ctypes.data, None, eb, 256)
    lib.nvm_jfast5_batch(ph, cp, jp, vm, n, *pcg(4242), k.ctypes.data, 0, ws)
    return int(np.count_nonzero(np.any(a != k, axis=1))) == 0

def time_native(lib, ph, cp, vm, jp, nm, ws):
    wb = np.zeros(nm, np.uint8)
    for ms in (12345, 1, 777, 2026, 99991):                    # warm caches like gate_k harness
        lib.nvm_jfast2f_batch(ph, cp, jp, vm, 8000, *pcg(ms), wb.ctypes.data, 1, ws)
        lib.nvm_jfast2g_batch(ph, cp, jp, vm, 8000, *pcg(ms), wb.ctypes.data, 1, ws)
        lib.nvm_jfast5_batch(ph, cp, jp, vm, 8000, *pcg(ms), wb.ctypes.data, 1, ws)
    def one():
        t0 = time.perf_counter()
        lib.nvm_jfast5_batch(ph, cp, jp, vm, SHOTS, *pcg(777), wb.ctypes.data, 1, ws)
        return (time.perf_counter()-t0)/SHOTS*1e9
    one()                                                       # warmup rep
    return statistics.median([one() for _ in range(3)])

def main():
    lib = bind()
    lib.nvm_j2e_noise_skip.argtypes = [ctypes.c_int]; lib.nvm_j2e_noise_skip(1)
    lib.nvm_j2e_time.argtypes = [ctypes.c_int]; lib.nvm_j2e_time(0)
    ws = (ctypes.c_long*20)()
    rows = []
    print(f"wall @ {SHOTS:,} shots (ns/shot).  Clifft=probed+extrapolated (shot-independent);")
    print(f"native=MDAM Gate-K FAST (cmode5), MEASURED at real 1M (cross-shot cache amortization).")
    print(f"{'circuit':18}{'n':>4}{'k':>4}{'clifft ns/sh':>13}{'native ns/sh':>13}{'native/clifft':>14}{'cl_probe':>9}")
    for b in BENCHES:
        text = open(f"{ROOT}/qec_bench/circuits/{b}.stim").read()
        prog = clifft.compile(text); k = getattr(prog, "peak_rank", 0); n = prog.num_qubits
        if k > 26:                       # off-axis d5: 2^k > 2^26 -> Clifft can't sample either (OOM/hang)
            print(f"{b:18}{n:>4}{str(k):>4}{'CANNOT RUN (2^'+str(k)+')':>27}", flush=True)
            rows.append([b, n, k, "", "", "", "", "", "", "cannot run (2^%d)" % k]); continue
        progD = clifft_progD(text)
        cl_ns, cl_n = time_clifft(progD)
        nat_ns = None; note = ""
        if b in NATIVE_TRY:
            try:
                t = translate(prog); nm = t["num_meas"]
                ph, cp, vm, jp, eb = setup_native(lib, t)
                if native_bitexact(lib, ph, cp, vm, jp, nm, ws, eb):
                    nat_ns = time_native(lib, ph, cp, vm, jp, nm, ws)
                else:
                    note = "FAST not bit-exact"
            except Exception as e:
                note = f"native err: {type(e).__name__}"
        else:
            note = "translate-unsupported"
        ratio = (nat_ns/cl_ns) if nat_ns else None
        ns_str = f"{nat_ns:.0f}" if nat_ns else f"N/A ({note})"
        r_str = f"{ratio:.2f}x" if ratio else "-"
        print(f"{b:18}{n:>4}{str(k):>4}{cl_ns:>13.0f}{ns_str:>13}{r_str:>14}{cl_n:>9}")
        rows.append([b, n, k, round(cl_ns,1), (round(nat_ns,1) if nat_ns else ""),
                     (round(ratio,3) if ratio else ""), f"extrap(probe={cl_n})",
                     round(cl_ns*SHOTS/1e9,3), (round(nat_ns*SHOTS/1e9,3) if nat_ns else ""), note])
    out = f"{ROOT}/results/benchmark_comparison/wall_table.csv"
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["circuit","n_qubits","peak_rank","clifft_ns_per_shot","native_ns_per_shot",
                    "native_over_clifft","clifft_measured_at_1M","clifft_total_1M_s","native_total_1M_s","note"])
        w.writerows(rows)
    print(f"\n-> {out}")

if __name__ == "__main__":
    main()
