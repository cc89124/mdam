#!/usr/bin/env python
"""Gate L2 -- native authoritative run() bit-exact vs Python bounded run_shot, on coherent + cultivation
benchmarks (whatever translate() now accepts AND peak_rank <= 26 so the Python oracle can run too)."""
import os, sys, ctypes
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import clifft
from mdam.backend.backend import NearCliffordBackend
from verify_mdam_oneshot import translate, make_prog, pcg, load_lib

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
KMAX = 26
ALL = ["coherent_d3_r1", "coherent_d3_r3", "coherent_d5_r1", "coherent_d5_r5", "surface_d7_r7",
       "coherent_rx_d3_r1", "coherent_rx_d3_r3", "coherent_rx_d5_r1", "coherent_rx_d5_r5",
       "coherent_ry_d3_r1", "coherent_ry_d3_r3", "coherent_ry_d5_r1", "coherent_ry_d5_r5",
       "cultivation_d3", "cultivation_d5", "distillation"]


def seeds(n):
    fixed = [1, 7, 42, 123, 999]
    rs = np.random.RandomState(2026)
    return fixed + [int(x) for x in rs.randint(0, 2**31 - 1, size=max(0, n - len(fixed)))]


def verify(bench, lib, nseed):
    text = open(f"{ROOT}/qec_bench/circuits/{bench}.stim").read()
    # Gate L Tier-3 DIRECT: FUSED compile (default passes) — the authoritative localized MDAM algorithm.
    # Native supports the fused frame-keyed U2/U4 directly (decomposition precomputed per in_state), which
    # PRESERVES the measurement-core localization (maxM).  De-fuse (bytecode_passes=None) de-localizes the
    # high-rank R_Z benches (d5_r5: maxM 12->31, 39MB->31GB) so it is NOT the authoritative path.
    prog = clifft.compile(text)
    k = getattr(prog, "peak_rank", 0)
    try:
        t = translate(prog)
    except Exception as e:
        return ("translate-fail", k, str(e).replace("unsupported opcode ", ""))
    if k > KMAX:
        return ("infeasible", k, f"2^{k} > 2^{KMAX} (oracle cannot run)")
    nm = t["num_meas"]
    ph = make_prog(lib, t); vm = lib.nvm_mdam_vm_create(ph)
    npass = 0; first = None
    for sd in seeds(nseed):
        be = NearCliffordBackend(clifft_axis_bounded=True, drop_dead=False, structure_once=False)
        o = be._reset
        be._reset = lambda prog, _o=o, _b=be: (_o(prog), setattr(_b.nc, "_compiled_core", True))[0]
        rec = be.run_shot(prog, sd)
        pyvec = np.zeros(nm, np.uint8)
        for c, b in rec.items():
            if 0 <= c < nm: pyvec[c] = b
        out = np.zeros(nm, np.uint8)
        draws = ctypes.c_ulonglong(); comp = ctypes.c_int(); orac = ctypes.c_int()
        eb = ctypes.create_string_buffer(256)
        shi, slo, ihi, ilo = pcg(sd)
        rc = lib.nvm_mdam_run(ph, vm, shi, slo, ihi, ilo, out.ctypes.data,
                              ctypes.byref(draws), ctypes.byref(comp), ctypes.byref(orac), eb, 256)
        if rc != 0:
            return ("native-error", k, eb.value.decode())
        if np.array_equal(pyvec, out):
            npass += 1
        elif first is None:
            d = np.where(pyvec != out)[0]
            first = f"seed {sd} idx {d[:5].tolist()} py={pyvec[d[0]]} nat={out[d[0]]} (comp={comp.value} orac={orac.value})"
    return ("ok" if npass == len(seeds(nseed)) else "MISMATCH", k,
            f"{npass}/{len(seeds(nseed))}" + (f"  first={first}" if first else ""))


def main():
    nseed = int(sys.argv[2]) if len(sys.argv) > 2 else 25
    targets = [sys.argv[1]] if len(sys.argv) > 1 and sys.argv[1] != "all" else ALL
    lib = load_lib()
    print(f"Gate L2: native authoritative vs Python bounded ({nseed} seeds)")
    print(f"{'circuit':20}{'k':>4}  {'status':14} detail")
    print("-" * 90)
    nok = nfeas = 0
    for b in targets:
        status, k, detail = verify(b, lib, nseed)
        if status in ("ok", "MISMATCH"): nfeas += 1
        if status == "ok": nok += 1
        flag = {"ok": "PASS", "MISMATCH": "**FAIL**", "infeasible": "skip(k>26)",
                "translate-fail": "no-translate", "native-error": "ERR"}[status]
        print(f"{b:20}{k:>4}  {flag:14} {detail}")
    print("-" * 90)
    print(f"bit-exact: {nok}/{nfeas} feasible benchmarks")
    sys.exit(0 if nok == nfeas else 1)


if __name__ == "__main__":
    main()
