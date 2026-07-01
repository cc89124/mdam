#!/usr/bin/env python
"""Gate L3 diagnostic: compare the native vs Python RNG draw sequence (kind 0=double/1=bounded + value)
for a benchmark+seed, and print the FIRST divergence -- that draw is the desync point and tells us which
measurement (and whether kind=classification or value=Born) caused it."""
import os, sys, ctypes
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import clifft
import mdam.backend.backend as bk
from mdam.backend.backend import NearCliffordBackend
from verify_mdam_oneshot import translate, make_prog, pcg, load_lib

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BENCH = sys.argv[1] if len(sys.argv) > 1 else "coherent_rx_d3_r3"
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 42


class RngLog:
    def __init__(self, real): self.real = real; self.log = []
    def random(self, *a, **k):
        v = self.real.random(*a, **k); self.log.append((0, float(v))); return v
    def integers(self, *a, **k):
        v = self.real.integers(*a, **k)
        self.log.append((1, float(np.asarray(v).reshape(-1)[0]) if np.ndim(v) else float(v))); return v
    def __getattr__(self, name): return getattr(self.real, name)


def native_draws(lib, ph, vm, seed, nm):
    lib.nvm_mdam_run_logged.restype = ctypes.c_int
    lib.nvm_mdam_run_logged.argtypes = [ctypes.c_void_p, ctypes.c_void_p] + [ctypes.c_uint64]*4 + \
        [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    MAXN = 200000
    out = np.zeros(nm, np.uint8); kinds = np.zeros(MAXN, np.int32); vals = np.zeros(MAXN, np.float64)
    shi, slo, ihi, ilo = pcg(seed)
    n = lib.nvm_mdam_run_logged(ph, vm, shi, slo, ihi, ilo, out.ctypes.data,
                                kinds.ctypes.data, vals.ctypes.data, MAXN)
    return out, [(int(kinds[i]), float(vals[i])) for i in range(n)]


def main():
    prog = clifft.compile(open(f"{ROOT}/qec_bench/circuits/{BENCH}.stim").read())
    t = translate(prog); nm = t["num_meas"]
    lib = load_lib(); ph = make_prog(lib, t); vm = lib.nvm_mdam_vm_create(ph)
    nout, ndraws = native_draws(lib, ph, vm, SEED, nm)

    # Python with rng draw logging (patch default_rng for this run only)
    orig = bk.np.random.default_rng
    bk.np.random.default_rng = lambda seed: RngLog(orig(seed))
    be = NearCliffordBackend(clifft_axis_bounded=True, drop_dead=False, structure_once=False)
    o = be._reset
    be._reset = lambda prog, _o=o, _b=be: (_o(prog), setattr(_b.nc, "_compiled_core", True))[0]
    rec = be.run_shot(prog, SEED)
    bk.np.random.default_rng = orig
    pdraws = be.nc.rng.log              # the rng that actually drove the shot (== self.nc.rng)
    pyvec = np.zeros(nm, np.uint8)
    for c, b in rec.items():
        if 0 <= c < nm: pyvec[c] = b

    print(f"=== {BENCH} seed {SEED} ===")
    print(f"records match: {np.array_equal(pyvec, nout)}  (py sum={pyvec.sum()} nat sum={nout.sum()})")
    print(f"draws: native={len(ndraws)}  python={len(pdraws)}")
    KIND = {0: "double", 1: "bndint"}
    first = None
    for i in range(min(len(ndraws), len(pdraws))):
        nk, nv = ndraws[i]; pk, pv = pdraws[i]
        if nk != pk or abs(nv - pv) > 1e-9:
            first = i; break
    if first is None and len(ndraws) != len(pdraws):
        first = min(len(ndraws), len(pdraws))
    if first is None:
        print("ALL DRAWS MATCH (desync is NOT in the rng stream -> value/record bug downstream)")
    else:
        lo = max(0, first - 3)
        print(f"FIRST DRAW DIVERGENCE at index {first}:")
        print(f"  {'idx':>5} {'native':>22} {'python':>22}")
        for i in range(lo, min(first + 4, max(len(ndraws), len(pdraws)))):
            ns = f"{KIND.get(ndraws[i][0],'?')}={ndraws[i][1]:.6f}" if i < len(ndraws) else "--"
            ps = f"{KIND.get(pdraws[i][0],'?')}={pdraws[i][1]:.6f}" if i < len(pdraws) else "--"
            mark = " <-- DIVERGE" if i == first else ""
            print(f"  {i:>5} {ns:>22} {ps:>22}{mark}")


if __name__ == "__main__":
    main()
