#!/usr/bin/env python
"""Gate D §4/§6 — native batch sampling vs authoritative Python backend.sample().

One Python->C++ call (nvm_mdam_sample_batch) runs the whole shot batch in C++:
the master PCG64 (state,inc) is handed in once; native expands per-shot seeds
(master.integers(0,2**63-1) Lemire-64 -> SeedSequence -> PCG64), runs each shot
in place, and writes the [shots, num_measurements] uint8 record buffer directly.
No Python shot loop, no per-shot Python objects."""
import os, sys, ctypes
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import clifft
from mdam.backend.backend import NearCliffordBackend
from verify_mdam_oneshot import translate, load_lib, make_prog, pcg, BENCH, _ROOT

def make_backend(prog):
    be = NearCliffordBackend(clifft_axis_bounded=True, drop_dead=False, structure_once=False)
    o = be._reset
    be._reset = lambda prog, _o=o, _b=be: (_o(prog), setattr(_b.nc, "_compiled_core", True))[0]
    return be

def add_batch_sig(lib):
    P = ctypes.c_void_p
    lib.nvm_mdam_sample_batch.restype = ctypes.c_int
    lib.nvm_mdam_sample_batch.argtypes = [P, P, ctypes.c_uint64] + [ctypes.c_uint64]*4 + [P, P, P, ctypes.c_int]
    return lib

def _batch_master(lib, ph, vm, shots, master_words, num_meas):
    """Single Python->C++ call given the master PCG64 (state,inc) words directly.
    Returns (records[shots,num_meas], stats dict incl. master continuation words)."""
    out = np.zeros((shots, num_meas), dtype=np.uint8)   # preallocated, contiguous
    stats = np.zeros(8, dtype=np.uint64)                # 4 stats + 4 master continuation words
    err = ctypes.create_string_buffer(256)
    mshi, mslo, mihi, milo = master_words
    rc = lib.nvm_mdam_sample_batch(ph, vm, shots, mshi, mslo, mihi, milo,
                                   out.ctypes.data, stats.ctypes.data, err, 256)
    if rc != 0:
        raise RuntimeError(f"native batch error (shot {stats[3]}): {err.value.decode()}")
    cont = (int(stats[4]), int(stats[5]), int(stats[6]), int(stats[7]))
    return out, dict(draws=int(stats[0]), compiled=int(stats[1]), oracle=int(stats[2]), cont=cont)

def native_batch(lib, ph, vm, shots, seed, num_meas):
    """Derive the master from `seed` (== np.random.default_rng(seed)) and run the batch."""
    return _batch_master(lib, ph, vm, shots, pcg(seed), num_meas)

def py_sample(be, prog, shots, seed, num_meas):
    return be.sample(prog, shots, seed=seed, num_measurements=num_meas)

def main():
    prog = clifft.compile(open(os.path.join(_ROOT, f"qec_bench/circuits/{BENCH}.stim")).read())
    t = translate(prog); nm = t["num_meas"]
    lib = add_batch_sig(load_lib()); ph = make_prog(lib, t); vm = lib.nvm_mdam_vm_create(ph)
    be = make_backend(prog)
    # Gate F-B: optionally run the whole suite in a region-compiler mode (env MDAM_FB=1/2/3).
    fb = int(os.environ.get("MDAM_FB", "0"))
    if fb:
        lib.nvm_mdam_vm_set_fb.restype = None
        lib.nvm_mdam_vm_set_fb.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.nvm_mdam_vm_set_fb(vm, fb)
        print(f"[MDAM_FB={fb}: region-compiler mode {'OFF COMPILE SHADOW FAST'.split()[fb]}]")
    f5 = int(os.environ.get("MDAM_F5", "0"))
    if f5:
        lib.nvm_mdam_vm_set_f5.restype = None; lib.nvm_mdam_vm_set_f5.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.nvm_mdam_vm_set_f5(vm, f5); print(f"[MDAM_F5={f5}: commit consume-skip]")

    fails = 0

    # ---- §4.1 one-shot equivalence: batch(1) == authoritative sample(1) ----
    print("== §4.1 one-shot equivalence ==")
    for S in [1, 7, 42, 123, 999]:
        nat, _ = native_batch(lib, ph, vm, 1, S, nm)
        py = py_sample(be, prog, 1, S, nm)
        ok = np.array_equal(nat, py)
        fails += (not ok)
        print(f"  seed {S}: shots=1 {'OK' if ok else 'MISMATCH'}")

    # ---- §4.2 batch equivalence across sizes ----
    print("== §4.2 batch equivalence (sizes) ==")
    for shots in [1, 2, 3, 10, 100, 1000, 10000]:
        S = 42
        nat, st = native_batch(lib, ph, vm, shots, S, nm)
        py = py_sample(be, prog, shots, S, nm)
        ok = np.array_equal(nat, py)
        fails += (not ok)
        extra = ""
        if not ok:
            d = np.where(np.any(nat != py, axis=1))[0]
            extra = f" first diff shot={d[0]} nat={nat[d[0]]} py={py[d[0]]}"
        print(f"  shots={shots:6d} seed={S}: {'OK' if ok else 'MISMATCH'}{extra}  (draws/shot~{st['draws']//max(1,shots)}, compiled={st['compiled']//max(1,shots)}, oracle={st['oracle']//max(1,shots)})")

    # ---- §4.3 repeated-call determinism ----
    print("== §4.3 repeated-call determinism ==")
    for S in [123, 999]:
        a, _ = native_batch(lib, ph, vm, 1000, S, nm)
        b, _ = native_batch(lib, ph, vm, 1000, S, nm)
        ok = np.array_equal(a, b)
        fails += (not ok)
        print(f"  seed {S}: batch(1000)==batch(1000) {'OK' if ok else 'MISMATCH'}")

    # ---- §4.4 batch splitting: one continuous master stream ----
    # batch(2000, S) == batch(1000, S) + batch(1000, continuation_state).  The master is a single
    # PCG64 stream; native returns its continuation (state,inc) so a split resumes the SAME stream.
    print("== §4.4 batch splitting (continuation stream) ==")
    for S in [42, 999]:
        full, _ = native_batch(lib, ph, vm, 2000, S, nm)
        first, st1 = native_batch(lib, ph, vm, 1000, S, nm)
        second, _ = _batch_master(lib, ph, vm, 1000, st1["cont"], nm)
        split = np.concatenate([first, second], axis=0)
        ok = np.array_equal(full, split)
        fails += (not ok)
        print(f"  seed {S}: batch(2000)==batch(1000)+batch(1000,cont) {'OK' if ok else 'MISMATCH'}")

    # ---- §4.5 state-leakage: shots must be independent of batch grouping ----
    # Authoritative sample uses ONE master stream; per-shot seed = master.integers().
    # So batch(N, S) row i == the i-th independent shot regardless of N.  Verify N=100 rows
    # are a prefix-stable function: batch(100,S)[:50] == batch(50,S).
    print("== §4.5 grouping invariance (no cross-shot state leak) ==")
    for S in [7, 2026]:
        big, _ = native_batch(lib, ph, vm, 100, S, nm)
        small, _ = native_batch(lib, ph, vm, 50, S, nm)
        ok = np.array_equal(big[:50], small)
        fails += (not ok)
        print(f"  seed {S}: batch(100)[:50]==batch(50) {'OK' if ok else 'MISMATCH'}")

    # ---- §6 correctness at scale: fixed + random seeds, >=100 shots, + 10000-shot authoritative ----
    print("== §6 correctness at scale ==")
    rs = np.random.RandomState(31)
    seeds = [1, 7, 42, 123, 999] + [int(x) for x in rs.randint(0, 2**31 - 1, size=100)]
    nbad = 0
    for S in seeds:
        nat, _ = native_batch(lib, ph, vm, 100, S, nm)
        py = py_sample(be, prog, 100, S, nm)
        if not np.array_equal(nat, py):
            nbad += 1
            if nbad <= 5:
                d = np.where(np.any(nat != py, axis=1))[0]
                print(f"  MISMATCH seed={S} shot={d[0]}")
    print(f"  105 seeds x 100 shots: {len(seeds)-nbad}/{len(seeds)} batch-exact")
    fails += nbad

    nat, st = native_batch(lib, ph, vm, 10000, 12345, nm)
    py = py_sample(be, prog, 10000, 12345, nm)
    ok = np.array_equal(nat, py)
    fails += (not ok)
    print(f"  10000-shot authoritative (seed 12345): {'OK' if ok else 'MISMATCH'}  draws={st['draws']} compiled={st['compiled']} oracle={st['oracle']}")

    print(f"\n{'ALL BATCH CHECKS PASS' if fails==0 else f'{fails} FAILURES'}")
    return 0 if fails == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
