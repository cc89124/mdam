#!/usr/bin/env python
"""Native-only driver to emit the C++ canonical fingerprint dump (FPN lines on stderr).

Builds the bench FUSED (clifft.compile), translates + makes the native prog, and calls
nvm_mdam_run for ONE seed with MDAM_FPDUMP=1 (the native VM dumps pullback(Z_q) per step to
stderr) and MDAM_FPSTOP=<step> so the run stops early (before the de-localization OOM).

The native dump (FPN <i>) is the state AFTER executing step i; it aligns to Python's FPP <i+1>.

Env: MDAM_BENCH (default coherent_d5_r5), MDAM_SEED (default 7), plus MDAM_FPDUMP / MDAM_FPSTOP
which are read by the native .so (set them in the shell before running this).
"""
import os, sys, ctypes
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import clifft
import verify_mdam_oneshot as V

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BENCH = os.environ.get("MDAM_BENCH", "coherent_d5_r5")
SEED = int(os.environ.get("MDAM_SEED", "7"))

prog = clifft.compile(open(os.path.join(_ROOT, f"qec_bench/circuits/{BENCH}.stim")).read())
t = V.translate(prog)
nm = t["num_meas"]
lib = V.load_lib()
ph = V.make_prog(lib, t)
vm = lib.nvm_mdam_vm_create(ph)

out = np.zeros(nm, np.uint8)
draws = ctypes.c_ulonglong(); comp = ctypes.c_int(); orac = ctypes.c_int()
errbuf = ctypes.create_string_buffer(256)
shi, slo, ihi, ilo = V.pcg(SEED)
rc = lib.nvm_mdam_run(ph, vm, shi, slo, ihi, ilo, out.ctypes.data,
                      ctypes.byref(draws), ctypes.byref(comp), ctypes.byref(orac), errbuf, 256)
sys.stderr.write(f"[fp_native_run] rc={rc} err={errbuf.value.decode()!r} "
                 f"bench={BENCH} seed={SEED} draws={draws.value}\n")
