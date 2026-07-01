#!/usr/bin/env python
"""Python-reference canonical fingerprint dump for the C++/Python inverse-frame divergence hunt.

Compiles a bench FUSED (clifft.compile), runs NearCliffordBackend.run_shot(prog, seed) with a
step_recorder that, at the TOP of each opcode step (state = after steps 0..step-1), pulls back Z_i
through the stabilizer inverse frame for every born qubit i and prints (xprime,zprime) bitmasks.

Output line per step (to stdout):
    FPP <step> name=<opname> a1=<a1> a2=<a2> i:<xp>:<zp> i:<xp>:<zp> ...
where xp/zp are integer bitmasks over qubit ids (phase ignored).  The state is BEFORE the named
prog instruction runs (top-of-step recorder).

Env: MDAM_BENCH (default coherent_d5_r5), MDAM_SEED (default 7), MDAM_FPMAXSTEP (stop dumping past).
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import clifft
from mdam.backend.backend import NearCliffordBackend, _opname

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BENCH = os.environ.get("MDAM_BENCH", "coherent_d5_r5")
SEED = int(os.environ.get("MDAM_SEED", "7"))
MAXSTEP = int(os.environ.get("MDAM_FPMAXSTEP", "1400"))

prog = clifft.compile(open(os.path.join(_ROOT, f"qec_bench/circuits/{BENCH}.stim")).read())

be = NearCliffordBackend(clifft_axis_bounded=True, drop_dead=False, structure_once=False)
o = be._reset
be._reset = lambda prog, _o=o, _b=be: (_o(prog), setattr(_b.nc, "_compiled_core", True))[0]

out = sys.stdout
meta_printed = [False]

def rec(step, b):
    if step > MAXSTEP:
        return
    nc = b.nc
    n = nc.n
    if not meta_printed[0]:
        out.write(f"FPMETA n={n}\n")
        meta_printed[0] = True
    if step < len(prog):
        inst = prog[step]; nm = _opname(inst.opcode); a1 = int(inst.axis_1); a2 = int(inst.axis_2)
    else:
        nm = "END"; a1 = a2 = -1
    parts = [f"FPP {step} name={nm} a1={a1} a2={a2}"]
    for i in range(n):
        xp, zp, _ = nc._pullback(0, 1 << i)
        parts.append(f"{i}:{xp:x}:{zp:x}")
    out.write(" ".join(parts) + "\n")

be.run_shot(prog, SEED, step_recorder=rec)
out.flush()
sys.stderr.write(f"[fp_python_dump] done bench={BENCH} seed={SEED} maxstep={MAXSTEP}\n")
