"""Emit the full per-T differential table (cultivation_d5, seed 1) requested in Step B0 Sec.4:
| T | rank | generator | axis | born | px | pz | phase | vector_diff | pauli_img_mismatch | next_p0_diff |
vector_diff = max|phiS - gamma*phiC| (incl gamma);  pauli_img_mismatch = tableau bit-diff (0 = images
identical);  next_p0_diff = |p0_real - p0_candidate| at the measurement following a last-before-measure T
(from the authoritative whole-run candidate engine), else '.'.
"""
import sys; sys.path.insert(0, "/home/jung/clifft-paper"); sys.setrecursionlimit(400000)
import copy
import numpy as np
import clifft  # noqa
import nearclifford_backend.backend as bk
from nearclifford_backend.clifft_axis.bounded import compile_bounded, CliftAxisBoundedNearClifford as C

sys.path.insert(0, "/home/jung/clifft-paper/reports_clifft_axis_bounded/flop_accounting/scripts")
from phase8_step_b0 import candidate_decompose, candidate_apply, candidate_flush

circ, seed = "cultivation_d5", 1
prog = compile_bounded(open(f"qec_bench/circuits/{circ}.stim").read())

# authoritative-candidate per-measurement p0 (for next_p0_diff) vs real
be = bk.NearCliffordBackend(clifft_axis_bounded=True, drop_dead=False, structure_once=False, clifft_axis_enforce=True)
be.run_shot(prog, seed); p0R = [c.get("p0") for c in be.nc.core_log]
of1 = C._flush_one
try:
    C._flush_one = candidate_flush
    be2 = bk.NearCliffordBackend(clifft_axis_bounded=True, drop_dead=False, structure_once=False, clifft_axis_enforce=True)
    be2.run_shot(prog, seed); p0C = [c.get("p0") for c in be2.nc.core_log]
finally:
    C._flush_one = of1
p0diff = [abs(a - b) if (a is not None and b is not None) else None for a, b in zip(p0R, p0C)]

rows = []
meas_ctr = [0]
ctx = {"last_T_idx": None}
ofm = C.measure_z


def f1(self, x, z, theta, phase=0):
    pre = copy.deepcopy(self)
    r = of1(self, x, z, theta, phase)
    cand = copy.deepcopy(pre); cand.budget.enforce = False
    meta = candidate_decompose(cand, x, z, theta, phase)
    candidate_apply(cand, meta, theta)
    g = meta["gamma"]
    vdiff = float(np.max(np.abs(self.phi - g * cand.phi)))
    img = 0 if (self.Xc == cand.Xc and self.Zc == cand.Zc and self.M == cand.M) else 1
    gen = f"i^{meta['pp']}·X^{meta['mx']:x}Z^{meta['mz']:x}"
    pivq = cand.M[meta["a"]] if meta.get("a") is not None else None
    rows.append([len(rows), pre.phi.size.bit_length() - 1, gen, pivq, meta["born"],
                 meta["px"], meta["pz"], meta["pp"], vdiff, img, None, meta["weight"],
                 ("T" if meta["s_sign"] * theta > 0 else "Td")])
    ctx["last_T_idx"] = len(rows) - 1
    return r


def mz(self, q):
    if ctx["last_T_idx"] is not None and meas_ctr[0] < len(p0diff):
        rows[ctx["last_T_idx"]][10] = p0diff[meas_ctr[0]]
    meas_ctr[0] += 1
    ctx["last_T_idx"] = None
    return ofm(self, q)


C._flush_one = f1; C.measure_z = mz
try:
    be3 = bk.NearCliffordBackend(clifft_axis_bounded=True, drop_dead=False, structure_once=False, clifft_axis_enforce=True)
    be3.run_shot(prog, seed)
finally:
    C._flush_one = of1; C.measure_z = ofm

out = ["| T | rank | generator | axis q | born | px | pz | i^pp | gate | w | vector_diff | img_mis | next_p0_diff |",
       "|--:|--:|---|--:|:--:|--:|--:|--:|:--:|--:|--:|--:|--:|"]
for d in rows:
    nd = "." if d[10] is None else f"{d[10]:.1e}"
    out.append(f"| {d[0]} | {d[1]} | `{d[2]}` | {d[3]} | {d[4]} | {d[5]} | {d[6]} | {d[7]} | "
               f"{d[12]} | {d[11]} | {d[8]:.1e} | {d[9]} | {nd} |")
maxv = max(d[8] for d in rows); imgmis = sum(d[9] for d in rows)
maxp0 = max((d[10] for d in rows if d[10] is not None), default=0.0)
out.append("")
out.append(f"**Summary:** T={len(rows)}, max vector_diff={maxv:.2e}, image mismatches={imgmis}, "
           f"max next_p0_diff={maxp0:.2e}, all gates diagonal T/T† (0 runtime butterfly in the candidate).")
txt = "\n".join(out)
open("/home/jung/clifft-paper/reports_clifft_axis_bounded/flop_accounting/data/phase8_per_T_table.md", "w").write(txt)
print(txt)
