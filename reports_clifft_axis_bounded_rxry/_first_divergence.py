"""Deterministic FIRST-DIVERGENCE localizer (no sampling).  Run the bounded engine on the
pure-coherent d3_r1, capture the realized record h and each measurement's engine outcome + Born
p0.  Compute the bounded REALIZED conditional P(record bit i | prefix).  Compute clifft's EXACT
realized conditional from record_probabilities (marginalizing free suffix bits).  The first
measurement where they differ is the first physical-state divergence."""
import sys
import numpy as np
sys.path.insert(0, "/home/jung/clifft-paper")
sys.setrecursionlimit(400000)
import stim, itertools
import clifft
import nearclifford_backend.backend as bk
from nearclifford_backend.clifft_axis.bounded import compile_bounded, CliftAxisBoundedNearClifford as B

def make_coherent_only(d, r, ang=0.02):
    c = stim.Circuit.generated("surface_code:rotated_memory_z", rounds=r, distance=d,
                               after_clifford_depolarization=1e-3, after_reset_flip_probability=0.0,
                               before_measure_flip_probability=0.0, before_round_data_depolarization=0.0)
    out = []
    for l in str(c).split("\n"):
        s = l.strip()
        if s.startswith(("DETECTOR", "OBSERVABLE_INCLUDE", "SHIFT_COORDS", "QUBIT_COORDS")):
            continue                                 # strip annotations (record_probabilities)
        if s.startswith(("DEPOLARIZE1(", "DEPOLARIZE2(")):
            out.append(f"R_Y({ang}) {s.split(')')[1].strip()}")
        else:
            out.append(l)
    return "\n".join(out)

prog = compile_bounded(make_coherent_only(3, 1))
NM = prog.num_measurements

# capture per-measurement (engine outcome, Born p0) and the realized record
CAP = []
o_mz = B.measure_z
def mz(self, q):
    out = o_mz(self, q)
    p0 = self.core_log[-1]["p0"] if (self.log_cores and self.core_log) else None
    CAP.append((out, p0))
    return out
B.measure_z = mz
o_reset = bk.NearCliffordBackend._reset
def reset(self, prog):
    o_reset(self, prog)
    if getattr(self, "clifft_axis_bounded", False):
        self.nc.log_cores = True; self.nc.core_log = []
bk.NearCliffordBackend._reset = reset

be = bk.NearCliffordBackend(clifft_axis_bounded=True, drop_dead=False,
                            structure_once=False, clifft_axis_enforce=True)
rec = be.run_shot(prog, 3)
h = np.array([rec.get(i, 0) for i in range(NM)], dtype=np.uint8)
print("realized record h =", list(h))

# bounded realized conditional P(record_i = h_i | prefix): the engine produced outcome `out`
# (engine bit) with prob p0 if out==0 else 1-p0; the record bit is a deterministic fn of `out`,
# so the probability of the realized record bit == probability the engine produced `out`.
rc_b = []
for (out, p0) in CAP:
    if p0 is None:
        rc_b.append(None)
    else:
        rc_b.append(p0 if out == 0 else (1.0 - p0))

# clifft exact realized conditional: M[k] = P(meas_0..k-1 = h[0:k]); cond_i = M[i+1]/M[i].
def M(k):
    if k == 0:
        return 1.0
    free = NM - k
    if free > 13:                     # cap enumeration cost; syndrome region (small k) skipped
        return None
    recs = np.tile(h, (1 << free, 1)).astype(np.uint8)
    for j, bits in enumerate(itertools.product((0, 1), repeat=free)):
        recs[j, k:] = bits
    p = np.asarray(clifft.record_probabilities(prog, recs))
    return float(p.sum())

print(f"\n{'i':>3} {'h_i':>4} {'bounded P(real)':>15} {'clifft P(real)':>15} {'|Δ|':>9}")
prevM = None
for i in range(NM):
    Mi = M(i); Mi1 = M(i + 1)
    cc = (Mi1 / Mi) if (Mi and Mi1 and Mi > 1e-15) else None
    bb = rc_b[i]
    if cc is None or bb is None:
        print(f"{i:>3} {int(h[i]):>4} {str(round(bb,4) if bb else bb):>15} "
              f"{'(skip)':>15} {'-':>9}")
        continue
    d = abs(bb - cc)
    flag = "  <== FIRST DIVERGENCE" if d > 0.01 else ""
    print(f"{i:>3} {int(h[i]):>4} {bb:>15.5f} {cc:>15.5f} {d:>9.5f}{flag}")
