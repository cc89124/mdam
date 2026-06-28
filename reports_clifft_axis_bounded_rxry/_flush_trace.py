"""Instrument ONE shot of the coherent-only d3_r1 to see what the bounded engine does with the
pending R_Y rotations at each measurement -- especially the first diverging one (final data
measurements, meas index 8..16).  Logs per measurement: |pending| before, the core flushed
(generators x/z/phase/theta), |M|, and the Born p0.  Reveals whether the coherent R_Y content
reaches the measured data qubit or is dropped/commuted-through."""
import sys
import numpy as np
sys.path.insert(0, "/home/jung/clifft-paper")
sys.setrecursionlimit(400000)
import stim
import clifft
import nearclifford_backend.backend as bk
from nearclifford_backend.clifft_axis import engine as em
from nearclifford_backend.clifft_axis.bounded import compile_bounded, CliftAxisBoundedNearClifford as B

def make_coherent_only(d, r, ang=0.02):
    c = stim.Circuit.generated("surface_code:rotated_memory_z", rounds=r, distance=d,
                               after_clifford_depolarization=1e-3, after_reset_flip_probability=0.0,
                               before_measure_flip_probability=0.0, before_round_data_depolarization=0.0)
    out = []
    for line in str(c).split("\n"):
        s = line.strip()
        if s.startswith("DEPOLARIZE1(") or s.startswith("DEPOLARIZE2("):
            out.append(f"R_Y({ang}) {s.split(')')[1].strip()}")
        else:
            out.append(line)
    return "\n".join(out)

prog = compile_bounded(make_coherent_only(3, 1))

LOG = []
o_do = em.CliftAxisNearClifford._do_flush
def do_flush(self, qx, qz, flush):
    LOG.append(("flush", len(self.pending), [(r[0], r[1], r[2], round(r[3], 4)) for r in flush]))
    return o_do(self, qx, qz, flush)
em.CliftAxisNearClifford._do_flush = do_flush

o_mz = B.measure_z
mctr = {"i": 0}
def mz(self, q):
    npend = len(self.pending); nM = len(self.M)
    out = o_mz(self, q)
    p0 = self.core_log[-1]["p0"] if (self.log_cores and self.core_log) else None
    LOG.append(("meas", mctr["i"], dict(q=q, pend_before=npend, M_before=nM,
                                        M_after=len(self.M), p0=p0)))
    mctr["i"] += 1
    return out
B.measure_z = mz

be = bk.NearCliffordBackend(clifft_axis_bounded=True, drop_dead=False,
                            structure_once=False, clifft_axis_enforce=True)
be.nc_log_cores = True
# enable log_cores on the engine instance created during run_shot:
o_reset = bk.NearCliffordBackend._reset
def reset(self, prog):
    o_reset(self, prog)
    if getattr(self, "clifft_axis_bounded", False):
        self.nc.log_cores = True; self.nc.core_log = []
bk.NearCliffordBackend._reset = reset

rec = be.run_shot(prog, 3)
print("measurement record:", [rec.get(i, 0) for i in range(prog.num_measurements)])
print("\n--- per-measurement trace (focus indices 8..16 = data) ---")
mi = -1
for entry in LOG:
    if entry[0] == "meas":
        mi = entry[1]
        d = entry[2]
        print(f"MEAS {mi:2d}: q={d['q']:2d}  pending_before={d['pend_before']:2d}  "
              f"M {d['M_before']}->{d['M_after']}  p0={d['p0']}")
    else:
        npend, gens = entry[1], entry[2]
        # show generators as signed-pauli-ish
        def lbl(g):
            x, z, p = g[0], g[1], g[2]
            return f"(x={x:#x},z={z:#x},φ={p},θ={g[3]})"
        print(f"   flush: {len(gens)} gens from {npend} pending: "
              f"{[lbl(g) for g in gens][:4]}{'...' if len(gens)>4 else ''}")
