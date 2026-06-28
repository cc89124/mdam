"""Empirically resolve the measurement structure of compiled coherent_ry_d3_r1:
how many measure_z (active) vs frame-only (dormant) fire at runtime, in what order,
which cidx each writes, and the per-active Born p0. Decides the exact-oracle pairing."""
import sys
sys.path.insert(0, "/home/jung/clifft-paper")
sys.setrecursionlimit(400000)
import nearclifford_backend.backend as bk
import ttn_backend.frame_layer as fl
from nearclifford_backend.clifft_axis.bounded import compile_bounded, CliftAxisBoundedNearClifford as B

EV = []  # ordered event log

o_mz = B.measure_z
def mz(self, q):
    b = o_mz(self, q)
    p0 = self.core_log[-1]["p0"] if (getattr(self, "log_cores", False) and self.core_log) else None
    EV.append(("mz", q, int(b), p0))
    return b
B.measure_z = mz

o_sx = fl.PauliFrame.set_xz
def sx(self, s, x, z=0):
    EV.append(("sx", int(s), int(x) & 1))
    return o_sx(self, s, x, z)
fl.PauliFrame.set_xz = sx

o_reset = bk.NearCliffordBackend._reset
def reset(self, prog):
    o_reset(self, prog)
    if getattr(self, "clifft_axis_bounded", False):
        self.nc.log_cores = True
        self.nc.core_log = []
bk.NearCliffordBackend._reset = reset

prog = compile_bounded(open("qec_bench/circuits/coherent_ry_d3_r1.stim").read())
print("num_measurements:", prog.num_measurements)
be = bk.NearCliffordBackend(clifft_axis_bounded=True, drop_dead=False,
                            structure_once=False, clifft_axis_enforce=True)
rec = be.run_shot(prog, 3)
print("record entries:", len(rec), "->", {k: rec[k] for k in sorted(rec)})
nmz = sum(1 for e in EV if e[0] == "mz")
nsx = sum(1 for e in EV if e[0] == "sx")
print(f"measure_z calls = {nmz}   set_xz calls = {nsx}")
print("\nevent stream (first 60):")
for e in EV[:60]:
    print("  ", e)
