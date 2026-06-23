"""Phase-1 bit-exact NEW-vs-OLD on the remaining (higher-rank) circuits -- the gold-standard
correctness check, and FASTER than _purge_verify (which re-runs the heavy original compress
on top).  Both OLD (reconstructed pre-Phase-1) and NEW run at production speed; per seed we
compare the full measurement record, the peak resident rank (memory bound), and the Born p0
sequence.  rec_mismatch==0 AND rank_mismatch==0 AND p0~0 == the optimization changed nothing.
"""
import sys, signal
sys.path.insert(0, "/home/jung/clifft-paper")
sys.setrecursionlimit(400000)
import numpy as np
import nearclifford_backend.backend as bk
from nearclifford_backend.simulator import pauli_commute
from nearclifford_backend.clifft_axis.bounded import compile_bounded, CliftAxisBoundedNearClifford as C

NEW = C.measure_z


def OLD(self, q):
    self._flush_core(0, 1 << q)
    Pm = (0, 1 << q, 0)
    magset = set(self.M)
    anti_s = [i for i in range(self.n)
              if i not in magset and not pauli_commute(self.Zc[i], Pm)]
    M_before = len(self.M)
    p0 = None
    if anti_s:
        out = self._ag_measure(Pm, anti_s); branch = "stabilizer"
    else:
        xp, zp, pp = self._pullback(0, 1 << q)
        r, sign = self._localize_to_Z(xp, zp, pp, prefer=q)
        if r is None:
            p0 = max(0.0, min(1.0, (1.0 + sign) / 2.0))
            out = 0 if float(self.rng.random()) < p0 else 1
        else:
            jr = self.M.index(r)
            p0r = self._branch_sqnorm(jr, 0)
            p0 = p0r if sign > 0 else (1.0 - p0r)
            out = 0 if float(self.rng.random()) < p0 else 1
            plus_bit = 0 if sign > 0 else 1
            keepbit = plus_bit if out == 0 else (1 - plus_bit)
            v = self.phi.reshape(-1, 2, 1 << jr)
            v[:, 1 - keepbit, :] = 0.0
            nrm2 = self._sqnorm_1d(self.phi)
            if nrm2 > 1e-24:
                self.phi /= nrm2 ** 0.5
            self._compress_magic()
        branch = "magic"
    self._reduce_full()
    if len(self.M) > self.max_M:
        self.max_M = len(self.M)
    self.budget.note_resident(self.phi.size, "post-reduce")
    if self.log_cores:
        self.core_log.append(dict(meas=self._meas_log_ctr, branch=branch, M_before=M_before,
                                  M_after=len(self.M), p0=p0, peak_live_words=self.budget.peak))
    self._meas_log_ctr += 1
    return out


def run(circ, seed, old):
    prog = compile_bounded(open(f"qec_bench/circuits/{circ}.stim").read())
    C.measure_z = OLD if old else NEW
    try:
        be = bk.NearCliffordBackend(clifft_axis_bounded=True, drop_dead=False,
                                    structure_once=False, clifft_axis_enforce=True)
        rec = dict(be.run_shot(prog, seed))
        pk = be.nc.budget.peak_resident.bit_length() - 1
        p0 = [c.get("p0") for c in be.nc.core_log if c.get("p0") is not None]
    finally:
        C.measure_z = NEW
    return rec, pk, p0


class TO(Exception):
    pass


signal.signal(signal.SIGALRM, lambda *a: (_ for _ in ()).throw(TO()))

# (circuit, seeds) -- heaviest get 1 seed
CIRCS = [("coherent_ry_d5_r1", 2), ("coherent_rx_d5_r1", 2), ("coherent_ry_d5_r5", 1),
         ("coherent_rx_d5_r5", 1), ("coherent_d7_r1", 2), ("coherent_d7_r7", 1),
         ("surface_d7_r7", 1)]
print("=== Phase-1 bit-exact NEW-vs-OLD, heavy circuits ===")
for circ, ns in CIRCS:
    rm = km = 0
    p0max = 0.0
    signal.alarm(420)
    try:
        for s in range(1, ns + 1):
            ro, pko, q0 = run(circ, s, True)
            rn, pkn, q1 = run(circ, s, False)
            if ro != rn:
                rm += 1
            if pko != pkn:
                km += 1
            p0max = max([p0max] + [abs(a - b) for a, b in zip(q0, q1)]) if len(q0) == len(q1) else 1.0
        ok = rm == 0 and km == 0 and p0max < 1e-9
        print(f"  {circ:20} seeds={ns}  rec_mismatch={rm}  rank_mismatch={km}  "
              f"max|dp0|={p0max:.1e}  {'PASS' if ok else 'FAIL'}", flush=True)
    except TO:
        print(f"  {circ:20} TIMEOUT (>420s)", flush=True)
    except Exception as e:
        print(f"  {circ:20} ERR {type(e).__name__}: {e}", flush=True)
    finally:
        signal.alarm(0)
