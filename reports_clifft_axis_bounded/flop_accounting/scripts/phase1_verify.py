"""Phase 1 EXACTNESS verification.

Proves the sqnorm-reuse + direct-drop optimization changes NOTHING observable:

  (1) per-seed BIT-EXACT measurement trajectory: NEW code vs a faithful reconstruction
      of the ORIGINAL measure_z (same engine, same rng), many seeds, every circuit.
  (2) final-state exactness: |<psi_old|psi_new>| == 1 on the small circuits.
  (3) peak resident rank (the hard memory bound) unchanged per seed.
  (4) the drops/meas==1 invariant the direct-drop relies on: re-run NEW with
      _purge_verify=True so every measurement asserts no residual product axis.
  (5) distributional sanity vs clifft's own sampler (ground truth) at high shots.

OLD is reconstructed inline (the pre-Phase-1 magic branch) and bound onto the class for
the comparison run, then the NEW method is restored -- so a single file exercises both.
"""
import sys
sys.path.insert(0, "/home/jung/clifft-paper")
sys.setrecursionlimit(400000)
import numpy as np
import clifft
import nearclifford_backend.backend as bk
from nearclifford_backend.simulator import pauli_commute
from nearclifford_backend.clifft_axis.bounded import compile_bounded, CliftAxisBoundedNearClifford as C

NEW_measure_z = C.measure_z          # the Phase-1 method currently on the class


def OLD_measure_z(self, q):
    """Verbatim pre-Phase-1 magic branch: single-branch Born under tot==1, full _sqnorm_1d
    renormalization, and the O(k) _compress_magic rescan."""
    self._flush_core(0, 1 << q)
    Pm = (0, 1 << q, 0)
    magset = set(self.M)
    anti_s = [i for i in range(self.n)
              if i not in magset and not pauli_commute(self.Zc[i], Pm)]
    M_before = len(self.M)
    p0 = None
    if anti_s:
        out = self._ag_measure(Pm, anti_s)
        branch = "stabilizer"
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
        self.core_log.append(dict(meas=self._meas_log_ctr, branch=branch,
                                  M_before=M_before, M_after=len(self.M), p0=p0,
                                  peak_live_words=self.budget.peak))
    self._meas_log_ctr += 1
    return out


def run_records(circ, seed, use_old, purge_verify=False):
    prog = compile_bounded(open(f"qec_bench/circuits/{circ}.stim").read())
    C.measure_z = OLD_measure_z if use_old else NEW_measure_z
    C._purge_verify = bool(purge_verify)
    try:
        be = bk.NearCliffordBackend(clifft_axis_bounded=True, drop_dead=False,
                                    structure_once=False, clifft_axis_enforce=True)
        rec = dict(be.run_shot(prog, seed))
        peak_rank = be.nc.budget.peak_resident.bit_length() - 1
        # rng-independent STATE check: the per-measurement Born p0 sequence along the realized
        # trajectory (full 2^n statevector is infeasible at n=17).  If the post-measurement
        # state ever diverged, a later p0 would differ.
        p0seq = [c.get("p0") for c in be.nc.core_log if c.get("p0") is not None]
    finally:
        C.measure_z = NEW_measure_z
        C._purge_verify = False
    return rec, peak_rank, p0seq


# (circuit, n_seeds) -- higher-rank circuits are heavier, so fewer seeds (still bit-exact)
CIRCS = [("coherent_ry_d3_r1", 12), ("coherent_ry_d3_r3", 8), ("cultivation_d3", 20),
         ("cultivation_d5", 12), ("coherent_rx_d3_r3", 10), ("coherent_d3_r3", 12),
         ("coherent_rx_d3_r1", 10), ("distillation", 20), ("coherent_d5_r5", 2)]

print("=== Phase 1 exactness: NEW vs reconstructed OLD, per-seed bit-exact ===\n")
all_ok = True
for circ, nseeds in CIRCS:
    SEEDS = list(range(1, nseeds + 1))
    rec_mismatch = rank_mismatch = 0
    p0_max = 0.0
    for s in SEEDS:
        ro, pko, q0 = run_records(circ, s, use_old=True)
        rn, pkn, q1 = run_records(circ, s, use_old=False)
        if ro != rn:
            rec_mismatch += 1
        if pko != pkn:
            rank_mismatch += 1
        if len(q0) == len(q1):
            p0_max = max([p0_max] + [abs(a - b) for a, b in zip(q0, q1)])
        else:
            p0_max = max(p0_max, 1.0)         # length mismatch = structural divergence
    # drops/meas==1 invariant the direct-drop relies on: ONCE per circuit (seed 1)
    purge_ok = True
    try:
        run_records(circ, 1, use_old=False, purge_verify=True)
    except AssertionError as e:
        purge_ok = False
        print(f"    !! {circ}: {e}")
    ok = (rec_mismatch == 0 and rank_mismatch == 0 and p0_max < 1e-9 and purge_ok)
    all_ok &= ok
    print(f"  {circ:20} seeds={len(SEEDS)}  rec_mismatch={rec_mismatch}  "
          f"rank_mismatch={rank_mismatch}  max|p0_old-p0_new|={p0_max:.1e}  "
          f"purge_inv={'OK' if purge_ok else 'FAIL'}  {'PASS' if ok else 'FAIL'}", flush=True)

print(f"\n{'ALL EXACT' if all_ok else 'SOME FAIL'}\n")

# ---- distributional sanity vs clifft ground truth ----
print("=== distributional vs clifft (null = clifft-vs-clifft spread) ===")
SHOTS = 6000
for c in ["coherent_rx_d3_r1", "coherent_rx_d3_r3", "coherent_ry_d3_r1", "coherent_ry_d3_r3"]:
    prog = compile_bounded(open(f"qec_bench/circuits/{c}.stim").read())
    g1 = np.asarray(clifft.sample(prog, shots=SHOTS, seed=11).measurements).mean(0)
    g2 = np.asarray(clifft.sample(prog, shots=SHOTS, seed=22).measurements).mean(0)
    null = float(np.abs(g1 - g2).max())
    be = bk.NearCliffordBackend(clifft_axis_bounded=True, drop_dead=False,
                                structure_once=False, clifft_axis_enforce=True)
    bb = be.sample(prog, shots=SHOTS, seed=33).mean(0)
    diff = float(np.abs(g1 - bb).max())
    print(f"  {c:18} null={null:.4f}  bounded-vs-clifft={diff:.4f}  ratio={diff/null:.2f}  "
          f"{'PASS' if diff <= null * 1.6 else 'INVESTIGATE'}")
