"""Phase 4 (ANALYSIS ONLY) -- cultivation_d5 T-generator sequence + H-amortization analysis.

Question: must each of the 91 T's pay its own localization H (F = q(4+3)2^r), or can a shared basis
amortize H across a run of rotations (F = (4B + 3q)2^r with B << q the # of basis transitions)?

Traces, for every pending-rotation flush (the materialized T's), the pulled-back PHYSICAL Pauli
generator P_i = U_C^dag L_i U_C, its support / symplectic (x,z) / active rank / whether it is
pure-Z in the current basis / commutation with the previous generator / the flush-batch it belongs
to (a batch = the rotations flushed together right before one measurement -- same frame, same |phi>).

Then computes:
  * per-batch: do all generators pairwise commute (=> one Clifford V diagonalizes the whole batch)?
  * B_batched = # batches that contain an off-diagonal generator (=> need >=1 H)  -- the achievable
    once-per-batch amortization.
  * B_runs = ordered maximal pairwise-commuting run partition of the whole 91-sequence (a LOWER
    bound on basis transitions if measurements did NOT force breaks).
  * projected FLOP for policy A (per-rotation), B (per-batch), C (commuting-run lower bound),
    vs clifft's diagonal-T baseline (3 q 2^r).
NO kernel/state is modified; _flush_one is wrapped read-only (it recomputes the pullback itself).
"""
import sys
sys.path.insert(0, "/home/jung/clifft-paper")
sys.setrecursionlimit(400000)
import clifft  # noqa
import nearclifford_backend.backend as bk
from nearclifford_backend.clifft_axis.bounded import compile_bounded, CliftAxisBoundedNearClifford as C

CIRC = "coherent_d5_r5" if len(sys.argv) > 1 and sys.argv[1] == "d5" else "cultivation_d5"
SEED = 1


def symp(p, q):
    """symplectic inner product of physical Paulis p=(x,z), q=(x,z); commute iff ==0 (mod 2)."""
    x1, z1 = p; x2, z2 = q
    return ((x1 & z2).bit_count() + (x2 & z1).bit_count()) & 1


def gf2_rank(vecs):
    """GF(2) rank of a list of bitmask integers (= min # Hadamards to simultaneously diagonalize a
    commuting Pauli set is <= rank of its X-block, after free CNOT/CZ/S reduction)."""
    basis = []
    for v in vecs:
        for b in basis:
            v = min(v, v ^ b)
        if v:
            basis.append(v); basis.sort(reverse=True)
    return len(basis)


events = []          # list of dicts per flushed rotation
batch = [0]
last_meas_kind = ["start"]

prog = compile_bounded(open(f"qec_bench/circuits/{CIRC}.stim").read())

orig_flush_one = C._flush_one
orig_flush_core = C._flush_core
orig_measure = C.measure_z
orig_ag = C._ag_measure


def flush_one(self, x, z, theta, phase=0):
    xp, zp, pp = self._pullback(x, z)
    # restrict to magic support over physical qubits (the dense register)
    supp = [qq for qq in range(self.n) if ((xp >> qq) & 1) or ((zp >> qq) & 1)]
    xw = int(xp).bit_count(); zw = int(zp).bit_count()
    events.append(dict(idx=len(events), batch=batch[0], xp=int(xp), zp=int(zp),
                       supp=tuple(supp), xw=xw, zw=zw, rank=len(self.M),
                       pureZ=(xp == 0), frame_ver=self._frame_ver,
                       since=last_meas_kind[0]))
    return orig_flush_one(self, x, z, theta, phase)


def flush_core(self, qx, qz):
    r = orig_flush_core(self, qx, qz)
    batch[0] += 1                       # a flush-batch boundary (one per measurement)
    return r


def measure_z(self, q):
    out = orig_measure(self, q)
    last_meas_kind[0] = "magic-meas"
    return out


def ag_measure(self, Pm, anti):
    last_meas_kind[0] = "AG-stab-meas"
    return orig_ag(self, Pm, anti)


C._flush_one = flush_one
C._flush_core = flush_core
C.measure_z = measure_z
C._ag_measure = ag_measure
try:
    be = bk.NearCliffordBackend(clifft_axis_bounded=True, drop_dead=False,
                                structure_once=False, clifft_axis_enforce=True)
    be.run_shot(prog, SEED)
finally:
    C._flush_one = orig_flush_one
    C._flush_core = orig_flush_core
    C.measure_z = orig_measure
    C._ag_measure = orig_ag

q = len(events)
print(f"=== {CIRC}: {q} pending-rotation flushes (T materializations) ===\n")

# ---- per-generator table ----
print(f"{'i':>3}{'batch':>6}{'rank':>5}{'Xw':>4}{'Zw':>4}{'pureZ':>6}{'supp':>14}"
      f"{'cmuPrev':>8}{'since':>14}")
prev = None
for e in events:
    cmu = "" if prev is None else ("commute" if symp((prev['xp'], prev['zp']),
                                                      (e['xp'], e['zp'])) == 0 else "ANTI")
    print(f"{e['idx']:>3}{e['batch']:>6}{e['rank']:>5}{e['xw']:>4}{e['zw']:>4}"
          f"{str(e['pureZ']):>6}{str(e['supp'])[:14]:>14}{cmu:>8}{e['since']:>14}")
    prev = e

# ---- per-batch commuting analysis ----
from collections import defaultdict
byb = defaultdict(list)
for e in events:
    byb[e['batch']].append(e)
print(f"\n=== per flush-batch (a batch = rotations flushed before one measurement, SAME frame) ===")
print(f"{'batch':>6}{'nrot':>6}{'allCommute':>12}{'anyOffdiag':>12}{'rank':>6}{'needH(1/0)':>12}")
B_batched = 0
total_offdiag = 0
for b in sorted(byb):
    es = byb[b]
    allc = all(symp((a['xp'], a['zp']), (c['xp'], c['zp'])) == 0
               for i, a in enumerate(es) for c in es[i + 1:])
    anyoff = any(not e['pureZ'] for e in es)
    needH = 1 if anyoff else 0
    B_batched += needH
    total_offdiag += sum(1 for e in es if not e['pureZ'])
    print(f"{b:>6}{len(es):>6}{str(allc):>12}{str(anyoff):>12}{es[0]['rank']:>6}{needH:>12}")

# ---- ordered maximal commuting-run partition (lower bound ignoring measurement breaks) ----
runs = []
cur = []
for e in events:
    if all(symp((e['xp'], e['zp']), (g['xp'], g['zp'])) == 0 for g in cur):
        cur.append(e)
    else:
        runs.append(cur); cur = [e]
if cur:
    runs.append(cur)
B_runs = sum(1 for r in runs if any(not e['pureZ'] for e in r))

# ---- minimum-H counts: H per batch = GF(2) rank of the batch's X-block (xp vectors) ----
Hbatch = {b: gf2_rank([e['xp'] for e in byb[b]]) for b in byb}
Hbatch_total = sum(Hbatch.values())
Hrun = [gf2_rank([e['xp'] for e in r]) for r in runs]
Hrun_total = sum(Hrun)
Hall = gf2_rank([e['xp'] for e in events])        # X-rank of the WHOLE commuting set (1-basis ideal)


# ---- projected FLOP (use each rotation's own 2^rank; H sweep=4, diagonal=3) ----
def f2(r):
    return 1 << r
F_A = sum((4 + 3) * f2(e['rank']) for e in events)                    # per-rotation: 1 H each
# per-batch simultaneous diagonalization: Xrank(batch) H's at the batch max rank + 3*2^r per rotation
F_B = sum(Hbatch[b] * 4 * f2(max(g['rank'] for g in byb[b])) for b in byb) \
      + sum(3 * f2(e['rank']) for e in events)
F_C = sum(Hrun[i] * 4 * f2(max(g['rank'] for g in r)) for i, r in enumerate(runs)) \
      + sum(3 * f2(e['rank']) for e in events)
F_clf = sum(3 * f2(e['rank']) for e in events)                       # clifft diagonal-T (no H)
F_cur = sum(12 * f2(e['rank']) for e in events)                      # current butterfly c=12

print(f"\n  per-batch min-H (GF2 X-rank): " + ", ".join(f"b{b}:{Hbatch[b]}(n{len(byb[b])})"
      for b in sorted(byb)))
print(f"\n=== H-amortization verdict ({CIRC}) ===")
print(f"  q (rotations)                    = {q}")
print(f"  off-diagonal generators          = {total_offdiag}/{q}")
print(f"  #flush-batches                   = {len(byb)}")
print(f"  H-sweeps, per-rotation policy    = {q}   (1 per rotation)")
print(f"  H-sweeps, per-batch simul-diag   = {Hbatch_total}   (Sigma GF2 X-rank per batch)")
print(f"  H-sweeps, commuting-run          = {Hrun_total}   ({len(runs)} runs, ranks {Hrun})")
print(f"  H-sweeps, single global basis    = {Hall}   (X-rank of all 91, ignores meas/rank changes)")
print(f"\n  projected FLOP (Sigma over each rotation's own 2^rank):")
print(f"    current butterfly (c=12)       = {F_cur/1e3:8.1f}k   ({F_cur/F_clf:.2f}x clifft)")
print(f"    A per-rotation  (4q+3q)        = {F_A/1e3:8.1f}k   ({F_A/F_clf:.2f}x clifft)  H={q}")
print(f"    B per-batch     (4*SigmaXrank+3q) = {F_B/1e3:8.1f}k   ({F_B/F_clf:.2f}x clifft)  H={Hbatch_total}")
print(f"    C commuting-run                = {F_C/1e3:8.1f}k   ({F_C/F_clf:.2f}x clifft)  H={Hrun_total}")
print(f"    clifft diagonal-T (3q)         = {F_clf/1e3:8.1f}k   (1.00x)")
