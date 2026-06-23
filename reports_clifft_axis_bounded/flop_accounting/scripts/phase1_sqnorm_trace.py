"""Phase 1 instrumentation: trace WHERE _branch_sqnorm fires in the bounded
measurement/purge path, per measurement, BEFORE vs AFTER the sqnorm-reuse fix.

For each measurement it records: meas id, rank before, Born sqnorm calls,
normalization full-sweep calls, purge (compress) sqnorm calls, rank after.
Aggregates per circuit: measurements, sqnorm calls, sqnorm/meas, sqnorm FLOP,
total FLOP -- the exact table the Phase 1 spec asks for.

sqnorm classification (by call site, tagged with an _in_purge flag the harness
toggles around _compress_magic):
  * born          : measure_z's p0r = _branch_sqnorm(jr,0)   (the ONE legitimate Born)
  * normalization : _sqnorm_1d(self.phi) full sweep after projection (uncharged today)
  * purge         : _drop_axis_inplace's sq0/sq1 inside _compress_magic (the explosion)

Run with no args for the summary table; --trace <circuit> for the per-measurement trace.
"""
import sys, math
sys.path.insert(0, "/home/jung/clifft-paper")
sys.setrecursionlimit(400000)
from collections import defaultdict
import numpy as np
import nearclifford_backend.backend as bk
from nearclifford_backend.clifft_axis import budget as _bud
from nearclifford_backend.clifft_axis.bounded import compile_bounded, CliftAxisBoundedNearClifford

# same FLOP coefficients as flop_attribution.py (budget.charge `where` -> FLOP/word)
COEF = {'rot:offdiag': 12, 'rot:offdiag-scalar': 12, 'collapse:offdiag': 12,
        'rot:diag': 6, 'rot:diag0': 6, 'rot:diag-scalar': 6, 'collapse:diag': 6, 'collapse:diag0': 6,
        'meas': 10, 'exp': 10, 'reduce:verify': 10, 'sqnorm': 2, 'normalize': 2, 'purge:h': 5, 'purge:s': 3,
        'purge:cnot': 0, 'reduce:cnot': 0, 'drop': 0, 'promote': 0, 'reduce:gf2scan': 0,
        'init': 0, 'post-reduce': 0}

CIRCS = ["coherent_ry_d3_r1", "coherent_ry_d3_r3", "cultivation_d3", "cultivation_d5",
         "coherent_rx_d3_r3", "coherent_d5_r5"]


def H(x):
    a = abs(x)
    for u, s in ((1e9, 'G'), (1e6, 'M'), (1e3, 'k')):
        if a >= u:
            return f"{x/u:.2f}{s}"
    return f"{x:.0f}"


def run(circ, seed=1, want_trace=False):
    prog = compile_bounded(open(f"qec_bench/circuits/{circ}.stim").read())

    # ---- tag sqnorm call sites ----
    Cls = CliftAxisBoundedNearClifford
    orig_branch = Cls._branch_sqnorm
    orig_sq1d = Cls._sqnorm_1d
    orig_compress = Cls._compress_magic
    orig_measure = Cls.measure_z

    trace = []           # per-measurement dicts
    cur = {}             # measurement-in-progress counters

    def branch(self, j, brnch):
        ph = 'purge' if getattr(self, '_in_purge', False) else 'born'
        cur[ph] = cur.get(ph, 0) + 1
        cur['_words_' + ph] = cur.get('_words_' + ph, 0) + int(self.phi.size)
        return orig_branch(self, j, brnch)

    def sq1d(arr):
        # static method: count via a module-level latch
        cur['normalization'] = cur.get('normalization', 0) + 1
        cur['_words_normalization'] = cur.get('_words_normalization', 0) + int(arr.size)
        return orig_sq1d(arr)

    def compress(self):
        self._in_purge = True
        try:
            return orig_compress(self)
        finally:
            self._in_purge = False

    def measure(self, q):
        cur.clear()
        rank_before = self.phi.size.bit_length() - 1
        out = orig_measure(self, q)
        rank_after = self.phi.size.bit_length() - 1
        trace.append(dict(meas=len(trace), rank_before=rank_before, rank_after=rank_after,
                          born=cur.get('born', 0), normalization=cur.get('normalization', 0),
                          purge=cur.get('purge', 0),
                          words_born=cur.get('_words_born', 0),
                          words_norm=cur.get('_words_normalization', 0),
                          words_purge=cur.get('_words_purge', 0)))
        return out

    Cls._branch_sqnorm = branch
    Cls._sqnorm_1d = staticmethod(sq1d)
    Cls._compress_magic = compress
    Cls.measure_z = measure

    # ---- total FLOP via budget.charge hook ----
    cat_flop = defaultdict(float)
    orig_charge = _bud.DenseMemoryBudget.charge

    def charge(self, resident, transient=0, where=""):
        N = int(resident)
        cat_flop[where] += COEF.get(where, 0) * N
        if where.startswith('collapse'):
            cat_flop['normalization_deferred'] += 6 * N
        return orig_charge(self, resident, transient, where)

    _bud.DenseMemoryBudget.charge = charge
    try:
        be = bk.NearCliffordBackend(clifft_axis_bounded=True, drop_dead=False,
                                    structure_once=False, clifft_axis_enforce=True)
        be.run_shot(prog, seed)
    finally:
        Cls._branch_sqnorm = orig_branch
        Cls._sqnorm_1d = staticmethod(orig_sq1d)
        Cls._compress_magic = orig_compress
        Cls.measure_z = orig_measure
        _bud.DenseMemoryBudget.charge = orig_charge

    total_flop = sum(cat_flop.values())
    sqnorm_flop = cat_flop.get('sqnorm', 0.0)
    nm = len(trace)
    sq_calls = sum(t['born'] + t['purge'] for t in trace)     # charged sqnorm = born + purge
    born_calls = sum(t['born'] for t in trace)
    purge_calls = sum(t['purge'] for t in trace)
    norm_calls = sum(t['normalization'] for t in trace)
    if want_trace:
        return trace
    return dict(circ=circ, meas=nm, sq_calls=sq_calls, born=born_calls, purge=purge_calls,
                norm=norm_calls, sq_per_meas=(sq_calls / nm if nm else 0),
                sqnorm_flop=sqnorm_flop, total_flop=total_flop, peak=prog.peak_rank)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--trace":
        circ = sys.argv[2]
        tr = run(circ, want_trace=True)
        print(f"\n=== per-measurement sqnorm trace: {circ} ===")
        print(f"{'meas':>4}{'r_before':>9}{'born':>6}{'norm':>6}{'purge':>7}{'r_after':>8}")
        for t in tr:
            print(f"{t['meas']:>4}{t['rank_before']:>9}{t['born']:>6}{t['normalization']:>6}"
                  f"{t['purge']:>7}{t['rank_after']:>8}")
        sys.exit(0)

    print(f"\n{'circuit':22}{'meas':>5}{'sqnorm':>8}{'born':>6}{'purge':>7}{'norm':>6}"
          f"{'sq/meas':>9}{'sqnormFLOP':>12}{'totalFLOP':>12}")
    print("-" * 88)
    for circ in CIRCS:
        r = run(circ)
        print(f"{r['circ']:22}{r['meas']:>5}{r['sq_calls']:>8}{r['born']:>6}{r['purge']:>7}"
              f"{r['norm']:>6}{r['sq_per_meas']:>9.1f}{H(r['sqnorm_flop']):>12}{H(r['total_flop']):>12}")
