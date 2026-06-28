"""Per-step ACTIVE-STATE trace for the clifft_axis_bounded LIVE backend, in the SAME format
as reports/per_step_active_state/fused_va_generate.py (one row per runtime step; columns
step,n_active,<eng>_resident_qubits,<eng>_transient_qubits,<eng>_resident_dim,<eng>_transient_dim).

Engine = `CliftAxisBoundedNearClifford` (the canonical bounded near-Clifford backend): one
dense magic register, localize-and-drop measurements, hard budget peak amplitude words <=
2^k_clifft, capacity buffer (no ndarray.resize), einsum-view norms, slack-aware rotations.
`resident` = settled magic rank len(M) between measurements; `transient` = peak magic rank
reached DURING a measurement (the flush/promote peak before the drop).  `n_active` =
len(slot2id) (active qubits) -- the Clifft active-state baseline, identical to the fused-VA
report's because it is a backend quantity.

Reproduce: clifft_env/bin/python reports_clifft_axis_bounded/bounded_generate.py
"""
import os
import sys

os.chdir("/home/jung/clifft-paper")
sys.path.insert(0, "/home/jung/clifft-paper")
sys.setrecursionlimit(400000)
import clifft
import nearclifford_backend.backend as bk
from nearclifford_backend.backend import count_idents
from nearclifford_backend.clifft_axis.bounded import CliftAxisBoundedNearClifford
from nearclifford_backend.virtual_axis.bench_memory import clifft_k

CIRCS = ['coherent_d3_r1', 'coherent_d3_r3', 'coherent_d5_r1', 'coherent_d5_r5',
         'cultivation_d3', 'cultivation_d5', 'distillation', 'surface_d7_r7']
OUT = 'reports_clifft_axis_bounded'


def extract(circ, seed=1):
    prog = clifft.compile(open(f'qec_bench/circuits/{circ}.stim').read())   # default compile
    n = count_idents(prog)
    rows = []
    cur = {}
    peakM = {'v': 0}
    o_prom = CliftAxisBoundedNearClifford._promote
    o_mz = CliftAxisBoundedNearClifford.measure_z

    def prom(self, q):                              # track the per-measurement rank high-water
        o_prom(self, q)
        if len(self.M) > peakM['v']:
            peakM['v'] = len(self.M)

    def mz(self, q):
        peakM['v'] = len(self.M)
        out = o_mz(self, q)
        cur['W'] = max(peakM['v'], len(self.M))     # transient at THIS measurement step
        return out

    CliftAxisBoundedNearClifford._promote = prom
    CliftAxisBoundedNearClifford.measure_z = mz

    def rec(step, be):
        na = len(be.slot2id)
        res = len(be.nc.M)
        tr = cur.pop('W', res)                      # W set if a measurement happened since last rec
        rows.append((step, na, res, tr))
    try:
        be = bk.NearCliffordBackend(clifft_axis_bounded=True, drop_dead=False,
                                    structure_once=False, clifft_axis_enforce=True)
        be.run_shot(prog, seed, step_recorder=rec)
        peak_res = be.nc.max_M
    finally:
        CliftAxisBoundedNearClifford._promote = o_prom
        CliftAxisBoundedNearClifford.measure_z = o_mz
    peak_ws = max((tr for (_, _, _, tr) in rows), default=0)
    return n, rows, peak_ws, peak_res


print(f"{'circuit':16}{'clifft_k':>9}{'bnd_transient':>14}{'bnd_resident':>14}{'steps':>7}")
summ = []
for c in CIRCS:
    try:
        k = clifft_k(c)
    except Exception:
        k = '?'
    n, rows, pw, pr = extract(c)
    with open(f'{OUT}/bounded_{c}_per_step.csv', 'w') as f:
        f.write('step,n_active,bounded_resident_qubits,bounded_transient_qubits,'
                'bounded_resident_dim,bounded_transient_dim\n')
        for (s, na, res, tr) in rows:
            f.write(f'{s},{na},{res},{tr},{1 << res},{1 << tr}\n')
    summ.append((c, k, pw, pr, len(rows)))
    print(f"{c:16}{str(k):>9}{pw:>14}{pr:>14}{len(rows):>7}")

with open(f'{OUT}/BOUNDED_SUMMARY.md', 'w') as f:
    f.write("# Per-step ACTIVE-STATE: clifft_axis_bounded LIVE backend\n\n")
    f.write("Peak active-state size of the canonical bounded near-Clifford engine "
            "(`CliftAxisBoundedNearClifford`, hard budget peak amplitude words <= 2^k_clifft). "
            "**transient** = peak materialized magic rank during a measurement (flush/promote "
            "peak before the localize-and-drop); **resident** = settled magic rank between "
            "measurements.  Per-step traces: `bounded_<circuit>_per_step.csv`.\n\n")
    f.write("| circuit | Clifft k | bounded transient (qubits) | bounded resident (qubits) | "
            "bounded transient dim | saving 2^(k-transient) |\n")
    f.write("|---|--:|--:|--:|--:|--:|\n")
    for (c, k, pw, pr, ns) in summ:
        sav = f"2^{k - pw}" if isinstance(k, int) else "?"
        f.write(f"| {c} | {k} | {pw} | {pr} | 2^{pw} | {sav} |\n")
print("WROTE", OUT + "/BOUNDED_SUMMARY.md and per-step CSVs")
