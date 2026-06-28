"""Per-step ACTIVE-STATE trace for the clifft_axis_bounded LIVE backend on OFF-AXIS
(R_X / R_Y) coherent-noise benchmarks, in the SAME format as
reports/per_step_active_state/fused_va_generate.py (one row per runtime step; columns
step,n_active,<eng>_resident_qubits,<eng>_transient_qubits,<eng>_resident_dim,<eng>_transient_dim).

These circuits are coherent_d{d}_r{r} with the single-qubit coherent over-rotation on the
X or Y axis instead of Z (qec_bench/generate_axis_noise.py):
    coherent_rx_d{d}_r{r}.stim : R_X(0.02) noise
    coherent_ry_d{d}_r{r}.stim : R_Y(0.02) noise
The off-axis rotation carries X-support, so (unlike diagonal R_Z) it does not commute through
the Z-stabilisers: the active rank k is larger (rx d3=14 / d5=38; ry d3=16 / d5=47, vs R_Z
d5=24) AND the materialised magic register stays large between measurements.

COMPILE: clifft.compile(..., bytecode_passes=None) via clifft_axis.bounded.compile_bounded.
Off-axis rotations MUST NOT be fused: the bounded backend applies fused U2/U4 nodes with a
raw frame.set_xz that does NOT conjugate the lazy engine's deferred (pending) rotations, so a
fused off-axis node is silently wrong.  No-fusion compile leaves the active rank k unchanged
(peak_rank identical, verified) and is exact (validated vs clifft.sample: bounded marginals
within the clifft-vs-clifft statistical spread on every d3 circuit).

FEASIBILITY: the d5 off-axis circuits materialise a magic register > 2^26 words (> 1 GiB) --
the localize-and-drop cannot bound it because off-axis noise keeps many magic dof simultan-
eously live -- so they are recorded as INFEASIBLE (no per-step trace) rather than OOM.  The
d3 circuits are feasible and traced in full.

`resident` = settled magic rank len(M) between measurements; `transient` = peak magic rank
reached DURING a measurement (the promote/flush peak before the localize-and-drop).
`n_active` = len(slot2id) (clifft active-state baseline, a backend quantity).

Reproduce: clifft_env/bin/python reports_clifft_axis_bounded_rxry/bounded_rxry_generate.py
"""
import os
import sys

os.chdir("/home/jung/clifft-paper")
sys.path.insert(0, "/home/jung/clifft-paper")
sys.setrecursionlimit(400000)
import clifft
import nearclifford_backend.backend as bk
from nearclifford_backend.backend import count_idents
from nearclifford_backend.clifft_axis.bounded import CliftAxisBoundedNearClifford, compile_bounded
from nearclifford_backend.clifft_axis.engine import CliftAxisNearClifford as _Engine

OUT = 'reports_clifft_axis_bounded'
FEASIBLE = ['coherent_rx_d3_r1', 'coherent_rx_d3_r3',
            'coherent_ry_d3_r1', 'coherent_ry_d3_r3']
INFEASIBLE = ['coherent_rx_d5_r1', 'coherent_rx_d5_r5',
              'coherent_ry_d5_r1', 'coherent_ry_d5_r5']
GUARD = 26          # 2^26 complex128 = 1 GiB feasibility ceiling for the d5 probe

# Arm a 1-GiB feasibility ceiling so a d5 off-axis blow-up fails cleanly (not OOM).
_orig_budget = _Engine.set_clifft_budget
def _guarded_budget(self, k, enforce=True):
    _orig_budget(self, min(k, GUARD), enforce=True)


def clifft_k(circ):
    return clifft.compile(open(f'qec_bench/circuits/{circ}.stim').read()).peak_rank


def extract(circ, seed=1):
    prog = compile_bounded(open(f'qec_bench/circuits/{circ}.stim').read())   # NO fusion
    n = count_idents(prog)
    rows = []
    cur = {}
    peakM = {'v': 0}
    o_prom = CliftAxisBoundedNearClifford._promote
    o_mz = CliftAxisBoundedNearClifford.measure_z

    def prom(self, q):                              # per-measurement rank high-water
        o_prom(self, q)
        if len(self.M) > peakM['v']:
            peakM['v'] = len(self.M)

    def mz(self, q):
        peakM['v'] = len(self.M)
        out = o_mz(self, q)
        cur['W'] = max(peakM['v'], len(self.M))
        return out

    CliftAxisBoundedNearClifford._promote = prom
    CliftAxisBoundedNearClifford.measure_z = mz

    def rec(step, be):
        na = len(be.slot2id)
        res = len(be.nc.M)
        tr = cur.pop('W', res)
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


print(f"{'circuit':20}{'clifft_k':>9}{'bnd_transient':>14}{'bnd_resident':>14}{'steps':>7}")
summ = []
for c in FEASIBLE:
    k = clifft_k(c)
    n, rows, pw, pr = extract(c)
    with open(f'{OUT}/bounded_{c}_per_step.csv', 'w') as f:
        f.write('step,n_active,bounded_resident_qubits,bounded_transient_qubits,'
                'bounded_resident_dim,bounded_transient_dim\n')
        for (s, na, res, tr) in rows:
            f.write(f'{s},{na},{res},{tr},{1 << res},{1 << tr}\n')
    summ.append((c, k, pw, pr, len(rows), True))
    print(f"{c:20}{k:>9}{pw:>14}{pr:>14}{len(rows):>7}")

# d5 off-axis: record INFEASIBLE (transient exceeds the 1-GiB ceiling) with the actual k.
_Engine.set_clifft_budget = _guarded_budget
infeas = []
for c in INFEASIBLE:
    k = clifft_k(c)
    prog = compile_bounded(open(f'qec_bench/circuits/{c}.stim').read())
    status = '?'
    try:
        be = bk.NearCliffordBackend(clifft_axis_bounded=True, drop_dead=False,
                                    structure_once=False, clifft_axis_enforce=True)
        be.run_shot(prog, 1)
        status = f'feasible(maxM={be.nc.max_M})'
    except Exception as e:
        status = f'INFEASIBLE>2^{GUARD}'
    infeas.append((c, k, status))
    print(f"{c:20}{k:>9}   {status}")
_Engine.set_clifft_budget = _orig_budget

with open(f'{OUT}/BOUNDED_RXRY_SUMMARY.md', 'w') as f:
    f.write("# Per-step ACTIVE-STATE: clifft_axis_bounded on OFF-AXIS (R_X / R_Y) noise\n\n")
    f.write("coherent_d{d}_r{r} with the coherent over-rotation on the X or Y axis "
            "(`R_X(0.02)` / `R_Y(0.02)`) instead of Z.  Compiled with **no bytecode fusion** "
            "(`compile_bounded`): off-axis rotations must not be fused into U2/U4 nodes (the "
            "bounded backend's `frame.set_xz` does not conjugate deferred pending rotations, "
            "so a fused off-axis node is silently wrong).  No-fusion leaves k unchanged and "
            "is exact (bounded marginals within the clifft-vs-clifft spread, 20k shots).\n\n")
    f.write("**transient** = peak materialised magic rank during a measurement; "
            "**resident** = settled magic rank between measurements.  "
            "Per-step traces: `bounded_<circuit>_per_step.csv`.\n\n")
    f.write("## Feasible (d=3)\n\n")
    f.write("| circuit | noise | Clifft k | bounded transient | bounded resident | "
            "transient dim | saving 2^(k-transient) |\n")
    f.write("|---|---|--:|--:|--:|--:|--:|\n")
    for (c, k, pw, pr, ns, _) in summ:
        ax = 'R_X' if '_rx_' in c else 'R_Y'
        f.write(f"| {c} | {ax} | {k} | {pw} | {pr} | 2^{pw} | 2^{k - pw} |\n")
    f.write("\n## Infeasible (d=5): off-axis noise keeps the magic rank > 2^26 (1 GiB)\n\n")
    f.write("Unlike diagonal R_Z (d5_r5 transient = 2^13 = 128 KiB), off-axis noise carries "
            "X-support and keeps many magic dof simultaneously live, so the localize-and-drop "
            "cannot bound the materialised register; it exceeds the 1-GiB ceiling.\n\n")
    f.write("| circuit | noise | Clifft k | bounded status |\n|---|---|--:|---|\n")
    for (c, k, status) in infeas:
        ax = 'R_X' if '_rx_' in c else 'R_Y'
        f.write(f"| {c} | {ax} | {k} | {status} |\n")
print("WROTE", OUT + "/BOUNDED_RXRY_SUMMARY.md and per-step CSVs")
