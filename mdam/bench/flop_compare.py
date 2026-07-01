"""Full-benchmark FLOP / traffic table via the VALIDATED budget.charge hook.
bounded = ALGORITHMIC FLOPs (stated convention).  clifft = MODELED (compiled .so, charged at 2^k).
Off-axis d5 (R_X/R_Y, k=38/47 > 2^26) are reported INFEASIBLE (not run)."""
import sys, time, math, csv, os
sys.path.insert(0, "/home/jung/clifft-paper")
sys.setrecursionlimit(400000)
import clifft
import mdam.backend.backend as bk
from mdam.backend.clifft_axis import budget as _bud
from mdam.backend.clifft_axis.bounded import compile_bounded

# ---- Clifft FLOP from CLIFFT'S OWN schedule (not modeled from MDAM events) ----
# Clifft evolves its dense active register EAGERLY per the compiled, outcome-independent active_k_history;
# each dense active-register op at instruction i costs C(op) * 2^{active_k_history[i]}.  (MDAM, by contrast,
# materializes lazily, so MDAM dense FLOP can be 0 where Clifft's is large.)  Per-element FLOP convention
# matches the bounded side: offdiag complex op = 12, diagonal phase = 6, permutation = 0, measurement = 12.
_CL_OFFD = {"ARRAY_ROT","ARRAY_H","ARRAY_U2","ARRAY_U4","EXPAND_ROT","EXPAND_T","EXPAND_T_DAG"}
_CL_DIAG = {"ARRAY_T","ARRAY_T_DAG","ARRAY_S","ARRAY_CZ","ARRAY_MULTI_CZ"}
_CL_MEAS = {"MEAS_ACTIVE_DIAGONAL","MEAS_ACTIVE_INTERFERE","SWAP_MEAS_INTERFERE"}  # collapse on the active register
_CL_PERM = {"ARRAY_CNOT","ARRAY_MULTI_CNOT","ARRAY_SWAP"}   # index permutation: 0 dense FLOP

def clifft_flop_from_schedule(cprog):
    """Clifft dense FLOP/shot from its own active_k_history + dense-op schedule (deterministic per circuit)."""
    akh = list(cprog.active_k_history); F = 0.0; n = len(cprog)
    for i in range(n):
        op = str(cprog[i].opcode).replace("Opcode.OP_", "")
        k = akh[i] if i < len(akh) else 0
        if op in _CL_OFFD or op in _CL_MEAS: F += 12.0 * (1 << k)
        elif op in _CL_DIAG:                 F += 6.0  * (1 << k)
        # _CL_PERM and all FRAME_*/NOISE/APPLY_PAULI/MEAS_DORMANT/DETECTOR ops touch only the symbolic
        # stabilizer frame (shared, not counted on either side) -> 0 dense FLOP
    return F

VALID = {  # (flop, R, W, clifft-shared) per CHARGED element
    'rot:offdiag': (12, 1, 1, 1), 'rot:offdiag-scalar': (12, 1, 1, 1), 'collapse:offdiag': (12, 1, 1, 1),
    'rot:diag': (6, 1, 1, 1), 'rot:diag0': (6, 1, 1, 1), 'rot:diag-scalar': (6, 1, 1, 1),
    'collapse:diag': (6, 1, 1, 1), 'collapse:diag0': (6, 1, 1, 1),
    'meas': (10, 1.5, 0, 1), 'exp': (10, 1.5, 0, 1), 'reduce:verify': (10, 1.5, 0, 0),
    'sqnorm': (2, 0.5, 0, 1), 'purge:h': (5, 1, 1, 0), 'purge:s': (3, 0.5, 0.5, 0),
    'purge:cnot': (0, 0.5, 0.5, 0), 'reduce:cnot': (0, 0.5, 0.5, 0),
    'drop': (0, 0.5, 0.5, 0), 'promote': (0, 0, 1, 0),
    'reduce:gf2scan': (0, 1, 0, 0), 'init': (0, 0, 0, 0), 'post-reduce': (0, 0, 0, 0),
}


class Prod:
    def __init__(self):
        self.fb = self.fc = self.R = self.W = 0.0; self.kern = {}; self.inv = 0; self._o = None
    def enable(self):
        self._o = _bud.DenseMemoryBudget.charge; P = self; orig = self._o
        def charge(self, resident, transient=0, where=""):
            f, rc, wc, sh = VALID.get(where, (0, 0, 0, 0)); N = int(resident)
            P.fb += f * N; P.R += rc * N; P.W += wc * N; P.inv += 1
            d = P.kern.setdefault(where, [0.0, 0]); d[0] += f * N; d[1] += 1
            if sh: P.fc += f * self.cap
            if where.startswith('collapse'):
                P.fb += 6 * N; P.R += N; P.W += N; P.fc += 6 * self.cap
            return orig(self, resident, transient, where)
        _bud.DenseMemoryBudget.charge = charge
    def disable(self): _bud.DenseMemoryBudget.charge = self._o


def axis_of(c):
    if 'rx' in c: return 'R_X'
    if 'ry' in c: return 'R_Y'
    if c.startswith('cultivation') or c.startswith('distill'): return 'T'
    return 'R_Z'


def state_volume(circ):
    p = f"reports_clifft_axis_bounded/bounded_{circ}_per_step.csv"
    if not os.path.exists(p): return None, None
    Sb = Sc = 0
    for r in csv.DictReader(open(p)):
        Sb += 1 << int(r['bounded_resident_qubits']); Sc += 1 << int(r['n_active'])
    return Sb, Sc


def run(circ, seed=1, kmax=26):
    prog = compile_bounded(open(f"qec_bench/circuits/{circ}.stim").read())
    if prog.peak_rank > kmax:
        return dict(circ=circ, k=prog.peak_rank, infeasible=True)
    P = Prod(); P.enable(); t0 = time.time()
    try:
        be = bk.NearCliffordBackend(clifft_axis_bounded=True, drop_dead=False,
                                    structure_once=False, clifft_axis_enforce=True)
        be.run_shot(prog, seed)
        # fused measurement core: the FLOP driver (n_U Pauli-sum terms x 2^r_out) is NOT a budget.charge
        # event -- it is recorded in the engine's own _fused_log (survivor_ops = n_U * 2^r_out).  Add it,
        # modeling clifft at the full peak (n_U * 2^k).  C_FUSED cancels in the ratio (Pauli term ~ collapse).
        C_FUSED = 12
        flog = getattr(be.nc, "_fused_log", None) or []
        P.fb += C_FUSED * sum(int(d["survivor_ops"]) for d in flog)
        P.fc += C_FUSED * sum(int(d["n_U_terms"]) * (1 << prog.peak_rank) for d in flog)
    finally:
        P.disable()
    dt = time.time() - t0; Sb, Sc = state_volume(circ)
    return dict(circ=circ, k=prog.peak_rank, maxM=be.last_max_M, fb=P.fb, fc=P.fc,
                R=P.R, W=P.W, bytes=16 * (P.R + P.W), inv=P.inv, kern=P.kern, dt=dt,
                Sb=Sb, Sc=Sc, nm=prog.num_measurements, infeasible=False)


def run_avg(circ, nseed=10, kmax=26):
    """Average per-shot FLOP/traffic over nseed seeds (magic firing is stochastic;
    a single shot is not representative).  Reports the mean per-shot quantities."""
    rs = [run(circ, seed=s, kmax=kmax) for s in range(1, nseed + 1)]
    if rs[0].get('infeasible'):
        return rs[0]
    n = len(rs)
    avg = lambda key: sum(r[key] for r in rs) / n
    return dict(circ=circ, k=rs[0]['k'], nm=rs[0]['nm'], nseed=n, infeasible=False,
                maxM=max(r['maxM'] for r in rs), fb=avg('fb'), fc=avg('fc'),
                R=avg('R'), W=avg('W'), bytes=avg('bytes'), inv=avg('inv'), dt=avg('dt'),
                Sb=rs[0]['Sb'], Sc=rs[0]['Sc'])


def H(x):
    if x is None: return "-"
    for u, s in ((1e12, 'T'), (1e9, 'G'), (1e6, 'M'), (1e3, 'k')):
        if abs(x) >= u: return f"{x/u:.2f}{s}"
    return f"{x:.0f}"


CIRCS = ["coherent_d3_r1", "coherent_d3_r3", "coherent_d5_r1", "coherent_d5_r5",
         "surface_d7_r7",
         "coherent_rx_d3_r1", "coherent_rx_d3_r3", "coherent_rx_d5_r1", "coherent_rx_d5_r5",
         "coherent_ry_d3_r1", "coherent_ry_d3_r3", "coherent_ry_d5_r1", "coherent_ry_d5_r5",
         "cultivation_d3", "cultivation_d5", "distillation"]

NSEED = int(os.environ.get("FLOP_NSEED", "10"))
rows = []
print(f"FLOP per shot.  bounded(MDAM) = mean over {NSEED} seeds, REAL dynamic events (incl. fused core).")
print(f"clifft = ITS OWN active_k_history dense-op schedule (deterministic; outcome-independent).")
print(f"{'circuit':18}{'ax':4}{'k':>3}{'maxM':>6}{'bnd FLOP':>11}{'clifft FLOP':>12}{'F_cl/F_bn':>20}{'ms':>7}")
for c in CIRCS:
    ax = axis_of(c)
    text = open(f"qec_bench/circuits/{c}.stim").read()
    cprog = clifft.compile(text)
    k = cprog.peak_rank
    fc = clifft_flop_from_schedule(cprog)                 # Clifft dense FLOP from its own schedule (always)
    if k > 26:                                            # neither engine can RUN above 2^26
        print(f"{c:18}{ax:4}{k:>3}  CANNOT RUN (2^{k}): clifft sched FLOP={H(fc)} (also needs 2^{k} memory)")
        rows.append([c, ax, k, "cannot_run", "", round(fc, 1), "", ""]); continue
    r = run_avg(c, NSEED)
    fb = r["fb"]
    if fb:
        fr = fc/fb; fr_s = f"{fr:.1f}x"
        rr = f"{fr:.2f}"
    else:
        fr_s = "inf (MDAM 0 dense)"; rr = "inf"          # MDAM materialized nothing; Clifft did fc
    print(f"{c:18}{ax:4}{k:>3}{r['maxM']:>6}{H(fb):>11}{H(fc):>12}{fr_s:>20}{r['dt']*1e3:>7.0f}")
    rows.append([c, ax, k, r["maxM"], round(fb, 1), round(fc, 1), rr, f"{r['dt']*1e3:.0f}"])

with open("results/benchmark_comparison/flop_table.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["circuit", "axis", "k_clifft", "max_M_mdam", "bounded_FLOP_per_shot",
                "clifft_FLOP_per_shot", "F_cl_over_F_bn", "mdam_ms"])
    w.writerows(rows)
print(f"\n-> results/benchmark_comparison/flop_table.csv  (bounded=mean of {NSEED} seeds; clifft=own schedule)")
