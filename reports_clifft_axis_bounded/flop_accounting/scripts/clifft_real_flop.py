"""REAL clifft algorithmic FLOP via the C++ CostMeter (instrumented _clifft_core.abi3.so).

Replaces the flat-peak shared-event REFERENCE (which assumed clifft holds 2^k every shared event).
Each clifft dense kernel now records its own primitive-op counts at its serial entry point; we apply
the SAME FLOP convention used for bounded:  cmul=6, rcmul=2, cadd=2, sqmag=4, vdot=8.

clifft native compile is FUSED (array_u2/u4); bounded native compile is unfused. Each backend's FLOP
is summed over ITS OWN real kernel events on the SAME circuit (seed 1, 1 shot) -> apples-to-apples.

Non-invasive: meter off vs on gives identical samples (asserted).  Internal consistency (§4): every
kernel's primitive totals equal the exact multiple of sum_pow2k its arithmetic demands (asserted).
"""
import sys, csv, os
sys.path.insert(0, "/home/jung/clifft-paper")
sys.setrecursionlimit(400000)
import numpy as np
import clifft
from clifft import _clifft_core as cc

CONV = dict(cmul=6, rcmul=2, cadd=2, sqmag=4, vdot=8)        # shared FLOP convention
def kflop(s):
    return sum(CONV[k] * s[k] for k in CONV)

# §4 internal-consistency: each kernel's primitive totals must be an EXACT multiple of sum_pow2k
# (S=sum of 2^k over calls).  half-sum = S/2, quarter-sum not derivable from S alone so we check
# the kernels whose counts are pure multiples of S or S/2.
def check_consistency(name, s):
    S = s['sum_pow2k']
    if S == 0:
        return True, "S=0"
    # Half-subspace kernels (rot/s/t): cmul = 2^(k-1) per call, so cmul = (S - n_k0)/2 where
    # n_k0 = #calls at active_k=0 (rotation on a dormant axis -> 0 array work, 0 FLOP, correct).
    # Invariant: 0 <= S - 2*cmul <= calls.
    HALF = {'array_rot', 'array_s', 'array_s_dag', 'array_t', 'array_t_dag'}
    if name in HALF:
        d = S - 2 * s['cmul']
        ok = (0 <= d <= s['calls']) and s['cmul'] == s['processed']
        return ok, "ok" if ok else f"S-2cmul={d} not in [0,{s['calls']}]"
    exp = {
        'array_h':     dict(rcmul=S, cadd=S),
        'array_u2':    dict(cmul=2*S, cadd=S),
        'array_u4':    dict(cmul=4*S, cadd=3*S),
        'expand':      dict(cmul=0),
        'expand_t':    dict(cmul=S), 'expand_t_dag': dict(cmul=S), 'expand_rot': dict(cmul=S),
        'meas_diagonal': dict(sqmag=S),
        'meas_interfere':      dict(sqmag=S, rcmul=S//2, cadd=S + S//2),
        'swap_meas_interfere': dict(sqmag=S, rcmul=S//2, cadd=S + S//2),
        'exp_val':     dict(sqmag=S, rcmul=S, vdot=S),
        'array_cnot':  dict(cmul=0, cadd=0), 'array_swap': dict(cmul=0, cadd=0),
        'array_multi_cnot': dict(cmul=0, cadd=0),
    }.get(name)
    if exp is None:
        return True, "n/a"               # cz/multi_cz use quarter-sum (not derivable from S)
    for prim, want in exp.items():
        if s[prim] != want:
            return False, f"{prim} {s[prim]}!={want}"
    return True, "ok"


def clifft_real(circ, seed=1, shots=1):
    prog = clifft.compile(open(f"qec_bench/circuits/{circ}.stim").read())   # NATIVE fused
    # non-invasive: meter off vs on -> identical samples
    cc.cost_meter_enable(False)
    m_off = np.asarray(clifft.sample(prog, 32, 12345).measurements)
    cc.cost_meter_reset(); cc.cost_meter_enable(True)
    m_on = np.asarray(clifft.sample(prog, 32, 12345).measurements)
    cc.cost_meter_enable(False)
    noninv = np.array_equal(m_off, m_on)
    # measured snapshot for the comparison run
    cc.cost_meter_reset(); cc.cost_meter_enable(True)
    clifft.sample(prog, shots, seed)
    cc.cost_meter_enable(False)
    snap = cc.cost_meter_snapshot()
    snap = {k: dict(v) for k, v in snap.items()}
    consistent = all(check_consistency(n, s)[0] for n, s in snap.items())
    fails = [(n, check_consistency(n, s)[1]) for n, s in snap.items() if not check_consistency(n, s)[0]]
    tot = sum(kflop(s) for s in snap.values())
    return dict(circ=circ, k=prog.peak_rank, nm=prog.num_measurements, snap=snap,
                flop=tot, noninv=noninv, consistent=consistent, fails=fails)


def load_bounded():
    """bounded REAL algorithmic FLOP + the OLD flat-peak reference, from flop_all.csv."""
    p = "reports_clifft_axis_bounded/flop_accounting/data/flop_all.csv"
    out = {}
    for r in csv.DictReader(open(p)):
        out[r['circuit']] = r
    return out


def H(x):
    if x is None: return "-"
    for u, s in ((1e9, 'G'), (1e6, 'M'), (1e3, 'k')):
        if abs(x) >= u: return f"{x/u:.2f}{s}"
    return f"{x:.0f}"


CIRCS = ["coherent_d3_r1", "coherent_d3_r3", "coherent_d5_r1", "coherent_d5_r5",
         "coherent_rx_d3_r1", "coherent_rx_d3_r3",
         "coherent_ry_d3_r1", "coherent_ry_d3_r3",
         "cultivation_d3", "cultivation_d5", "distillation"]

bnd = load_bounded()
rows = []
print("=== REAL clifft FLOP (instrumented .so) vs bounded REAL FLOP  [same convention, seed 1, 1 shot] ===")
print(f"{'circuit':18}{'k':>3}{'clifft REAL':>12}{'bounded REAL':>13}{'cl/bnd':>8}{'NI':>4}{'cons':>5}")
for c in CIRCS:
    r = clifft_real(c)
    b = bnd.get(c, {})
    bf = float(b['bounded_FLOP']) if b.get('bounded_FLOP') not in (None, '', 'INFEASIBLE') else None
    cl = r['flop']
    ratio = (cl / bf) if bf else float('nan')
    print(f"{c:18}{r['k']:>3}{H(cl):>12}{H(bf):>13}{ratio:>8.2f}"
          f"{'OK' if r['noninv'] else 'XX':>4}{'OK' if r['consistent'] else 'XX':>5}")
    if r['fails']:
        print("     CONSISTENCY FAIL:", r['fails'])
    rows.append([c, r['k'], r['nm'], int(cl), int(bf) if bf else '', f"{ratio:.3f}" if bf else ''])

with open("reports_clifft_axis_bounded/flop_accounting/data/clifft_real_flop.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["circuit", "k", "num_meas", "clifft_REAL_FLOP", "bounded_REAL_FLOP",
                "clifft_over_bounded"])
    w.writerows(rows)
print("\n-> reports_clifft_axis_bounded/flop_accounting/data/clifft_real_flop.csv")

# per-kernel breakdown for the two corrected R_Y circuits
for c in ("coherent_ry_d3_r1", "coherent_ry_d3_r3"):
    r = clifft_real(c)
    print(f"\n=== {c}: clifft REAL per-kernel ===")
    for n, s in sorted(r['snap'].items(), key=lambda kv: -kflop(kv[1])):
        print(f"  {n:20} calls={s['calls']:>4} rank_max={s['rank_max']:>2} FLOP={H(kflop(s)):>9}")
    print(f"  TOTAL = {H(r['flop'])}")
