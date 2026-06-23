"""A. Per-measurement FLOP + resident-rank curve, via the VALIDATED budget.charge hook.

For each dense kernel event we record (event_idx, resident_rank=log2(N), flop, where) and
bucket the work into measurement EPOCHS: every charge accumulates into a "pending" bucket that
CLOSES when a Born expectation ('meas') fires -- i.e. the bucket holds all rotations/reductions
performed to prepare-and-execute measurement i, plus the measurement itself.

This makes the core mechanism visible:
    low-rank epoch  -> little FLOP
    rank climbs to r_max just before a measurement -> unavoidable full-state work
    measurement -> drop -> rank falls again.

bounded FLOP = ALGORITHMIC (validated convention).  clifft = MODELED: clifft holds 2^k every
shared event (no localize-and-drop), so clifft FLOP/epoch = (shared coeff-sum at that epoch)*2^k.

Outputs per circuit:
  reports_clifft_axis_bounded_rxry/flop_rank_trace_<circ>.png   (rank sawtooth + per-event FLOP)
  reports_clifft_axis_bounded_rxry/flop_by_rank_<circ>.png      (FLOP histogram over rank)
  reports_clifft_axis_bounded_rxry/per_measurement_<circ>.csv
"""
import sys, csv, math, os
sys.path.insert(0, "/home/jung/clifft-paper")
sys.setrecursionlimit(400000)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nearclifford_backend.backend as bk
from nearclifford_backend.clifft_axis import budget as _bud
from nearclifford_backend.clifft_axis.bounded import compile_bounded

# run from /home/jung/clifft-paper with /home/jung/clifft_env/bin/python
OUT = "reports_clifft_axis_bounded/flop_accounting/figures"   # PNGs
DAT = "reports_clifft_axis_bounded/flop_accounting/data"      # per_rank CSVs

# (flop, clifft-shared) per element of the CHARGED N -- VALIDATED coeffs (== flop_production.py)
COEFF = {
    'rot:offdiag': (12, 1), 'rot:offdiag-scalar': (12, 1), 'collapse:offdiag': (12, 1),
    'rot:diag': (6, 1), 'rot:diag0': (6, 1), 'rot:diag-scalar': (6, 1),
    'collapse:diag': (6, 1), 'collapse:diag0': (6, 1),
    'meas': (10, 1), 'exp': (10, 1), 'reduce:verify': (10, 0),
    'sqnorm': (2, 1),                       # 4*(N/2)/N
    'purge:h': (5, 0), 'purge:s': (3, 0),
    'purge:cnot': (0, 0), 'reduce:cnot': (0, 0), 'drop': (0, 0), 'promote': (0, 0),
    'reduce:gf2scan': (0, 0), 'init': (0, 0), 'post-reduce': (0, 0),
}


class Rec:
    def __init__(self):
        self.events = []     # (idx, rank, flop_bnd, flop_cl_share_coeff, where)
        self.cap = None
        self._orig = None

    def enable(self):
        self._orig = _bud.DenseMemoryBudget.charge
        R = self; orig = self._orig

        def charge(self, resident, transient=0, where=""):
            f, shared = COEFF.get(where, (0, 0))
            N = int(resident)
            R.cap = self.cap
            rank = int(round(math.log2(N))) if N >= 1 else 0
            fb = f * N
            fc_coeff = (f if shared else 0)
            if where.startswith('collapse'):       # modeled norm+renorm (6N)
                fb += 6 * N
                fc_coeff += 6
            R.events.append((len(R.events), rank, fb, fc_coeff, where))
            return orig(self, resident, transient, where)
        _bud.DenseMemoryBudget.charge = charge

    def disable(self):
        _bud.DenseMemoryBudget.charge = self._orig


def run(circ, seed=1):
    prog = compile_bounded(open(f"qec_bench/circuits/{circ}.stim").read())
    R = Rec(); R.enable()
    try:
        be = bk.NearCliffordBackend(clifft_axis_bounded=True, drop_dead=False,
                                    structure_once=False, clifft_axis_enforce=True)
        be.run_shot(prog, seed)
    finally:
        R.disable()
    return prog, be, R


def per_rank(R):
    """FLOP performed at each resident rank: bounded(2^r) vs clifft-modeled(2^k).
    The magic register is NOT measured per-stabilizer (those are Clifford); instead the rank
    rises to peak as measured qubits enter the register, then the measured-magic purge peels it
    back down -- one rise-and-fall mountain.  The meaningful decomposition is therefore by RANK."""
    cap = R.cap or 1
    d = {}                                   # rank -> [flop_bnd, flop_cl, n_events]
    for (_i, rank, fbnd, fcc, _w) in R.events:
        e = d.setdefault(rank, [0.0, 0.0, 0]); e[0] += fbnd; e[1] += fcc * cap; e[2] += 1
    return d


def H(x):
    for u, s in ((1e12, 'T'), (1e9, 'G'), (1e6, 'M'), (1e3, 'k')):
        if abs(x) >= u: return f"{x/u:.2f}{s}"
    return f"{x:.0f}"


def plot_circuit(circ, ax_label):
    prog, be, R = run(circ)
    k = prog.peak_rank
    cap = R.cap or 1
    evs = R.events
    idx = [e[0] for e in evs]
    rank = [e[1] for e in evs]
    fb = [e[2] for e in evs]
    fc = [e[3] * cap for e in evs]
    cum_b = np.cumsum(fb); cum_c = np.cumsum(fc)
    pr = per_rank(R)
    by_rank = {rk: [pr[rk][0], pr[rk][1]] for rk in pr}        # (flop_bnd, flop_cl)
    peak = max(rank) if rank else 0
    # win decomposition: irreducible peak (r==peak) vs slack shoulder (r==peak-1) vs localized tail
    fb_peak = pr.get(peak, [0, 0, 0])[0];  fc_peak = pr.get(peak, [0, 0, 0])[1]
    fb_sh = pr.get(peak - 1, [0, 0, 0])[0]; fc_sh = pr.get(peak - 1, [0, 0, 0])[1]
    fb_tail = sum(pr[r][0] for r in pr if r < peak - 1)
    fc_tail = sum(pr[r][1] for r in pr if r < peak - 1)
    peak_loc = idx[rank.index(peak)] if peak in rank else 0    # event where peak first reached
    ratio_tot = (cum_c[-1] / cum_b[-1]) if len(cum_b) and cum_b[-1] else 1.0
    logy = ratio_tot > 50                                      # log scale when gap spans >1.7 decades

    # ---- Figure 1: rank trace + per-event FLOP + cumulative ----
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                 gridspec_kw=dict(height_ratios=[1, 1.1]))
    a1.step(idx, rank, where='post', color='C0', lw=1.4, label='bounded resident rank $r_t$')
    a1.axhline(k, color='C3', ls='--', lw=1.4, label=f'clifft active $k={k}$ (held every step)')
    a1.fill_between(idx, rank, step='post', alpha=0.12, color='C0')
    a1.set_ylabel('magic rank'); a1.set_ylim(-0.5, k + 1)
    a1.axvline(peak_loc, color='0.5', ls=':', lw=1)
    a1.annotate('rank climbs as measured\nqubits enter register', xy=(peak_loc*0.45, peak*0.55),
                fontsize=8, color='0.35', ha='center')
    a1.annotate('measured-magic purge\npeels rank back down', xy=(peak_loc + (len(idx)-peak_loc)*0.45,
                peak*0.55), fontsize=8, color='0.35', ha='center')
    a1.legend(loc='lower center', fontsize=9, ncol=2)
    a1.set_title(f'{circ}  ({ax_label})   peak $r_{{max}}={peak}$,  '
                 f'clifft $k={k}$   |   FLOP: bounded {H(cum_b[-1] if len(cum_b) else 0)} '
                 f'vs clifft(modeled) {H(cum_c[-1] if len(cum_c) else 0)}', fontsize=10)
    # per-event FLOP as stems colored by rank
    sc = a2.scatter(idx, fb, c=rank, cmap='viridis', s=14, vmin=0, vmax=k, zorder=3)
    a2.vlines(idx, 0, fb, color='0.8', lw=0.5, zorder=1)
    a2.set_ylabel('per-event bounded FLOP'); a2.set_xlabel('dense-kernel event index (exec order)')
    cb = fig.colorbar(sc, ax=a2, pad=0.01); cb.set_label('rank at event')
    a2b = a2.twinx()
    a2b.plot(idx, cum_b, color='C0', lw=1.6, label='cum bounded (validated)')
    a2b.plot(idx, cum_c, color='C3', lw=1.6, ls='--', label='cum clifft (modeled)')
    a2b.set_ylabel('cumulative FLOP' + (' [log]' if logy else '')); a2b.legend(loc='upper left', fontsize=8)
    if logy:
        a2b.set_yscale('log'); a2.set_yscale('symlog')
    fig.tight_layout()
    p1 = f"{OUT}/flop_rank_trace_{circ}.png"; fig.savefig(p1, dpi=130); plt.close(fig)

    # ---- Figure 2: FLOP-by-rank histogram (bounded vs clifft-modeled) ----
    fig2, ax = plt.subplots(figsize=(8, 4.6))
    ranks = sorted(by_rank)
    bvals = [by_rank[r][0] for r in ranks]
    cvals = [by_rank[r][1] for r in ranks]
    x = np.arange(len(ranks)); w = 0.4
    ax.bar(x - w/2, bvals, w, color='C0', label='bounded (algorithmic)')
    ax.bar(x + w/2, cvals, w, color='C3', alpha=0.75, label='clifft (modeled, $2^k$)')
    ax.set_xticks(x); ax.set_xticklabels(ranks)
    ax.set_xlabel('resident magic rank $r$')
    ax.set_ylabel('FLOP performed at rank $r$' + (' [log]' if logy else ''))
    ax.set_title(f'{circ} ({ax_label}): where does the dense arithmetic happen?  '
                 f'($r_{{max}}={peak}$, clifft $k={k}$)')
    if logy: ax.set_yscale('log')
    ax.legend()
    for xi, b in zip(x, bvals):
        if b > 0: ax.text(xi - w/2, b, H(b), ha='center', va='bottom', fontsize=7, rotation=90)
    fig2.tight_layout()
    p2 = f"{OUT}/flop_by_rank_{circ}.png"; fig2.savefig(p2, dpi=130); plt.close(fig2)

    # ---- CSV per resident rank (the meaningful decomposition) ----
    p3 = f"{DAT}/per_rank_{circ}.csv"
    with open(p3, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["rank", "n_events", "flop_bounded", "flop_clifft_modeled", "win_ratio_cl_bn"])
        for rk in sorted(pr):
            fbe, fce, ne = pr[rk]
            wr.writerow([rk, ne, int(fbe), int(fce), f"{fce/fbe:.2f}" if fbe else ""])

    tot = sum(fb)
    return dict(circ=circ, ax=ax_label, k=k, peak=peak, tot_b=tot,
                tot_c=cum_c[-1] if len(cum_c) else 0, n_meas=prog.num_measurements,
                fb_peak=fb_peak, fc_peak=fc_peak, fb_sh=fb_sh, fc_sh=fc_sh,
                fb_tail=fb_tail, fc_tail=fc_tail, frac_peak=(fb_peak/tot if tot else 0),
                p1=p1, p2=p2, p3=p3, by_rank=by_rank, cap=cap)


CIRCS = [("coherent_ry_d3_r1", "R_Y"), ("coherent_ry_d3_r3", "R_Y"),
         ("coherent_rx_d3_r1", "R_X"), ("coherent_rx_d3_r3", "R_X"),
         ("coherent_d3_r3", "R_Z"), ("coherent_d5_r5", "R_Z"),
         ("cultivation_d3", "T"), ("cultivation_d5", "T"), ("distillation", "T")]
print("=== A. per-rank FLOP decomposition + rank-mountain curve ===")
print(f"{'circuit':18}{'ax':4}{'k':>3}{'peak':>5}{'#meas':>6}"
      f"{'bnd FLOP':>10}{'cl FLOP':>10}{'F_cl/F_bn':>10}")
print("    win-decomposition  [peak r= irreducible | shoulder r-1= ~2x | tail r<peak-1= localized]")
for c, ax in CIRCS:
    r = plot_circuit(c, ax)
    fr = r['tot_c'] / r['tot_b'] if r['tot_b'] else float('nan')
    print(f"\n{r['circ']:18}{r['ax']:4}{r['k']:>3}{r['peak']:>5}{r['n_meas']:>6}"
          f"{H(r['tot_b']):>10}{H(r['tot_c']):>10}{fr:>9.1f}x")
    def w(a, b): return f"{b/a:.1f}x" if a else "inf"
    print(f"    peak  r={r['peak']:>2}:  bnd {H(r['fb_peak']):>8}  cl {H(r['fc_peak']):>8}  -> {w(r['fb_peak'],r['fc_peak']):>5}  (irreducible full-state work)")
    print(f"    shldr r={r['peak']-1:>2}:  bnd {H(r['fb_sh']):>8}  cl {H(r['fc_sh']):>8}  -> {w(r['fb_sh'],r['fc_sh']):>5}  (slack-1 vectorized, ~2x)")
    print(f"    tail  r<{r['peak']-1:>2}:  bnd {H(r['fb_tail']):>8}  cl {H(r['fc_tail']):>8}  -> {w(r['fb_tail'],r['fc_tail']):>5}  (localize-and-drop: the win)")
    print(f"    -> {r['p1']}  |  {r['p2']}  |  {r['p3']}")
