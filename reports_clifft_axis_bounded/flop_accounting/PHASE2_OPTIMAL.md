# Phase 2B — optimal path activation + verification (sections 1–4)

Activated: `clifft_axis` engines set `_inv_enabled=True` (global `NearClifford` default stays
`False`); bounded localization switched to 1-H frame-fold (`_loc_undo=False`). Both fallbacks
preserved (`_pullback_via_basis`, `_loc_undo=True`).

## §2 warmed-shot integrity (phase2_warmshot_inv.py) — PASS
- `_inv_enabled` stays `True` on every warmed shot; fresh engine per shot re-inits Ax/Az to the
  identity images, counters zero. Global `NearClifford` default confirmed `False`.
- Per-shot full `_pullback_basis` rebuilds are STABLE (not growing): ry_d3_r1=0, cultivation_d5=0,
  distillation=0/1, rx_d3_r3=3, d5_r5=1. The **only** trigger of a full rebuild is line 417 inside
  `_ag_measure` (stabilizer-measurement AG-projection) — proven by grep (sole `_inv_dirty=True`
  site). Frame-fold-induced recompute = **0**.

## §3 correctness (phase2_correctness.py + phase1_verify.py) — ALL PASS
- SHADOW verify (incremental inverse-frame vs GF(2) basis, every pullback, frame-fold mode):
  **0 mismatch on all 9 circuits**.
- 3-mode bit-exact (P1 butterfly / P2-undo / P2-fold+inv): records + peak rank **bit-identical**;
  `max|Δp0| ≤ 1.7e-15` (machine-epsilon FP reassociation, not the R_Y-style frame discrepancy);
  memory bound held (peak ≤ cap) every circuit; rotation-once held; residual-product clean.
- Phase-1 suite (NEW vs reconstructed pre-Phase-1 measure_z): **ALL EXACT** — records/rank/p0
  match, purge invariant OK. Optimal-path activation did not disturb the committed Phase-1 baseline.
- Regression coverage: RY sign (coherent_ry_*), CZ conj (cultivation_*), i^p phase (pp checked in
  every shadow pullback tuple), measured-axis demotion (AG-measure rebuild path, shadow-checked
  post-projection), residual product (distillation) — all inside the above.

## §4 performance — 5 circuits × modes (phase2_perf_5circ.txt)

`pb_recmp` = expensive O(n²) GF(2) recomputes actually run (inv OFF: cache misses; inv ON:
AG-measure lazy rebuilds). CNOT/SWAP = 0 FLOP but counted as sweeps + perms + traffic.

| circuit | mode | FLOP | wall(ms) | sweeps | traffic | perms | pb_recmp | peakK |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ry_d3_r1 | clifft-unfused | 12.29M | 2.16 | 217 | 8.09M | 13 | 0 | 16 |
| | P1 | 16.90M | 212.6 | 207 | 4.30M | 54 | 0 | 16 |
| | P2undo | 18.41M | 124.0 | 549 | 25.84M | 278 | 0 | 16 |
| | P2fold-noI | 12.91M | 90.6 | 293 | 9.32M | 81 | **51** | 16 |
| | **P2fold+inv** | **12.91M** | **96.3** | 293 | 9.32M | 81 | **0** | 16 |
| ry_d3_r3 | clifft-unfused | 36.55M | 5.50 | 579 | 24.11M | 34 | 0 | 16 |
| | P1 | 48.04M | 648.7 | 419 | 10.96M | 106 | 0 | 16 |
| | P2undo | 50.47M | 542.7 | 1009 | 46.22M | 488 | 0 | 16 |
| | P2fold-noI | 40.26M | 364.7 | 731 | 25.68M | 221 | 118 | 16 |
| | **P2fold+inv** | **40.26M** | **355.5** | 731 | 25.68M | 221 | **0** | 16 |
| rx_d3_r3 | clifft-unfused | 2.59M | 0.44 | 340 | 2.71M | 133 | 0 | 14 |
| | P1 | 1.31M | 25.4 | 318 | 431k | 85 | 3 | 12 |
| | **P2fold+inv** | **1.30M** | **25.6** | 330 | 530k | 93 | 3 | 12 |
| d5_r5 | clifft-unfused | 17.99G | 10463.7 | 1554 | 30.60G | 817 | 0 | 24 |
| | P1 | 25.10M | 198.9 | 822 | 6.47M | 168 | 1 | 13 |
| | P2undo | 23.43M | 188.6 | 1668 | 17.40M | 492 | 1 | 13 |
| | P2fold-noI | 20.75M | 335.3 | 1417 | 14.87M | 467 | **220** | 13 |
| | **P2fold+inv** | **20.75M** | **225.5** | 1417 | 14.87M | 467 | **1** | 13 |
| cultivation_d5 | clifft-unfused | 212.82k | 0.11 | 425 | 530k | 292 | 0 | 10 |
| | P1 / **P2fold+inv** | **727.70k** | 106 | 213 | 172k | 58 | 0 | 10 |

### Targets
- **ry_d3_r1 — 1-H FLOP gain retained: YES.** P2fold+inv 12.91M = 1.05× clifft-unfused (P1 was
  1.38×; P2undo 18.41M = 2-H). FLOP −24% vs P1, wall −55% vs P1.
- **ry_d3_r3 — FLOP < clifft-unfused?** P2fold+inv 40.26M = 1.10× unfused (P1 1.31×). Closes most
  of the gap but still 10% above unfused (the irreducible 1 extra H per off-diagonal rotation).
  Wall −45% vs P1; FLOP −16% vs P1.
- **d5_r5 — pullback-recompute regression eliminated: YES.** The documented root cause (frame-fold
  right_* → O(n²) `_pullback_basis` recompute, 94% of overhead) is gone: recompute **220 → 1**,
  wall **335 → 225 ms (−33%)**. FLOP −17% vs P1. *Residual:* P2fold+inv 225 ms is still +13% over
  P1 (199 ms) and +20% over P2undo (189 ms) — NOT pullback (now 1), but the localizer's extra
  strided sweeps + the O(n) incremental inverse-frame update at n=72. (Irrelevant to clifft: bounded
  beats clifft-unfused 46× on wall, 867× on FLOP here.)
- **rx_d3_r3 — win retained: YES.** 1.30M = 0.50× clifft-unfused, wall ≈ P1 (tie).
- **cultivation_d5 — FLOP unchanged by Phase 2** (localizer not triggered: T is diagonal R_Z, no
  off-diagonal X-rotation to localize). Gap = the diagonal-T accounting (727.70k vs 212.82k = 3.4×
  FLOP) attributed in Phase 2A — but bounded touches 1/3 the memory (172k vs 530k traffic, 213 vs
  425 sweeps). Off-diagonal localizer cannot address it.

### Net (at the initial 2^12 gate)
Frame-fold + inverse-frame is **FLOP-optimal on all 5** and **wall-best on the off-diagonal-rotation
regime (RY d3, the localizer's purpose)**. On d5_r5 (n=72) it wins FLOP (−17%) but P2undo is
wall-fastest (−16% over P1) — a FLOP↔wall tradeoff from the localizer's strided-sweep / O(n)
inverse-frame Python cost at large n, with the recompute regression itself fully eliminated.

## §4.1 gate tuning (phase2_gate_sweep.py) — `_loc_min_size` 2^12 → 2^14

The localizer's wall cost is driven by **n** (the O(n) inverse-frame update per fold), but the gate
keys on **rank/phi.size**. The sweep showed: d5_r5 (peak rank 13, phi.size ≤ 2^13) needs threshold
≥ 2^14 to disengage; ry_d3_r1's localizations are all at rank ≥ 14 so 2^14 keeps its full FLOP win;
ry_d3_r3 loses its rank-12/13 localizations (40.26M → 43.59M) but still beats P1. **`_loc_min_size`
set to 2^14** — disengage d5_r5's large-n/low-rank regime (wall recovered), keep the RY rank-16
regime. Both fallbacks preserved. Correctness re-verified at 2^14: ALL PASS (d5_r5 now `|Δp0|=0`,
localizer off → identical to butterfly).

Production 5-circuit (2^14 gate, P2fold+inv):
| circuit | FLOP | wall(ms) | vs clifft-unf | pb_recmp | note |
|---|---:|---:|---:|---:|---|
| ry_d3_r1 | 12.85M | 91.8 | 1.05× | 0 | 1-H FLOP win retained |
| ry_d3_r3 | 43.59M | 483 | 1.19× | 0 | < P1 48.04M, wall −25% |
| rx_d3_r3 | 1.31M | 26.3 | 0.51× | 3 | win retained |
| d5_r5 | 25.10M | 194 | 716× less | 1 | localizer gated off; inv-frame still −15% on butterfly (228→194ms, recmp 60→1) |
| cultivation_d5 | 727.70k | 103 | 3.42× | 0 | diagonal-T accounting (Phase 2A); not localizable |

## §5 final 9-circuit table (phase2_final_table.py, production 2^14 gate)

| circuit | clf-fused | clf-unfused | bnd-orig | Phase 1 | Phase 2 FLOP | Phase 2 wall | peak b/c | P2<unf |
|---|---:|---:|---:|---:|---:|---:|---:|:--:|
| coherent_ry_d3_r1 | 24.40M | 12.29M | 22.16M | 16.90M | 12.85M | 92.2ms | 16/16 | no (1.05×) |
| coherent_ry_d3_r3 | 72.66M | 36.55M | 64.65M | 48.04M | 43.59M | 469.1ms | 16/16 | no (1.19×) |
| cultivation_d3 | 1.78k | 1.80k | 6.17k | 5.47k | 5.47k | 5.4ms | 4/4 | no (3.04×) |
| cultivation_d5 | 211.15k | 212.82k | 833.59k | 727.70k | 727.70k | 103.3ms | 10/10 | no (3.42×) |
| coherent_rx_d3_r1 | 837.61k | 870.38k | 516.89k | 248.63k | 248.63k | 7.8ms | 11/14 | **YES** 0.29× |
| coherent_rx_d3_r3 | 2.49M | 2.59M | 3.27M | 1.31M | 1.31M | 26.3ms | 12/14 | **YES** 0.51× |
| coherent_d3_r3 | 59.28k | 52.56k | 21.41k | 15.50k | 15.50k | 8.9ms | 5/8 | **YES** 0.29× |
| coherent_d5_r5 | 18.93G | 17.99G | 49.61M | 25.10M | 25.10M | 190.4ms | 13/24 | **YES** 716× |
| distillation | 1.88k | 1.90k | 1.57k | 1.27k | 1.27k | 12.9ms | 4/5 | **YES** 0.67× |

**Aggregates:**
- Phase 2 beats Clifft-unfused: **5/9** circuits. The 4 that don't: 2 RY (1.05× / 1.19× — the
  irreducible 1-H-per-off-diagonal-rotation cost, matching the Phase-2A parity finding) and 2
  cultivation (3.0× / 3.4× — diagonal-T accounting, not addressable by an off-diagonal localizer;
  bounded still touches ⅓ the memory traffic).
- **Phase 2 FLOP vs Phase 1: −9.2%** (83.85M / 92.35M, sum). **wall vs Phase 1: −25.7%**.
- **Peak resident-rank UNCHANGED on all 9** (Phase-1 memory bound invariant held).
- Pullback FULL recomputes (Phase 2, all 9): **8 total — ALL from AG-measure lazy rebuild
  (stabilizer-projection, irreducible; clifft pays the same Gottesman-Knill projection).
  Frame-fold-induced recompute = 0** (the d5_r5 regression cause). Inverse-frame: 3412 updates,
  1145 O(1) lookups.

## §6 commit-gate assessment
| gate | status |
|---|---|
| 전체 exactness PASS | ✅ correctness ALL PASS + Phase-1 suite ALL EXACT |
| shadow mismatch 0 | ✅ all 9 circuits, every pullback |
| full pullback rebuild 정상 경로 0회 | ✅ frame-fold/Clifford-evolution-induced = **0**; the 8 total are the design's accepted AG-projection lazy rebuild (no incremental rule, rebuilds ONCE per stabilizer measurement) |
| d5_r5 wall 회귀 제거 | ✅ 194 ms (≈ P1); inverse-frame even −15% vs pre-inv butterfly (228→194), recmp 60→1 |
| RY 1-H FLOP 이득 유지 | ✅ ry_d3_r1 12.85M = 1.05× clifft-unfused |
| Phase 1 memory bound 불변 | ✅ peak rank UNCHANGED all 9 |
| fallback 경로 보존 | ✅ `_pullback_via_basis` + `_loc_undo=True` both retained |
