# FLOP comparison — MDAM (bounded) vs Clifft

**Scope: PER-SHOT FLOP, single shot, NO cross-shot reuse/dedup/caching.**
This table is the cost of **one** shot. It is **not** a 1M-shot run.
- **MDAM** = mean over 10 seeds (magic firing is stochastic, so one shot is not representative).
- **Clifft** = deterministic (its `active_k_history` schedule is outcome-independent → same every shot).
- **No deduplication, no cross-shot caching on either side.** Each shot is counted as if computed from
  scratch. Naive 1M-shot total = per-shot × 10⁶; the `Clifft / MDAM` ratio is shot-count-independent.
- The **"1M shots"** label belongs **only** to the wall-time table ([`wall_table.md`](wall_table.md)),
  which uses the **native VM** with Gate-K **cross-shot edge caching** (≈99% hit on cultivation_d3).
  That cache reuse is **NOT** reflected here — so this per-shot FLOP table is the *conservative*
  (no-reuse) view; amortizing repeated transitions over 1M shots would favor MDAM further.

All benchmarks except `coherent_d7_*` (excluded per request).
Code: [`mdam/bench/flop_compare.py`](../../mdam/bench/flop_compare.py) · raw: [`flop_table.csv`](flop_table.csv).

> **CORRECTION (this supersedes an earlier version).** The earlier table modeled `clifft FLOP = (MDAM
> dense-event count) × 2^k`. That is wrong: it makes Clifft's FLOP follow MDAM's schedule, so whenever MDAM
> did 0 dense work it reported Clifft = 0 too — which is false (Clifft still evolves its dense register).
> **Now each engine is counted from its OWN real schedule:** MDAM from its actual dynamic events (incl. the
> fused measurement core), Clifft from its own `active_k_history` dense-op schedule (deterministic).

| circuit | axis | k | maxM (MDAM) | MDAM FLOP/shot | Clifft FLOP/shot | **Clifft / MDAM** | note |
|---|---|--:|--:|--:|--:|--:|---|
| coherent_d5_r5 | R_Z | 24 | 12 | 137.5 M | 69.26 G | **503.7×** | MDAM wins big (localization) |
| coherent_d3_r1 | R_Z | 5 | 0 | **0** | 8.51 k | **∞** | MDAM skips dense entirely; Clifft doesn't |
| coherent_d5_r1 | R_Z | 13 | 0 | **0** | 6.49 M | **∞** | MDAM skips dense entirely; Clifft doesn't |
| coherent_d3_r3 | R_Z | 8 | 4 | 50.52 k | 190.5 k | **3.8×** | MDAM wins |
| coherent_rx_d3_r1 | R_X | 14 | 10 | 780.8 k | 2.58 M | **3.3×** | MDAM wins |
| coherent_ry_d3_r1 | R_Y | 16 | 16 | 11.74 M | 20.80 M | **1.8×** | near parity (peak band is a wash) |
| coherent_ry_d3_r3 | R_Y | 16 | 16 | 56.80 M | 62.34 M | **1.1×** | near parity |
| cultivation_d3 | T | 4 | 3 | 4.22 k | 3.79 k | **0.9×** | MDAM uses *more* FLOP (see note) |
| coherent_rx_d3_r3 | R_X | 14 | 11 | 18.76 M | 7.69 M | **0.4×** | MDAM uses *more* FLOP |
| distillation | T | 5 | 3 | 9.68 k | 4.80 k | **0.5×** | MDAM uses *more* FLOP |
| cultivation_d5 | T | 10 | 9 | 6.51 M | 436.0 k | **0.07×** | MDAM uses *more* FLOP |
| surface_d7_r7 | R_Z | 0 | 0 | 0 | 0 | — | pure Clifford, both 0 |
| coherent_rx_d5_r1 | R_X | 38 | — | cannot run | 82.1 T | — | both need 2³⁸ memory |
| coherent_rx_d5_r5 | R_X | 38 | — | cannot run | 410.6 T | — | both need 2³⁸ memory |
| coherent_ry_d5_r1 | R_Y | 47 | — | cannot run | 79665 T | — | both need 2⁴⁷ memory |
| coherent_ry_d5_r5 | R_Y | 47 | — | cannot run | 398323 T | — | both need 2⁴⁷ memory |

`Clifft / MDAM` > 1 ⇒ MDAM does fewer FLOP (MDAM wins). < 1 ⇒ MDAM does **more** FLOP.

## What the corrected numbers actually say
- **R_Z high-rank (coherent_d5_r5, k=24): 503× real FLOP win.** Clifft materializes the full `2^24` register;
  MDAM localizes to peak `2^12`. This is MDAM's genuine advantage regime.
- **r1 circuits (coherent_d3_r1, coherent_d5_r1): MDAM does literally 0 dense FLOP; Clifft does real work
  (8.5 k / 6.5 M).** Clifft's `active_k_history` is an outcome-independent compiled schedule, so it evolves the
  dense register regardless; MDAM's lazy, measurement-driven materialization skips it entirely on these
  trajectories (`maxM = 0`). This is the **opposite** of the earlier "both 0" — MDAM wins decisively here.
- **R_Y (coherent_ry): ~1.1–1.8× (near parity).** Off-axis magic keeps the active rank at the peak, so there
  is little to localize.
- **Magic / T-saturated (cultivation_d3/d5, distillation) and R_X r3: MDAM does MORE FLOP (ratio < 1).** This is
  expected and matches the prior **Gate G** finding ("cultivation is a structural lose case for MDAM"): for
  magic-saturated circuits the measurements probe the magic register directly via MDAM's exact Pauli-sum core
  (`n_U · 2^{r_out}`), which costs more arithmetic than Clifft's active-register evolution. **MDAM's advantage
  on these circuits is MEMORY (bounded peak rank), not FLOP.**
- **off-axis d5 (R_X/R_Y, k=38/47): neither engine can run** (would need 2³⁸–2⁴⁷ ≈ 4 TB–2 PB). Clifft's
  scheduled FLOP is shown (astronomical) to illustrate; it also cannot fit in memory.

## Methodology & honest caveats
- **MDAM FLOP** = real dynamic events from a run (charge-hook rotation/collapse at the resident rank `2^r`)
  **plus the fused measurement core** `n_U · 2^{r_out}` from the engine's own `_fused_log`. Mean over 10 seeds.
- **Clifft FLOP** = `Σ_i C(op_i) · 2^{active_k_history[i]}` over Clifft's dense active-register ops
  (`ARRAY_*`, `EXPAND_*`, `MEAS_ACTIVE_*`, `SWAP_MEAS_INTERFERE`). Deterministic (outcome-independent).
  Frame/Clifford/dormant ops touch only the symbolic stabilizer backbone → 0 dense FLOP on both sides.
- **Per-element FLOP convention** (both sides): off-diagonal complex op = 12, diagonal phase = 6, permutation
  (CNOT/SWAP) = 0, measurement collapse = 12. The ratio is robust to this constant where the two engines do
  the *same kind* of op; for the magic circuits the engines use *different* measurement mechanisms (MDAM's
  Pauli-sum vs Clifft's active-register collapse), so the magic ratios are convention-sensitive — read them as
  "MDAM trades more FLOP for less memory," not as a precise factor.
- Clifft's core is a compiled `.so`; its per-op FLOP is still assigned by the convention above, not measured
  from the binary. What IS measured directly from Clifft is its **active-rank schedule** (`active_k_history`).
