# Why does bounded use less peak memory than Clifft yet MORE FLOP?

Measured, not estimated. Three runs per circuit under ONE FLOP convention (cmul=6, rcmul=2, cadd=2,
sqmag=4, vdot=8; compile-time matrix algebra and memcpy excluded): **A** Clifft fused, **B** Clifft
UNFUSED (`bytecode_passes=None`, the architecture-fair baseline — fusion is ruled out as a cause),
**C** bounded. Both backends bucketed into the SAME categories from their REAL kernel events
(Clifft via the C++ CostMeter, bounded via the `budget.charge` hook). Seed 1, 1 shot.

Data: `data/flop_attribution.csv`. Harness: `scripts/flop_attribution.py`.

## One-sentence conclusion

> bounded's peak memory shrinks (`max 2^r`), but total FLOP `= Σ_i c_i 2^{r_i} + F_aux` is governed by
> **per-amplitude cost `c_i` and event count**, not peak rank. bounded loses on FLOP because (1) its
> rotations are **off-diagonal full-array** sweeps (12·2^r) where Clifft's are **diagonal half-array**
> (3·2^r) — a ~4× per-rotation penalty, and (2) its measured-magic purge recomputes a **branch sqnorm
> at every rank of the descent** (~22–28 per measurement) where Clifft does one Born per measurement;
> it partially RECOVERS via frame-deferral (Cliffords cost 0 on the array). Only when the rank gap is
> exponential (d5_r5, 2^{11}) does localization swamp these overheads.

## The two backends have nearly DISJOINT category profiles (the root structural fact)

| | Clifft (unfused) | bounded |
|---|---|---|
| rotations | **diagonal** half-array `array_rot` (3·2^r) | **off-diagonal** full-array `rot:offdiag` (12·2^r) |
| Cliffords (H/S) | applied **on the array** (`array_gate`) | **deferred to the frame** (0 FLOP) |
| Born | `sqmag` inside one `meas_interfere`/measurement | standalone `sqnorm` × ~22–28 per measurement (purge descent) |
| projection | fold inside meas | strided-slice zero (0 FLOP) |
| normalization | O(1) into `gamma_` | O(1) |
| measured-magic purge | **none** | `purge:h/s` (W_M localization) |

## Per-circuit 100% attribution (clifft-unfused B vs bounded C)

### coherent_ry_d3_r1 — equal peak (16=16), bounded +9.66M (1.79×)
| cause | Δ FLOP | note |
|---|--:|---|
| off-diagonal full-array rotation | **+10.80M** | 57 rot each; bnd 14.92M (12·2^r) vs clf 4.12M (3·2^r), ~3.6× |
| measured-magic Born/purge | **+5.92M** | bnd sqnorm 5.65M (375 calls!) + purge 1.38M vs clf born+proj 1.11M |
| Clifford frame-deferral (bounded SAVES) | **−7.05M** | clf array_gate 7.05M (113 calls) vs bnd 0 |
| **net** | **+9.66M** | = +10.80 +5.92 −7.05 |

### cultivation_d5 — equal peak (10=10), bounded +608k (3.86×)
| cause | Δ FLOP | % of +608k |
|---|--:|--:|
| off-diagonal magic-gate (T as offdiag rot) | **+504k** | 83% — bnd offdiag 684k vs clf array_gate(T) 180k |
| measured-magic Born/purge | **+104k** | 17% — bnd sqnorm 118k + purge 18k vs clf born+proj 33k |
| **net** | **+608k** | 100% |
(No rank saving — peak=k. No Clifford-deferral offset — clf's gate work IS the T magic, not separable Cliffords.)

### coherent_rx_d3_r3 — **smaller peak (12<14) yet bounded +546k (1.21×)** ← THE counterexample
| cause | Δ FLOP | note |
|---|--:|---|
| rotation rank-localization (bounded SAVES) | **−385k** | bnd 1.025M (rank≤12) vs clf 1.41M (rank≤14): 2^2 rank gain > 4× offdiag penalty |
| Clifford frame-deferral (bounded SAVES) | **−140k** | clf array_gate 140k vs bnd 0 |
| measured-magic Born/purge (bounded LOSES) | **+1.07M** | bnd sqnorm 2.07M (**912 calls**) + purge 40k vs clf born+proj 1.04M |
| **net** | **+546k** | rank saved 525k, but sqnorm overhead 1070k exceeds it 2× |

> **coherent_rx_d3_r3: bounded shrank peak state ~4× (2^12 vs 2^14) but total FLOP grew 1.21×, because
> rank reduction saved 525k FLOP (rotations 385k + Cliffords 140k) yet the measured-magic sqnorm/purge
> added 1.07M FLOP — the auxiliary cost exceeds the rank saving by ~2×.**

### coherent_d5_r5 (R_Z) — positive control: smaller peak (13≪24), bounded 372× CHEAPER
| cause | Δ FLOP | note |
|---|--:|---|
| rotation rank-localization | **−16.89G** | bnd 22.71M (rank≤13) vs clf 16.91G (rank≤24): 2^11 gain swamps 4× offdiag penalty |
| array_gate + born localization | **−1.07G** | clf 1.07G vs bnd 0 / folded |
| measured-magic sqnorm/purge (bounded extra) | +25.6M | negligible vs 17.94G |
| **net** | **−17.94G** | exponential rank gain dominates all overheads |

## Peak memory vs total FLOP — the four numbers side by side

| circuit | peak rank (bnd/clf) | peak bytes (bnd/clf) | W1=Σ2^r (bnd/clf-unf) | total FLOP (bnd/clf-unf) |
|---|---|---|---|---|
| coherent_ry_d3_r1 | 16/16 | 1MB/1MB | (≤clf) /4.96M | 21.95M / **12.29M** |
| cultivation_d5 | 10/10 | 16KB/16KB | 143k / 295k | 821k / **213k** |
| coherent_rx_d3_r3 | **12**/14 | **64KB**/256KB | 1.23M / 1.59M | 3.13M / **2.59M** |
| coherent_d5_r5 | **13**/24 | **128KB**/256MB | **15.8M** / 18.1G | **48.3M** / 17.99G |

**Decoupling proven:** in every row bounded's **W1 (Σ2^r) is ≤ Clifft's** — bounded never touches more
total amplitude-volume. Yet its FLOP is higher (except d5_r5) because `c_i` (off-diagonal 12 vs
diagonal 3 = 4×) and the sqnorm event count are larger. **peak↓ and W1↓ do not imply FLOP↓.**

## Hypotheses — confirmed / rejected (measured)

| hypothesis | verdict | evidence |
|---|---|---|
| A. bounded re-traverses more state (Σ2^r larger) | **REJECTED** | bounded W1 ≤ Clifft W1 in ALL circuits (e.g. cult_d5 143k<295k) |
| B. off-diagonal rotation penalty | **CONFIRMED, dominant** | bnd rot 12·2^r vs clf 3·2^r = ~4× (offdiag×2 + full-vs-half-array×2); ry_d3_r1 +10.8M |
| C. repeated Born/sqnorm | **CONFIRMED, major** | 375 sqnorm/17meas (ry), 912/33 (rx_d3_r3) ≈ 22–28× per measurement, spread over the rank mountain |
| D. purge/demotion | **CONFIRMED, MINOR** | purge:h/s = 1.38M/21.95M=6% (ry), 18k (cult), 40k (rx_d3_r3) — small |
| E. duplicate rotation flush | **REJECTED** | rotation count matches (ry: 57 bnd == 57 clf); each non-Clifford applied once |
| F. transient promote/drop cost | **REJECTED (FLOP)** | promote/drop = 0 arithmetic FLOP (memory traffic only) |
| G. accounting mismatch | **CONTROLLED** | same convention; diagonal=half-array, offdiag=full-array verified in both; norm O(1) both; compile-time & memcpy excluded |
| "fusion is the cause" | **REJECTED** | Clifft UNFUSED is the baseline and is still ≤ bounded; fusion *raises* Clifft FLOP |

## Optimization priority (by measured recoverable FLOP)

| optimization | recoverable FLOP (ry_d3_r1) | % of bounded | correctness risk | difficulty |
|---|--:|--:|---|---|
| keep rotations **diagonal half-array** (avoid frame→off-axis; skip \|0> half) | up to **+10.8M→~+0** | ~49% | medium (must re-derive frame so rotation axis stays Z) | high |
| amortize **measured-magic sqnorm** (one Born/measurement, not per-rank) | up to **~5M** | ~23% | medium (purge must not re-probe each rank) | medium |
| (already have) **rank localization** | — | the only real win (d5_r5 372×) | none | done |
| Born/projection fusion (one pass) | small | <5% | low | low |
| bounded-aware U2/U4 fusion | **negative** (raises FLOP) | — | — | not worth it for FLOP |

The two genuine FLOP levers are **diagonal rotations** and **sqnorm amortization**; fusion is NOT a
FLOP lever (it trades +FLOP for −memory-passes).

## Code locations

| cost | file:line | current behavior | fix candidate |
|---|---|---|---|
| off-diagonal rotation | `clifft_axis/engine.py:213` `_pauli_lincomb_inplace(...,":offdiag")` charge `8*CH` butterfly | frame conjugation puts rot axis on X/Y → 12·2^r full-array | keep frame so pending stays Z-diagonal → 3·2^r half-array (the clifft form) |
| diagonal rotation (full-array) | `engine.py:266` `_pauli_lincomb_inplace(...,"rot")` | two-multiplier scale over ALL 2^r | skip the \|0> subspace (process 2^{r-1}) |
| repeated sqnorm | `engine.py:138-146` `_branch_sqnorm` (charge `phi.size,"sqnorm"`), called from `bounded.py:125-126,235` + `engine.py:401-402` during the purge descent | ~22–28 branch-norms per measurement, one at each rank | compute Born once at the measurement rank; reuse across the descent |
| purge (minor) | `engine.py` `_h_axis`/`_s_axis` (`purge:h/s`) | W_M Clifford localization per measured axis | leave (small); only matters if rotation+sqnorm fixed |
| Clifford deferral (bounded's WIN) | frame ops in `lazy.py` (0 FLOP) | H/S folded into PauliFrame, never on the array | keep — this is why bounded beats clifft-unfused on Clifford work |

Clifft-side instrumentation: `/home/jung/clifft/src/clifft/util/cost_meter.{h,cc}` + `svm_kernels.inl`
records + `bindings.cc` (`cost_meter_*`). Non-invasive (meter off==on samples) and internally
consistent (exact primitive multiples) — verified for all circuits.
