# Per-rank FLOP mechanism — full validated benchmark set

**What this shows.** bounded's FLOP advantage comes from a *rank mountain*: non-Clifford effects are
delayed, dense kernels run only at the rank required by the current **magic-relevant** measurement
dependency, and the magic register is purged back down — rather than clifft holding its peak active
rank `2^k` throughout. The advantage decomposes per resident rank into three bands.

## Labels (kept distinct everywhere)
- **bounded FLOP = validated algorithmic FLOP** (stated convention; hook == direct-event meter exact
  at r=1..6 + real circuits + unit-calls; meter on/off → record & max_M bit-identical).
- **clifft FLOP = modeled** (compiled `_clifft_core.abi3.so`, not instrumentable; each clifft-shared
  event charged at the full `2^k`).
- **rank / state-volume = trace-derived** (from the actual resident-rank sequence).

## Fidelity (prerequisite — all validated)
R_Y d3_r1/r3 EXACT Born (per-meas 2.55e-15, joint 1.4e-13); R_X EXACT Born + cross-entropy NEW=OLD;
R_Z + T (cultivation/distillation) cz-fix no-op proof + prior trajectory-EXACT.

## The structure is one rise-and-fall mountain, not a per-measurement sawtooth
Most QEC measurements are **stabilizer** measurements handled by the Clifford frame — they do **not**
move the magic rank. The rank climbs `0→…→r_max` as magic-relevant axes enter the dense register,
peaks, then the measured-magic purge peels axes off one by one back to 0. Decomposition is therefore
by **rank**, not by measurement epoch.

## Per-rank win decomposition (bounded `2^r` vs clifft-modeled `2^k`, same event set)

| circuit | ax | k | r_max | **peak band** (r=r_max) | **shoulder** (r-1, ~2×) | **tail** (r<r_max-1) | **blended** | regime |
|---|---|--:|--:|--:|--:|--:|--:|---|
| coherent_d5_r5 | R_Z | 24 | **13** | **2048×** | 4090× | 27259× | **3307×** | r_max ≪ k |
| coherent_rx_d3_r1 | R_X | 14 | 11 | 8.0× | 15.4× | 95.0× | **33.9×** | r_max < k |
| coherent_rx_d3_r3 | R_X | 14 | 12 | 4.0× | 7.9× | 25.0× | **15.8×** | r_max < k |
| coherent_d3_r3 | R_Z | 8 | 5 | 8.0× | 15.8× | 44.0× | **13.4×** | r_max < k |
| coherent_ry_d3_r1 | R_Y | 16 | **16** | 0.9× | 1.9× | 17.5× | **4.3×** | r_max = k |
| coherent_ry_d3_r3 | R_Y | 16 | **16** | 1.0× | 2.0× | 14.4× | **3.8×** | r_max = k |
| distillation | T | 5 | 4 | 1.4× | 2.6× | 8.9× | **2.9×** | magic-saturated |
| cultivation_d5 | T | 10 | **10** | 1.0× | 2.0× | 16.3× | **2.0×** | magic-saturated |
| cultivation_d3 | T | 4 | **4** | 0.9× | 2.0× | 4.7× | **1.2×** | magic-saturated |

(maxM=0 circuits — coherent_d3_r1/d5_r1/d7_r1 R_Z, surface_d7_r7 — build no magic register: 0 FLOP.
Off-axis d5 R_X/R_Y k=38/47 and R_Z d7_r7 k=48 are INFEASIBLE >2^26.)

## Reading it — two effects, three regimes

The advantage is the sum of **two distinct effects**:
1. **peak-rank compression** — bounded's peak `2^{r_max}` is below clifft's `2^k`. Active iff `r_max < k`.
2. **dense-computation localization** — even below the peak, the rise/fall tail runs at `2^r ≪ 2^k`.

| regime | example | peak band | net |
|---|---|---|---|
| **r_max ≪ k** | d5_r5 (R_Z) | wins 2048× | both effects → **3307×** |
| **r_max < k** | rx_d3, d3_r3 (R_Z) | wins 4–8× | both → **13–34×** |
| **r_max = k** | ry_d3, cultivation | **wash (≈1×)** | localization only → **1.2–4.3×** |

So R_Y (r_max=k=16) gets **no peak-memory benefit** and no larger-circuit feasibility; its ~4× is
**pure computation localization** of the cheap tail. R_X / R_Z d5_r5 get **both**, hence the huge ratios.

## Why state-volume (Σ2^r) overstates the FLOP ratio
state-volume weights every step equally; FLOP is `Σ_i c_i 2^{r_i}` with per-amplitude kernel cost
`c_i`. The descent clusters many expensive off-diagonal rotations near the top of the mountain, so FLOP
is more top-heavy than step-count. For R_Y the top is pinned at clifft's own k → state-volume 10.7×
but FLOP 4.3×. state-volume is a good direction-of-effect proxy, not the FLOP reduction itself.

## Honest claim wording
> bounded delays non-Clifford effects and executes dense kernels only at the rank required by the
> current magic-relevant measurement dependency, producing a rank mountain rather than maintaining
> clifft's peak active rank `2^k` throughout. When `r_max < k` this compresses both peak memory and
> dense FLOPs (up to 3307× at d5_r5); when `r_max = k` (R_Y) peak memory is unchanged but the cheap
> low-rank majority is still localized, reducing dense FLOPs ~4×. clifft FLOP is a matched
> operation-count model (compiled core, not instrumented); `ms` is bounded-only and Python-bound —
> wall-clock speedup must be measured separately.

## Artifacts (per circuit, in `reports_clifft_axis_bounded_rxry/`)
`flop_rank_trace_<circ>.png` (rank mountain + cumulative FLOP), `flop_by_rank_<circ>.png` (per-rank
histogram; log-y when the gap >50×), `per_rank_<circ>.csv`. Full benchmark table: `flop_all.csv` /
`flop_all.py`. Production summary: `flop_production.py`. Generator: `flop_per_measurement.py`
(coeffs identical to the validated production hook). Run with `/home/jung/clifft_env/bin/python`.

Circuits covered: coherent_ry_d3_r1, coherent_ry_d3_r3, coherent_rx_d3_r1, coherent_rx_d3_r3,
coherent_d3_r3, coherent_d5_r5, cultivation_d3, cultivation_d5, distillation.
