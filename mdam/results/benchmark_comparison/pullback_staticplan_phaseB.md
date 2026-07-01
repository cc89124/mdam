# Pullback → StaticPlan Phase B: IMPLEMENTED, bit-exact, but wall-NEUTRAL — and WHY

**TL;DR.** Phase B (replace the live pullback `F†PF` substitution with `out_mask = StaticTable[key]`,
`phase = c_static + Σ ax/az phase mod 4`) is implemented, **bit-exact (20 000 shots × 4 benches, shadow
mask+phase violations 0)**, and the affine model is genuinely **2.85× cheaper per call** than the live
substitution. But the **wall is neutral (±0.2 %)** on every benchmark. Calibrated direct measurement
(rdtsc @ 3.70 GHz) shows why: **the live substitution is only 0.4–1.1 % of wall.** The earlier
`wall_breakdown.md` figure of "pullback ≈ 26 %" was a **PROFILE-build ISKIP artifact** (the per-op timer
inflated the region 11–30×). The real reducible lever in this region is the **inverse-frame REBUILD**
(`rebuild_inverse_frame`), which is **30 % of cult_d3 wall and 15 % of distillation wall** — and Phase B
does not touch it.

## What was built (default OFF; authoritative path unchanged)

- `native_magic_state.hpp`: `PbStaticEnt` (static out-mask, `c_static` residual, support index lists),
  `pb_lookup()` with a **direct-mapped cache** (4096 slots) in front of an `unordered_map` (the per-call
  hash/bucket/node-chase was the bottleneck; element pointers are rehash-stable, never erased).
- Fast path inside `NativeDenseEngineState::pullback()`: lookup → static mask → affine mod-4 phase.
  Shadow mode verifies static-vs-live every call; Phase-A mode (static mask, live phase) for ablation.
- API: `nvm_pb_static / _shadow / _phase` (the three flags), `_reset`, `_stats`, `_shadow_fail`.
  Timing: `nvm_pb_time` + `nvm_rdtsc` (rebuild/subst/lookup/affine cycle split).

## Correctness (the gate)

`baseline (static OFF)` vs `static-PB (affine phase) + shadow`, seeds 1/7/42/123/999 × 4000 shots:

| bench | shots | record mism | shadow mask viol | shadow phase viol | keys | result |
|---|---:|---:|---:|---:|---:|:--:|
| cultivation_d3 | 20 000 | 0 | 0 | 0 | 31 | PASS |
| cultivation_d5 | 20 000 | 0 | 0 | 0 | 91 | PASS |
| distillation | 20 000 | 0 | 0 | 0 | 15 | PASS |
| coherent_d5_r5 | 20 000 | 0 | 0 | 0 | 341 | PASS |

The Phase-B premise (Step 1 mask-static + affine mod-4 phase) is not just provable — it's **bit-exact in
practice**.

## Per-call: the affine model IS cheaper than the substitution

Direct rdtsc, cult_d5 (the subst-heavy case):

| | cyc/call |
|---|---:|
| live substitution `inverse_frame.pullback` | 90 |
| static: direct-mapped lookup | 47 |
| static: mask-copy + affine phase sum | 32 |
| **static total (lookup+affine)** | **78** |
| **affine alone (if lookup were free)** | **32 → 2.85× cheaper than subst** |

The math delivered. The direct-mapped cache brought lookup from 97→47 cyc (the 40-byte key hash still
dominates), making static (78) finally < subst (90). So Phase B is a per-call win **and** wall-neutral —
because the region it lives in is tiny.

## The calibrated truth (rdtsc @ 3.70 GHz, ns/cyc = 0.2706) — this corrects wall_breakdown.md

| bench | wall ns/shot | REBUILD ns (×/shot) | REBUILD % wall | SUBST ns (×/shot) | **SUBST % wall** | pullback total % |
|---|---:|---:|---:|---:|---:|---:|
| cultivation_d3 | 14 715 | 4 444 (5.00) | **30.2 %** | 166 (34.0) | **1.1 %** | 31.3 % |
| cultivation_d5 | 196 058 | 187 (0.02) | 0.1 % | 1 722 (106.0) | **0.9 %** | 1.0 % |
| distillation | 17 146 | 2 485 (4.00) | **14.5 %** | 67 (14.8) | **0.4 %** | 14.9 % |

- **Substitution (what Phase B replaces) = 0.4–1.1 % of wall.** Perfect elimination saves < 1 %. Confirmed
  neutral wall: cult_d3 0.999×, cult_d5 1.001×, distillation 1.002×.
- **The "26 % pullback" in wall_breakdown.md was a PROFILE artifact.** The PROFILE build's PLAN_PULLBACK
  region read 36 497 ns/shot for cult_d5; direct rdtsc says the substitution is 1 722 ns — an 11× ISKIP
  inflation. Trust the calibrated rdtsc number, not the PROFILE region, for "% of wall".

## The real lever (redirect): the inverse-frame REBUILD

`rebuild_inverse_frame` = `build_inverse_basis` (O(n²) GF(2) elimination) + 2n × `pullback_from_basis`.
It fires once per magic boundary (cult_d3 5×/shot, distillation 4×/shot) at ~3285 / 2296 cyc each:
**30.2 % of cult_d3 wall, 14.5 % of distillation wall.** Phase B does not touch it.

Gate J already established the inverse-frame **masks are shot-static; only `phase_pack` is the dynamic
carried state** (`reconstruct_inverse` from static masks + carried phases). So the rebuild is **recomputing
shot-static masks every boundary** — a StaticPlan for the rebuild (cache masks per boundary, recompute only
phases) could remove most of that 30 %/15 %. **Caveat:** this is adjacent to the forbidden F4/imem
plan_cache zone and must be a clean redesign (à la Gate M), not a revival of the buggy cache.

cult_d5 and coherent_d5_r5 are **dense-bound** (pullback total ~1 %); the 2^r lincomb dense apply
dominates. Neither rebuild nor subst caching helps them — the ~1.19× dense floor stands.

## Status

- Phase B: **DONE, correct, default-OFF, wall-neutral.** Kept as a verified-correct capability + the
  rebuild/subst timing instrumentation needed for the next step.
- Next lever (pending direction): **inverse-frame rebuild StaticPlan** — the real 30 %/15 % prize for the
  control-plane-bound benches (cult_d3, distillation). NOT the substitution.
