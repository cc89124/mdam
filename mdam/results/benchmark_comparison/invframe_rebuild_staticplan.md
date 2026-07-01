# Inverse-frame REBUILD StaticPlan (clean-room) — the real control-plane lever

**TL;DR.** After Phase B (pullback substitution) proved wall-neutral — the substitution was only ~1 % of
wall, the "26 %" was a PROFILE artifact — calibrated rdtsc pointed at the real lever: the **inverse-frame
rebuild** (`rebuild_inverse_frame`), 30.2 % of cult_d3 wall and 14.5 % of distillation wall. This is a
**clean-room** StaticPlan for that rebuild (separate file `native_invframe_static.hpp`, separate `rb_*`
flags, separate checker; NOTHING from F4/imem/old plan_cache). Result: **bit-exact** (shadow + fast,
record mismatch 0 all benches) and a **real wall win**: cult_d3 7.00×→**5.09×** (−27.3 %), distillation
1.62×→**1.41×** (−12.5 %); cult_d5 neutral (its rebuild is 0.1 % of wall).

## Why this is the right target (recap from pullback_staticplan_phaseB.md)

`rebuild_inverse_frame` = `build_inverse_basis` (O(n²) GF(2) elimination) + 2n × `pullback_from_basis`.
Calibrated rdtsc @ 3.70 GHz: 30.2 % of cult_d3 wall, 14.5 % of distillation wall. The pullback
substitution Phase B optimized was ~1 %. So the rebuild is the prize.

## The premise (from the rebuild math) and the KEY insight

The rebuild output MASKS depend only on the tableau MASKS (`build_inverse_basis`/`pullback_from_basis`
read getx/getz); the output PHASE is `out_phase = c_static[g] − Σ_{j∈coeff(g)} tableau_phase[j] (mod 4)`
(from `res.phase = R.phase − Q.phase`, Q carrying the tableau generator phases).

**KEY: the right key is the tableau-MASK signature, NOT a frame-epoch index.** A first attempt keyed by
rebuild-index failed (cult_d3 passed, but distillation/cult_d5/d5_r5 had mask violations — their rebuild
count or tableau-mask state at a given index varies across shots, because measurement-dependent Clifford
branching changes the tableau masks, not just the signs). Re-keying by a content signature of the tableau
masks (O(n) FNV, masks only) makes the output a deterministic pure function of the key.

## Step 1+2 — de-risk gate (key = tableau-mask signature), seeds 1/7/42/123/999 × 2500

| bench | rebuilds/shot | distinct sigs | mask viol | phase-affine viol | eligible | saturation |
|---|---:|---:|---:|---:|:--:|:--:|
| cultivation_d3 | 5 | 5 | 0 | 0 | YES | saturated |
| cultivation_d5 | 1 | 16 | 0 | 0 | YES | saturated |
| distillation | 4 | 5 | 0 | 0 | YES | saturated |
| coherent_d5_r5 | 12 | 15 | 0 | 0 | YES | saturated |

mask viol 0 ⇒ the signature is a SUFFICIENT key (output mask = pure fn of tableau masks). phase-affine
viol 0 ⇒ phase is exactly `c_static − Σ coeff·tableau_phase (mod 4)`. distinct sigs SATURATE (5–16,
didn't grow 1→5 seeds) ⇒ a tiny, memory-bounded plan set. The inverse-frame masks have FAR fewer distinct
states than the full measurement-boundary state (cult_d5 had 95 k boundary qkeys in Gate M but only 16
mask signatures here — masks-only + no magic register + no phases collapses the state space).

## Implementation (clean-room, default OFF, authoritative path unchanged)

- `native_invframe_static.hpp`: `RbGenPlan` (static out-mask + `c_static` + coeff support), `RbPlan`
  (per-generator ax/az), `rb_plan_map` (sig → plan, unordered_map, pointer-stable, never erased),
  flags `rb_static_on`/`rb_static_shadow`, shadow-fail record.
- `rebuild_inverse_frame`: HIT (sig in map, built) → skip `build_inverse_basis` + all `pullback_from_basis`;
  fill each generator from the cached mask + affine phase. MISS → live rebuild + capture plan. SHADOW →
  live rebuild + verify the plan vs live for every generator (mask+phase), abort-record on mismatch.
- C API `nvm_rb_static`/`_shadow`/`_reset`/`_stats`/`_shadow_fail`; checker `nvm_rb_cap`/`_stats`/`_count_hist`.
- Shares NOTHING with F4/imem/old plan_cache. The de-risk checker + shadow are the verifiers; performance
  judged ONLY by calibrated rdtsc clean wall (never PROFILE %-of-wall, never cmode-vs-cmode).

## Correctness — seeds 1/7/42/123/999 (the gate)

baseline (rb OFF) vs static-invframe; both SHADOW (live+verify, no skip) and FAST (skip rebuild):

| bench | shots | shadow rec mism | shadow fail | FAST rec mism | plans | hit% |
|---|---:|---:|---:|---:|---:|---:|
| cultivation_d3 | 20 000 | 0 | 0 | 0 | 5 | 100.0 % |
| cultivation_d5 | 7 500 | 0 | 0 | 0 | 16 | 92.5 % |
| distillation | 20 000 | 0 | 0 | 0 | 5 | 100.0 % |
| coherent_d5_r5 | 3 000 | 0 | 0 | 0 | 15 | 100.0 % |

Both the verify path and the production skip-the-rebuild path are bit-exact.

## Performance — calibrated rdtsc @ 3.70 GHz (ns/cyc 0.2706), median-11 clean wall

| bench | rebuild ns base→static | rebuild reduced | wall base→static | wall speedup | ratio vs clifft |
|---|---|---:|---|---:|---|
| **cultivation_d3** | 4445 → 403 | **91 %** | 14998 → 10906 | **1.375× (−27.3 %)** | 7.00 → **5.09** |
| **distillation** | 2446 → 263 | **89 %** | 17896 → 15658 | **1.143× (−12.5 %)** | 1.62 → **1.41** |
| cultivation_d5 | 194 → 16 | 92 % | 204767 → 205184 | 0.998× (neutral) | 2.49 → 2.49 |
| **coherent_d5_r5** | 746558 → 9869 | **99 %** | 11.18 ms → 10.43 ms | **1.072× (−6.7 %)** | 15 plans, 100 % hit |

The rebuild StaticPlan removes 89–99 % of the rebuild cost on every bench; the **wall win tracks the
rebuild's share of wall** (cult_d3 30 %→saved 27 %; distillation 14.5 %→saved 12.5 %; d5_r5 6.7 %→saved 6.7 %;
cult_d5 0.1 %→neutral). Even d5_r5 (dense-bound, k=24) benefits — it runs 12 rebuilds/shot at n=72
(746 µs/shot of rebuild). `static+shadow` is slower (runs live + verify) — verification mode, not production.

## Success criteria (met)

- cult_d3: rebuild region reduced ≥70 % → **91 %** ✓; wall 14.7 µs → **10.9 µs** (target "10–11 µs") ✓.
- distillation: rebuild 14.5 % removed → **−12.5 % wall** ✓; plain path 1.62×→1.41× (fblock 0.79× path is
  separate, default-OFF rb_static does not touch it) ✓.
- cult_d5: no-regression ✓ (neutral, rebuild 0.1 % of wall). d5_r5: **bonus −6.7 %** (12 rebuilds/shot at
  n=72 made its rebuild 6.7 % of wall — 99 % removed).

## Remaining bottleneck after static-invframe (next candidates)

cult_d3 now 5.09× (Clifft 2.14 µs is a hard floor — win not expected, bottleneck-removal demonstrated).
Remaining control plane: per-op frame conjugation (the Gate N fblock target, generalizable beyond
distillation), opcode dispatch superinstruction, noise/RNG bookkeeping. cult_d5/d5_r5 stay dense-bound.
