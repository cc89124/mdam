# Pullback → StaticPlan, Step 1: mask shot-invariance (PROVEN)

**Goal:** stop recomputing `P' = F†PF` every shot; fix the pulled-back **mask** at compile time (StaticPlan),
compute only **sign/phase** at runtime. Target: pullback `F†PF` (cult_d3 ~26 %, cult_d5 ~26 % of wall, the #1
control region in `wall_breakdown.md`).

**Premise to prove first (de-risk before building):** for each `(boundary mp, request kind, input physical
Pauli)`, is the pulled-back `(x,z)` MASK identical across all shots/seeds (only sign/phase dynamic)?

## Method

In-C++ aggregated invariance checker (default OFF, `nvm_pb_cap`): `pullback()` records, per key
`(mp, kind, input x/z masks)`, the first-seen output mask + phase, and counts any later mask disagreement
(`mask_viol`) and phase variation. kinds: 0 PLAN_Pm, 1 oracle_Pm, 2 flush_pullback, 3 PLAN_rot.
Seeds 1/7/42/123/999 × 2500 shots = **12 500 shots/bench**.

## Table 1 — Pullback mask invariance

| bench | request kind | calls/shot | unique masks | **mask viol** | phase varies | static-plan? |
|---|---|---:|---:|---:|:--:|:--:|
| cultivation_d3 | PLAN_Pm | 3.99 | 5 | **0** | yes | **YES** |
| cultivation_d3 | oracle_Pm | 1.00 | 1 | **0** | yes | **YES** |
| cultivation_d3 | flush_pullback | 1.03 | 5 | **0** | yes | **YES** |
| cultivation_d3 | PLAN_rot | 27.97 | 22 | **0** | yes | **YES** |
| cultivation_d5 | PLAN_Pm | 4.99 | 15 | **0** | yes | **YES** |
| cultivation_d5 | oracle_Pm | 10.00 | 10 | **0** | yes | **YES** |
| cultivation_d5 | flush_pullback | 33.08 | 37 | **0** | yes | **YES** |
| cultivation_d5 | PLAN_rot | 57.93 | 72 | **0** | yes | **YES** |
| distillation | PLAN_Pm | 4.00 | 5 | **0** | no | **YES** |
| distillation | oracle_Pm | 0.78 | 1 | **0** | yes | **YES** |
| distillation | PLAN_rot | 10.00 | 10 | **0** | no | **YES** |

**Result: 0 mask violations across every (bench × kind).** The mask is 100 % shot-static; only the phase is
dynamic. Unique masks are few (5–72), so the StaticPlan table is tiny. **All request kinds are StaticPlan-eligible.**

## Strategic finding — Phase A alone does NOT win; the prize needs Phase B

The pullback `inverse_frame.pullback(P)` computes the **mask and the phase together** in one O(weight) pass.
Therefore:

- **Phase A (static mask, phase via the existing pullback)** saves only the *consumers* of the mask —
  promote-set, M-layout, `mx/mz` reduction (the "F layout/mask" region) — **not** the pullback core itself,
  because the pullback still runs to get the phase. Modest win.
- **Phase B (affine phase)** is what removes the pullback `F†PF` itself (the 26 % region): compute the phase as a
  parity over a static mask of a compact **inverse-frame phase vector**, with no O(weight) substitution.

So the next de-risk gate (before building Phase B) is the **phase premise**: is the pulled-back phase an affine
(XOR / mod-4) function of the inverse-frame phase vector? (Gate J showed the phase is NOT a pure-query of
*outcomes* — the amplitude-dependent `drop_residual_products` fold_x — but the phase-vs-*frame-phase-vector*
question is different and is the one that matters here.)

## Step 2 — data structure (designed, not yet built)

```
struct StaticPulledPauli {            // one per (boundary, kind, input-Pauli) key — ~5-72 per kind
    uint32_t mp; uint8_t kind;
    uint64_t in_x[2], in_z[2];        // key
    uint64_t out_x[2], out_z[2];      // PROVEN shot-static (Table 1)
    uint8_t  static_phase;            // Phase B: base
    uint64_t phase_affine_mask[2];    // Phase B: phase = static_phase ^ parity(inv_phase_vec & mask)
};
struct PullbackStaticPlan { std::map<Key, StaticPulledPauli> tab; };  // lookup (mp,kind,in) -> entry
```

Runtime: `e = plan.lookup(mp, kind, P); xmask=e.out_x; zmask=e.out_z; phase = e.static_phase ^ parity(inv_phase_vec & e.phase_affine_mask)`.

## Status / next

- Step 1 (mask invariance) **PROVEN** — all kinds eligible, 0 violations.
- Next gate: **verify the Phase-B phase premise** (phase = affine of inverse-frame phase vector) before building,
  since Phase A alone yields only the small layout win. If the phase is affine → build Phase B (the 26 % prize).
  If not → Phase A (mask+layout static, phase kept live) is the bounded fallback.
- Instrumentation is default-OFF (`pb_cap_on` false; authoritative output unchanged — `pb_kind` writes are
  side-effect-free, only read inside the capture branch).
