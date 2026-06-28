# Phase 5 — Why clifft's T is 0-Hadamard, and can bounded reach parity? (analysis only)

Resolves whether cultivation's residual (4.0× butterfly / 2.33× per-rot / 1.64× batched) can reach
clifft parity (1.0×, **0 array_h**). **Verdict: Case A — parity (0 runtime H) IS achievable for
cultivation.** The residual is not a representation limit; clifft achieves 0 array_h on the IDENTICAL
factored representation. Sources: clifft C++ (quoted below), scripts/phase4_generator_trace.py,
/tmp/lifetime.py.

## 1. clifft's array_t is a TRUE diagonal phase — verified from source, not assumed

`exec_array_t` (svm_kernels.inl:970-987) calls `apply_phase_waterfall`, a **pure diagonal phase
multiply** on the half-array where the axis bit = 1: one complex-mul per amplitude over 2^(k-1) ⟹
**c = 3·2^k**. NO 2×2 butterfly, no hidden off-diagonal. CostMeter cross-check: cultivation array_t
FLOP/sum_pow2k = **3.00** exactly. The name is not the evidence; the kernel arithmetic and the meter
both confirm c=3.

When `p_x[v]=1` (the Pauli frame has an X on the axis): clifft applies **T† (still diagonal,
conjugate phase)** and γ absorbs the phase (svm_kernels.inl:982-987). Still c=3, still diagonal — the
X is handled at the scalar/frame level, no off-diagonal cost is hidden.

## 2. The mechanism — clifft keeps a PAULI residue frame; the H is absorbed at axis creation

- clifft stores only **per-axis Pauli bits p_x/p_z** for active axes (svm.h:156-157), NOT a full
  Clifford tableau. The amplitude array is kept in the **Z-basis (computational)**.
- **A single-qubit T is diagonal under ANY single-qubit Pauli frame**: T·X = X·T† and T·Z = Z·T —
  both diagonal. So whether the frame Pauli on an axis is X, Z, or Y, clifft applies T (or T†)
  **diagonally, no Hadamard.** This is why q5/q14 (which carry both X- and Z-type generators over
  their life) cost clifft **0** H — both are diagonal under their Pauli frame.
- The genuine **non-Pauli Clifford (Hadamard) content** — the state-prep H that creates a magic axis
  in the X-basis — is **absorbed at axis EXPANSION**: `route_to_active_z` (backend.cc:362-389) emits
  `FRAME_H + EXPAND` (axis born in |+⟩) for a dormant pivot, **not** `array_h`. `EXPAND` is a 0-FLOP
  copy; materializing the axis in |+⟩ vs |0⟩ is the same cost. So the prep-H is paid at 0.
- `array_h` (c=7, active-axis basis change) fires ONLY when an X-basis rotation hits an ALREADY-ACTIVE
  pivot. **Cultivation never triggers it ⟹ 0 array_h** (measured). The CNOT structure is array_cnot
  (0 FLOP permutation) and the Clifford localization is folded into clifft's own symbolic
  `virtual_frame` (backend.cc:418-517) — clifft, like bounded, keeps a symbolic Clifford frame for
  CNOT/CZ/S localization; the difference is ONLY where the Hadamard goes.

**Gate-level trace, one cultivation T:**
```
clifft:  physical T_q → localize_pauli folds CNOT/CZ/S into virtual_frame (free)
                      → route_to_active_z: pivot dormant → FRAME_H (Pauli, O(1)) + EXPAND (|+⟩, 0 FLOP)
                      → array_t on pivot, p_x-aware → DIAGONAL phase, c=3·2^k.   H on array = 0.
bounded: physical T_q → _pullback through FULL Clifford frame U_C → OFF-DIAGONAL Pauli (X-character)
                      → butterfly c=12 (or localize: collapse + 1 array H + diagonal, c=7).  H = 1/rot.
```
The ONLY difference: clifft keeps the Hadamard OUT of the generator (absorbed at expansion, Pauli
residue frame); bounded pulls the full Clifford frame INTO the generator (→ off-diagonal → runtime H).

## 3. Where each backend pays the Clifford H-content (the answer to "why is bounded's H extra")

| H-content | clifft | bounded (current) |
|---|---|---|
| state-prep H (creates magic axis basis) | absorbed at EXPAND (|+⟩), 0 FLOP | carried in symbolic U_C → re-applied as runtime localization H |
| Pauli frame X/Z on active axis | p_x/p_z + diagonal-T kernel (T/T†), 0 H | merged into generator via pullback → off-diagonal |
| CNOT/CZ/S localization | virtual_frame (symbolic, free) | inverse-frame (symbolic, free) — SAME |
| active-axis non-Pauli basis change | array_h (c=7) — **0 occurrences in cultivation** | n/a (would be the localization H) |

bounded's 47 batched H's (and the c=12 butterfly) are entirely the first two rows: it merges the
prep-H and the Pauli frame into the rotation generator instead of (a) absorbing the prep-H at promote
and (b) keeping a Pauli residue handled by a Pauli-aware diagonal-T.

## 4. Lifetime / full-register basis structure (cultivation, /tmp/lifetime.py)

Per-qubit single-qubit Pauli type across its lifetime: **14 of 16 qubits are pure-X** (one promote-
basis diagonalizes all their T's); only **q5, q14** carry both X and Z single-qubit types. In a
naive promote-basis-only scheme those 2 would force a basis flip; but under clifft's Pauli-aware
diagonal-T both X and Z are diagonal, so even they cost 0 H. The multi-qubit (weight 3,5,7)
generators are products of single-qubit Paulis: with each qubit in its own promote-basis they become
Z-strings → free CNOT-collapse → diagonal, 0 H. **No full-register entangling H basis is required** —
the diagonalizing Clifford factors into per-axis (expansion-absorbed) Hadamards + free frame-CNOT/CZ.

## 5. Three execution policies

| policy | runtime H-sweeps | FLOP × clifft | rank | exactness | architecture change |
|---|---:|---:|---|---|---|
| 1. batch simul-diag (Phase 4) | 47 | 1.64× | unchanged | exact | small (batch the localizer) |
| 2. lifetime promote-basis | ~2–4 (only q5/q14 flips) | ~1.05× | unchanged¹ | exact | medium (choose promote basis) |
| 3. clifft-equivalent: Pauli-residue frame + p_x/p_z-aware diagonal-T + EXPAND-absorbed H + free frame-CNOT | **0** | **1.00× (parity)** | unchanged¹ | large (frame re-architecture) |

¹ in-register basis choice and promote-basis are FREE of rank cost (Phase-4 §5: `_h_axis` etc. keep
`r=len(M)`; `_promote` sets the new axis's basis at creation for 0 extra). The rank advantage is
ORTHOGONAL and preserved — for circuits with headroom (d5_r5, r=13<k=24) the diagonal T's run at r,
still ≪ clifft.

## 6. The seven questions

1. **clifft cultivation T = 3·2^k diagonal?** YES — source (pure phase waterfall) + CostMeter (3.00).
2. **Still c=3 with p_x on?** YES — p_x selects T† (diagonal); no hidden 2×2.
3. **Can bounded apply the same frame-aware diagonal matrix at rank r?** YES in principle — identical
   factored representation; needs a Pauli-residue frame + p_x/p_z-aware diagonal-T (Policy 3).
4. **If so, why does bounded currently need a separate H?** Because it pulls the T generator back
   through the FULL Clifford frame (merging prep-H + Pauli into the generator → off-diagonal) and
   materializes axes in Z-basis instead of absorbing the prep-H at EXPAND.
5. **If impossible, what invariant blocks it?** NONE structural — clifft is the existence proof on the
   identical representation. The "block" is bounded's full-symbolic-Clifford-frame architecture, a
   design choice, not a representation invariant.
6. **Can promote-basis ALONE reduce 47 H → 0?** Not quite — promote-basis fixes the 14 single-basis
   qubits but leaves q5/q14's X↔Z as a flip (~2–4 H). Reaching exactly 0 also needs the Pauli-aware
   diagonal-T kernel (clifft's array_t p_x/p_z), which makes those flips diagonal too. **Promote-basis
   + Pauli-aware-T → 0.**
7. **Exact minimum H / lower bound (cultivation)?** **0** — clifft achieves 0 array_h end-to-end on
   cultivation, so the exact minimum runtime H for this circuit is 0. (For a general circuit the lower
   bound = clifft's array_h count = the active-axis non-Pauli basis changes not absorbable at
   expansion; cultivation happens to have 0 of these.)

## 7. Verdict — Case A (parity achievable for cultivation)

bounded CAN reach F_bnd ≤ F_clf (exact parity, 0 runtime H) on cultivation, using clifft-equivalent
frame discipline: **Pauli-residue frame on active axes + p_x/p_z-aware diagonal-T + state-prep H
absorbed at promote/EXPAND + CNOT/CZ/S folded into the symbolic frame (free).** This is proven
possible by clifft's measured 0 array_h on the identical factored representation, and the rank
advantage is preserved (orthogonal, in-register/promote basis is rank-free).

**Honest scope:** (i) this is *cultivation-specific* — it relies on clifft's 0 array_h, which holds
because cultivation's Hadamards are all prep-H (expansion-absorbable); a circuit that forces active-
axis non-Pauli basis changes would give clifft array_h > 0 and bounded the same, a conditional bound.
(ii) Parity requires the Policy-3 frame re-architecture (not the small Phase-4 batching); the
realizable-with-small-change result remains 1.64× (Policy 1) or ~1.05× (Policy 2). (iii) No
implementation is done here per instruction — this establishes the achievable floor (0 H) and the
constructive discipline, overturning the earlier "(7/3) structural" claim definitively.

## 8. Constructive schedule (design only; NOT implemented)

For each magic axis: at `_promote`, materialize it in the basis that diagonalizes the non-Pauli part
of its first generator (free — the axis is |0⟩); keep its subsequent X/Z Pauli evolution in a Pauli
residue (p_x/p_z) and apply each T via a p_x/p_z-aware diagonal half-array kernel (T or T†, c=3); fold
all CNOT/CZ/S into the existing symbolic inverse-frame (free) and collapse multi-qubit Z-strings with
free CNOT permutation before the diagonal rotation. No butterfly, no per-rotation H, no size gate, no
T-specific 2×2 kernel — the rotation kernel is the existing `rot:diaghalf` (c=3); the new piece is the
Pauli-residue frame + promote-basis bookkeeping. Projected cultivation FLOP = clifft's 164.9k (parity);
projected traffic ≤ clifft (bounded keeps CNOT symbolic vs clifft's array_cnot).
