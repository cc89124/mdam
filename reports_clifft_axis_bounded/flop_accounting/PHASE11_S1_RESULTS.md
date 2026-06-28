# Step C / S1 — behavior-neutral shadow factorization: RESULTS

**S1 is behavior-neutral (shadow ON ≡ shadow OFF on all 9 circuits) and the factorization
`U_C = C_sym·P_res·B` reconstructs by construction. The operative test — Z-preservation /
diagonalizability of the T generators — is 89/91 (97%) on cultivation_d5 with the canonical
born-basis-on-U_C rule, AG=0.** The 2 residuals are the q14 X/Z conflict and they pin down a precise
scoping fact for S2 (below). No engine change, no dispatch, no authoritative swap. a05843e / tag /
butterfly / localizer / `_pullback_via_basis` / Policy-3 default-off preserved. Script:
`scripts/phase11_s1_shadow.py`.

## 1. Canonical decomposition rules (fixed, deterministic)

- active-axis order = `self.M` (promotion order); dormant order = qubit index.
- **born-basis pivot rule:** at each promote of qubit q, record B_q from the triggering pullback's
  character on q — pure-X→H (born-X), Y→S†H (born-Y), pure-Z→I (born-Z); re-promote re-records.
- **P_res** = `i^p ⊗_{a∈M} X_a^{px}Z_a^{pz}` (axes in M order, i-power on the product).
- **C_sym** holds CNOT/CZ/S/SWAP (Z-preserving); **B** holds the per-axis Hadamard/born; global phase in γ.
- generator of a rotation/measurement on lab q through C_sym = `B·(U_C†Z_qU_C)·B†` up to a P_res sign;
  **Z-only on the active axes ⟺ diagonal T** (the §3 pullback-direction Z-preservation check).

## 2. Reconstruction is automatic (and what that means)

Defining `C_sym·P_res ≔ U_C·B⁻¹` makes `C_sym·P_res·B = U_C` **identically** — so reconstruction is
exact at every mutation by construction; there is no "reconstruction mismatch" to find. The *content* of
S1 is therefore whether the leftover `V = U_C·B⁻¹` **cleanly splits** into a Z-preserving C_sym and a
Pauli P_res — equivalently, whether each T's generator is Z-only after the born basis. That is the
89/91 result.

## 3. Deliverables (the §7 checklist)

| deliverable | result |
|---|---|
| shadow mismatch total (records/rank/p0, shadow ON vs OFF) | **0 — IDENTICAL on all 9 circuits** (behavior-neutral) |
| first mismatch event | none (reconstruction automatic); Z-preservation first fails at cultivation_d5 **T#73** |
| 9-circuit reconstruction | **all reconstruct** (automatic); Z-preservation by class (table below) |
| cultivation diagonalizable T | **89/91 (d5, 97%)**, **27/29 (d3, 93%)**, **AG=0** |
| measurement-boundary invariant (cultivation) | 0 AG-measures ⟹ **0 non-Pauli injection** at all 15 boundaries; 0 re-promotes; the 2 residuals are *within-batch*, not boundary events |
| canonicalization rules | §1 (fixed, deterministic) |
| extra metadata / runtime overhead | one born-basis dict (1 entry/axis); **0 runtime overhead** (read-only hooks) |
| updated FLOP projection | §5 |

**Per-circuit Z-preservation & classification:**

| circuit | diag/T | AG | shadow≡plain | class |
|---|--:|--:|:--:|---|
| cultivation_d5 | 356/364 (97%) | 0 | ✓ | near-parity |
| cultivation_d3 | 216/232 (93%) | 0 | ✓ | near-parity |
| coherent_d3_r3 | 342/390 | 24 | ✓ | AG-heavy (conditional) |
| coherent_d5_r5 | 710/806 | 24 | ✓ | AG-heavy (conditional) |
| distillation | 40/80 | 3 | ✓ | AG-heavy (conditional) |
| coherent_rx_d3_r1/r3 | 116/164, 276/420 | 12 | ✓ | R_X off-axis fallback |
| coherent_ry_d3_r1/r3 | 84/342, 104/612 | 0 | ✓ | R_Y off-axis fallback |

R_X/R_Y land much lower (off-axis generators are genuinely non-diagonal — the existing
localizer/butterfly fallback, as designed). AG-heavy circuits inject non-Pauli Cliffords (the §8 case-2
regime). cultivation is the clean diagonal regime.

## 4. The 2 residuals — a precise scoping finding for S2

cultivation_d5 non-diagonal T's: **T#73 (q5,q14, +π/4)** and **T#74 (q5,q14,q15, −π/4)**, both with a
**pure-Z generator on axis 14**. Axis 14 (= q14) appears in **42 generators: 38 pure-X, 2 pure-Z, 0 Y**;
it is born-X (correct for 38), and the 2 pure-Z ones cannot be diagonalized by the same static basis (X
and Z are conjugate — Phase-5's "q5/q14 carry both X and Z", now exact).

**Why this matters:** the born-basis-on-U_C shortcut operates on bounded's **already-pulled** U_C, into
which the entangling CNOTs are baked. Those CNOTs rotate T#73/74's generator to put Z on axis 14. The
**full design keeps the entangling CNOTs symbolic in an independent C_sym** (never pulled into the
generator), which is exactly how **clifft reaches 91/91 with 0 array_h** (census: array_t 45 +
array_t_dag 46 = 91, 0 array_h). So:

- **born-on-U_C (what S1 can test on the existing frame): 89/91.**
- **independent C_sym with symbolic CNOTs (the full design = clifft): 91/91, proven by clifft's 0 array_h.**

S1's job was to surface this *before* implementation, and it did: **S2 must maintain C_sym as its own
tableau (CNOTs folded symbolically, never into the generator), not derive it from U_C.** The 89/91
shortcut is a lower bound; the design's 91/91 target stands (clifft is the existence proof).

## 5. Updated FLOP projection

| scheme | diag T | runtime H / array_h | born | proj. FLOP (cult_d5) | × clifft |
|---|--:|--:|--:|--:|--:|
| current bounded (butterfly) | 0 | 0 | — | 727.7k | 3.42 |
| B1 born-X (folded into U_C) | 26 | 0 (butterfly fallback) | 16·c4 | 719.1k | 3.38 |
| **S1-shortcut realizable** (89 diag + 2 localize, born free) | 89 | 2 (localize c≈7) | 16 free | **≈ 221k** | **≈1.04** |
| **full design = independent C_sym** (clifft) | 91 | 0 | 16 free (expand) | **212.8k** | **1.00 (parity)** |

The S1-shortcut already collapses 3.42× → ~1.04× (near-parity); the last 2 residuals (≈0.04×) close only
with the independent C_sym of S2.

## 6. Verdict and the S2 ask

S1 passes its gate: behavior-neutral (0 mismatch), reconstruction exact, canonical rules fixed, and the
diagonalizable structure characterized (cultivation 89/91, AG=0, the 2 residuals pinned to the q14 X/Z
conflict). The key design refinement: **C_sym must be an independent symbolic tableau (entangling CNOTs
never pulled into the generator)** to reach the clifft-proven 91/91 parity — the born-on-U_C shortcut
tops at 89/91. No S2 dispatch / EXPAND change / localizer deletion / flag-default change until this S1
result is approved.
