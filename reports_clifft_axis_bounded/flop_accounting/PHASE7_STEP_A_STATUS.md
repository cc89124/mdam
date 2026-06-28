# Step A status — behavior-neutral shadow (honest report, NO engine change)

Scope: the behavior-neutral verification only. NO T/T† dispatch, NO EXPAND change, NO gate, NO
soft-reset. a05843e / tag / fallbacks preserved. Script: scripts/phase7_step_a.py,
data/phase7_step_a.txt.

## What IS rigorously established

1. **Baseline equivalence — ALL 9 circuits PASS (observation-only).** Records, peak rank, and the
   per-measurement Born p0 are bit-identical with vs without the shadow hooks ⟹ the harness does not
   perturb the dense state, FLOP, or localizer (verification point 9). ✓
2. **Policy-3 precondition holds.** Every pending T's pulled-back generator is a SINGLE Pauli (the
   pullback of one logical Pauli through a Clifford is one Pauli). A single nonzero Pauli is ALWAYS
   single-axis-localizable by a complete Clifford (one born-H + free CNOT/CZ collapse). So each T is a
   single-axis rotation, the precondition for diagonal T/T† dispatch. (The harness's `singlePauli`
   counter under-reports because it builds the active mask BEFORE promotion of X-support qubits — a
   counting artifact, not a precondition failure; the mathematical statement above is exact.)
3. **cultivation has 0 AG-measures.** Measurements take the magic path (array-fold + Pauli) and never
   multiply a non-Pauli Clifford into the frame ⟹ across all 15 measurement boundaries the active-axis
   Clifford content comes only from gates (the §6 hypothesis). ry_d3_r1 also 0 AG; rx/coherent_d3_r3/
   d5_r5/distillation have AG>0 (12/24/24/3) — those are the §9-case-2 regime needing canonicalization.
4. **Collapse-first heuristic coverage = 78/91** for cultivation; the other 13 single-Pauli generators
   need a complete localization (still ONE born-H) — they fall back to the butterfly in today's code.
5. **Cost notation (clarified, matching the existing meter):**
   - `array_h` ALONE = **4·2^r** (purge:h)
   - diagonal R_Z/T ALONE = **3·2^r** (rot:diaghalf / clifft array_rot)
   - **H + diagonal rotation = 7·2^r** (the localizer's per-off-diagonal-rotation total)
   - off-diagonal butterfly = **12·2^r** (rot:offdiag)

## The honest finding — a faithful shadow needs a FULL Clifford split

The verification asked for (point 1) phase-exact reconstruction of every active (X_i, Z_i) image from
`C_outer + P_res + born-basis`. I attempted two shortcuts and BOTH fail:
- **Local single-qubit symplectic extraction** (read born off the frame's bits on qubit q): FAILS.
  The local 2×2 is not constant and not even a clean single-qubit Clifford (e.g. `(0,0,0,0)` = the
  image of X_q is entirely off-qubit) because the frame ENTANGLES active axes via CNOT/CZ. The local
  bits mix entanglement, born, and residue — they do not isolate the born.
- **Per-generator collapse** (read born off each T's localization): FAILS to separate, because the
  generator's X-character mixes (a) the entangling frame, (b) the born Hadamard, and (c) the Pauli
  residue; the collapse-first H absorbs all three into one H and cannot attribute it.

**Conclusion:** a faithful, phase-exact `(C_outer entangling | per-axis born | Pauli residue)` shadow
requires an actual incremental Clifford factorization (maintain `C_outer` as its own tableau with H
diverted to per-axis borns, evolve via §2, reconstruct and compare each image). That is a substantial,
phase-sensitive implementation (the R_Y-frame-bug class lives exactly here). I did NOT fake it — this
harness verifies the preconditions + baseline-equivalence only, and I am flagging the gap rather than
reporting a shadow pass I did not achieve.

## Why the gap may not be worth closing as a standalone shadow

The shadow's job is to prove the split is exact. But **Step B's dispatch is verified by a STRONGER
test**: when the T/T† diagonal dispatch is actually applied, its correctness is checked by **bit-exact
records / rank / p0 vs a05843e** (the gold standard the whole project uses), which directly validates
the split without a separate shadow. The shadow would be a weaker, redundant check that costs a
high-risk Clifford-synthesis implementation. clifft's measured **0 array_h** already proves the split
EXISTS for cultivation; the precondition (every T single-Pauli) + 0-AG show the structure is present.

## Updated metrics (verification points)

| point | result |
|---|---|
| shadow mismatch count | n/a — full split-shadow not implemented (honest gap, see above) |
| first mismatch event | n/a |
| diagonal-dispatchable T / 91 | **91/91** (all single Paulis; precondition holds) ; heuristic 78/91 today |
| non-Pauli active residue per boundary | **0 AG-measures ⟹ 0 Clifford-injection** across 15 boundaries (frame-level); exact per-axis residue count needs the full split |
| record / rank / p0 vs existing | **identical, all 9 circuits** ✓ |
| rank / core schedule change | **0** (observation-only) ✓ |
| Policy-3 FLOP residual-0 | runtime H target 0 ; F_bnd ≈ 193.9k = 0.91× clifft (diagonal-T 164.9k + Born 28.95k) — from §8 |

## Recommendation / decision needed

Two ways to proceed (no implementation until you choose):
- **(A) Build the full incremental Clifford-split shadow** (C_outer tableau + per-axis born + residue,
  §2 evolution, phase-exact image reconstruction vs source-of-truth). Rigorous but high-effort/high-
  risk; closes point-1 exactly.
- **(B) Accept the Step-A gate as: baseline-equivalence (✓) + single-Pauli precondition (✓) + 0-AG
  structure (✓) + clifft existence proof + cost-notation (✓), and move to Step B** (T/T† diagonal
  dispatch behind a default-off flag), where correctness is verified directly by bit-exact records/
  rank/p0 vs a05843e — a stronger check than the shadow. The full split is then built incrementally as
  part of Step B's dispatch, validated by bit-exactness rather than a parallel shadow.

I recommend **(B)**: the bit-exact dispatch test in Step B subsumes the shadow's guarantee at lower
risk, and the Step-A preconditions are already met. But this is your call.
