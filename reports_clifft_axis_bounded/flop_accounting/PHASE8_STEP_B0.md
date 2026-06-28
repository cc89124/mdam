# Step B0 — event-level differential shadow (candidate Policy-3 diagonal vs the exact butterfly)

**Result: 91/91 PASS (and 364/364 across seeds).** The candidate Policy-3 DIAGONAL dispatch reproduces
the existing exact path's physical state at machine precision on every cultivation T, including the
γ bookkeeping, and — run authoritatively through every measurement — yields bit-identical records,
peak rank, and per-measurement Born p0. No engine change, no dispatch activation, a05843e / tag /
fallbacks preserved. Scripts: `scripts/phase8_step_b0.py`, `phase8_focus.py`, `phase8_table.py`;
data: `data/phase8_per_T_table.md`.

## 0. Why this answers the Step-A gap correctly (not the naive Option B)

The user rejected "verify only final record/rank/p0 then flip T/T† on": without a faithful residue
extraction that would risk re-introducing the R_Y/CZ/phase bug class. Step B0 instead **builds the
candidate diagonal path and compares the PHYSICAL STATE right after each T** against the existing exact
path used as source-of-truth — the stronger, event-level check the user specified.

**Source of truth.** cultivation's peak rank is 10 < the `_loc_min_size = 2^14` localizer gate, so the
authoritative path is the **off-diagonal butterfly** `_pauli_lincomb_inplace` (exp(−iθ/2·P̂) applied
directly, frame untouched). The candidate is compared against *that*.

**Faithful AND tractable comparison.** The candidate is built **frame-preserving** (apply V → diagonal
T/T† → undo V), so its tableau Xc/Zc is **bit-identical** to the butterfly's. For two bounded states
with the same U_C, `|Ψ₁⟩ = U_C(φ₁⊗|0⟩)` equals `γ|Ψ₂⟩ = γU_C(φ₂⊗|0⟩)` **iff** `φ₁ = γφ₂` (U_C unitary ⟹
injective). So comparing the 2^r magic register `φ` up-to-global-phase is **exactly** comparing the full
2^n physical states — no 2^16 materialization needed. For d3 (n=6) the full 2^6 statevector is ALSO
materialized through the independent U_C-matrix path and agrees to 2.2e-16, confirming the reduction.

## 1. The explicitly-constructed candidate (not "single-Pauli ⟹ dispatchable" hand-waving)

For each T the harness constructs, in the **magic-bit (mx,mz,pp) space the butterfly actually uses**:
chosen pivot axis `a`; the residual entangling Clifford = the free CNOT-collapse of the X- and Z-strings
onto `a`; the **born basis** (Z / X=H / Y=S†H) that diagonalizes the single-qubit residue on `a`;
(p_x, p_z) and the frame phase i^pp; and the T/T† dispatch + γ from the collapse sign. Then it applies
the **diagonal** gate T (p_x=0) or T† (p_x=1) and γ ← γ·e^{−i·s·θ/2}, and undoes V.

## 2. Verification results (all comparison points the user listed)

| point | cultivation_d5 (4 seeds, 364 T) | cultivation_d3 (8 seeds, 232 T) |
|---|---|---|
| candidate dispatch constructed | **364 / 364** | **232 / 232** |
| 1. dense state up to global phase | max residual **3.7e-15** | **3.3e-16** |
| 2. physical state incl. γ (`max|φ_S − γφ_C|`) | **3.7e-16** | **3.7e-16** |
| 3. active (X_i,Z_i) images (tableau bit-diff) | **0** | **0** |
| 4. random-Pauli ⟨P⟩ mismatch | **6.7e-16** | **6.7e-16** |
| 5. norm `||φ_S|−|φ_C||` | **3.3e-15** | **3.3e-16** |
| 6. next-measurement Born p0 (Part 2, through real meas) | **1.1e-16** | **5.6e-17** |
| 7. rank / active-axis map mismatch | **0** | **0** |
| INDEP full-statevector(2^6) up-to-phase / incl-γ | (n=16, via §0 reduction) | **2.2e-16 / 2.3e-16** |
| born-basis distribution | all **X** (X-prepped) | all **X** |

**Part 2 (whole-run authoritative candidate engine).** Making the diagonal dispatch authoritative on a
throwaway engine and running the FULL circuit through every measurement/drop gives, vs the real
butterfly engine: **records identical, peak rank identical, |M| before/after schedule identical,
per-measurement p0 identical** (max diff 1.1e-16) over all seeds — i.e. the dispatch is exact *through*
measurements, not just at the isolated T.

## 3. The dispatch RULE, verified (Sec.5)

**Gate identity** `T·X^xZ^z = ω^x·X^xZ^z·T^{(−1)^x}` (ω=e^{iπ/4}) — standalone matrix check, every residue:

| residue | I | Z | X | XZ(=−iY) |
|---|---|---|---|---|
| T → | T | T | **T†** | **T†** |
| ‖LHS−RHS‖ | 0 | 0 | 1.0e-17 | 1.0e-17 |

**Per-T rule-vs-actual** (cultivation_d5, all 91): the effective single-axis action
`exp(−iθ/2·s·Z_a)` equals `γ·(T or T†)` to **1.1e-16**. Clean correspondence: the **13 T† dispatches ⟺
exactly the 13 frame-phase pp=2 generators** (s=−1); the other 78 are T (pp=0, s=+1). **0 Y-residue** —
cultivation is purely X-prepped, so every born basis is X (matches Phase 4). Weights 3/5/7: 37/16/3
all dispatch exactly; 16 first-after-promote; 13 last-before-measure. q5 is pivot for 26 T's, q14 for 2;
all consistent.

## 4. Per-T table

Full 91-row table in `data/phase8_per_T_table.md` (columns: T, rank, generator, axis q, born, px, pz,
i^pp, gate, weight, vector_diff, img_mismatch, next_p0_diff). **Summary: max vector_diff 3.7e-16,
image mismatches 0, max next_p0_diff 3.7e-31, all 91 gates diagonal T/T† (0 butterfly in the candidate).**

## 5. Verdict and what it does / does NOT establish

**Establishes (this is the success branch of the user's §6):** the per-T diagonal-T/T†-via-residue
construction is **physically exact** — same dense state (incl. γ), same Pauli images, same norm, same
next-measurement p0, same rank/map — for **all 91** cultivation T's and through every measurement. The
Step-A worry ("can't separate born/residue/entangling") is resolved **per-T**: the separation is
recovered from the pulled-back generator's symbolic collapse (born basis = the one collapse-H; residue =
the collapse sign → T/T†; entangling part = the free CNOTs), and it reproduces the exact path bit-for-bit.

**Does NOT yet establish (this is Step B1, by design):** the FLOP **win**. The Step-B0 candidate still
*applies and then undoes* the born-H (frame-preserving, to make the comparison airtight); it does not
reduce runtime ops. The 0-runtime-H win requires the **persistent split** — born basis fixed at promote
(free), residue carried on the axis, diagonal T/T† (c=3) with no butterfly and no runtime H. Step B0 is
the safety gate proving that split is exact before it is built.

## 6. Recommended next step — Step B1 (persistent split, default-off, bit-exact)

Per the user's "91/91 통과 시" branch. Minimal mutation set (design in the cover message), implemented
behind a **default-off flag** and verified **bit-identical to a05843e** on records/rank/p0 across all 9
circuits, with the butterfly and both localizer fallbacks preserved. No final-FLOP claim until that
bit-exact gate passes.
