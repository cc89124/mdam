# Step C — full symbolic-Clifford frame: DESIGN ONLY (no implementation)

Goal: a bounded representation that reaches **FLOP ≤ clifft** by keeping the entangling Clifford in a
symbolic OUTER frame and the magic axes in their canonical Z-eigenbasis, so every magic rotation is a
diagonal T/T† (c=3) — exactly clifft's discipline, on the same factored representation, while preserving
bounded's rank advantage. NO implementation here. a05843e / Phase-2 tag / butterfly / localizer /
`_pullback_via_basis` / Policy-3 default-off all preserved. Grounded in the clifft kernel census
(`/tmp/clifft_kernel_census.py`) and the Step-B0/B1 traces.

Why Step B1 (per-axis born basis) was insufficient (measured): it folded the born-H into the **same**
full `U_C` and kept pulling the T generator through full `U_C`, so every subsequent circuit Clifford
re-entered the generator and re-rotated it off-diagonal — 26/91 diagonal, the rest butterfly, 0.99×
bounded. The fix is to pull the T generator through a **Z-preserving** outer frame only.

---

## 1. State invariant

$$|\Psi\rangle \;=\; \gamma\; C_{\mathrm{sym}}\; P_{\mathrm{res}}\;\bigl(|\phi\rangle_A \otimes |0\rangle_D\bigr)$$

- **A** = the ordered active magic axes (= current `self.M`); **D** = all other qubits, held in |0⟩.
- **|φ⟩_A**: the dense 2^|A| register in the **canonical magic basis** — each axis a's computational Z is
  the eigen-axis of its T-rotations. **Invariant (I):** every magic R_Z/T rotation on an axis is
  *diagonal* in |φ⟩_A. (Same dense object as today's `phi`, only the basis label differs.)
- **P_res** = `i^p ⊗_{a∈A} X_a^{px_a} Z_a^{pz_a}`: a Pauli over the active axes (per-axis 2 bits + one
  global i-power). It is the Pauli part of the frame carried symbolically. It commutes through a
  diagonal T by the **Step-B0-verified rule** `T·X^{px}Z^{pz} = ω^{px} X^{px}Z^{pz} T^{(-1)^{px}}`
  (ω=e^{iπ/4}), so a residue never forces an array touch — it only selects T vs T† and multiplies γ.
- **C_sym**: a symbolic Clifford **tableau over n qubits** (the existing Xc/Zc machinery), the OUTER
  entangling/coupling frame. **Invariant (C) — Z-preservation on A:** for every active axis a,
  `C_sym Z_a^{canon} C_sym†` is a **Z-only** Pauli string in the lab (no X-component on A). Equivalently
  the active-Z image columns of C_sym carry no X-bit on A. C_sym may contain arbitrary CNOT/CZ/S among
  axes and arbitrary coupling to D, but **no Hadamard that swaps an active axis's X↔Z** — those are
  absorbed at promote (born basis) or, when truly forced, trigger the one explicit fallback (§3, §8).

Invariant (C) is the whole game: it is exactly the condition under which the next theorem holds.

**Theorem (diagonality).** Under (C), the generator of a T/R_Z on any lab qubit q — `C_sym† Z_q C_sym`
restricted to A — is a pure Z-string ⟹ the rotation is diagonal in |φ⟩_A (after a 0-FLOP CNOT collapse to
a single Z_a). Proof: Z_q expands over the frame's Z-images; by (C) each active-axis Z-image is Z-only,
and CNOT/CZ/S (the only non-trivial C_sym content on A) map Z→Z-strings; so no X-bit on A appears. ∎

---

## 2. Exact equivalence with the current bounded representation

Today: `|Ψ⟩ = U_C (φ'_A ⊗ |0⟩_D)`, φ' in the Z-basis, U_C the full frame (tableau). Any Clifford factors
(Bruhat) as `U_C = C_sym · P_res · B`, where `B = ⊗_{a} B_a` is the per-axis born basis (the local
Clifford taking canonical→Z on each axis), P_res the residual Pauli on A, and C_sym the entangling rest
chosen to satisfy (C). Define `|φ⟩_A = B† φ'_A` (the same physical magic state, re-labelled into the
canonical basis). Then

`C_sym P_res (φ ⊗ 0) = C_sym P_res B (φ' ⊗ 0) = U_C (φ' ⊗ 0) = |Ψ⟩` (up to the tracked γ).

So the new tuple `(γ, C_sym, P_res, φ)` reconstructs the **identical** physical state. Every observable
— measurement records, Born p0, and the active rank |A| — is a function of |Ψ⟩ alone, hence **bit-exact**
with today's engine (basis choice is unobservable). The factorization is *maintained* incrementally by
§3; existence each step follows from closure of the Clifford group under the update rules.

---

## 3. Per-gate update rules

Notation: a gate on lab qubits; "fold into C_sym" = the existing `right_h/right_s/right_cx`-style
tableau update (O(n), 0 FLOP); "array op on axis a" = the existing strided kernel on |φ⟩.

| gate | rule | array cost |
|---|---|---|
| **CNOT(c,t)**, **CZ(a,b)**, **SWAP** | Z-preserving ⟹ fold into C_sym (symbolic). Coupling among axes / to D only relabels Z-images. | **0** (≙ clifft array_cnot/cz: 0 / c=0.5) |
| **S / S†** on axis a | Z-preserving (S Z S†=Z). If it is part of the canonical-basis bookkeeping (X↔Y) fold into C_sym; if it must act on the dense magic state (clifft's array_s) apply `_s_axis` (c=2/3). | 0 or c≈3 (≙ array_s) |
| **H on q ∈ D** (dormant/data) | fold into C_sym (symbolic). If this H is the prep that the next promote will turn into a magic axis, it *defines that axis's born basis* (absorbed free at promote). | **0** |
| **H on q ∈ A** (active axis) | would break (C). Check the predicate P(a) [below]. If absorbable (the H re-labels via P_res / pairs with C_sym to stay Z-preserving): fold symbolically. **Else (genuine non-Pauli active-axis basis change = clifft array_h): apply `_h_axis(a)` (re-base the axis), the ONE explicit fallback.** cultivation: **0** of these (clifft census: 0 array_h). | 0 or c=4 (≙ array_h, **0× in cultivation**) |
| **T / T† / R_Z(θ)** on lab q | pull Z_q through C_sym → Z-string on A (diagonal, by Thm). Free CNOT-collapse to a single Z_a (fold into C_sym, 0 FLOP). Apply diagonal `rot:diaghalf` on a (c=3); T vs T† and γ←γω from P_res[a].px (rule). **No butterfly, no runtime H.** | **c=3** (≙ array_t/array_t_dag) |
| **R_X / R_Y** on lab q | genuinely off-axis: generator is X/Y-type in the magic basis, NOT diagonalizable by any born basis ⟹ **fallback** to the existing localizer/butterfly (unchanged). | c=7/12 (≙ clifft U4) |
| **measure Z_q** | §5 | c=8 (≙ meas_interfere) |
| **promote q** | §6: materialize axis in born basis **free** (|+⟩/magic by copy, like clifft expand), extend C_sym/P_res by identity. | **0** (≙ expand: 0 FLOP) |
| **demote/drop a** | §6: localize-and-drop (existing), restrict C_sym/P_res to A∖{a}. | 0 |

**Predicate P(a) (H on an active axis is absorbable):** maintain (C) as a hard invariant; when a gate
would put an X-bit on an active-Z image, the engine restores (C) by the *minimal* array op (an
`_h_axis` on the offending axis). This makes the fallback **explicit, counted, and minimal** — its count
is provably clifft's `array_h` count (the active-axis non-Pauli basis changes not absorbable at
expansion). For cultivation that is 0; for the §8 regimes it is clifft's value, preserving ≤-clifft.

---

## 4. Why entangling Clifford stays symbolic and T is always diagonal

CNOT/CZ/SWAP/S are **Z-preserving**: they map active-axis Z-images to Z-strings, never introducing an
X-bit on A, so they live in C_sym at 0 FLOP (clifft: array_cnot 277 calls / 0 FLOP, array_cz 9 / c=0.5).
The **only** operator that can put X-character on an active axis is a Hadamard on that axis; by §3 those
are absorbed at promote (born basis) or are the explicit array_h fallback. Hence by the Theorem the T
generator is a pure Z-string on A for every rotation outside the fallback set ⟹ diagonal T/T† (c=3). The
entangling content never enters the generator because it is Z-preserving and stays in C_sym; it is only
*materialized as 0-FLOP permutations* (free CNOT-collapse) at the moment of the diagonal rotation.

---

## 5. Measurement: handling `C_sym† M C_sym`

Measure Z_q: pull M=Z_q through C_sym → by (C) a **Z-string on A** (plus a P_res sign) — *diagonal*, no
X-character. So the existing `_localize_to_Z` reduces to a **free CNOT-collapse** to a single Z_r (no
localizing Hadamard, because there is no X to turn into Z), then the Born is the existing strided
branch-sqnorm and the collapse a strided-slice zero. **No extra dense sweep** beyond the one full sqnorm
sweep already done per measurement (= clifft meas_interfere, c=8). An AG-measure (Z_q anticommuting with
a non-magic stabilizer) is handled by the inherited `_ag_measure` (frame-only, magic untouched), exactly
as today; it may inject a non-Pauli Clifford that later forces an array_h (the §8 case-2 regime).

---

## 6. promote / demote: symbolic restriction/extension and rank invariance

**promote(q):** |A|→|A|+1, |φ| doubles. Materialize the new axis in its born basis **for free** — the
EXPAND-style |+⟩ (or magic) fill is a copy+scale charged like clifft's `expand` (**0 FLOP**, replacing
B1's c=4 `_h_axis`). Extend C_sym and P_res by identity on the new axis. The born basis = the prep-H of
that qubit, absorbed here (not pulled into later generators).

**demote(a):** the existing measurement **localize-and-drop** (`_drop_localized` + residual sweep): after
a measurement fixes an axis to a product Z-eigenstate, drop it (φ halves, the dropped axis's symbolic
rows fold into C_sym via the existing X-fold). |A| therefore tracks clifft's `active_k_history`
(decreasing after each magic measurement) — **rank invariant preserved**.

**Rank-freeness of the symbolic discipline (Phase-4 §5, code-verified):** every C_sym fold and born-basis
choice is an in-register basis rotation of the *same* 2^|A| amplitudes (`_h_axis/_s_axis/_cnot_axes`
keep `r=len(M)`); only promote grows rank (+1) and only drop shrinks it. So the symbolic frame **does
not grow rank** — bounded keeps `|A| ≤ k_clifft`, and where it has headroom (d5_r5: 13<24) the diagonal
T's run at r ≪ clifft's k, an *orthogonal* advantage preserved.

---

## 7. cultivation_d5 projection (grounded in the clifft census = the existence proof)

The design replicates clifft's discipline on the identical factored representation, so its event counts
are clifft's measured counts (clifft = the reference symbolic-frame implementation; Phase-5 confirmed it
achieves these on this representation). From `/tmp/clifft_kernel_census.py` (cultivation_d5, k=10):

| event | new-design count | c | FLOP | ≙ current bounded |
|---|--:|--:|--:|---|
| diagonal **T** (array_t) | 45 | 3 | 99.8k | butterfly c=12 |
| diagonal **T†** (array_t_dag) | 46 | 3 | 74.4k | butterfly c=12 |
| **runtime H / butterfly** | **0** | — | **0** | 65 butterflies today |
| array_s | 2 | 3 | 3.3k | — |
| born materialize (expand) | 16 | 0 | **0** | B1 paid 16·c=4 |
| array_cnot (free permutations) | 277 | 0 | 0 | purge:cnot 0 |
| array_cz | 9 | 0.5 | 2.2k | — |
| meas_interfere | 15 | 8 | 33.1k | meas c=10 |
| **projected total FLOP** | | | **212.8k** | bounded 727.7k |

**Projected: 91/91 diagonal, 0 butterfly, 0 runtime H, 16 free born ⟹ FLOP = 212.8k = 1.00× clifft
(parity)**, vs current bounded 727.7k (3.42×) and B1 born-X 719.1k (3.38×). Event correspondence is 1:1
(each bounded T ↔ a clifft array_t/array_t_dag; each promote ↔ a free expand; each entangler ↔ a free
array_cnot). Rank: bounded peak 10 = clifft peak 10 (cultivation ties on rank; the FLOP, not the memory,
is the target here).

(cultivation_d3 cross-check, k=4: clifft total 1.80k; design projects the same 29 diagonal T's / 0 H.)

---

## 8. Generality boundary

| regime | behaviour | bound |
|---|---|---|
| **R_Z / T + Clifford, no active-axis H** (cultivation) | all rotations diagonal; 0 forced array_h | **parity, FLOP = clifft** |
| **active-axis non-Pauli basis change** (clifft array_h > 0) | explicit `_h_axis` fallback, count = clifft's array_h | bounded matches clifft kernel-for-kernel ⟹ **≤ clifft** (plus rank headroom) |
| **AG-measure-heavy** (distillation, coherent_*: AG>0) | inherited `_ag_measure` (frame-only); injected non-Pauli Cliffords may force the §8 array_h above | ≤ clifft on the diagonal part; array_h tracks clifft |
| **R_X / R_Y off-axis** (coherent_rx/ry, d5_r5 RY) | generator genuinely off-diagonal ⟹ existing **localizer/butterfly fallback** (unchanged) | NOT claimed as parity; current behaviour preserved |

So the parity claim is scoped to the **diagonal (R_Z/T + Clifford)** regime; everything else degrades
*gracefully* to the existing exact fallbacks, never worse than today.

---

## 9. Implementation phases, each gated by differential/bit-exact shadow

All phases are **default-off** (`clifft_axis_policy3`/a new `clifft_axis_symframe` flag) and reuse the
Step-B0 differential shadow + a05843e bit-exact harness as oracles; nothing authoritative changes until
the final gate.

- **S1 — shadow factorization (no behaviour change).** Maintain C_sym / P_res / born-basis ALONGSIDE the
  existing U_C, non-authoritative. Per gate, assert invariant (C) and that `C_sym·P_res·B` reconstructs
  U_C. Validate on all 9 circuits. *Gate:* factorization exact, (C) holds for cultivation with 0 forced
  array_h (= clifft).
- **S2 — diagonal T routing (differential).** Compute each T's generator through C_sym; confirm it is a
  Z-string for cultivation (91/91) and off-diagonal→fallback for R_X/R_Y. Compare the resulting state to
  the butterfly source-of-truth per-T (the existing phase8 harness). *Gate:* 91/91 diagonal, machine-
  precision state match, fallbacks correct.
- **S3 — free born materialization.** Replace the c=4 `_h_axis` born with the expand-style |+⟩ copy
  (0 FLOP). *Gate:* born cost 0, state unchanged.
- **S4 — measurement / promote / demote integration.** Wire the diagonal-measurement, free promote, and
  localize-and-drop to the symbolic frame. *Gate:* records / rank / p0 **bit-exact vs a05843e** on all 9
  circuits (allowing the documented state-exact core-log reclassification on AG-heavy circuits, §B1).
- **S5 — authoritative under default-off flag + FLOP.** Flip the flag, measure FLOP vs clifft. *Gate:*
  cultivation FLOP ≤ clifft (target 212.8k = parity); no regression elsewhere; all fallbacks intact.

No code, no flag flip, no fallback deletion, no localizer ungate, no threshold change until this design
is approved. The risk (this is the R_Y-frame-bug class) is contained by S1–S2 being pure shadow/
differential checks before any authoritative change.
