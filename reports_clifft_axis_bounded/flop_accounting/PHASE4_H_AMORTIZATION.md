# Phase 4 — Is cultivation's localization-H amortizable? (analysis only, no implementation)

Tests whether the cultivation 91-T residual comes from `F = q(4+3)2^r` (one H per rotation) or can be
batched to `F = (4B + 3q)2^r` with `B ≪ q` basis transitions. **It CORRECTS the Phase-3 claim that
cultivation is a hard structural counterexample at (7/3)·F_clf.** Source: scripts/phase4_generator_
trace.py, data/phase4_cultivation_generators.txt (real `_flush_one` pullback trace).

## 1. The 91 generators (cultivation_d5)

Every materialized T pulls back to a PHYSICAL Pauli generator P_i = U_C† Z_{q_i} U_C. The trace
(full table in the data file) shows:
- **All 91 are off-diagonal** (X-character present) in bounded's Z-basis magic register — this is
  why the current code uses the c=12 butterfly.
- **Within each flush-batch, all generators pairwise COMMUTE** — VERIFIED: **0 within-batch
  anticommuting pairs** (a batch = the rotations flushed together before one measurement, under the
  SAME frame; pullback of mutually-commuting Z-type generators by a fixed Clifford commutes).
- **The full 91 do NOT all pairwise commute** — VERIFIED: **388 anticommuting pairs out of 4095**,
  ALL cross-batch (e.g. batch-pairs (5,7)=104, (5,8)=52). A measurement between batches mutates the
  frame, so generators from different batches generally anticommute. The earlier phrasing "all 91
  pairwise commute" was WRONG (it would imply 1 run; there are anticommuting pairs).
- The greedy ordered partition into *pairwise-commuting* runs gives **2 runs (73 + 18)**, but run 1
  spans batches 0–6 and therefore CROSSES measurement boundaries (axis drop / promote / projection)
  — so it is NOT a realizable single basis (see §4). Within-batch is the only measurement-free unit.
- They are X-prepped (early ones pure-X: P_0 = X_0, P_4 = X_0X_1X_2): the magic states are cultivated
  in the X-basis, so the T's Z-generator appears as an X-string in the computational magic register.

## 2. Simultaneous diagonalization — two distinct quantities: transitions B vs H-sweeps Σh

TERMINOLOGY (the earlier draft conflated these): a **basis transition** builds one diagonalizing
Clifford V; that single V can contain **multiple Hadamard sweeps** — `h_j` = GF(2) rank of the
X-block of the generators diagonalized at transition j (CNOT/CZ/S reduction is free, 0-FLOP). The
FLOP-relevant cost is the **H-sweep total Σh_j**, NOT the transition count B:

    F = Σ_j 4·h_j·2^{r_j}   +   Σ_i 3·2^{r_i}        (H-sweeps)           (diagonal rotations)

| policy | basis transitions B | H-sweeps Σh | realizable? |
|---|---:|---:|---|
| A. per-rotation (current localizer) | 91 | **91** (h_j=1 each) | yes |
| B. **per flush-batch** simul-diag | 14 | **47** (Σ_batch X-rank) | **YES — credible** (each batch frame-fixed, within-batch commuting, measurement-free) |
| C. 2 commuting runs | 2 | 25 (X-ranks 16+9) | **NO — candidate only** (run 1 crosses 6 measurements) |
| (single global basis) | 1 | 16 | unrealizable (only ≤10 of 16 qubits live at once) |

Per-batch X-ranks: b0:4(n8) b1:3(n4) b2:2(n2) b4:10(n19) **b5:10(n38)** b6:3(n4)… — the big batch
(38 rotations) needs only **10** H-sweeps, not 38. **Σh ≪ q is real for the realizable per-batch
policy (47 ≪ 91).**

## 3. Three-policy FLOP comparison (Σ over each rotation's own 2^rank; H=4, diag=3)

| policy | FLOP | × clifft | H-sweeps | status |
|---|---:|---:|---:|---|
| current butterfly (c=12) | 659.6k | 4.00× | — | current |
| A. per-rotation localize | 384.8k | 2.33× | 91 | (Phase-3 assumption) |
| **B. per-batch simul-diag** | **269.9k** | **1.64×** | 47 | **credible & realizable** |
| C. 2 commuting runs | 267.3k | 1.62× | 25 | candidate (unproven across meas.) |
| clifft diagonal-T (3q, **0 H**) | 164.9k | 1.00× | 0 | measured |

The credible, realizable improvement is **4.00× → 1.64×** (per-batch). The 1.62× (25-H) figure is a
candidate/lower-bound — it assumes a basis held across 6 measurements, which is not established.

## 4. Verdict on the core question — **conclusion 3 (partially amortizable), NOT 2 (structural)**

The Phase-3 figure (7/3 = 2.33×) assumed policy A (one H per rotation). **That residual is
implementational.** Per-batch simultaneous diagonalization (within-batch generators are VERIFIED
pairwise-commuting, frame-fixed, measurement-free) amortizes H-sweeps **91 → 47**, dropping the
residual to **~1.64× clifft** — exact and realizable (a smarter localization schedule over the
existing in-place Clifford kernels; the batch boundaries ARE the measurements, so nothing crosses a
measurement). Pushing further (25 H-sweeps / 1.62×) would require holding a basis ACROSS
measurements — NOT established (§1), so it is a candidate, not a result.

**It is NOT Σh = q forced.** To claim that one would have to show every T needs its own H; the trace
disproves it (the 38-rotation batch shares ONE 10-H-sweep basis). But it is also NOT "91 → 25
guaranteed": the full set has 388 cross-batch anticommuting pairs, so a single run-spanning basis is
not free.

The **remaining ~1.6× → 1.0× gap** is the B ≈ 25–47 H-sweeps bounded still does where clifft does
**zero** array_h (clifft: confirmed 0 `array_h` on cultivation; it applies the X-prepped T's diagonally
via its p_x-aware `array_t` kernel — the magic axis is stored in the X-eigenbasis, so the T is diagonal
with no Hadamard). Whether bounded can reach clifft's 0-H is the **materialization-basis question**:
choose each magic qubit's promote-basis (free at materialization) so its T's land diagonal. clifft
proves 0-H is achievable in *some* representation; for bounded's compressed register it is plausible
but bounded by the multi-qubit coupling (Xw=3,5,7 generators) and intervening measurements — an
offline schedule problem, not yet proven. **So the honest structural floor is ≤ 1.6×, possibly →1.0×;
cultivation is not a clean 2.33× counterexample.**

## 5. Rank-loss claim — REFUTED (code-verified)

The four operations are distinct and must not be conflated:
| operation | rank effect | code |
|---|---|---|
| **in-r-axis basis change** (apply Clifford V to the magic register) | **rank UNCHANGED** | `_h_axis`/`_s_axis`/`_cnot_axes` reshape `self.phi` in place; `r = len(self.M)` untouched |
| new-axis materialization | rank +1 | `_promote`: `self.M.append(q)`, buffer 2^r→2^{r+1} |
| full physical Clifford (all n qubits) | rank → n (= clifft) | not done by bounded |
| symbolic frame gauge change | no array | `right_*`, 0-FLOP |

An in-place unitary on a rank-r register is a basis rotation of the SAME 2^r amplitudes ⟹ **does not
grow rank**. So bounded can apply the amortized diagonalizing Cliffords (the B H-sweeps) **in place at
r ≤ k without losing the rank advantage.** The claim "applying Cliffords to the state ⟹ become clifft ⟹
lose rank" conflates the in-register basis change (free of rank cost) with full physical
materialization (rank cost) — they are different. The batched-localization amortization is therefore
compatible with the bounded representation.

## 6. Deliverables summary

1. **91-generator sequence:** data/phase4_cultivation_generators.txt — all off-diagonal,
   **within-batch pairwise commuting (0 anticommuting); full set 388 cross-batch anticommuting pairs**,
   X-prepped, 14 flush-batches.
2. **Commuting statistics:** every batch internally pairwise-commuting (0 within-batch anticommuting);
   greedy ordered partition = 2 pairwise-commuting runs (73+18) but they cross measurements; per-batch
   X-ranks 1–10 (the 38-rotation batch needs 10 H-sweeps, not 38).
3. **H-sweeps Σh (the FLOP driver) and transitions B:** per-batch **Σh = 47** (B=14 transitions) —
   CREDIBLE & realizable; 2-run Σh = 25 (B=2) — CANDIDATE, crosses measurements; single-basis Σh = 16
   — unrealizable (≤10 of 16 qubits live at once). **Realizable: Σh = 47 ≪ q = 91.**
4. **Projected FLOP:** per-rotation 2.33× / **per-batch 1.64× (credible)** / 2-run 1.62× (candidate) /
   clifft 1.00×.
5. **Final judgment:** the 2.33× residual is **implementational** (per-rotation H policy); per-batch
   simultaneous diagonalization is exact/realizable and drops it to **1.64×**. Whether the remaining
   1.64× → 1.0× is reducible (clifft does 0 array_h) is OPEN — the materialization-basis question, not
   a proven structural floor. **Conclusion (3), not (2); and the credible gain is 4.0× → 1.64×, with
   25-H / 1.62× and the 1.0× parity both still unproven.**
6. **Lower-bound argument (the residual that IS forced under in-register basis only):** with the
   magic register fixed in the Z-basis and only in-register Cliffords, each measurement-batch must
   spend ≥ the GF(2) X-rank of its generators in H-sweeps (you cannot diagonalize d independent
   X-directions with fewer than d Hadamards). That gives the realizable per-batch **Σh = 47**. It is
   strictly below the per-rotation 91 and the c=12 butterfly. It vanishes toward clifft's 0 only if the
   promote-basis is chosen to pre-diagonalize the future T's (Phase-5 offline-schedule question) —
   unresolved.
7. **Simplest amortized schedule (design only, NOT implemented):** at each measurement flush, collect
   the pending (commuting) rotations, build ONE Clifford V that simultaneously diagonalizes them
   (free CNOT/CZ/S reduction + GF(2)-X-rank Hadamards), apply V in place (rank unchanged), apply all
   rotations as diagonal half-array `rot:diaghalf` (c=3), then proceed to the measurement. This
   replaces 38 butterflies (c=12) or 38 per-rotation H's with 10 shared H's + 38 diagonal rotations.
   No new T kernel, no size gate, no per-circuit branch — just batch the existing localizer over the
   commuting set instead of per-rotation.
