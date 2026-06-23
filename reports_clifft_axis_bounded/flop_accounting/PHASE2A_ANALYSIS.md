# Phase 2A — analysis & design (NO general diagonalization code until approved)

Baseline: Phase 1 frozen at commit `a0e67f7`, tag `phase1-sqnorm`. Everything below is
**measured or derived**, not assumed. The four review deliverables (§9 of the spec):
§1 post-Phase-1 attribution, §2 clifft's real rotation cost, §3 the state-preserving
transform, §4 design A/B/C + recommendation.

---

## 0. Resolving the paradox first — "memory smaller, FLOP larger" is NOT fundamental

The user's objection: bounded's amplitude-volume `W1 = Σ2^r` is always ≤ clifft's, so FLOP
larger than clifft "makes no sense." **It is not fundamental — it is bounded's rotation
KERNEL being suboptimal.** The "12·2^r vs 3·2^k" framing was wrong because it ignored what
clifft actually pays to make a rotation diagonal. Measured facts (clifft C++ source):

- clifft `array_cnot` / `array_swap` = `CLIFFT_COST(...,0,0,0,0,0)` = **0 FLOP** (pure
  permutation — amplitude moves only). [`svm_kernels.inl:126,374`]
- clifft `array_h` = `cadd=2^k` + `rcmul=2^k` = **4·2^k**. [`svm_kernels.inl:649`]
- clifft `array_rot` = `cmul=2^{k-1}` = **3·2^k** (diagonal half-array). [`svm_kernels.inl:1072`]

So clifft turning a **weight-w** off-axis Pauli rotation into a diagonal one costs:

> **(w−1) CNOT [0 FLOP, free permutation] + 1 array_h [4·2^k] + 1 array_rot [3·2^k] ≈ 7·2^k,
> independent of w.**

bounded instead applies the whole thing as **one off-diagonal butterfly = 12·2^r**
([`engine.py:213`], `rot:offdiag`). So the true per-rotation gap at equal rank is **12 vs 7 ≈
1.7×**, not 4×. The paradox dissolves: bounded does an unnecessary full butterfly where clifft
collapses the X-string with free CNOTs and pays a single H. Phase 2 = adopt clifft's pattern.

---

## 1. Post-Phase-1 FLOP attribution (3 backends, exact ΔF)

One FLOP convention (cmul6 rcmul2 cadd2 sqmag4 vdot8), compile-time matrix algebra & memcpy
excluded, complete totals (incl. the normalization divide). Harness:
`scripts/phase2a_attribution.py`; data `data/phase2a_attribution.csv`. Categories from each
backend's REAL kernel events. ΔF = F(bounded P1) − F(clifft-unfused), decomposed per category;
the decomposition sums **EXACTLY** to the measured gap (residual 0) on every circuit.

| circuit | clifft-unf | bnd orig | bnd P1 | **gap (P1−clf)** | = diag_rot | + offdiag_rot | + array_Cliff | + Born/sqnorm |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| coherent_ry_d3_r1 | 12.29M | 22.47M | 17.22M | **+4.93M** | −4.12M | **+14.92M** | −5.68M | −0.20M |
| coherent_ry_d3_r3 | 36.55M | 64.96M | 48.36M | **+11.81M** | −12.34M | **+44.23M** | −19.50M | −0.57M |
| cultivation_d3 | 1.80k | 6.23k | 5.53k | **+3.74k** | 0 | **+4.82k** | −1.03k | −0.06k |
| cultivation_d5 | 212.82k | 837.69k | 731.79k | **+518.97k** | 0 | **+684.41k** | −161.30k | −4.14k |
| coherent_rx_d3_r1 | 870.38k | 522.36k | 254.10k | **−616.27k** | −466.55k | +190.34k | −31.71k | −308.35k |
| coherent_rx_d3_r3 | 2.59M | 3.28M | 1.33M | **−1.26M** | −1.39M | +1.01M | −99.46k | −781.44k |
| coherent_d3_r3 | 52.56k | 21.44k | 15.53k | **−37.04k** | −42.95k | +12.36k | −3.69k | −2.75k |
| coherent_d5_r5 | 17.99G | 49.62M | 25.11M | **−17.96G** | −16.91G | +20.74M | −536.70M | −534.81M |
| distillation | 1.90k | 1.86k | 1.56k | **−0.34k** | +0.15k | +0.46k | −0.73k | −0.22k |

**Answer to "is the remaining RY/cultivation gap entirely off-diagonal rotation?"** — by the
exact decomposition: the gap = **offdiag_rot (the only positive term)** MINUS the savings
bounded already banks in diag_rot, array_Clifford, and (post-Phase-1) Born. For RY/cult the
offdiag term exceeds the savings → net loss; everywhere else the savings dominate → bounded
wins. **There is NO unexplained residual** (every row sums exactly). Two structural notes:

- bounded's **array_Clifford is far BELOW clifft's** (ry_d3_r1 1.38M vs 7.05M; d5_r5 0.04M vs
  537M) — bounded's frame-deferral genuinely saves Clifford work. clifft pays array_h/cnot to
  localize; bounded pays one big butterfly instead.
- Post-Phase-1 **Born is now a bounded WIN** in every circuit (ry_d3_r1 0.92M vs clifft 1.11M;
  d5_r5 2.36M vs 537M). The Phase-1 sqnorm removal already flipped that category.

So the **single** lever left is `offdiag_rot`. It is large because the kernel is a 12·2^r
butterfly, not because bounded touches more state (W1 ≤ clifft always).

---

## 2. clifft's REAL rotation localization — call path and per-rotation cost

Traced in `/home/jung/clifft/src/clifft/`. clifft's factored state is **identical in form to
bounded's**: `|ψ⟩ = γ · U_C · P · (|φ⟩_A ⊗ |0⟩_D)` (U_C = Clifford/tableau frame, P = Pauli
frame, |φ⟩ = dense active register) [`svm.cc:714-724`]. The mechanism:

| step | clifft fn / file | dense sweep? | FLOP | frame/tableau update |
|---|---|---|---|---|
| pull generator back to virtual Z | `frontend.cc:469-496` `trace_rz` + `prepend_H_XZ/YZ` | **no** | 0 | inverse tableau (compile-time) — RX/RY's X/Y→Z H is folded here, **not** on the array |
| collapse multi-qubit X-string | `backend.cc:418-517` `localize_pauli` → `emit_cnot/cz` | only if pivot **active** | **0** (`array_cnot`/`cz`/`swap` are permutations) | `frame_cnot`/`frame_cz` when dormant |
| route pivot to Z | `backend.cc:362-389` `route_to_active_z` → `array_h` | yes, **iff active & X-basis** | **4·2^k** (one H) | else `frame_h` (dormant) — free |
| diagonal rotation | `svm_kernels.inl:1072` `exec_array_rot` (reads `p_x` for sign) | yes | **3·2^k** | Pauli-frame `p_x[v]` flips the angle sign — an X-frame axis stays **diagonal** |
| **per off-axis rotation total** | | | **≈ 7·2^k** | (weight-independent; CNOTs free) |

Key sub-findings:
1. **clifft does NOT get diagonal rotations for free.** It pays one `array_h` (4·2^k) per
   off-axis rotation whose pivot is active, plus free CNOTs. The "array_Clifford" bucket in §1
   (clifft 7.05M for ry_d3_r1) *is* mostly these rotation-localization H's + measurement H's.
2. **The Pauli frame handles the X/Y character diagonally** (`exec_array_rot` with `p_x`): an
   axis that is physically X/Y appears as a Z-rotation with a sign flip — still half-array
   diagonal. No butterfly. This is the trick bounded is missing.
3. **CNOTs/SWAPs are free** in both backends (permutations, 0 FLOP), so collapsing a weight-w
   X-string to one axis is free; only the final single H costs.
4. Fused vs unfused localize identically; fusion only bundles the diagonal rotation with
   neighbours into U2/U4 (a memory-pass, *raises* FLOP) — irrelevant to localization.

**The boxed question — "what does the basis change cost beyond 3·2^k?"** Answer: **one
array_h = 4·2^k per off-axis rotation** (CNOTs free, S negligible). Real per-rotation total
≈ **7·2^k**, vs bounded's current **12·2^r**.

---

## 3. State-preserving transform (the invariant) — and it REUSES verified code

bounded's representation (exactly as in code):

  |Ψ⟩ = γ · C · F · U_pend · (|φ⟩_A ⊗ |0⟩_D)

C = Clifford frame (tableau, `Zc`/`Xc`), F = Pauli frame, U_pend = deferred rotations,
|φ⟩ = dense magic register on the active axes.

Flushing a rotation `R_P(θ)` (P a physical Pauli). The operator that must act on the core,
after the anticommuting-core flush, is the pullback `R_{P'}(θ)` with
**P' = (C F)† P (C F) = i^{pp} X^{mx} Z^{mz}** on the magic register — this is exactly bounded's
existing `_pullback` + `_masks` ([`engine.py:256-258`]). Today, if `mx≠0`, bounded applies
`R_{P'}` as an **off-diagonal butterfly** (`_pauli_lincomb_inplace(..,":offdiag")`, 12·2^r).

**Diagonalization.** Pick a magic-register Clifford **V** with **V P' V† = ±Z_a** (single
diagonal axis a). Then `R_{P'}(θ) = V† R_{Z_a}(θ) V`, so

  R_P |Ψ⟩ = γ C F U_pend **V† R_{Z_a}(θ) V** |φ⟩.

Absorb V into the frame instead of undoing it on the array. Define

  |φ_new⟩ = R_{Z_a}(θ) · V|φ⟩       (apply V to the array, then a **diagonal** R_Z),
  C_new such that C_new F_new = C F V†   (right-multiply the frame by V†).

Then **|Ψ_new⟩ = γ C_new F_new |φ_new⟩ = γ C F V† R_{Z_a} V |φ⟩ = R_P|Ψ⟩** ∎ — the physical
state is preserved exactly, and V is applied to the dense array **once** (not undone).

What each object does (must all hold, else the state drifts — these are the old RY/CZ bug
sites, to be re-asserted):
- **|φ⟩** → `V|φ⟩` then a diagonal `R_{Z_a}(±θ)` (sign from the i^{pp}/frame parity).
- **C (Clifford frame)** → `C · V†` via `right_h/right_s/right_cx` (the SAME calls
  `_localize_to_Z` already uses for measurement).
- **F (Pauli frame)** → unchanged in form; the i^{pp} phase sets the R_Z **angle sign**.
- **U_pend (other deferred rotations)** → their generators must be conjugated by V (pullback
  through the new C is automatic since C changed; pending stored in lab frame are re-pulled at
  their own flush). Must verify no double-conjugation (the CZ bug).
- **measurement observables** → pulled back through the new C at measure time (automatic).
- **γ (global phase)** → carries the i^{pp} phase of P'; no separate sweep.
- **active/dormant mapping** → V acts only on active axes; dormant stay |0⟩.

**Cost is NOT zero (per the spec's warning).** V = (CNOT-collapse of the weight-w X-string,
**free**) + (one H to send the single residual X→Z, **4·2^r**) + (S if Y, small). The CNOTs
are free permutations; the **single H is a real 4·2^r dense sweep that MUST be counted.** Net
per off-axis rotation: **V (≈4·2^r) + R_Z (3·2^r) = 7·2^r**, replacing 12·2^r.

**This reuses bounded's already-verified localizer.** `_localize_to_Z` ([`bounded.py:179`])
already does exactly "localize a magic-register Pauli to ±Z_a, apply to φ, fold V† into the
frame via right_*, track the sign by conjugating P through V" — and it survived the RY/CZ
frame-bug fixes. Phase 2 applies the *same* routine to a rotation generator instead of a
measurement observable, then calls the **existing** diagonal kernel.

**Caveat to prove in 2B (not assume):** the verified `_localize_to_Z` applies an H to *each*
support qubit *then* collapses (w H's). To hit clifft's **1-H** cost it must be reordered to
**collapse-with-CNOTs-first, then one H** (clifft's order). The 1-H vs w-H choice is the whole
ballgame (see §4 sensitivity) and must be measured, not assumed.

---

## 4. Designs A / B / C

Per-rotation dense cost (rank r, weight w, CNOTs free):

| design | rotation FLOP | basis-change FLOP | extra sweeps | memory-bound risk | ledger difficulty | exactness risk |
|---|--:|--:|--:|---|---|---|
| **A** per-rotation localize→diag (reuse `_localize_to_Z`, frame-kept) | 3·2^r | **4·2^r** (1 H, if collapse-first) | 1 H/rot | none (V in place, CNOT permute) | low–med (reuse proven code) | med (frame; reuse cuts it) |
| **B** canonicalize whole core then run | 3·2^r·(diag share) | amortized 1 transform/core | few/core | none | **high** (symplectic elim, anticommuting pairs can't all be Z) | high |
| **C** offline precompiled core template | 3·2^r | runtime-bound dense ops only | per template | none | med (compile-time heavy) | med (binding bugs) |

**Sensitivity that decides everything — 1-H vs w-H localization.** Projected bounded total
using the model "offdiag 12·2^r → diag 3·2^r + H·(#H)·2^r·4" :

| circuit | clifft-unf | bnd P1 | **A, 1-H** (7·2^r) | A, naive w-H | verdict (1-H) |
|---|--:|--:|--:|--:|:--:|
| coherent_ry_d3_r1 | 12.29M | 17.22M | **11.0M (0.90×)** | ~17M (≈1.4×) | ✓ win |
| coherent_ry_d3_r3 | 36.55M | 48.36M | **29.9M (0.82×)** | ~48M | ✓ win |
| cultivation_d3 | 1.80k | 5.53k | **3.5k (1.96×)** | ~5k | ✗ still lose |
| cultivation_d5 | 212.82k | 731.79k | **447k (2.10×)** | ~730k | ✗ still lose |
| coherent_rx_d3_r1 | 870.38k | 254.10k | **175k (0.20×)** | ~254k | ✓ win more |
| coherent_rx_d3_r3 | 2.59M | 1.33M | **0.91M (0.35×)** | ~1.3M | ✓ win more |
| coherent_d3_r3 | 52.56k | 15.53k | **10.4k (0.20×)** | ~15k | ✓ win more |
| coherent_d5_r5 | 17.99G | 25.11M | **16.5M (0.0009×)** | ~25M | ✓ unchanged huge win |
| distillation | 1.90k | 1.56k | **1.37k (0.72×)** | ~1.5k | ✓ win |

Two honest conclusions:
1. **The 1-H (collapse-first) localization is mandatory.** With naive w-H, the added H's
   roughly cancel the off-diagonal saving (no win). The entire Phase-2 payoff hinges on
   collapsing the X-string with free CNOTs **before** the single H — exactly clifft's order.
2. **RY flips to a win (0.82–0.90×); RX/d5_r5/distillation win more; cultivation still loses
   (~2×).** cultivation has *no rank advantage* (peak = k) and clifft's T-magic localizes very
   cheaply (clifft total is array_Clifford-dominated, 213k); bounded's magic-register T is
   inherently the off-axis kind. Phase 2 improves cult 3.4×→2.1× but does **not** undercut
   clifft there. I will NOT claim otherwise.

**Recommendation: Design A (collapse-first 1-H localization, reusing `_localize_to_Z`,
frame-kept).** Lowest total FLOP, reuses the one piece of code already hardened against the
RY/CZ frame bugs, no memory-bound risk (V applied in place, CNOTs are permutations), and the
ledger update is the existing `right_*` path. Design B's symplectic canonicalization is
higher exactness risk for no extra FLOP win (anticommuting generators force off-diagonal pairs
anyway). Design C only helps if a core's rotations share a basis — QEC noise rotations sit on
different qubits, so the template rarely amortizes; keep C as a *later* optimization if 2B
shows the frame-kept V inflates subsequent pullbacks.

---

## What I am asking you to approve

1. **§1** — the post-Phase-1 attribution and that the *only* positive gap term is `offdiag_rot`
   (exact, residual 0).
2. **§2** — clifft's real per-rotation cost is **≈7·2^k (1 array_h + diag, CNOTs free)**, so the
   honest gap is ~1.7× per rotation, not 4×.
3. **§3** — the frame-absorption transform `C_new F_new = C F V†`, `|φ_new⟩ = R_{Z_a} V|φ⟩`,
   reusing the verified `_localize_to_Z`, with the H cost (4·2^r) explicitly counted.
4. **§4** — **Design A, collapse-first 1-H localization**, with the honest projection: RY/RX/
   d5_r5/distillation win, **cultivation stays ~2× above clifft**.

On approval I will implement **Phase 2B step-by-step** (already-Z rotations → single-axis X/Y →
multi-axis), each step bit-exact vs Phase-1, counting every added H in the total, and re-running
the full exactness suite (record/rank/p0/memory-bound + the RY/CZ/i^p regression cases). I will
**not** write the general diagonalization until you approve this design.
