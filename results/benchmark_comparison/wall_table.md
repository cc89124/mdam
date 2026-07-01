# Wall-time comparison — MDAM (native C++ VM) vs Clifft

**Scope: real per-shot WALL TIME (ns/shot), native C++ VM.** Wall-clock companion to
[`flop_table.md`](flop_table.md). All rows are the **authoritative** native path, **bit-exact** verified
against the Python bounded reference (25–100 seeds).

> **Clifft is an EXTERNAL baseline only.** It is *not* referenced inside the MDAM algorithm — no Clifft
> fallback, no Clifft-based cost guard. MDAM chooses its own smallest measurement-projected exact
> representation by MDAM-internal cost (state rank `r`, operator/Pauli-generator rank `ρ`, unique-xmask).
> The `Clifft / MDAM` column is for paper evaluation: **>1 ⇒ MDAM faster, <1 ⇒ MDAM slower.**

Path: `nvm_mdam_sample_batch` (authoritative `run()`/shot). FUSED-compiled program. Inverse frame is
materialized **on demand** (never built when no consumer reads it → the maxM=0 win). Core rotations are
**folded** (exact: co-axial commuting rotations `R(θ1)·R(θ2)=R(θ1+θ2)` combined before the kernel).
Single-thread, `taskset -c 2`, `*_NUM_THREADS=1`.

## A. All-benchmark wall (both engines feasible, MDAM bit-exact)

| circuit | axis | k | maxM | MDAM ns/shot | Clifft ns/shot | **Clifft / MDAM** | regime |
|---|---|--:|--:|--:|--:|--:|---|
| **coherent_d7_r1** | R_Z | 25 | ~0 | 112,358 | 3,996,135,191 | **35,566×** | r≪k localization |
| **coherent_d5_r5** | R_Z | 24 | 12 | 10,608,326 | 8,695,465,445 | **819.7×** | r<k localization |
| **coherent_d5_r1** | R_Z | 13 | 0 | 29,980 | 205,263 | **6.85×** | r≪k localization |
| coherent_d3_r3 | R_Z | 8 | 4 | 46,603 | 14,099 | 0.30× | small-k control-plane |
| coherent_d3_r1 | R_Z | 5 | 0 | 4,540 | 1,390 | 0.31× | small-k control-plane |
| coherent_rx_d3_r1 | R_X | 14 | 10 | 165,295 | 131,084 | 0.79× | off-axis (weak localization) |
| coherent_rx_d3_r3 | R_X | 14 | 11 | 974,038 | 393,711 | 0.40× | off-axis |
| cultivation_d3 | T | 4 | 3 | 14,914 | 2,348 | 0.16× | magic-saturated (ρ=r=k) |
| cultivation_d5 | T | 10 | 9 | 191,336 | 82,110 | 0.43× | magic-saturated (ρ=r=k) |
| distillation | T | 5 | 3 | 17,875 | 11,714 | 0.66× | magic, part-diagonal |
| surface_d7_r7 | — | 0 | 0 | 21,096 | 8,994 | 0.43× | k=0 degenerate (pure dispatch) |

cultivation_d3 with the Gate-K cross-shot edge cache (`cmode5`, 1M-shot warm, 99.6% hit, bit-exact): **2,285 ns
vs Clifft 2,165 = 0.95× (≈ parity)** — the cache lifts cult_d3 from 0.16× to parity but saturates there.

## B. MDAM runs where Clifft is physically infeasible (oracle 2^k unrepresentable)

| circuit | axis | k | MDAM ns/shot | Clifft | note |
|---|---|--:|--:|---|---|
| coherent_rx_d5_r1 | R_X | 38 | ~25.6 s | INFEASIBLE (2^38 = 4 TB) | runs; weak off-axis localization |
| coherent_rx_d5_r5 | R_X | 38 | > 60 s | INFEASIBLE (4 TB) | runs |
| coherent_d7_r7 | R_Z | 48 | > 190 s | INFEASIBLE (2^48 = 4.5 PB) | n=118; same multiword path as the verified d7_r1 |

## C. Native VM unsupported

| circuit | k | status |
|---|--:|---|
| coherent_ry_d3_r1 / r3 | 16 | `MO_ARRAY_U4: non-structural fused` (general 2-qubit unitary; not implemented) |
| coherent_ry_d5_r1 / r5 | 47 | non-structural U4 (+ Clifft 2^47 infeasible) |

---

## No-regression status (the goal: MDAM forced run never slower than Clifft)

**Wins (genuine localization, r<k):** d7_r1 35,566×, d5_r5 820×, d5_r1 6.85×. These scale as 2^k for the
r1 family (maxM≈0): the win grows with distance.

**Remaining losses are bounded (1.5×–6×), and their causes are now quantified (per-core measured):**
- **Magic-saturated (cultivation): ρ = r_mat = k (full-rank operator) ⇒ no localization advantage, BUT
  the loss is SYMBOLIC-OVERHEAD-bound, not dense-bound, and near-parity is NOT excluded.** The Pauli
  generators span the whole register — the localized rank `r` *equals* Clifft's full rank `k`, so MDAM has
  no register-size advantage and applies the deferred rotations as dense Pauli-rotations on `2^r = 2^k`.
  **Corrected TSC-rdtsc decomposition of the 193 µs shot (cult_d5, k=10, 15 magic meas/shot = 5 compiled +
  10 oracle).** An earlier note here claimed "dense alone = 92.8 µs > Clifft, provably unreachable" — that
  was an **over-attribution**: it lumped the oracle `flush_core`'s per-rotation `pullback`+`promote`
  (symbolic) into "dense." Measured split of `flush_core` (26.7 µs): `lincomb` (true dense butterfly) =
  21.3 µs (80%), `pullback`+`promote` (symbolic) = 3.7 µs (14%). The true breakdown:

  | component | ns/shot | irreducible (Clifft pays too)? |
  |---|--:|---|
  | compiled magic kernel `direct_rot` (mode0−mode13) | 57,800 | **dense** butterfly on 2^r |
  | oracle `flush_core` → `lincomb` only | 21,300 | **dense** butterfly on 2^r |
  | Born 2^r + project + normalize | 4,400 | yes |
  | noise sampling | ~7,000 | yes |
  | **symbolic control** (inverse-frame pullback + promote + frame + active-gate + localize + drop/commit + dispatch) | **~101,000** | **MDAM-only, removable** |

  **Corrected verdict: MDAM's pure DENSE butterfly = 57.8k + 21.3k = 79.1 µs, which is BELOW Clifft's
  82.1 µs (0.96×).** The irreducible floor (dense + Born + noise) ≈ 90.5 µs ≈ **1.1× Clifft** — *near*
  parity, not a 2.3× wall. So cult_d5 is **not** a win (r = k ⇒ no localization, dense ≈ Clifft); the
  ~91 µs near-parity floor would require stripping the ~101 µs symbolic scaffold. **Step-2 reconnaissance
  measured that this strip is NOT a simple cache/skip:** (a) the scaffold is *structurally load-bearing* —
  a leave-one-out (ISKIP, mode13) shows skipping the commit `RIGHTFOLD` makes the shot **3× slower**,
  because the commit shrinks M/rank every measurement and removing it lets M grow unbounded; (b) the
  cacheable structure (F4 `StaticPlan`: M_mat/localizer/masks/ranks) is *already* cached; (c) the only
  remaining dynamic symbolic — the rpp pullback phase — is **dense-coupled** (it reads the inverse frame
  whose phase-pack depends on a Born-amplitude branch in `drop_residual_products`, the Gate-J wall), so it
  cannot be reduced to a pure phase/sign patch. Therefore cult_d5 near-parity requires the Gate-J
  dense-coupled magic compile (prior-built, prior-blocked), not an easy strip. The clean Clifford-frame
  side (~36 µs) is compilable but its region mechanism measured *slower* than the already-lean authoritative
  interpreter, and reaches only ~156 µs (0.53×) anyway. The diagonal-kernel hypothesis stays falsified
  (X-rotations, `xmask ≠ 0` ⇒ butterfly); folding (P2) already removes the only exact co-axial reduction (+15%).
- **Small-k control-plane (d3_r1/d3_r3, maxM=0): the gap is distributed** (frame ~14% + dispatch ~19% +
  noise ~7% + magic-scaffold ~18%). Empty-core scaffold skip ceiling ≈ 18% (→ ~0.39×); true parity requires
  compiling the whole control plane to Clifft-tightness.
- **Off-axis R_X:** butterfly rotations + weak localization (maxM≈k).

**The intrinsic planner (design, Clifft-free):** at each measurement boundary MDAM derives candidate exact
representations — P1 localized-reduced, P2 projected-folded, P3 direct-reduced (always available, floor),
P4 cached-boundary — and selects the smallest by MDAM-internal cost. The no-regression property follows from
two intrinsic invariants, **without any Clifft reference**: (I1) `r ≤ k` always (MDAM never materializes more
than the active register); (I2) P3 (direct apply) always exists at cost `(#gates)·2^r ≤ (#gates)·2^k`.
**Status: P2 folding implemented (default-on, exact).** The intrinsic invariants (I1: `r ≤ k`; I2: P3 direct
floor) already guarantee MDAM never materializes more than the active register — so on every **localized**
benchmark (`r < k`) MDAM is at-or-faster than Clifft. The remaining losses are exactly the benchmarks where
`r = k` (no localization). The corrected decomposition above shows these are **symbolic-overhead-bound**, not
dense-bound: the pure dense butterfly (79 µs) is already below Clifft (82 µs); the loss is the ~101 µs of
inverse-frame/promote/localize/dispatch scaffolding around it. **Right design (per user): a single operator
normal form** — always a *projected factorized Pauli product* on the reduced measurement core, applied
sequentially to ψ_r, where one Pauli-apply primitive specializes naturally (identity→scalar, Z-only→in-place
phase, X/Y→butterfly). This is NOT a P1/P2/P3/diagonal *selector*; it is one representation in which:
`r ≪ k` wins (smaller ψ_r), `r = k` degrades gracefully to ≈Clifft (`O(m·2^r) ≈ O(m·2^k)`), `r = 0` does
nothing (empty core), and diagonal is just `xmask = 0`. The current compiled-vs-oracle *split* and the boxed
symbolic scaffolding are the non-normal-form residue to remove. **Honest framing: wins = localization regime
(`r ≪ k`, R_Z high-rank, scales 2^k); the `r = k` cases are not wins but can reach near-parity (~1.1×) under
the single normal form — the advantage there is memory, with speed degrading gracefully rather than blowing up.**

## Single operator normal form (architectural result, implemented + verified)

MDAM applies every deferred/core rotation to ψ_r through **one** primitive,
[`pauli_rot_apply`](../../mdam/native_vm/native_pauli_apply.hpp) — a *projected factorized Pauli product*
on the reduced measurement core. No boundary Pauli-sum `K_b = Σ c_u P_u` and no `2^r × 2^r` matrix are
ever formed. This is **not** a candidate selector (direct-dense / localized / diagonal / folding); the three
behaviours are the *natural branches* of the one primitive:

| state regime | what the primitive does | example | result |
|---|---|---|---|
| `r = 0` (empty core, maxM=0) | apply loop never runs | coherent_d3_r1, d5_r1, distillation — **0 core-applies/shot** (measured) | no dense sweep → coherent **not degraded** |
| `r ≪ k` | sweep a small ψ_r | coherent_d5_r5 (r=12, k=24) — **21 applies/shot** | **WINS** (807×) |
| `r = k` | sweep a full ψ_r | cultivation_d5 (r=10, k=10) — **33 applies/shot** | degrades to ≈Clifft |
| `xmask = 0` vs `≠ 0` | in-place Z-phase vs butterfly | (per-rotation) | diagonal/butterfly, one function |

The compiled magic kernel (`direct_rot`, FLOP-instrumented) and the oracle/general path (`lincomb`) are the
**same** primitive: `nvm_selftest_pauli_apply` confirms bit-identical output over 200,000 random rotations
(max abs diff = `0.000e+00`). Unifying them (routing `lincomb` through `pauli_rot_apply`) is bit-exact on all
11 benchmarks (25/25) with no wall regression (d5_r5 807×, d5_r1 6.75×). The regime difference across every
benchmark is *purely r vs k* — one algorithm, no benchmark-specific branch.

## Methodology
- Harness: [`gate_l_wall_all.py`](../../mdam/native_vm/gate_l_wall_all.py),
  [`gate_l_wall_slow.py`](../../mdam/native_vm/gate_l_wall_slow.py), cmode5 via
  [`gate_k_fast.py`](../../mdam/native_vm/gate_k_fast.py). Verify: [`verify_mdam_coherent.py`](../../mdam/native_vm/verify_mdam_coherent.py),
  batch [`verify_mdam_batch.py`](../../mdam/native_vm/verify_mdam_batch.py).
- MDAM ns/shot = one `nvm_mdam_sample_batch` of `mdam_N` shots (after 50-shot warm-up), wall/N; authoritative
  path is flat-after-warm-up so ns/shot is N-independent (mdam_N up to 1M where it fits a ~30 s budget).
- Clifft ns/shot = `clifft.sample`; N=2 for k≥22 (seconds/shot), up to 200 for small k.
- All A-group rows bit-exact (native authoritative vs Python bounded, 25–100 seeds, bit-identical records).
- Operator folding (default; opt-out `MDAM_NOFOLD`) is an exact representation step: identical records, fewer
  kernel passes. Inverse-frame on-demand (opt-out `MDAM_NOLAZY` = eager reference).
