# Fused measurement integration — removing the +1 measured-axis transient (coherent_d3_r3)

**Goal (this step):** put the verified exact fused measurement into the authoritative MDAM
backend so the measurement runs `phi_in ∈ ℂ^{2^{r_in}} → phi_out ∈ ℂ^{2^{r_out}}` directly,
with the measured (newly-promoted) axis **never** materialized as a dense-array axis. No FLOP
/ Pauli-sum-count optimization, no rank-reduction change, no new approximation. Scope is
`coherent_d3_r3`; generalization to other circuits is the explicit next step.

## What changed (code locations)

All in `nearclifford_backend/clifft_axis/bounded.py`:

| Location | Change |
|---|---|
| module top | `_fpopc`, `_fmul`, `_fdag`, `_apply_xz` — symbolic Pauli-sum algebra over the *would-be* `r_mat` magic-bit layout (used only to contract the measured axis; the `2^{r_mat}` array is never built). |
| class flag | `_fused_measure = True` — production default. |
| `_fused_core_entries` | the anticommuting core for `Z_q` **without** advancing `_meas_ctr` / touching `pending` (mirrors `_flush_core`'s fast-table-else-live-scan selection). |
| `_fused_setup` | read-only plan: virtual flush (replay the promote order to get `M_mat`), build `U_core` as a Pauli sum, the measured Pauli `M'`. Returns `None` (→ oracle) unless the case is **diagonal-magic with a newly-promoted measured axis**. |
| `_fused_born` | `p0 = (1 + ⟨M'(t)⟩_in)/2`, `M'(t)=U_core^†M'U_core`, keeping only terms with no `X` on the new axes, evaluated on `phi_in` (`2^{r_in}`). |
| `_fused_survivor` | `K_b\|phi_in⟩` built directly on the `r_out` register (`2^{r_out}`) via the localizer parity mask (measured axis summed out). |
| `_fused_commit` | sample (one `rng.random`, oracle convention), normalize, the **same swap-pop axis bookkeeping** → `M_out`, the **same frame folds** (`right_cx` per Z-support control + `\|1⟩`-branch `X_q` fold), residual-product sweep, budget/log. |
| `measure_z` | dispatch: on real shots (not the structure-discovery pass, not `resource_only`) try `_fused_setup`; if it applies, `_fused_commit`; **else the existing materialize-localize-drop body runs verbatim** (now the verification oracle). |

## Removed allocation path (production, for the targeted case)

For every diagonal-magic measurement whose measured axis is newly promoted — **100% of the 8
magic measurements in `coherent_d3_r3`** — the production path no longer executes:

```
promote measured axis  →  2^{r_out+1} state allocation  →  apply core rotations
                       →  Born on the materialized axis  →  drop the measured axis
```

It executes instead:

```
Born p0 on phi_in (2^{r_in})  →  outcome sampling  →  K_b survivor on 2^{r_out}
```

The materialize-localize-drop code is retained **only as the oracle** (`_fused_measure = False`);
it is not a production fallback for the targeted case (it is never reached for d3_r3's magic
measurements). It still serves the genuinely-different measurement types the fused fast path
does not target (stabilizer / dormant / deterministic / off-diagonal / **measured axis already
resident**) — those have no `r_out+1` transient to remove.

## Per-core exactness (coherent_d3_r3) — fused vs oracle, bit-identical

Standalone replay (`/tmp/proto_fused_integ.py`, authoritative outcome replayed): all 8 magic-Z
cores reproduce **M ordering, Xc, Zc frame, and phi (up to global phase) and p0** to machine
precision (`|Δphi| ≤ 1.1e-16`, `|Δp0| ≤ 2.2e-16`).

End-to-end (`/tmp/test_fused_integ.py`, fused samples its own trajectory), 8 seeds:

| check | result |
|---|---|
| measurement record trajectory | identical every seed |
| final `M` ordering / `Xc` / `Zc` | identical every seed |
| final magic state `phi` (≤ global phase) | `\|Δ\| ≤ 1.1e-16` |
| peak dense rank | **fused 4, oracle 5** |

Ground truth: the fused bounded backend's record distribution matches **clifft.sample** within
sampling error (0 of 33 slots beyond 5σ at N=6000).

Regression (`/tmp/test_fused_regress.py`, all feasible benchmarks): records + final state +
frame bit-identical to the oracle on every circuit; fused peak dense rank ≤ oracle everywhere.
Incidental win: `coherent_rx_d3_r1` 11→10. The resident-measured-axis circuits (`coherent_d5_r5`,
`cultivation_d3/d5`, `distillation`) correctly fall back (no peak change) — the deferred
generalization.

## Largest dense-array exponent

Per fused core the only exponential allocation is the `2^{r_out}` survivor (and the `2^{r_in}`
`phi_in`/Born temporaries, `r_out ≥ r_in`). Hard invariants asserted every core:
`measurement_axis_materialized = False`, `largest_dense_array_exponent = r_out`, and **no
`2^{r_mat}=2^{r_out+1}` allocation**. Confirmed by the budget: `peak_dense_rank = 4` (fused) vs
`5` (oracle) — the resident/output peak, not the old transient peak.

```
 meas  r_in  r_mat_ref  r_out  largest_dense_array_exp  meas_axis_materialized
   4     0       3        2            2                     False
   5     2       4        3            3                     False
   6     3       5        4            4                     False
   7..11 4       5        4            4                     False
```

## Caveat (point D)

The dense exponential object shrinks (`r_out` vs `r_out+1`), but the branch map is built from a
symbolic Pauli sum whose **working memory** (Python dicts of `#U_core` terms) is non-exponential
yet, at these tiny ranks, exceeds the dense saving in *total* tracemalloc. That is the FLOP /
Pauli-sum-length concern, not the dense-array bound (quantified below).

---

# Generalization: resident measured axis (B2) — coherent_d5_r5 13→12

The first integration handled only a **fresh** measured axis (`m >= r_in`).  Classifying every
benchmark core (`/tmp/classify_cores.py`) split the removable transients (`r_mat > max(r_in,r_out)`)
into two:

* **fresh** measured axis (`r_in → r_in+1 → r_out`) — handled in step 1.
* **resident** measured axis (`m < r_in`, already inside `phi_in`; the core promotes only WORK
  axes, `r_in → r_mat = r_out+1 → r_out`).  This is what pins `coherent_d5_r5` at peak 13.

Two fixes generalized the fused path:

1. **Resident survivor** (`_fused_survivor_resident`).  Input has no `(x)|0>_m`; for each output
   index the measured-axis bit is **gathered** at the localizer-forced value
   `m_pre = keepbit XOR parity(Z-support controls)` (a select, not a zero-mask).  Still layout-A,
   still `2^{r_out}`.  `_fused_born` and the M-ordering / frame folds were already general.
2. **Localizer pivot, not q.**  When the measured qubit `q` has no `Z` on itself, the oracle's
   `_localize_to_Z(prefer=q)` collapses `M'` onto and drops `supp[0]`, **not `q`** (cultivation;
   the d5_r5 B1 tail).  `_fused_setup` now computes the true pivot `r = q if q∈supp else supp[0]`
   and keys the drop / frame fold / X-fold on `r`.

## Application rate and peak (4 seeds, bit-identical to oracle throughout)

| Benchmark | diag-removable | off-diag-removable | **fused rate (supported)** | peak before | peak after |
|---|--:|--:|:--:|--:|--:|
| coherent_d3_r3 | 8 | 0 | **8/8** | 5 | **4** |
| coherent_d5_r5 | 48 | 0 | **48/48** | 13 | **12** |
| cultivation_d3 | 1 | 1 | **1/1** | 4 | 4 |
| cultivation_d5 | 3 | 1 | **3/3** | 10 | 10 |
| distillation | 0 | 0 | — | 4 | 4 |
| coherent_rx_d3_r1 | 2 | 1 | **2/2** | 11 | **10** |
| coherent_ry_d3_r3 | 1 | 4 | **1/1** | 16 | 16 |

**100% fused application on every supported (diagonal-removable) core, all benchmarks.**  Records,
final state (≤ global phase), and frame (`Xc`/`Zc`) bit-identical to the oracle on every seed.

The peaks that do *not* drop are **not** fused failures: they are **off-diagonal** measured Paulis
(`Mx != 0`, X-character on a magic axis) which the diagonal fast path correctly excludes
(cultivation's 1 off-diag core; ry's 4) — or **non-removable** B1 cores whose peak *is* the input
(distillation's rank-4 resident state; nothing to remove).  Off-diagonal is the identified next
sub-target.

## d5_r5 deep verification (`/tmp/d5r5_deep.py`)

| check | result |
|---|---|
| fused-eligible cores | 48 (all removable B2) |
| **both branches** b=0 AND b=1 exact vs 2^{r_mat} reference | **48/48** (fidelity defect ≤ 3.3e-15) |
| p0 match | 48/48 (max \|Δp0\| = 2.8e-15) |
| invariants (pivot dropped, peak < r_mat) | 48/48 |
| **peak_dense_rank (real shots)** | **12 — zero 2¹³ allocations** |

## FLOP / Pauli-sum cost (section 8)

Fused trades FLOP for the 2× peak-memory reduction.  Per fused core, amplitude-touches:
`fused ≈ #U_core · 2^{r_out} + #M'(t) · 2^{r_in}` vs oracle `≈ #rot · 2^{r_mat}`.

| quantity | d5_r5 range | note |
|---|---|---|
| `#U_core` Pauli terms | 16 – 256 (median 64) | grows ~`2^{#core rotations}`; the point-D working set |
| `#M'(t)` Born terms | 8 – 128 | |
| fused / oracle amplitude-touch ratio | median **4.8×**, up to **14.5×** | the FLOP cost of the symbolic contraction |

So the dense exponential object halves (`2^{r_mat} → 2^{r_out}`) at the cost of ~5× more
(non-exponential) amplitude-touches dominated by `#U_core`.  Reducing `#U_core` (Pauli-sum length)
is the FLOP-optimization lever, deferred by request.

---

# Off-diagonal generalization (cultivation, distillation)

The off-diagonal measured Pauli (`Mx != 0`, X/Y on a magic axis) reduces to the diagonal machinery
by **folding the localizer's H/S into the core**: the oracle's `_localize_to_Z` applies `H` (X→Z) /
`S†,H` (Y→Z) to diagonalise M' before the CNOT collapse.  Writing those as Pauli sums
(`H=(X+Z)/√2`, `S†=((1−i)I+(1+i)Z)/2`) and forming `U' = W_HS·U_core`, `M'' = W_HS M' W_HS†` (pure
Z), the off-diagonal case becomes the diagonal `_fused_survivor`/`_fused_born` with `(U', M'')`;
only the frame fold gains the `right_h`/`right_s` gates.  Diagonal is the empty-`W_HS` special case.

Two correctness fixes were required:

1. **Post-flush stabilizer check.**  `anti_s` must be computed against the **post-flush** register
   (as the oracle does), not pre-flush — the core flush can promote the very qubit the measured
   Pauli anticommutes with, turning an apparent stabilizer measurement into a magic off-diagonal one
   (cultivation, distillation).
2. **Frame-change guard.**  At rank ≥ `_loc_min_size` (2¹⁴) the flush *itself* localizes an
   off-diagonal **rotation** (`_flush_offdiag_localized`), mutating the frame mid-flush.  The fused
   plan reads the pre-flush frame, so if any core rotation would trigger that, it falls back to the
   oracle (ry's high-rank cores).  A `_fused_max_terms` cap likewise falls back before a pathological
   Pauli-sum blow-up — both correctness-neutral hang guards.

## Final peaks (all bit-identical to oracle, multi-seed)

| Benchmark | peak before → after | note |
|---|:--:|---|
| coherent_d3_r3 | 5 → **4** | fresh diagonal |
| coherent_d5_r5 | 13 → **12** | resident diagonal (48 cores) |
| cultivation_d3 | 4 → **3** | off-diagonal |
| cultivation_d5 | 10 → **9** | off-diagonal (15/15 both-branch exact, ≤3.3e-16) |
| distillation | 4 → **3** | off-diagonal / post-flush-magic |
| coherent_rx_d3_r1 | 11 → **10** | mixed |
| coherent_ry_d3_r3 | 16 → 16 | off-diagonal cores high-rank → frame-change fallback |

**6 of 7 benchmarks reduced; all 7 bit-identical (records, state ≤ global phase, Xc/Zc).**  ry is
the sole non-reduction: its off-diagonal cores live at rank ≥14 where the rotation localizer mutates
the frame, so they fall back (correct, not reduced).  Handling that needs a frame-aware virtual
flush — the remaining sub-target.

---

# ry and the frame-aware virtual flush — why it is NOT a frame problem

The remaining ry non-reduction was investigated for a frame-aware virtual flush (symbolically
reproducing the `F0 →W1† F1 →…` evolution the rotation localizer drives).  The frame *tracking* is
cheap, but it does not help: ry's measurement cores contain **up to 27 magic rotations** (meas 1:
rank 2→15, 27 localized rotations).  The symbolic Pauli-sum of that many rotations grows toward
`2^{#rotations}` (≈2²⁷) terms — whether the rotations are kept off-diagonal (direct) or localized
to diagonal — versus the oracle's **bounded** `2^{r_mat}=2¹⁶` dense materialization.  The bottleneck
is the **Pauli-sum length, not the frame.**  Removing the +1 transient saves a factor 2 in the dense
object; the symbolic contraction costs an exponential in the rotation count.  So ry cannot be closed
by the symbolic branch-map method — it would need a *dense* localized flush at `2^{r_out}` (the
oracle's technique minus the measured axis), a different implementation.  The `_fused_max_terms` cap
correctly falls these cores back.

# Full-circuit cost (task 2) — the overhead is real at current scales

| Benchmark | magic cores | fused | fallback | #U max | #U avg | oracle ms | fused ms | **runtime ×** | peak rank o→f | **peak KB o→f** |
|---|--:|--:|--:|--:|--:|--:|--:|:--:|:--:|:--:|
| coherent_d3_r3 | 12 | 12 | 0 | 64 | 24 | 8.8 | 18.8 | 2.15× | 5→4 | 29→42 |
| coherent_d5_r5 | 60 | 60 | 0 | 256 | 59 | 183 | 562 | 3.08× | 13→12 | 333→648 |
| cultivation_d3 | 5 | 5 | 0 | 16 | 9 | 5.9 | 6.9 | 1.18× | 4→3 | 1262→1262 |
| cultivation_d5 | 15 | 15 | 0 | 1024 | 73 | 90 | 657 | 7.32× | 10→9 | 8602→8604 |
| distillation | 5 | 5 | 0 | 64 | 27 | 11.8 | 19.2 | 1.63× | 4→3 | 1214→1216 |
| coherent_rx_d3_r1 | 14 | 14 | 0 | 64 | 13 | 7.6 | 15.3 | 2.01× | 11→10 | 131→155 |
| coherent_ry_d3_r3 | 33 | 27 | 6 | 512 | 28 | 480 | 757 | 1.58× | 16→16 | 419→4133 |

**Conclusion — the overhead is LARGE, not negligible:**
* **Runtime:** fused is **1.2× – 7.3× slower** (worst: cultivation_d5 7.3×, d5_r5 3.1×).
* **Real peak bytes:** fused is **equal-to-2× higher** where the magic register dominates (d5_r5
  333→648 KB, ry 419→4133 KB); negligible where other structures dominate (cultivation, distillation).
* The fused path **does** reduce the dense-array EXPONENT (the formal `2^{k}` feasibility bound — the
  MDAM memory claim, met and verified).  But at these scales (k ≤ 16) the **symbolic Pauli-sum
  working set + survivor-build intermediates exceed the factor-2 dense saving**, so *total* peak
  bytes and runtime are worse.  The win is asymptotic (large k, where `2^k` is the wall); the
  constant-factor overhead dominates here.

Per the task-2 criterion this is the **"overhead large → optimize"** branch: the symbolic
contraction (#U_core, which scales ~`2^{#core rotations}`) is the genuine bottleneck, confirmed at
circuit scale — not a micro-optimization guess.  The next research step is a **factorized /
dense-localized execution** of the branch map (bound the Pauli-sum, or do the localized flush densely
at `2^{r_out}`), which would also be the route to closing ry.

## Status

* fresh + resident, **diagonal and off-diagonal** measured axis: **integrated, bit-identical** on
  every benchmark; 100% fused on every tractable removable core; **6/7 peaks reduced** (d3_r3 5→4,
  d5_r5 13→12, cultivation_d3 4→3, cultivation_d5 10→9, distillation 4→3, rx 11→10).
* **dense-array exponent reduced and verified** (both branches, zero over-`max(r_in,r_out)` alloc).
* **cost measured:** runtime 1.2–7.3×, peak bytes up to 2× (10× ry) higher at current scale — the
  formal bound improves, the symbolic constant factor dominates → factorized execution is the
  warranted next optimization (and the route to ry's rank ≥14 cores).
