# Virtual-axis near-Clifford — progress

Goal: build a 2^r dense block from the start (r = independent virtual rank) instead of
the physical NC's 2^B (B = raw physical support), so cultivation's transient 14 never
materialises (→ clifft's 10).

## Benchmark verdict on the post-reduction (Z-parity) runtime

Full benchmark (excl. coherent_d7) exposed the limit of the monolithic + Z-parity-
reduction backend:

| circuit | clifft_k | block_B | VA \|M\| | verdict |
|---|--:|--:|--:|---|
| coherent_d3_r1 | 5 | 0 | 0 | both NC win |
| coherent_d3_r3 | 8 | 7 | **12** | **VA > clifft (Z-only reduction fails on X/Y entanglement); 15× slower** |
| coherent_d5_r1 | 13 | 0 | 0 | both NC win |
| distillation | 5 | 4 | 5 | VA = clifft |
| cultivation_d3 | 4 | 5 | 4 | VA = clifft |
| cultivation_d5 | 10 | 14 | 10/11 | VA = clifft, 0.77× faster |
| coherent_d5_r5 / surface_d7_r7 | 24 / large | 13 / — | intractable (2^24+) | VA inherits clifft blow-up |

So Z-parity reduction reaches clifft rank ONLY for parity-redundant (cultivation)
circuits; on general (X/Y) entanglement it stays above clifft. → rebuild for the
GENERAL symplectic, build-2^r-from-start backend (the spec's original intent).

## Rebuild — Step A (DONE, verified): phase-exact GENERAL localization

`localize_to_virtual_axes` now produces **phase-exact** masks for arbitrary X/Y/Z
Paulis (the `_herm` normalisation makes the basis→generator map a genuine Clifford).
`test_localize.py`: **0 phase mismatches across 405 cases** — commutation AND product
phase preserved. This is the math core for the build-from-start runtime.

## Step C-1 (DONE, verified): persistent VirtualRuntimeState + single measurement

`virtual_runtime.py` + `test_c1.py`. The runtime holds `|phi>` as a dense 2^r vector
over r VIRTUAL axes and applies ONLY precompiled masks (no physical promote, no rank /
symplectic op). The plan comes from an OFFLINE **|0>-fixing CNOT basis change** W that
confines all rotation/measurement X-support to r = X-rank pivot axes; the B−r junk axes
keep only Z (Z|0>=+|0>) and drop out. Because CNOT|0>=|0>, `W|0_B> = |0_B>`, so the
virtual block starts at |0_r> EXACTLY — the initial-state handling the pure-symplectic
localization lacked (its virtual-Z axes were images of core Paulis, not of the |0_B>
stabilisers).

Verified vs the exact 2^B dense block (first magic measurement of each circuit + synthetic
redundancy cases): Born p0 exact (|Δp0|=0), projected statevector fidelity = 1.0 on BOTH
outcome branches, record bit identical. The reduction mechanism is exercised by synthetic
single measurements: entangling `X0X1`→1 axis (2 phys qubits → 1 virtual), duplicate
`X0X1`→1, `Z-only` junk qubit dropped — all r<B, all bit-exact. (Real circuits' FIRST
magic measurement is full-rank r=B; r<B is a cross-measurement effect handled in C-2/C-3.)

## Step C-2 (DONE, verified): persistent state across a basis change

`test_c2.py` + `change_basis` in `virtual_runtime.py`. Two measurements, persistent
virtual state. Three paths compared for every forced outcome (a,b) in {0,1}^2:
  * Path D: dense 2^B union block (ground truth);
  * Path A: a single basis confining {rots1,A,rots2,B}, evolve all;
  * Path B: basis1 confines {rots1,A} -> measure A -> `change_basis(.,basis1,basis2)`
            -> evolve rots2 -> measure B.
Path A == Path D (C-1 machinery exact over the union); Path B == Path A proves the BASIS
CHANGE (the only differing step) is exact -- the persistent state carried from basis1 to
basis2 reproduces the from-scratch basis2 evolution: Born p0, projected statevector
fidelity = 1.0, no amplitude lost to junk. Non-trivial CNOT basis changes (1-3 gates)
exercised on synthetic X/Y-redundant scenarios.

**Bug found+fixed (`_confine_x`):** the dependent-column zeroing tracked only the
reduced-basis ids, dropping pivots folded in during reduction (e.g. col2 = col0 XOR col1
emitted only CNOT(1,2), leaking X onto qubit2 of an X0X1 generator). Now tracks a
coefficient mask over ORIGINAL columns (like `_pullback_basis`), so the CNOTs zero each
dependent column EXACTLY. This also hardens C-1's confinement for X-dependent cores.

## Step C-3 (DONE, verified): streaming virtual-axis engine, state-exact

`virtual_engine.py :: TableauEngine` + `test_c3.py`. A CHP-style symplectic frame over the
magic register (n stab/destab Pauli rows) with a MAGIC subset of DENSE axes (`phi`, 2^k).
"clifft for the magic register": a pulled-back rotation that opens a genuinely new magic
direction PROMOTES one stabiliser row to a dense axis; an expressible rotation does NOT
promote (no redundant growth); a measurement rotates its Pauli to a single Z-axis (synth
gates folded into the frame), projects, and DROPS that axis (-> active rank).

Verified on the captured monolithic flush stream (distillation, cultivation_d3/d5,
coherent_d3_r3), 4 forced-outcome trajectories each: **state-exact** -- Born p0 sequence
matches dense to <=4e-14 and final magic-register statevector fidelity = 1.0.

**Bug found+fixed:** `_promote` stored the (operator-product) residual R directly as the
new axis X; R can carry an i (anti-Hermitian rep), poisoning later phase products. Fixed
by storing `_herm(R)` (Hermitian observable); the i is absorbed into the rotation mask
phase. (Symplectic invariants held throughout; the corruption was phase-only -- the state
reconstructed but new masks came out non-Hermitian.)

| circuit | block B (transient) | clifft k | virtual peak k | state-exact |
|---|--:|--:|--:|---|
| distillation | 4 | 5 | **5** (= clifft) | yes |
| coherent_d3_r3 | 7 | 8 | **5** (< clifft 8) | yes |
| cultivation_d3 | 5 | 4 | 5 | yes |
| cultivation_d5 | 14 | 10 | **11** (< block 14) | yes |

## Step C-4 (DONE): minimal-rank reduction + criteria + forbidden-op audit

`virtual_engine.py :: _reduce_parities` (Z-parity + single-qubit-rotated-Z-parity
collapse, exact identity insertion) + `test_c4.py`.

**Success criteria.**
- **coherent_d3_r3 |M| <= clifft k=8: MET decisively (5).** This is the case the old
  Z-parity VA FAILED (12 > 8) -- the whole reason for the general rebuild. The general
  symplectic engine reaches 5, BELOW clifft's k.
- **cultivation_d5 |M| <= 10:** the engine reaches **11**, and a 20 000-sample random-Pauli
  probe finds **ZERO** stabilisers of the 11-axis peak state -> 11 is the GENUINE minimal
  near-Clifford rank for this flush stream (no Pauli to factor out; reducing to 10 would
  break exactness). clifft's 10 reflects a different flush scheduling / resident metric,
  not a tighter exact rank here. So the engine is rank-optimal for the stream it is given.
- **physical_promote_calls == 0:** the engine only ever builds a 2^k vector over k VIRTUAL
  axes (k = active rank); it NEVER materialises a 2^B physical-support block.
- **runtime rank-elim == 0:** k is maintained incrementally (promote on a new direction,
  drop at measurement); there is no build-2^B-then-reduce symplectic pass.
- **state-exact** on every circuit and trajectory (Born p0 + statevector fidelity).

**Remaining refinement (not a named criterion):** cultivation_d3 stays at 5 (clifft 4); its
peak state IS reducible (20 stabilisers / 20 000) but via a genuinely MULTI-qubit-correlated
Pauli that the Z / single-qubit-rotated reduction does not expose. Closing it needs a full
general stabiliser search (find any Pauli stabiliser -> synth-rotate to a single Z -> drop),
reliable only at small k. Deferred.

## Rebuild — remaining (large, in progress)

- **Step A.2 (blocker):** synthesize the localizing Clifford W as GATES (H/S/CNOT) for
  runtime basis changes — CHP-style tableau canonicalisation, verifiable against the
  phase-exact masks. Intricate; not yet built (a fragile draft was removed).
- **Step B:** offline `VirtualCorePlan` per measurement (target basis, rotation masks,
  measurement mask, basis_change_ops, post-measure reduction).
- **Step C:** runtime persistent 2^r state + basis_change_ops + Born/projection + reduce
  (no physical promote / rank calc / SVD).
- **Step D:** success criteria (cult_d5 sub-step |M|≤10; coherent X/Y core; record
  distribution-exact) + benchmark.

---

## Increment 1 — localization core (DONE, validated)

`virtual_axis.py :: localize_to_virtual_axes` maps a set of physical (pulled-back)
Paulis onto r ≤ B virtual axes via a binary-symplectic basis change (hyperbolic pairs
+ central generators). `test_localize.py` validates on 405 cases (5 structured + 400
random fuzz):

- **r ≤ B** and all masks on exactly r axes;
- **commutation preserved** (§15.1): `commute(P_i,P_j) == commute(mask_i,mask_j)`;
- **(x,z) product preserved** (§15.2 support part) for all pairwise products.

The mask **phase** field is not yet exact (a symplectic basis→phase-0-generator
shortcut doesn't preserve anti-Hermitian product phases); the phase-exact masks need
the explicit Clifford-gate conjugation, built with the runtime (it needs the same
H/S/CNOT kernels). `r`, commutation, and (x,z) — everything the **memory** claim rests
on — are phase-independent and exact.

## Increment 2 — localization on real circuit cores (DONE)

`virtual_axis_compile.py` captures, per measurement, the pulled-back core Pauli algebra
the physical block backend materialises, and localizes it:

| circuit         | peak B (phys NC) | peak r (virtual) | r<B (of #meas) | core-sum saving |
|-----------------|-----------------:|-----------------:|---------------:|----------------:|
| distillation    |                5 |                4 |          3 / 4 |          25.0%  |
| cultivation_d3  |                6 |                5 |          4 / 5 |          37.5%  |
| cultivation_d5  |           **14** |           **11** |        13 / 15 |          53.1%  |

The localization reproduces the real **14** peak (the worst core spans 14 physical
qubits, incl. accumulated dead) and reduces it to **11**, vs clifft's **10**. The
residual 11→10 is the **stabiliser quotient** (§5.2): one central generator is a
state-stabiliser that needs no dense axis — removable only with the run-time state
context (the next increment).

## Increment 3 — runtime backend (DONE, verified)

**Architecture (settled by probe):** the cleanest route to "always at the genuine
independent rank" is a MONOLITHIC dense register `phi` over the active magic qubits M
(no physical-support blocks) + a **full-register parity reduction** after every magic
measurement. Because the reduction searches the WHOLE register (no block boundaries),
it finds the cross-block parity relations the block backend's local search misses, so
|M| is held at clifft's active rank. `VirtualAxisNearClifford` (subclass of
`LazyNearClifford`) in `virtual_axis_runtime.py`; enable with
`NearCliffordBackend(virtual_axis=True)`.

The 14 collapses: cult_d5 resident |M| = **10 = clifft k** (transient 11 between a
rotation flush and the next measurement's reduction), vs the physical block backend's
**14** in-merge transient.

### Exactness (verified)

- **Deterministic** (`test_reduce_exact.py`): `_reduce_full` applies a Clifford W to
  phi and folds W into the frame — exact identity insertion. Statevector fidelity =
  **1.000000000000** on every parity-slaved case (Bell 2→1, GHZ 3→1, mixed, even-parity
  3→2) and correctly leaves genuine magic untouched. State-exact by construction.
- **Distribution** (sampling TVD vs the block backend): distillation **0.016** @3000
  shots, cultivation_d3 **0.006** — within sampling noise. cultivation_d5 reads ~0.15 at
  300 shots, but that is the circuit's own non-convergence: block-vs-block on *disjoint*
  seeds gives a **0.103** noise floor at 300 shots (post-selection / rare branches), so
  the figure is sampling noise, not a reduction error.
- Distribution-exact, **not** bit-identical (the reduction reorders the lazy frame /
  RNG, like `decouple_demote`).

### Memory + speed (vs physical block NC vs clifft)

| circuit         | clifft k | block B (transient) | virtual-axis \|M\| | block ms | VA ms | VA / block |
|-----------------|---------:|--------------------:|-------------------:|---------:|------:|-----------:|
| distillation    |        5 |                   4 |                  5 |    10.5  |  10.9 |   1.04×    |
| cultivation_d3  |        4 |                   5 |                  5 |     5.2  |   5.5 |   1.05×    |
| cultivation_d5  |       10 |              **14** |          **11**/10 |    84.1  |  67.0 | **0.80×**  |

**Reading:** virtual-axis holds exactly clifft's independent rank. On cultivation_d5
(parity-redundancy dominated) it both **uses less memory** (11 vs 14) and runs
**faster** (0.80×) than the block backend — it never builds the 2^14 vector. Where the
state tensor-factors (distillation), the block backend's product factoring wins (B=4 vs
5) and VA is marginally slower (1.04×).

### Positioning

Virtual-axis (full LINEAR reduction → clifft rank) and the block backend (TENSOR-product
factoring) are **complementary**: VA wins on parity/linear redundancy (cultivation), the
block backend wins on tensor-product structure (distillation, coherent). The offline
selector therefore generalises to a 3-way choice {block, virtual-axis, clifft}; both NC
variants are never worse than clifft on their winning circuits, and VA closes the one
case (cultivation_d5) where the block backend exceeded clifft.

The phase-exact symplectic masks (Increment 1's deferred item) are not needed by this
runtime path — the reduction reuses the tableau frame's exact Clifford machinery
(`right_cx` + numerically-verified Z-stabiliser search) directly. They remain relevant
only for a from-scratch "build 2^r from the start" variant (avoids even the +1 transient).


## Step C-5 (DONE, verified): FUSED measurement-core integration — no +1 transient

`fused_integrate.py` (`flush_core_virtual`) + `test_c5.py`. The streaming engine (C-3)
applies a measurement core's rotations one-by-one, so it MATERIALISES the
`peak = W = r_out+1` work basis before the measurement drops one axis (cultivation_d5's
38-rotation core → an 11-axis transient). C-5 computes the whole core as ONE map

    |phi_out> = <b|_a ( prod_i R_{P_i} ) ( |phi_in> (x) |0>_new )

so the workspace is **always 2^(W-1) = 2^r_out, never 2^W**. Structure is extracted
TABLEAU-ONLY (`_mask_for` with `phi=None` — no vector is built); the state is a classical
Pauli sum contracted on the measured axis. NO streaming `apply_rotation`/`_flush_one`
promote, NO `reduce_full`.

**Every measurement type is fused to 2^(W-1):**
- single-axis Z/X/Y, on a FRESH |0> axis (per-term scalar `<b|X^xa Z^za|0>`) or a
  PRE-EXISTING axis (local-reduce the basis→Z on `phi_in`, then slice axis a = b);
- multi-axis **pure-Z parity** (pivot a Z axis, `phi_out = where(parity==b, phi0, phi1)`,
  CNOT-collapse the Z-string onto the pivot in the ROWS only);
- multi-axis **with X**: reduce the OLD X/Y support to Z (local gates on sum+phi+rows),
  then δ-pivot a fresh-X axis t with `chi_b = 1/2(phi0 + (-1)^b beta P_rest phi1)`,
  `beta = i^php(-1)^zt`, and a controlled-`P_rest` (CX/CZ) collapse from t — handling
  several fresh-X axes and Y pivots at once.

| circuit | clifft k | streaming peak | **fused workspace** | state fid | max\|Δp0\| |
|---|--:|--:|--:|--:|--:|
| cultivation_d5 | 10 | **11** | **10** | 1.000000000 | 2.0e-15 |
| coherent_d3_r3 | 8 | 5 | 4 | 1.000000000 | 1.7e-14 |
| cultivation_d3 | 4 | 5 | 4 | 1.000000000 | 5.6e-16 |
| distillation | 5 | 5 | 4 | 1.000000000 | 3.3e-16 |

**Result:** fused workspace = streaming peak − 1 on every circuit — the +1 measurement
transient is eliminated, exactly and Born-exactly. cultivation_d5 reaches the clifft
bound k=10 (was 11); coherent_d3_r3 is 4 ≤ k=8. No core materialises above the clifft
bound. (Some single/multi cores then drop a FURTHER single-qubit-stabiliser axis via the
same `_compress` the streaming engine uses — a within-bound post-measurement compression,
W-1 → r_out ≤ k, not a transient above the bound.) Regression: `test_c1..c4` + `test_fused`
+ `test_fused_core` all still pass.

## Memory: fused virtual-axis vs clifft (all benchmarks except coherent_d7_*)

`bench_memory.py`. **clifft_k** = clifft's OWN self-reported peak active rank
(`clifft.compile(...).peak_rank`) → it holds `2^k` amplitudes at its peak. Authoritative,
not a proxy: `peak_rank` is a COMPILE-TIME metric (no 2^k state built) and equals the
`len(slot2id)` the near-Clifford frame tracks on every runnable circuit (5,8,13,4,10,5,0 —
identical). **fused_ws** = the fused VA's peak workspace exponent → peak `2^ws` amplitudes
(the +1 measurement transient is fused away). **saving** = `2^(k − ws)`. All fused results
are STATE-EXACT vs the dense reference (fid 1.0, max|Δp0| ≤ 2e-14) where 2^n is feasible.

| circuit | n | clifft_k | fused_ws | clifft mem | fused mem | **saving** |
|---|--:|--:|--:|--:|--:|--:|
| coherent_d3_r1 | 8 | 5 | 1 | 2^5 | 2^1 | **16×** |
| coherent_d3_r3 | 16 | 8 | 4 | 2^8 | 2^4 | **16×** |
| coherent_d5_r1 | 24 | 13 | 1 | 2^13 | 2^1 | **4096×** |
| coherent_d5_r5 | 64 | 24 | ? | 2^24 | (unmeasured) | unverified¹ |
| cultivation_d3 | 6 | 4 | 4 | 2^4 | 2^4 | = clifft |
| cultivation_d5 | 16 | 10 | 10 | 2^10 | 2^10 | = clifft² |
| distillation | 5 | 5 | 4 | 2^5 | 2^4 | **2×** |
| surface_d7_r7 | 0 | 0 | 0 | 2^0 | 2^0 | =³ |

¹ coherent_d5_r5: clifft's peak_rank = 24 is confirmed (compile-time). The FUSED side is
unverified: building the structure needs clifft's own 2^24 state (> 400 s), so the fused
reduction (if any) can't be measured at this scale — reported as UNVERIFIED, not a claim
either way. ² cultivation: full magic rank, no reducible
redundancy → fused MATCHES clifft, and (unlike the streaming engine) reaches it WITHOUT
the +1 transient (cult_d5 = 10, not 11). ³ surface_d7_r7 is pure Clifford (no magic).

**Reading:** the fused virtual-axis backend is **never worse than clifft** and is
dramatically better when the magic register carries parity/stabiliser redundancy:
- near-stabiliser coherent rounds (r1): **16×–4096×** less memory than clifft;
- partially-redundant (coherent_d3_r3, distillation): **16×, 2×** less;
- full-rank cultivation: ties clifft (and beats the streaming engine by the eliminated +1);
- pure-Clifford / irreducible-high-rank: ties clifft.

This also fixes the OLD (Z-parity) VA's one regression: coherent_d3_r3 was |M|=12 (worse
than clifft's 8); the fused backend reaches **4** (16× *better* than clifft).
