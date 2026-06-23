# Phase 1 — eliminate the repeated measurement sqnorm (bounded backend)

> **Baseline frozen**: commit `a0e67f7`, tag `phase1-sqnorm` (branch `feat/multicnot-parity-rewrite`).
> Reproduce: `python reports_clifft_axis_bounded/flop_accounting/scripts/phase1_verify.py`
> (bit-exact NEW vs reconstructed-OLD), `…/phase1_sqnorm_trace.py` (sqnorm before/after),
> `…/phase1_bounded_flop.py` (FLOP before/after, all 9 circuits, complete normalize accounting).
> The original (pre-Phase-1) backend is reconstructed inline in those harnesses (`OLD_measure_z`),
> so both versions run from this one checkout. `_purge_verify=True` keeps the residual-product
> assertion live in verification mode.


Two changes, both **bit-exact** (verified per-seed vs the original code). Nothing observable
changes: measurement records, peak resident rank (the hard memory bound), and the Born
probability sequence are identical to fp; only the FLOP drops. clifft is **not touched**.

## What was wrong (measured)

Per magic measurement the bounded engine did `~22–34` `_branch_sqnorm` sweeps, of which
**95–98 % was the `_compress_magic` rescan**, not the Born:

| call site | role | per measurement |
|---|---|---|
| `measure_z`: `p0r = _branch_sqnorm(jr,0)` | Born | 1 (legitimate) |
| `measure_z`: `nrm2 = _sqnorm_1d(phi)` | renormalize (full sweep, uncharged) | 1 |
| `_compress_magic` → `_drop_axis_inplace` `sq0/sq1` | **find the product axis to drop** | **22–34** |

The compress `while changed: for a in range(len(M))` recomputes **both** branch norms for
**every** axis on **every** rescan — but a localized magic measurement drops **exactly one**
axis (`r`, the just-measured dof), confirmed by a drops/meas histogram of `{1: …}` on every
circuit. The 22–34 sweeps were spent re-finding an axis we already knew.

## The two fixes

**1. Drift-safe Born + normalization reuse** (`measure_z`, magic branch)
Compute **both** branch sqnorms `s0,s1` for the measured axis = exactly **one** full sweep;
`tot=s0+s1` is the true current norm² (drift self-correcting, as the old `_sqnorm_1d` was).
The two halves serve **both** the Born `p0=(s0|s1)/tot` **and** the post-projection
renormalization `nrm2 = s0|s1` — the separate full `_sqnorm_1d` sweep is gone (real work
1.5 sweeps → 1).

**2. Direct drop of the localized axis + support-gated residual sweep** (`_drop_localized`)
Drop `r` directly with the known `keepbit` (no sqnorm), then a **single cheap support pass**
(`OR`/`AND` of the nonzero-amplitude indices) flags any residual product axis, each confirmed
with the exact `branch_sqnorm < 1e-20` threshold. Replaces the O(k)-sqnorm-per-rank
`_compress_magic` rescan → **purge sqnorm 0**. The rare second product axis (e.g.
`distillation`, where a measurement disentangles a parity-slaved qubit on 6/20 seeds) is
caught exactly by the gate — the product-axis set is invariant under dropping product axes,
so one pass finds them all.

## Result (measured, seed 1, 1 shot)

| circuit | sqnorm calls | total FLOP | | clifft-unfused (fixed) |
|---|---|---|---|---|
| | before→after | **before → after** | Δ | (context, NOT remeasured) |
| coherent_ry_d3_r1 | 375→34 | 21.95M → **16.95M** | −23 % | 12.29M |
| coherent_ry_d3_r3 | 929→66 | 63.39M → **47.57M** | −25 % | — |
| cultivation_d5 | 241→30 | 821k → **724k** | −12 % | 213k |
| **coherent_rx_d3_r3** | 912→60 | 3.13M → **1.25M** | **−60 %** | **2.59M** |
| coherent_d5_r5 | 2484→120 | 48.27M → **24.44M** | −49 % | 17.99G |

**The rx_d3_r3 counterexample flips.** It was the case the prior analysis flagged: peak state
4× smaller than clifft yet FLOP 1.21× larger (3.13M vs 2.59M), because the measured-magic
sqnorm added ~1.07M. Phase 1 removes that → **1.25M, now below clifft-unfused (2.59M)**.
bounded now wins on BOTH peak memory and FLOP for this circuit.

Remaining bounded > clifft (ry, cultivation) is the **off-diagonal rotation** penalty
(12·2^r vs diagonal 3·2^r) — Phase 2's target, untouched here.

## Verification

* **bit-exact NEW vs reconstructed-OLD, per seed** (record + peak rank + Born p0):
  9 circuits **ALL EXACT** — rec/rank mismatch 0, `max|p0_old−p0_new| ≤ 6.7e-15`.
  Includes `distillation` (the residual-axis case: rec_mismatch 6→0 after the gate fix).
* **residual-product invariant**: `_purge_verify` re-runs the original `_compress_magic`
  after the fast path and asserts it drops nothing — OK on every verified circuit.
* **distributional vs clifft** (6000 shots, null = clifft-vs-clifft spread): rx_d3_r1 0.86,
  rx_d3_r3 1.57 — PASS.
* peak resident rank unchanged on every seed (hard memory bound preserved).

Harness: `scripts/phase1_sqnorm_trace.py`, `scripts/phase1_verify.py`,
`scripts/phase1_bounded_flop.py`. Code: `nearclifford_backend/clifft_axis/bounded.py`
(`measure_z` magic branch, `_drop_localized`/`_drop_localized_core`/`_support_bits`/
`_drop_residual_products`).
