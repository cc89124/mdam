# Dense-FLOP tables — which one to read

Two FLOP tables live here. They measure **different algorithms** and disagree by up to ~15× on the
magic-saturated benches. Read the native one.

| file | what it measures | status |
|---|---|---|
| `flop_table.csv` | the **old Python `clifft_axis_bounded`** backend, whose measurement builds a **fused Pauli-sum core** `K_b = Σ_u c_u P_u` (`survivor_ops = n_U · 2^r`) | **superseded / misleading** |
| `flop_table_native.csv` | the **native one-factor VM** (`native_vm/`), which applies each core rotation **one factor at a time** (`pauli_rot_apply`), never forming a Pauli sum (matches `MDAM_localized_computation.md` §6–§7) | **authoritative** |

Both charge Clifft from **its own** `active_k_history` dense-op schedule, and both use the same per-element
convention (offdiag=12, diag=6, perm=0, meas=12, per `2^rank`). Generator: `mdam/bench/native_flop_compare.py`
(native) ; `mdam/bench/flop_compare.py` (old).

## Why the old table over-counts

The old backend's fused core expands the boundary operator into `n_U` Pauli terms and sweeps each over
`2^r` → `O(n_U · 2^r)`. The native VM applies the core as a **product** of `m_core` factors → `O(m_core · 2^r)`
with `m_core ≪ n_U`. The Pauli-sum `n_U` is what inflates the old numbers. Concrete corrections:

| circuit | old `F_cl/F_bn` (Python-fused) | **native `total/clifft`** | verdict |
|---|---|---|---|
| `coherent_rx_d3_r3` | 0.41 (MDAM 2.4× **more**) | **0.16** (MDAM 6× **less**) | old table was wrong: native wins |
| `cultivation_d5` | 0.07 (MDAM 14.9× **more**) | **1.19** (≈ parity) | old 14.9× was all Pauli-sum |
| `cultivation_d3` | 0.90 | **1.35** | both ≈ parity, native slightly over |
| `distillation` | 0.50 | **0.48** | agree (≈ half Clifft) |

## What `flop_table_native.csv` shows

Native dense FLOP is split into `rot` (core rotation factors) | `collapse` (Born+project+norm) | `localizer`
(the `oracle_localize` Cliffords on the block). Clifft is split into `gate` (offdiag+diag) | `meas`.

- **`r ≪ k` (localized magic) → native ≪ Clifft.** `coherent_d3_r3` 0.16×, `rx_d3_r3` 0.16×,
  `coherent_d5_r5` ~6e-4×, and the `r1` family `0` dense (deterministic). These are the real wins.
- **`r ≈ k` (saturated) → native ≈ parity, ~10–35 % over.** `cultivation_d3` 1.35×, `cultivation_d5` 1.19×.
  The excess is **not** a Pauli-sum/replay/rank blow-up (native has none of those, verified in code). It is:
  1. **`rot`** — native applies the deferred core as a bundle at the measurement-time materialised rank
     `r_mat`, which is ≥ Clifft's *per-gate squeezed* rank (Clifft's `StatevectorSqueezePass` reorders so each
     gate hits a locally-minimal active rank). cult_d5 `rot` 450 k vs Clifft `gate` 386 k = **1.17×**; cult_d3
     **1.26×**. This is the dominant excess.
  2. **`localizer`** — pure extra (Clifft has none): a `±Z` localizer Clifford applied to the block. A direct
     `⟨φ|P_m|φ⟩` projector would avoid it (implementation choice, ~8–15 % of Clifft total).
  3. **`collapse`** is actually **cheaper** than Clifft's `meas` (native charges `2^{r_mat} ≤ 2^k`):
     cult_d5 31 k vs 50 k = 0.62×. So the measurement itself is not the excess.

## Bottom line

`C_native,dense = Θ(2^r)`, no asymptotic blow-up. For `r ≪ k` it is exponentially below Clifft (the win
regime). For `r ≈ k` it is ≈ parity, ~10–35 % over, from (1) bundled-core rank vs Clifft's squeezed per-gate
rank and (2) the separable localizer. Where the *wall-clock* gap exceeds the *FLOP* gap (cult_d5: 2.4× wall vs
1.19× FLOP) the remainder is **symbolic control plane**, not dense arithmetic.
