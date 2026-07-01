# MDAM — How the computation is done: measurement-localized near-Clifford simulation

**Scope.** This note records, mathematically and in detail, *how MDAM actually performs its arithmetic* —
specifically **why it only ever materializes the part of the state that the next measurement can see**, never
the full `2^n` (or even the full `2^k`) amplitude vector. It is grounded in the native engine
(`mdam/native_vm/`, `mdam/backend/clifft_axis/cpp/`) and the measured behaviour of the QEC benchmarks.

The one-sentence summary:

> MDAM keeps the Clifford part of the circuit **symbolic** (a tableau + a deferred Pauli/Clifford *frame*),
> keeps the non-Clifford rotations **deferred** as a product of Pauli rotations, and at each measurement
> materializes a **dense amplitude block only over the qubits that rotation chain actually couples to the
> measured observable** — the *measurement-relevant reduced core*, of dimension `2^r` with `r ≤ k`. When the
> non-Clifford content is local, `r ≪ k` and the work `O(m·2^r)` is exponentially smaller than the `O(2^k)`
> a full-register simulator (Clifft) pays.

---

## 1. Notation and primitives

- `n` qubits. The single-qubit Paulis `X, Y, Z`; an `n`-qubit Pauli is `P = i^{pp} X^{x} Z^{z}` with bit-masks
  `x, z ∈ {0,1}^n` and a phase exponent `pp ∈ {0,1,2,3}` (`i^{pp}`). In code a Pauli is a `PackedPauli`
  (`x`/`z` word-arrays + `phase`).
- **Pauli rotation** about axis `P` by angle `θ`:
  ```
  R_P(θ) = exp(−i (θ/2) P) = cos(θ/2)·I − i·sin(θ/2)·P .
  ```
  Because `P² = I`, two rotations about the *same* axis add: `R_P(θ₁)·R_P(θ₂) = R_P(θ₁+θ₂)` (this is the exact
  *folding* identity, §7.3).
- **Clifford conjugation** sends a Pauli to a (signed) Pauli: for any Clifford `C`, `C P C† = ± P'` with `P'`
  Pauli. This is the single algebraic fact that makes the whole scheme work.
- A **stabilizer state** `|s⟩` is the simultaneous `+1` eigenstate of `n` independent commuting Paulis; it is
  describable in `O(n²)` bits (a tableau), no amplitudes.

The target rotation convention matches the code exactly: in
[`native_pauli_apply.hpp`](native_vm/native_pauli_apply.hpp) a rotation is applied as
`R = α·I + β·(i^{pp} X^{x} Z^{z})` with `α = cos(θ/2)`, `β = −i·sin(θ/2)`.

---

## 2. The near-Clifford factorized state

A general `n`-qubit pure state needs `2^n` complex amplitudes. MDAM never stores that. It maintains the state in
the factorized form

```
            ┌── F : a Clifford "frame"  (tableau Xc/Zc + deferred Pauli layer), tracked SYMBOLICALLY, O(n²) bits
  |ψ⟩ = F · |χ⟩       where
            └── |χ⟩ : a "residual" that is a stabilizer state acted on by a product of deferred Pauli rotations.
```

The residual factorizes further as

```
  |χ⟩ = |φ⟩_M  ⊗  |s⟩_{rest}
```

where

- `M ⊆ {0,…,n−1}` is the set of **magic axes** — the qubits that currently carry non-Clifford amplitude;
- `|φ⟩_M ∈ ℂ^{2^r}` is a **dense amplitude block** of rank `r = |M|` (the `dense.resident` buffer);
- `|s⟩_{rest}` is a stabilizer state on the remaining `n − r` qubits (no amplitudes, tableau only).

So the *only* dense memory MDAM ever touches is `2^r`, and the whole game is keeping `r` as small as the
measurement allows. The three carriers:

| object | code | size | what it holds |
|---|---|---|---|
| stabilizer tableau | `NativeTableau` (`Xc/Zc`) | `O(n²)` bits | the Clifford frame `F` (image of each `X_i,Z_i`) |
| inverse frame | `NativeInverseFrame` (`Ax/Az`) | `O(n²)` bits | `F†` — used to pull Paulis *back* through `F` |
| deferred rotations | `pending` ledger | `O(#rot)` Paulis | the not-yet-applied `R_{P'_j}(θ_j)` |
| dense block | `dense.resident` | `2^r` complex | `|φ⟩_M`, the magic amplitudes |

---

## 3. Deferring the circuit: the frame and the pullback

Process the circuit gate by gate, maintaining the invariant `|ψ⟩ = F|χ⟩`.

**Clifford gate `G`.** `|ψ⟩ → G|ψ⟩ = (G F)|χ⟩`, so we absorb it into the frame, `F ← G F`. This is an
`O(n)` tableau/frame update — **no amplitudes touched**. (Code: `frame.h/cnot/cz/s_gate/swap`,
`engine.cx/cz/s/h_axis/…`.)

**Non-Clifford rotation `R_P(θ)`.** Apply it to `|ψ⟩`:

```
  R_P(θ)|ψ⟩ = R_P(θ) F |χ⟩ = F · ( F† R_P(θ) F ) |χ⟩ = F · R_{P'}(θ) |χ⟩ ,   P' = F† P F .
```

The rotation moves *through* the frame and reappears as a rotation about the **pulled-back axis**
`P' = F† P F`, acting on the residual. `P'` is a Pauli (Clifford conjugation), so we simply append `R_{P'}(θ)`
to the deferred ledger. Again **no amplitudes touched** — the rotation is stored, not executed.

Computing `P' = F†PF` is the **pullback** (`NativeDenseEngineState::pullback`): a GF(2) decomposition of the
logical Pauli over the frame's column basis, plus the phase. With the basis built once per frame mutation it is
`O(n²)` to build (`build_inverse_basis`) and `O(n)` per pulled Pauli (`pullback_from_basis`). The frame is only
materialized **lazily**, on the first pullback that actually reads it (`lazy_inverse`); a trajectory that never
reads the frame — e.g. a measurement-only Clifford circuit — pays **zero** rebuilds.

After the whole circuit, the residual is

```
  |χ⟩ = R_{P'_L}(θ_L) ⋯ R_{P'_1}(θ_1) |χ_0⟩ ,     |χ_0⟩ a stabilizer state,
```

a **product of Pauli rotations** — never expanded into a sum, never into a matrix.

---

## 4. Magic axes and the dense block — where the `2^r` lives

A rotation `R_{P'}(θ)` with a non-Clifford angle injects magic *only on the qubits in the support of `P'`*.
Define the magic-axis set `M` as the union of supports of the deferred rotations that have become entangling.
Two facts keep the dense block small:

1. **Stabilizer rotations are free.** If `P'` is a stabilizer of the current `|χ⟩` (i.e. `P'|χ⟩ = ±|χ⟩`), then
   `R_{P'}(θ)|χ⟩ = (cos(θ/2) ∓ i sin(θ/2))|χ⟩` is a *global phase* — no state change. Such rotations never grow
   `M`.
2. **Promotion is on-demand.** A qubit enters `M` (`promote`, rank `r → r+1`, the `dense.resident` buffer
   doubles) **only** when a rotation that must be materialized has `X`-support on it. Until then it lives in the
   stabilizer factor `|s⟩_{rest}` for free.

Hence `r = |M|` is the number of qubits the non-Clifford layer has actually entangled — the *intrinsic* magic
rank of the trajectory, bounded by the peak rank `k` that a full-register method would carry: **`r ≤ k`**,
with strict `r ≪ k` whenever the magic is geometrically local (the QEC win case).

---

## 5. Localization: computing only the measured state

This is the heart of the method. We never materialize all of `|χ⟩`. To measure `Z_q` we materialize **only the
sub-block of `|φ⟩_M` that the measurement can distinguish**, the *measurement-relevant reduced core*.

### 5.1 The measured observable, pulled back

Measuring `Z_q` on `|ψ⟩ = F|χ⟩` has Born statistics

```
  ⟨ψ| Z_q |ψ⟩ = ⟨χ| F† Z_q F |χ⟩ = ⟨χ| P_m |χ⟩ ,     P_m = F† Z_q F
```

so the measurement reduces to measuring the pulled-back Pauli `P_m` on the residual `|χ⟩`.

### 5.2 The reduced core: which rotations actually matter

Write `|χ⟩ = R_{P'_L}⋯R_{P'_1}|χ_0⟩`. A deferred rotation `R_{P'_j}` is **irrelevant to this measurement** if it
can be commuted past `P_m` and the remaining rotations without changing `⟨χ|P_m|χ⟩` — concretely, if `P'_j`
commutes with `P_m` *and* with every rotation that itself reaches `P_m`. Such rotations slide through and act on
`|χ_0⟩` as stabilizer phases, contributing nothing to the outcome distribution.

The rotations that **cannot** be commuted away form the **core**: the anticommuting/entangling closure rooted at
`P_m`,

```
  core(P_m) = smallest set S of deferred rotations such that
              P_m and all P'_j∈S are closed under "anticommutes-with", 
```

built by `dynamic_core_scr` (`native_magic_measure.hpp`) as a graph closure over the pending Paulis. Only the
qubits in `supp(core) ∪ supp(P_m)` need a dense amplitude; call this set `M_meas` and its size

```
  r_mat = | M_meas |   ≤ |M| = r   ≤ k .
```

**This is the localization.** The dense work for the measurement is confined to `ℂ^{2^{r_mat}}`, and `r_mat` is
the rank of the *local* sub-problem the measurement poses — typically far below the global peak `k`.

> The structural part of this core — its membership, the axis layout `M_mat`, the localizer gate list, and the
> rotation masks — is **shot-invariant** (a deterministic function of the static gate stream and the current
> tableau); MDAM caches it once (`StaticPlan`/F4 cache, `core_cache`). Only the rotation **phases/angles** and
> the **Born outcome** are recomputed per shot.

### 5.3 Materialize → measure → shrink

1. **Flush the core** (`oracle_flush_core` / the compiled `magic_execute`). For each rotation `e` in the core:
   pull its axis back (`pb = pullback(e.P)`), promote any new `X`-support qubits into `M_meas`, reduce the axis
   to masks `(mx, mz)` over the `M_meas` layout, and apply `R` to `|φ⟩` with the §7 primitive. Each rotation is
   consumed exactly once — **no re-materialization**.
2. **Localize** `P_m` to a single axis. A Clifford `W` (built from `H/S/CNOT` on the dense axes,
   `oracle_localize`) maps `P_m|_{M_meas} ↦ sign · Z_{r*}` for one chosen axis `r*`; `W` is applied to `|φ⟩` and
   folded into the frame (`right_h/s/cx`) so the global state is unchanged.
3. **Born + sample.** With `P_m` now `= ±Z_{r*}`, the outcome probability is a partial norm of the dense block:
   ```
   s_b = ‖ branch of |φ⟩ with bit r* = b ‖²  =  Σ_{ j : (j≫r*)&1 = b } |φ[j]|² ,    b∈{0,1}
   p₀ = (sign>0 ? s₀ : s₁) / (s₀ + s₁) .
   ```
   Draw one uniform `u`; outcome `= (u < p₀) ? 0 : 1`. (Code: `branch_sqnorm`.)
4. **Project + normalize.** Zero the killed branch, rescale by `1/√(kept norm)`.
5. **Drop the measured axis** (`drop_localized_core`, `drop_residual_products`): the measured qubit becomes a
   determined stabilizer, so it leaves `M`; rank shrinks `r_mat → r_mat − 1` and the dense buffer **halves**.
   *This commit step is what keeps `r` bounded over a long circuit* — it is structurally load-bearing, not
   bookkeeping.

The deterministic case (`P_m` has no magic support after localization, `r<0`) skips the dense work entirely:
the outcome is fixed by `sign` and one RNG draw, `O(1)`.

---

## 6. The single operator normal form

MDAM never builds a boundary operator as a Pauli sum `K_b = Σ_u c_u P_u`, and never an explicit `2^{r}×2^{r}`
matrix. The operator representation is **always** the factorized product

```
  |φ⟩  ←  ∏_{e ∈ core}  R_{P'_e}(θ_e)  |φ⟩
```

applied **one factor at a time** to the dense block, followed by the localize/Born/project/drop of §5.3. There
is no candidate selector (“if cultivation do dense, elif coherent do localized, …”); the regimes are just values
of `r`:

| regime | what the product does | cost |
|---|---|---|
| `r = 0` (empty core) | the product is empty → identity | **0** dense sweeps |
| `r ≪ k` | sweep a small `|φ⟩` | `O(m·2^r) ≪ O(2^k)` → **win** |
| `r = k` | sweep a full `|φ⟩` | `O(m·2^k)` → ≈ Clifft |

---

## 7. The Pauli-apply primitive (the only place arithmetic happens)

Every factor `R_{P'}(θ)` is applied to `|φ⟩ ∈ ℂ^{2^r}` by one primitive,
[`pauli_rot_apply`](native_vm/native_pauli_apply.hpp). With the pulled-back axis reduced to masks `(x,z)` over
the `M`-layout (`x = mx`, `z = mz`), phase `pp`, `α = cos(θ/2)`, `β = −i sin(θ/2)`, and `bph = β·i^{pp}`:

### 7.1 Diagonal branch (`x = 0`: identity or `Z`-only)
`P' = i^{pp}Z^{z}` is diagonal, so `R` multiplies each amplitude by a per-index phase:

```
  φ[j] ←  φ[j] · ( α + bph )      if  parity(j AND z) = 0
          φ[j] · ( α − bph )      if  parity(j AND z) = 1
```
Cost: `2^r` complex multiplies, one streaming pass. (`x = z = 0` is the global-phase / scalar case.)

### 7.2 Butterfly branch (`x ≠ 0`: `X`/`Y` support)
`R` couples each index `j` with its partner `j ⊕ x` (only the canonical `j` without the pivot bit is processed):

```
  let  piv = lowest set bit of x ,   for each j with (j AND piv)=0,  k = j ⊕ x:
      a = φ[j],  b = φ[k]
      s_j = (−1)^{parity(j AND z)},   s_k = (−1)^{parity(k AND z)}
      φ[j] ← α·a + bph·(s_k·b)
      φ[k] ← α·b + bph·(s_j·a)
```
Cost: `2^{r−1}` pairs, `O(2^r)`. This is the exact `2×2` rotation acting on each Pauli-conjugate amplitude pair.

### 7.3 Exact algebraic folding (no expansion)
Before sweeping, co-axial rotations are merged with `R_{P'}(θ₁)·R_{P'}(θ₂) = R_{P'}(θ₁+θ₂)` (same `(x,z,pp)`,
mutually commuting): `m` factors → `#distinct axes` factors, **exactly**, never forming a Pauli sum. This is a
factor-stream rewrite, not a separate execution path.

### 7.4 One primitive, two instantiations
The compiled fast kernel `direct_rot` (`cpp/mdm_core_executor.cpp`, FLOP-instrumented) and the general path
`lincomb` are the **same** math; `nvm_selftest_pauli_apply` confirms bit-identical output over `2·10⁵` random
rotations (max abs diff `= 0`). So §7.1–7.2 *are* the whole arithmetic core of MDAM.

---

## 8. Cost and the `r` vs `k` regimes

Per measurement the dense cost is

```
  C_meas = O( m_core · 2^{r_mat} ) ,
```
with `m_core` core factors (after folding) and `r_mat ≤ k`. Over a shot,
`C_shot = Σ_measurements C_meas`. A full-register simulator (Clifft) instead pays `Ω(2^k)` at the peak **on
every magic operation**, independent of locality. Therefore:

- **`r ≪ k` (localized magic).** `2^{r} ≪ 2^{k}`: MDAM is exponentially cheaper. Measured: `coherent_d5_r5`
  (`k=24`, `r≈12`) — **807×** faster; `coherent_d7_r1` (`k=25`, `r≈0`) — **35,566×**; the `r1` family scales as
  `2^k`, i.e. the advantage grows with code distance.
- **`r = k` (magic-saturated).** `2^{r} = 2^{k}`: no register-size advantage; MDAM does the same dense butterfly
  work as Clifft and degrades **gracefully** to ≈parity. Measured: `cultivation_d5` (`ρ = r = k = 10`) — the
  pure dense butterfly is `79.1 µs` vs Clifft `82.1 µs` (`0.96×`); the residual slowdown is symbolic frame
  scaffolding, not the dense arithmetic.
- **`r = 0` (no magic seen).** The factor product is empty: **zero** dense sweeps. Measured
  (`nvm_core_apply_count`): `coherent_d3_r1`, `coherent_d5_r1`, `distillation` all do `0` core-applies/shot — the
  measurement is a pure stabilizer projection. This is why a localized circuit is never *slower* for lack of
  dense work.

All three are the **same** algorithm at different `r`.

---

## 9. Worked micro-picture

Circuit on 6 qubits: a Clifford prelude `C`, one `T` on qubit 2, a Clifford `D`, then measure `Z_4`.

1. `C` absorbed: `F ← C`. No amplitudes. `r = 0`.
2. `T = R_{Z}(π/4)` on qubit 2: pull back `Z_2` → `P'_1 = F†Z_2F`. Append `R_{P'_1}(π/4)`. Still **no
   amplitudes** — deferred. `r = 0` (not yet materialized).
3. `D` absorbed: `F ← D F`.
4. Measure `Z_4`: `P_m = F†Z_4F`.
   - Build `core(P_m)`. If `P'_1` commutes with `P_m`: core empty, `r_mat = 0` → the outcome is a deterministic
     stabilizer projection, **no dense block built at all**.
   - If `P'_1` anticommutes: core `= {R_{P'_1}}`. Materialize `M_meas = supp` (say `r_mat = 1`, a `2¹ = 2`-vector),
     apply the one rotation by §7, localize, Born over 2 amplitudes, project, drop → back to `r = 0`.

   Either way the dense work is `2^{0}` or `2^{1}`, never `2^{6}`. A full simulator would carry `2^{6}`
   throughout.

---

## 10. Correspondence to the implementation

| step | function | file |
|---|---|---|
| Clifford absorb into frame | `frame.*`, `engine.cx/cz/s/h_axis` | `native_frame.hpp`, `native_magic_state.hpp` |
| pullback `P' = F†PF` | `pullback`, `build_inverse_basis`, `pullback_from_basis` | `native_magic_state.hpp` |
| lazy frame materialization | `lazy_inverse`, `rebuild_inverse_frame` | `native_magic_state.hpp` |
| deferred rotation ledger | `pending`, `promote` | `native_pending.hpp`, `native_magic_state.hpp` |
| measurement-relevant core | `dynamic_core_scr` | `native_magic_measure.hpp` |
| shot-invariant skeleton cache | `StaticPlan` (F4), `core_cache` | `native_magic_measure.hpp`, `native_mdam_shot.hpp` |
| flush core into `|φ⟩` | `oracle_flush_core` / `magic_execute` | `native_oracle_measure.hpp` / `native_magic_measure.hpp` |
| localize `P_m → ±Z_{r*}` | `oracle_localize`, `right_h/s/cx` | `native_oracle_measure.hpp`, `native_magic_state.hpp` |
| Born + project + normalize | `branch_sqnorm` + project loop | `native_oracle_measure.hpp` |
| drop measured axis (shrink `r`) | `drop_localized_core`, `drop_residual_products` | `native_magic_state.hpp` |
| **the one Pauli-apply primitive** | `pauli_rot_apply` (= `direct_rot` = `lincomb`) | `native_pauli_apply.hpp`, `cpp/mdm_core_executor.cpp` |
| folding `R(θ₁)R(θ₂)=R(θ₁+θ₂)` | `fold_core_rotations` | `native_magic_measure.hpp` |
| regime instrumentation | `nvm_core_apply_count`, `nvm_selftest_pauli_apply` | `native_mdam_vm.cpp` |

---

### Take-away

MDAM is exact (it reproduces the full-register samples bit-for-bit). Its speed comes entirely from **never
representing more of the state than the next measurement can resolve**: Cliffords stay symbolic, rotations stay
deferred as a Pauli *product*, and only the anticommuting closure rooted at the pulled-back observable is
materialized as a `2^{r_mat}`-amplitude block — swept once, measured, and shrunk. When the magic is local
(`r ≪ k`) this is exponentially cheaper than a `2^k` register; when it is global (`r = k`) it degrades to the
same dense work, never worse in arithmetic.
