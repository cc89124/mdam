# Phase 6 — Policy 3 precise design: Pauli-residue frame + p_x/p_z-diagonal T (analysis only)

Goal: port clifft's discipline (Pauli-residue frame, p_x/p_z-conditioned diagonal T/T†, dormant-axis
Hadamard absorbed at EXPAND) onto bounded's rank-r active register, so cultivation runs at
**runtime H = 0** and **F_bnd ≤ F_clf**, with the rank advantage preserved. Design + proofs only —
NO implementation, NO kernel/gate/soft-reset; a05843e / tag / fallbacks preserved.

Empirical anchors (this session): cultivation has **0 AG-measures** (measurements never inject
Clifford into the frame — phase6_incarnation.txt), all generators are physical-T (R_Z, diagonal),
clifft achieves **0 array_h** end-to-end (measured), bounded's Born is c=2 vs clifft c=8.

---

## §1 State invariant

    |Ψ⟩ = γ · C_outer · P_res · ( |φ⟩_A ⊗ |0⟩_D )

| symbol | meaning | current bounded field |
|---|---|---|
| γ | global scalar (phase + deferred norm) | implicit (norm) + global phase |
| **C_outer** | Clifford on the structure OUTSIDE the active register **and the H-free part on active axes**; localizing CNOT/CZ/S only, **no Hadamard on any active axis** | subset of `Xc/Zc` tableau `C` |
| **P_res** | per-active-axis Pauli residue `∏_v i^{p_v} X^{x_v} Z^{z_v}` | `F` (Pauli frame) + new per-axis `p_x[v]/p_z[v]` |
| **|φ⟩_A** | canonical active array, each axis stored in its **born eigenbasis** (the absorbed-H basis) | `phi` over `M` |
| |0⟩_D | dormant qubits | implicit |

The ONLY change vs today: the Hadamard content that currently lives in `C` (making pulled-back T
generators off-diagonal) is **moved** — onto the array at EXPAND (born basis) and into `P_res`
(Pauli part). Invariant: **C_outer contains no Hadamard acting on an active axis.** Then every
physical-T generator pulls back to a single-axis diagonal `Z_v` (up to the Pauli residue), never X.

Reduction to today's representation (exactness witness): expanding the born basis back to |0⟩ via the
absorbed Clifford and folding `p_x/p_z` into `C` recovers the current `γ C F |φ⟩` form bit-for-bit —
so Policy 3 is a *change of bookkeeping*, not of the state.

---

## §2 Operation update rules

Notation: axis v, residue on v is `i^{p} X^{x} Z^{z}` (x=p_x[v], z=p_z[v]). "sweep" = dense pass.

| op | array sweep | rank | residue (p_x,p_z) | γ / phase | pending |
|---|---|---|---|---|---|
| **promote/EXPAND(|e⟩)** | 1 write of 2^r new half (copy [+phase]); **NOT** a 4·2^r H | **+1** | new axis p=(0,0) in its born basis | /√2 (and ±1,±i for |∓⟩,|±i⟩) | unchanged |
| **H_v** (dormant→active) | absorbed into EXPAND (born |+⟩) | +1 | — | — | — |
| **H_v** (already active) | `_h_axis` **4·2^r** (the only genuine H; §9 case 2/3) | 0 | swap p_x↔p_z | +2·(p_x∧p_z) | conj pending by H |
| **S_v / S†** | none (Pauli rule) OR `_s_axis` 2·2^r if materialized | 0 | p_z ^= p_x ; phase ±i^{p_x} | i^{±p_x} | conj pending |
| **CNOT(c,t)** | none (frame-fold, free) | 0 | x_t^=x_c, z_c^=z_t | 0 | conj pending |
| **CZ(c,t)** | none (frame-fold) | 0 | z_c^=x_t, z_t^=x_c | 0 | conj pending |
| **SWAP** | none (index relabel) | 0 | swap axes | 0 | relabel |
| **Pauli X/Z corr.** | none | 0 | flip p_x / p_z | sign per anticommute | — |
| **T_v / T†_v** | **diagonal half-array 3·2^r** (T^{(-1)^{p_x}}) | 0 | unchanged | γ·e^{±iπ p_x/4} | — |
| **R_Z(θ)_v** | **diagonal half-array 3·2^r** (R_Z((-1)^{p_x}θ)) | 0 | unchanged | unchanged | — |
| **R_X/R_Y(θ)** | off-axis: needs born-X basis (EXPAND) or active H (§9 case 3) | 0/+1 | — | — | — |
| **measure Z_v** | Born `_branch_sqnorm` (c=2) + project + `_drop` | **−1** | resolve v; Pauli update others | norm into γ | — |
| **projection/normalize** | in-place /‖·‖ (c=2) | 0 | unchanged | norm into γ | — |
| **demotion (drop)** | compaction copy (0 FLOP) | −1 | drop v's residue | — | — |
| **copy/branch/reset** | buffer copy | 0 | deep-copy p_x/p_z/C_outer | — | deep-copy |

The rows with "none" in the array column are the wins: Clifford structure stays symbolic (C_outer +
residue), only **T/R_Z (diagonal, c=3)** and **Born (c=2)** touch the array. No butterfly, no H —
except §9 cases 2/3/4.

---

## §3 Exact diagonal T rule (phase-complete derivation)

Conventions: `T = diag(1, ω)`, `ω = e^{iπ/4}`; `X T X = ω T†` (since `X diag(1,ω) X = diag(ω,1) =
ω·diag(1, ω̄) = ω T†`); `Z T = T Z`. Then for residue `X^x Z^z`:

    T · X^x Z^z = ω^x · X^x Z^z · T^{(-1)^x}          (★)

Proof: x=0 ⟹ `T Z^z = Z^z T` (T,Z diagonal). x=1 ⟹ `T X Z^z = (T X) Z^z = ω X T† Z^z = ω X Z^z T†`
(T† commutes with Z). ∎  Including the residue scalar `i^p` (commutes): `T (i^p X^x Z^z) = i^p ω^x
(X^x Z^z) T^{(-1)^x}`.

**Consequence (the kernel dispatch):** applying T to the active axis with residue (x=p_x, z=p_z):

    array ← T^{(-1)^{p_x}} |φ⟩   (diagonal half-array, c=3) ;   γ ← γ·ω^{p_x} ;   residue UNCHANGED

So `p_x=0` → apply T; `p_x=1` → apply **T†** and absorb `ω=e^{iπ/4}` into γ. **`p_z` and `i^p` do
NOT change the kernel** (Z and the scalar commute with T) — they only ride along. This is byte-for-
byte clifft's `exec_array_t` (svm_kernels.inl:982-987: `px → apply_phase_waterfall(−1/√2) ;
multiply_phase(e^{iπ/4})`). T† mirror: `T† X = ω̄ X T`, so `p_x=1` → apply T, γ·ω̄.

**General R_Z(θ):** `R_Z(θ) = diag(e^{-iθ/2}, e^{iθ/2})`, `X R_Z(θ) X = R_Z(-θ)` (no global phase, det 1),
`Z R_Z = R_Z Z`. So

    R_Z(θ) · X^x Z^z = X^x Z^z · R_Z((-1)^x θ)        (no phase)

⟹ apply `R_Z((-1)^{p_x} θ)` diagonally (c=3), residue and γ unchanged. The existing `rot:diaghalf`
kernel already IS `R_Z` on a half-array; the only new logic is the sign flip `θ → (-1)^{p_x}θ`. This
also gives the frame rule `R_Z(θ) X = X R_Z(-θ)` requested: an X-residue flips the rotation sign,
nothing else.

---

## §4 EXPAND-absorbed eigenstate initialization (the Hadamard at 0 cost)

A dormant axis needing born-state |e⟩ is created by writing ONLY the new 2^r-element upper half — never
a 4·2^r butterfly over the whole array:

| born |e⟩ | upper-half write `arr[2^r+i] =` | lower `arr[i]` | norm | FLOP/elt |
|---|---|---|---|---|---|
| |0⟩ | 0 | φ[i] | 1 | 0 |
| |1⟩ | φ[i] ; then arr[i]=0 | 0 | 1 | 0 |
| |+⟩ | φ[i] | φ[i] | 1/√2 | 0 (copy) |
| |−⟩ | −φ[i] | φ[i] | 1/√2 | 2 (rcmul) |
| |+i⟩ | i·φ[i] | φ[i] | 1/√2 | 6 (cmul) |
| |−i⟩ | −i·φ[i] | φ[i] | 1/√2 | 6 (cmul) |

Each is one pass over 2^r words, rank r→r+1. Materializing |+⟩ costs the SAME as |0⟩ (a copy) — the
Hadamard is absorbed for free. Compare the avoided active-axis H: `_h_axis` is a 4·2^r butterfly over
the EXISTING array. So **EXPAND(|+⟩) replaces 4·2^r with 0** for the dormant-prep case. (This is
clifft's EXPAND/EXPAND_T, svm_kernels.inl:893/1018.) The born basis fixes which Pauli is diagonal on
that axis; the residue p_x/p_z then tracks Pauli evolution within that fixed basis.

---

## §5 weight-w generator handling (three event-level examples, cultivation)

Each axis stored in its born basis ⟹ a physical Pauli that matches the born basis acts as Z (diagonal)
on the array. Decompose: born-aligned single-qubit Paulis ⟹ Z-strings (free CNOT-collapse + diagonal);
genuine non-Pauli ⟹ dense H (only off-axis rotations, §9).

1. **weight-3 pure-X `X_a X_b X_c`** (batch 0, P_4): a,b,c born |+⟩ ⟹ each X acts as Z on its array →
   `Z_a Z_b Z_c` (diagonal). Free CNOT-collapse `CNOT(a,b),CNOT(a,c)` (frame, 0 FLOP) → `Z_a` →
   `rot:diaghalf` c=3. **Dense H = 0** (3 H's absorbed at a,b,c EXPAND).
2. **weight-2 mixed `X_a Z_b`** (batch 6): a born |+⟩ (X→Z), b born |0⟩ (Z→Z) ⟹ `Z_a Z_b` diagonal →
   free collapse → c=3. **Dense H = 0** (per-axis born bases differ, both diagonal).
3. **q5 X-type then Z-type** (batch 4 vs batch 8, same incarnation): both are physical T's; the X/Z is
   the **Pauli residue** p_x[5]/p_z[5] (NOT a basis change — q5 born once). §3 ⟹ X-type → T†(5) (p_x=1),
   Z-type → T(5) (p_x=0), both diagonal c=3. **Dense H = 0** — handled by T/T† dispatch, not an H.

The only thing that would force a dense H: a generator whose single-qubit Pauli on some axis is NOT
the born one and NOT a Pauli residue of it — i.e., a genuine off-axis rotation (R_X/R_Y), §9.

---

## §6 Measurement-boundary preservation (the crucial proof)

**Claim:** across every cultivation measurement, the state stays expressible as `γ C_outer P_res |φ⟩`
with **C_outer gaining no Hadamard on an active axis**, so runtime array_h stays 0.

**Proof (cultivation, empirically anchored):**
- (a) **Measurements inject no Clifford into the frame.** cultivation has **0 AG-measures**
  (phase6_incarnation.txt): every measurement takes the magic path (`_branch_sqnorm` Born + project +
  drop), which is an **array fold + Pauli residue resolution** — it updates `P_res` and drops an axis,
  it does NOT multiply a Clifford into `C_outer`. (The AG path — the only one that mutates `C` with a
  non-Pauli Clifford — is never taken here.) So C_outer evolves ONLY via gates.
- (b) **Every cultivation Hadamard is on a dormant axis.** clifft's measured **0 array_h** end-to-end
  ⟹ no Hadamard ever hits an already-active axis (clifft would emit array_h if one did, backend.cc:386).
  So all H's are prep (dormant→active), absorbed at EXPAND (§4), never entering C_outer-on-active.
- (a)+(b) ⟹ C_outer-on-active stays Hadamard-free across all 15 measurement boundaries ⟹ every
  physical-T pulls back to diagonal `Z_v` ⟹ **runtime array_h = 0** is maintained end-to-end. ∎

**Residue class at each boundary:** before/after every measurement the active state is
`Pauli-residue-only` (p_x/p_z), confirmed by 0 AG-measures. **Canonicalization is required only if an
AG-measure occurs** (a stabilizer measurement anticommuting with a non-magic stabilizer) — then a
non-Pauli Clifford enters and a one-shot re-diagonalization (or the Phase-2 localizer) is needed. That
cost is **the same one clifft pays** for the same measurement (clifft also updates its frame via
Gottesman-Knill). cultivation has zero such events.

---

## §7 Preservation of the existing bounded invariants

Policy 3 changes only the *bookkeeping split* of `C → (C_outer, born-basis, P_res)`; the physical state
is identical (§1 witness). Therefore:
- **core membership / active-axis birth-death:** unchanged — promote/drop happen at the same logical
  points; EXPAND-in-basis is the SAME promote event with a chosen born state (no extra promote). Verified
  structurally: the born-basis choice is made AT the existing `_promote`, adding no new axis.
- **resident/transient rank:** unchanged — born-basis is set on the new half at EXPAND (rank +1 only);
  in-register residue updates touch no rank (Phase-4 §5). Peak rank identical.
- **rotation-once:** each pending rotation still flushes exactly once (the dispatch chooses diagonal vs
  off-axis, never both).
- **exact measurement probability:** §1 witness ⟹ identical state ⟹ identical Born p0 (the T/T† and
  R_Z(±θ) rules are exact, §3). 
- **shot-invariant schedule:** the born-basis / residue rules are deterministic functions of the
  (offline-known) gate/measurement stream, independent of RNG.

No invariant requires a new axis promotion; the frame re-split is rank-neutral.

---

## §8 Cultivation projected result (virtual application, pre-implementation)

Applying the §2–§5 rules to the cultivation trace:

| quantity | Policy 3 (projected) | clifft-unfused | current bounded (P2) |
|---|---:|---:|---:|
| **runtime H-sweeps** | **0** | 0 | 91 (butterfly) / 47 (batch) |
| diagonal T/T† (c=3) | 91 | 91 | 0 |
| off-diagonal butterfly | 0 | 0 | 91 |
| diagonal-T FLOP (3·Σ2^r @ bnd ranks) | 164.9k | 174.3k | — |
| Born/measurement FLOP | 28.95k (sqnorm 20.68k + norm 8.27k, c=2) | 33.09k (meas_interfere, c=8) | 28.95k |
| array_s / array_cz | 0 (frame-routed) | 5.47k | 0 |
| **total FLOP** | **≈ 193.9k** | 212.8k | 727.7k |
| **ratio vs clifft** | **0.91×** | 1.00× | 3.42× |
| touched words | ≤ clifft (CNOT symbolic) | 265k | 98.7k |
| peak rank | 10 | 10 | 10 |

**Residual-0 attribution (F_bnd ≤ F_clf):** bnd diagonal-T 164.9k ≤ clf 174.3k (bnd ranks ≤ clf) [−9.4k]
+ bnd Born 28.95k ≤ clf meas 33.09k (c=2 vs c=8) [−4.1k] + bnd saves array_s/cz 5.47k (frame) [−5.5k] =
**F_bnd ≈ 193.9k = F_clf − 18.9k.** So Policy 3 not only reaches parity, it goes **BELOW** clifft
(0.91×) via bounded's cheaper Born + frame-routed Cliffords — while keeping H = 0. Target met.

---

## §9 Generality boundary (when 0-H parity holds vs when the localizer is still needed)

| regime | generator | 0-H parity? | mechanism / fallback |
|---|---|---|---|
| **1. dormant-axis H absorbable (cultivation)** | physical T (R_Z), prep-H on dormant axes, no AG-measure | **YES** | EXPAND-in-basis + p_x/p_z diagonal-T; F_bnd ≤ F_clf |
| **2. active-axis non-Pauli basis change** | T after an H on an already-active axis | conditional | clifft pays array_h (c=7); bounded pays one `_h_axis` (c=4) — bounded ≤ clifft iff r ≤ k (rank-advantaged). Not free, but not extra vs clifft. |
| **3. RX/RY noise** | rotation generator is genuinely X/Y (off-axis), not frame-induced | NO (needs 1 H) | the **Phase-2 collapse-first localizer is the fallback** (1 H + diagonal). On a fresh axis the H is EXPAND-absorbable (parity); on an active axis it is a real H (= clifft's array_h/u2). |
| **4. multi-axis mutually anticommuting rotations** | e.g. an X-rot and a Z-rot on the same axis interleaved | NO (≥1 H per basis) | anticommuting ⟹ no shared diagonal basis ⟹ Phase-2 localizer per basis; clifft equally pays (array_h / u2 / u4). |

**Conclusion:** 0-H parity is exactly regime 1 (all rotations diagonal-after-frame-absorption, H's
dormant-absorbable, no Clifford-injecting measurement). Regimes 2–4 are where the **existing Phase-2
localizer must remain as fallback** — and there bounded is ≤ clifft when rank-advantaged, never worse
in coefficient. So Policy 3 is an *additional diagonal fast-path*, not a replacement: keep the
localizer for off-axis/active-H/anticommuting cases.

---

## §10 Implementation plan (staged; do NOT start until approved)

- **Step A** — residue **shadow** state only: add `p_x[v]/p_z[v]` per active axis + `C_outer` split,
  maintained alongside the current `C/F`, with a cross-check that `(C_outer, born, P_res)` reconstructs
  the current `C F` frame bit-for-bit on every op (like the inverse-frame shadow-verify). No behavior
  change. Gate behind a flag, default off.
- **Step B** — T/T† **diagonal dispatch**: when a flushed rotation localizes to a single active axis
  with the generator already diagonal-up-to-residue, apply `rot:diaghalf` with `θ→(-1)^{p_x}θ` and
  `γ·ω^{p_x}` instead of the butterfly. Bit-exact vs current (records/rank/p0).
- **Step C** — EXPAND-absorbed Clifford-eigenstate init: `_promote(q, born=|e⟩)` writing the new half
  per §4; route dormant-axis prep-H into the born basis instead of `_h_axis` + frame-H.
- **Step D** — cultivation **runtime-H=0 verification**: assert 0 `purge:h`/butterfly, FLOP ≤ clifft,
  records/rank/p0 bit-exact vs a05843e.
- **Step E** — full measurement/demotion regression: 9-circuit record/rank/p0 bit-exact, shadow-verify
  0 mismatch, memory bound, rotation-once, AG-measure path canonicalization (regime 2/§6).
- **Step F** — fallback integration: keep the Phase-2 localizer for regimes 2–4 (off-axis/active-H/
  anticommuting); Policy 3 dispatch only when the diagonal precondition holds; never delete the
  localizer or the `_pullback_via_basis` path until the full exactness suite passes.

Preserve `a05843e`, `phase2-localizer-invframe`, and all fallbacks throughout.
