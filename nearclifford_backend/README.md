# nearclifford_backend — near-Clifford simulation of clifft circuits

A **complete, standalone** backend for clifft bytecode (sibling to the tensor-tree
`ttn_backend/`). It reproduces the exact `clifft.sample` measurement-record
distribution, but represents the active quantum state in **near-Clifford** form: a
Clifford "frame" carried as a stabilizer *tableau* (polynomial, GF(2) bit-ops) times
a small **dense magic register** that holds only the genuinely non-Clifford part of
the state.

This document is the conceptual + mathematical reference. For the cross-backend
memory / active-state / FLOP comparisons see `reports/per_step_active_state/`,
`reports/per_step_memory_3way/`, and `reports/per_step_flops/`.

---

## 1. The idea in one sentence

Every Clifford operation in a quantum circuit can be tracked by *relabelling Pauli
operators* (Gottesman–Knill, `O(n²)` bits) instead of by touching a state vector;
so if a circuit is "almost Clifford" — a stabilizer backbone with a few non-Clifford
rotations — you only ever need a dense vector over the handful of qubits the
rotations actually entangle, while the entire stabilizer structure stays in the
tableau **for free**.

---

## 2. State representation

### 2.1 The central decomposition

The active state is always kept in the factored form

```
|psi>  =  U_C ( (⊗_{i ∉ M} |0>_i)  ⊗  |phi>_M )
```

* **`U_C`** — a Clifford unitary, *never built as a matrix*. It is stored as a
  **tableau**: the images of the computational Pauli generators,

  ```
  Xc[i] = U_C X_i U_C^†,   Zc[i] = U_C Z_i U_C^†   (i = 0 … n-1)
  ```

  i.e. `2n` Paulis. Knowing where `U_C` sends every `X_i` and `Z_i` determines it
  completely (a Clifford is fixed by its action on the Pauli generators).

* **`M`** — the **magic register**: the ordered set of qubits that a non-Clifford
  rotation has had to take out of `|0>`. Every other qubit is still `|0>` *in the
  frame* and costs nothing.

* **`|phi>_M`** — a dense complex amplitude vector of dimension `2^{|M|}` over the
  magic qubits. This is the *only* object whose size grows exponentially, and it
  grows only in `|M|`, not in `n`.

The whole game is to keep `|M|` (later, the largest entangled *block* of `M`) as
small as the physics allows.

### 2.2 Pauli and tableau encoding

A Pauli on `n` qubits is stored as two `n`-bit masks plus a phase:

```
P  =  i^p · ∏_q  X_q^{x_q} Z_q^{z_q},      p ∈ {0,1,2,3}   (the factor is i^p)
```

— `(x, z, p)` with `x, z` Python big-ints (bit `q` = qubit `q`). Multiplication
(`pauli_mul`) and the commutation test (`pauli_commute`) are pure bit arithmetic:

```
P_a P_b = (x_a ⊕ x_b,  z_a ⊕ z_b,  p_a + p_b + 2·popcount(z_a & x_b)  mod 4)
[P_a, P_b] = 0   ⟺   popcount(x_a & z_b) ⊕ popcount(z_a & x_b) = 0     (symplectic)
```

The `2·popcount(z_a & x_b)` term is exactly the sign you pick up commuting the
`Z`-part of `a` past the `X`-part of `b` (`ZX = −XZ`). Everything downstream —
gates, pullback, measurement — is built from these two operations, so the Clifford
machinery is `O(n)`-word bit arithmetic with **zero floating point**.

---

## 3. Clifford gates are free (frame conjugation)

Appending a Clifford `G` (`U_C ← G U_C`) means every stored image becomes
`G·(image)·G^†`. Because images are Paulis, this is just a Pauli relabelling of the
`2n` rows — `O(n)` per gate, never touching `M` or `|phi>`:

```
H_q :  X_q ↔ Z_q,            Y_q → −Y_q
S_q :  Z_q' = z ⊕ (x_q ≪ q), phase += x_q·(+i)   (S^† : −i)        (S X S^† = Y)
CX(c,t): X_c → X_c X_t,  Z_t → Z_c Z_t   (X_t, Z_c unchanged)
CZ(a,b) = H_b · CX(a,b) · H_b
```

(See `simulator.py::h/s/cx/cz`.) clifft's `EXPAND` "birth" of a fresh active qubit
in `|+>` is likewise just a frame event. **No Clifford gate ever costs a FLOP** in
this representation — this is the structural reason the backend can be exponentially
cheaper than a dense or tensor simulator on stabilizer-heavy circuits.

---

## 4. Non-Clifford rotations

A rotation gate is `R_P(θ) = exp(−i θ P/2)` for some logical Pauli `P`. Applied to
`|psi> = U_C|base>`:

```
exp(−i θ P/2) · U_C |base>
   =  U_C · exp(−i θ (U_C^† P U_C)/2) · |base>
   =  U_C · exp(−i θ P'/2) |base>,        P' := U_C^† P U_C.
```

So the *only* thing to do is apply `exp(−i θ P'/2)` to the cheap base state
`(⊗|0>) ⊗ |phi>`. Three steps:

### 4.1 Pullback `P' = U_C^† P U_C`  (`_pullback`)

We need `P` in terms of the frame. The tableau columns `{Xc[i], Zc[i]}` are the
images of the computational generators and they generate the whole Pauli group, so
we solve a **symplectic GF(2) linear system**: find coefficient bits `b` with

```
∏_j (image_j)^{b_j}  =  P        (matching the (x,z) support of P; Gaussian
                                  elimination over the 2n image columns)
```

Because `image_j = U_C g_j U_C^†` (with `g_j` the computational generator `X_i`/`Z_i`),

```
P = ∏ image_j^{b_j} = U_C (∏ g_j^{b_j}) U_C^†   ⟹   P' = U_C^† P U_C = ∏ g_j^{b_j}.
```

The pulled-back support `(x', z')` is read off from the same coefficients applied to
the computational generators, and the phase is **exact**: it is the phase of the
computational-generator product minus the phase of the image product
(`(R.p − Q.p) mod 4` in the code). This is `O(n²)` bit-work and phase-exact
(verified against a dense reference to fidelity 1.0).

### 4.2 Promotion criterion

`P'` may have `X`/`Y` support (an `x'` bit) on a qubit that is still `|0>`. An `X`
on `|0>` leaves the computational basis, so that qubit **must** be promoted into the
magic register:

```
for each q with (x' >> q) & 1 and q ∉ M:   M.append(q);  |phi> ← |0> ⊗ |phi>
```

A `Z`-only action on a `|0>` qubit is a no-op (`Z|0> = |0>`) — *no promotion*. This
single asymmetry is the whole story of where the cost comes from (see §8.4): a
`Z`-type rotation on a `|0>`-frame qubit is free, an `X`/`Y`-type one is not.

### 4.3 The dense update

Restricted to `M`, `P'_M = i^{p'} X^{x'_M} Z^{z'_M}` acts on `|phi>` by an
index permutation + sign, in `O(2^{|M|})` and **without forming a matrix**
(`_apply_magic_pauli`): the `X`-mask permutes amplitudes (`idx ⊕ x'_M`), the
`Z`-mask supplies a `±1` parity sign, and `i^{p'}` an overall phase. Then

```
|phi>  ←  cos(θ/2) |phi>  −  i sin(θ/2) · P'_M |phi>.
```

---

## 5. Measurement

Measuring `Z_q` is where the representation pays off or promotes, depending on
**how the measured Pauli relates to the frame**.

The measured logical Pauli is `Pm = Z_q`. Two mutually exclusive cases
(`simulator.py::measure_z`):

### 5.1 Stabilizer path — Gottesman–Knill, *no promotion* (`_ag_measure`)

If `Z_q` **anticommutes with some non-magic stabilizer** `Zc[i]` (`i ∉ M`), the
outcome is a fair coin and the state update is a pure tableau relabelling:

```
out ~ Uniform{0,1}
pick a pivot p with Zc[p] anticommuting Pm;  Sp := Zc[p]
for every other row R (stab Zc[i], destab Xc[i]) anticommuting Pm:  R ← R · Sp
Xc[p] ← Sp;   Zc[p] ← (−1)^out · Pm
```

This is the standard stabilizer measurement: `O(n²)` **bit-ops, zero FLOP, and the
magic register is untouched**. Most syndrome measurements in a QEC circuit take this
path — which is why a distance-`d` code with thousands of stabilizer measurements
can run with a magic register of size 0.

### 5.2 Magic path — Born rule on `|phi>`

If `Z_q` commutes with all non-magic stabilizers, it pulls back to a Pauli `P'`
acting only on `|0>`-`Z` qubits and the magic register. Any residual non-magic
`X`-support promotes (as in §4.2); then it is a genuine measurement on `|phi>`:

```
v   = P'_M |phi>
⟨P'⟩ = Re ⟨phi|v⟩,   p0 = (1 + ⟨P'⟩)/2          (Born probability of +1)
out = 0 if rand() < p0 else 1,   sign = (−1)^out
|phi> ← normalize( ½ (|phi> + sign · v) )       (projector (I ± P'_M)/2)
```

This costs `O(2^{|M|})` FLOP (the `vdot`, the projection, the norm).

### 5.3 Compression (`_compress_magic` / block `factor`)

After a magic measurement a qubit may have collapsed to a product `|0>`/`|1>`. The
register is scanned and any disentangled qubit is dropped (its factor peeled off),
keeping `|M|` minimal. The block backend (§7) generalises this to peel *equatorial*
product factors as well, which is essential.

---

## 6. Lazy deferral — `LazyNearClifford` (`lazy.py`)

The base simulator promotes a qubit the instant a rotation touches it. But many
rotations **commute with the measurements that come later** and therefore never
affect any sampled outcome. Lazy deferral exploits this.

### 6.1 Deferred state form

All Cliffords are pushed to the right of the rotations:

```
|psi>  =  ( ∏_j R_{L_j}(θ_j) )  ·  U_C ( |0>_{∉M} ⊗ |phi>_M )
```

The rotations `R_{L_j}(θ_j)` are kept **pending**, as a list of generators
`L_j = (x, z, phase, θ)`.

### 6.2 Physical-frame generators and their conjugation

`L_j` is stored in the **physical (lab) frame**: an `R_Z` applied to qubit `q` is
`L = Z_q` *at application time*, with no pullback. When a later Clifford `G` arrives,
every pending generator is conjugated in place,

```
L_j ← G L_j G^†      (O(1) per pending Pauli per 1–2q gate; _conj_h/_conj_s/_conj_cx)
```

so the pending list always describes the rotations in the *current* physical frame.

### 6.3 Why the physical frame (not the pre-frame) is the right place

A pending generator must survive a measurement's tableau relabelling. Physical-frame
storage is **invariant** under it: if a measurement updates `U_C → U_C V`, the
physical operator `U_C P U_C^†` is unchanged, so a pending rotation keeps meaning the
same physical rotation. (Storing pre-frame Paulis would be wrong — the projection
relabels the `|0>` frame and silently corrupts deferred rotations.)

### 6.4 Anticommutation-connected core flush

At a measurement of physical `Z_q`, a pending rotation `R_L(θ)` may be **pushed
through the projector for free iff `L` commutes with `Z_q`** — and, transitively,
with everything already in the commuting set. So we flush exactly the
**anticommutation-connected component** of `Z_q` in the pending list
(`_core_indices`, a graph reachability over the anticommutation relation):

```
core = { j : L_j anticommutes Z_q, or anticommutes some L_k already in core }
flush every L_j ∈ core into the dense register (pull back, promote, apply);
keep the rest pending.
```

Concretely: a qubit born `|+>` carrying a coherent `R_Z` that is later read in `Z`
has `L = Z_q` **commuting** with the `Z_q` measurement → never flushed → genuinely
free. An ancilla whose `R_Z` is rotated to `X` by a syndrome-extraction `H`
anticommutes with its `Z` readout → flushed, but it is re-measured every round so the
live core stays small and bounded. This is what turns the naive "promote everything"
into "materialise only the anticommuting core".

---

## 7. Block-factored magic register (`block_magic.py`)

A monolithic `|phi>` of dimension `2^{|M|}` conflates *entanglement* with mere
*count*. A coherent `R_Z` on a `|+>`-born data qubit makes it non-stabilizer (magic),
but it stays an **unentangled, single-qubit equatorial** state `(|0> + e^{iα}|1>)`:
it should cost dimension 2, not double the whole vector. The monolithic vector can
peel product `|0>/|1>` factors but **not product equatorial factors**, so these
inflate one dense vector (`coherent_d5_r5` → `|M| > 20`).

### 7.1 Representation: a tensor product of entangled blocks

`MagicRegister` stores the magic part as

```
|phi>_M  =  ⊗_b  |v_b>_{Q_b}          (blocks Q_b partition M)
```

— a list of independent blocks `(Q_b, v_b)`, each `v_b` a dense vector of dimension
`2^{|Q_b|}`. The **live resource is the largest block**,

```
max_block()  =  max_b |Q_b|,
```

not `|M|`. Memory is `Σ_b 16·2^{|Q_b|}` bytes plus the polynomial tableau/pending
overhead.

### 7.2 The factoring map (`factor` / `_factor_block`)

After every operation, each block is scanned for a qubit `j` that factors out. For
the bipartition (qubit `j` | rest), reshape `v` to `(high, 2, low)` so the two slices
`b0 = v[:,0,:]`, `b1 = v[:,1,:]` are the `j=0` / `j=1` halves; this cut has **Schmidt
rank 1 ⟺ qubit `j` is unentangled from the rest**. The three rank-1 cases, detected
by norms / a single inner product (no SVD):

```
‖b1‖ ≈ 0                          → qubit j is |0>  : DROP it (rejoins the |0> frame)
‖b0‖ ≈ 0                          → qubit j is |1>  : split into its own dim-2 block
b1 = α·b0  (‖b1 − α b0‖ ≈ 0)      → product/equatorial: split (|0> + α|1>)/√… off
```

The last case is the crucial one the monolithic register cannot do: it peels a
**product equatorial qubit** to a dim-2 block, so coherent phases cost 2 each instead
of doubling the register. (`_factor_block` uses strided reshapes + `np.linalg.norm`
— `O(2^{|block|})`, never an SVD.)

### 7.3 Merge on multi-block operations (`_merge`)

A rotation/measurement whose support touches several blocks first merges them into
one block via Kronecker products (`v ← v2 ⊗ v`, qubits concatenated), applies the
operation, then re-factors. So blocks grow only when an operation genuinely entangles
them, and shrink again whenever the state allows.

### 7.4 What "live resource" means

`max_block()` is the size of the **largest genuinely-entangled non-Clifford core**.
`diag_peak_block.py` confirms (via an exact product-cut / finest-tensor-factorization
sweep) that e.g. the `coherent_d5_r5` peak-13 block is *irreducible* — not a tensor
product the factorizer is missing — so 13 is a genuine entangled core, not a
factoring artifact.

**Transient vs resident — the reported peak is the memory high-water mark.** That
peak-13 block is the **intra-step transient** high-water mark: a measurement's
anticommutation-core flush momentarily materialises the full entangled block (all
pending rotations applied + factored) *just before* the measurement projector
collapses it back, at which point the block **resident** at the step boundary settles
to 12 (and `factor()` peels the 13th qubit). Both are exact; for a **memory
feasibility** figure the transient is the honest, conservative number (the backend
genuinely allocated `16·2^13`=128 KB at that instant), so it is the MAIN figure
throughout; the settled resident 12 (`16·2^12`=64 KB) is reported alongside as a
secondary metric. `diag_peak_block.py` / `verify_block.py` (which sample `max_block`
at every `_bump`, i.e. intra-step) report the transient 13; the per-step report
curves additionally record the settled resident 12 via `take_step_peak()`.

---

## 8. Cost model and where it wins

There are **two distinct cost units**, and conflating them is the usual mistake:

| work | unit | who pays it | scaling |
|---|---|---|---|
| Clifford gates + stabilizer measurements (tableau/frame, pullback) | **GF(2) bit-ops** | all near-Clifford work | polynomial: gate `O(n)`, measurement `O(n²)` |
| dense magic-register evolution (`apply_rotation`, Born measure, kron) | **FLOP** (`matmul`-like) | only when magic is materialised | `O(2^{max_block})` |
| block factoring scan (`factor`) | **FLOP** (`norm`/`vdot`) | the price of the block representation | `O(|block|·2^{max_block})` |

### 8.1 Memory

```
mem = Σ_b 16·2^{|Q_b|}                      (magic blocks, the only exponential term)
    + 2n(2·⌈n/8⌉ + 1)                       (tableau: 2n Pauli images)
    + |pending|·(2·⌈n/8⌉ + 16)              (deferred rotations)
```

For a pure stabilizer circuit the first term is **0** and only the polynomial tableau
remains.

### 8.2 Active-state dimension

The dense-equivalent dimension is `2^{max_block}`, versus a dense baseline `2^k`
(`k` = concurrently active idents) and a tensor network's `stored/16`. The advantage
over the dense baseline is exactly

```
2^{k − max_block}.
```

### 8.3 Compute

`clifft`'s dense baseline applies every active gate to a `2^k` vector (all FLOP);
near-Clifford does the Clifford part in **bit-ops** and only the genuine magic in
FLOP. Empirically the FLOP it does spend is **dominated by the factoring `norm`
scan**, not the state update — the factoring is the price paid to keep the blocks
small (see `reports/per_step_flops/FLOPS_TABLE.*`).

### 8.4 The deciding factor: where the magic actually is

* **Coherent-error families (win, large).** The coherent rotations are *equatorial*
  (`Z`-type in the data basis), which (§4.2) **never promote** through a `Z`-basis
  syndrome measurement; the syndrome state's huge entanglement is pure *stabilizer*
  structure absorbed by the tableau for free. So `coherent_d5_r1 → max_block 0`,
  `coherent_d5_r5 → 13` transient / 12 resident (vs dense `k = 24`). The win is **stabilizer-free**, and it
  is simultaneously a memory win and a compute win (small blocks ⇒ few FLOP, and the
  rest is polynomial bit-ops).

* **All-magic families (factorable → win/parity; irreducible → limit, not a win).**
  `distillation` / `cultivation` inject genuine `T`/magic states — most active qubits
  are real magic, little for the Clifford frame to absorb. They used to *lose*
  (`max_block ≈ k` plus a pure-overhead factoring scan). Two fixes (§11) close most of
  that gap: touched-qubit factoring cuts the factoring FLOP, and the `_purge_redundant`
  `W_M` peel removes the dof each magic measurement consumes (so measured-out qubits no
  longer sit resident). Peak `max_block` drops `distillation` 5→3, `cultivation_d3`
  6→4, `cultivation_d5` 14→11, taking `distillation` from a memory loss (dense/NC 0.9×)
  to a **win** (2.2×) and `cultivation_d3` to parity. `cultivation_d5` improves
  0.1×→0.5× but **stays a 2× loss vs `clifft`** — its transient `max_block 2^11`
  exceeds `clifft`'s `2^10`, and even the settled resident `2^10` only *ties*; the
  magic is irreducible, so near-Clifford does **not** outperform `clifft` here (an
  honest limit case, not a win). The residual cost on these is the genuine non-Clifford
  magic, which no representation can remove.

  > **Be honest about the baseline (measured).** "dense" above means the **active-state**
  > baseline `2^k` (`k = clifft peak_rank`), *not* the full `2^N` statevector. Against
  > `2^N` every circuit wins astronomically (the physical ancillas are stabilizer and
  > absorbed: `distillation` `2^85/2^3`, `cultivation_d5` `2^42/2^11`). Against the
  > meaningful `2^k`/`clifft` baseline the all-magic picture is mixed and partly
  > negative — the only remaining lever is block factoring, which helps only if the
  > magic actually factors:
  >
  > | circuit | `N` | `k=peak_rank` | `max_block` | vs `2^N` | **vs `2^k` (clifft)** |
  > |---|--:|--:|--:|--:|--:|
  > | distillation | 85 | 5 | 3 | ~10²⁴× | **4× (win — magic factors)** |
  > | cultivation_d3 | 15 | 4 | 4 | 2050× | **1× (parity — irreducible)** |
  > | cultivation_d5 | 42 | 10 | 11 | 2×10⁹× | **0.5× (LOSS — irreducible + 1-qubit transient over-materialisation)** |
  > | coherent_d3_r3 | 26 | 8 | 5 | 2×10⁶× | 8× |
  > | coherent_d5_r5 | 64 | 24 | 13 | 2×10¹⁵× | **2050×** |
  >
  > So near-Clifford is **weakest at the all-magic limit** (where it converges to dense:
  > `cultivation_d5` is genuinely a 2× *loss* vs `clifft` — transient `2^11` vs `2^10`,
  > and even the resident `2^10` only ties; the resident drop `11→10` is recovery vs the
  > *eager* block backend, **not** vs `clifft`) and **strongest on stabilizer-rich
  > coherent circuits** (`d5_r5` 2050× vs `clifft`). Stating the all-magic loss plainly
  > makes the stabilizer-rich win credible.

* **Pure-Clifford + Pauli noise (`surface_d7_r7`).** Compiles to **zero** active
  idents — entirely Pauli-frame — so all of memory, active-state, and FLOP are ~0.

### 8.5 Where the lazy gain comes from: never-flushed (A) vs bounded core (B)

A reviewer's sharpest attack on lazy deferral: *"did you just put non-Clifford
rotations that no recorded measurement ever sees, and skip computing them?"* That
would be **Condition A** (a rotation whose physical generator commutes with every
subsequent measured Pauli → never flushed → free) — exact but trivial. The
defensible source is **Condition B** (rotations *are* flushed/observable, but each
measurement's anticommutation-connected core stays small). This implementation
performs **no** pending fusion/cancellation, so A and B are the *only* two sources.
We measured the split directly (instrument `apply_rotation` = total, `_flush_core`
= flushed core, leftover `pending` at shot end = never-flushed), block backend,
per shot:

| circuit | total rot | never (A) | flushed (B) | **never %** | max core (rot / qubits) | avg core qb | `Σ 2^core` |
|---|--:|--:|--:|:--:|--:|--:|--:|
| coherent_d3_r1 | 25 | 25 | 0 | **100 %** | 0 / 0 | 0 | 0 |
| coherent_d5_r1 | 82 | 82 | 0 | **100 %** | 0 / 0 | 0 | 0 |
| coherent_d3_r3 | 97 | 16 | 81 | **16.5 %** | 14 / 5 | 4.6 | 216 |
| coherent_d5_r5 | 546 | 47 | 499 | **8.6 %** | 16 / 13 | 12.1 | 3.3×10⁵ |
| distillation | 10 | 0 | 10 | **0 %** | 6 / 3 | 2.7 | 20 |
| cultivation_d3 | 29 | 0 | 29 | **0 %** | 14 / 4 | 3.4 | 60 |

**Reading.**
* **Boundary-free single-round (`*_r1`): pure Condition A** (100 % never-flushed,
  `max_block 0`). The equatorial `R_Z` on each `|+>`-born data qubit commutes with
  its `Z`-basis read for the whole circuit. This is the trivial/expected regime — it
  is honest (`|M|=0`, free) but **must not be the memory headline**. (NB it is still a
  real win even over `clifft`, which *expands* these legs: `coherent_d5_r1`
  `peak_rank 13` vs our `0` — but it is the shallow kind.)
* **Multi-round coherent (`d3_r3`, `d5_r5`): Condition B dominates** — 83 %/91 % of
  rotations are *flushed* (observable), only 8–17 % never-flushed. The gain is that
  each measurement's core stays bounded (`max_block` 5 / 13) and **does not grow to
  the total rotation count** (97 / 546): re-measuring each round severs the
  dependency graph, so `Σ 2^core` (216 / 3.3×10⁵) stays far below the monolithic
  `2^k` an eager backend would pay. This is the direct evidence for the strong claim:
  *non-Clifford effects couple to recorded measurements only in small local cores.*
* **All-magic (`distillation`, `cultivation_d3`): pure Condition B** (0 % never-flushed)
  — every rotation observable, materialised only in small cores (3 / 4).

So the headline efficiency is **Condition B**, not trivial skip; the `*_r1` family is
the only pure-A case and is labelled as the free/equatorial baseline. Three-way
working-set ladder (eager near-Clifford |M| → `clifft` `peak_rank` → our `max_block`):
`d3_r3` 16→8→5, `distillation` 5→5→3, `cultivation_d3` 6→4→4, `d5_r5` (eager
overflows)→24→13. The `clifft`→ours step is pure factoring (B); the eager→`clifft`
step is what `clifft` already recovers on its own.

---

## 9. Correctness and verification

All claims are phase-exact and validated, not asymptotic hand-waving:

* `scripts/verify_simulator.py` — core vs dense statevector: **fidelity 1.0**;
  measurement TVD < 0.006; the ZXZ single-qubit decomposition self-consistent to
  `~1e-16`.
* `scripts/verify_backend.py` — **end-to-end vs `clifft.sample`** on the real QEC
  circuits: per-measurement marginal agreement + peak live `|M|`.
* `scripts/verify_block.py` — the block register vs dense (fidelity 1.0) and vs
  `clifft`; includes `purge_case`, which checks the post-measurement statevector
  still equals the exact dense `Z`-projection after the `_purge_redundant` `W_M` peel
  (the measured-magic purge is state-exact, §11).
* `scripts/diag_peak_block.py` — exact finest-tensor-factorization sweep proving the
  peak block is irreducible (the magic is real, not a missed factorization).

```
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  /home/jung/clifft_env/bin/python -m nearclifford_backend.scripts.verify_simulator
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  /home/jung/clifft_env/bin/python -m nearclifford_backend.scripts.verify_block
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  /home/jung/clifft_env/bin/python -m nearclifford_backend.scripts.diag_peak_block coherent_d5_r5 4
```

**Verified live resource (block backend, vs clifft ground truth).** The MAIN figure is
the intra-step **transient** peak `max_block` (the memory high-water mark, §7.4); the
**resident** value in parentheses is the settled step-boundary block (it under-reports
the true peak and is the secondary metric):

| circuit | peak `max_block` (transient / resident) | dense `k` | note |
|---|---:|---:|---|
| coherent_d5_r1 (boundary-free) | **0** / 0 | 13 | pure stabilizer in Z basis — free |
| coherent_d3_r3 (multi-round) | **5** / 4 | 8 | matches clifft within sampling error (see below) |
| coherent_d5_r5 (multi-round) | **13** (irreducible) / 12 | 24 | transient `2^13`=128 KB (resident `2^12`=64 KB) vs TTN `χ=2^11`≈134 MB |
| distillation | **3** / 3 | 5 | all-magic, PASS (purge: was 5) |
| cultivation_d3 | **4** / 3 | 4 | all-magic, PASS (purge: was 6) |
| cultivation_d5 (all-magic limit) | **11** / 10 | 10 | **limit case — does NOT beat `clifft`**: transient `11 > k=10` (2× loss); resident `10` only ties (parity). The `11→10` drop is vs *eager*, not `clifft`. |

(Most circuits have transient = resident; the gap appears only where a measurement's
core-flush momentarily forms a block one qubit larger than the post-collapse residue —
`coherent_d3_r3` 5/4 and `coherent_d5_r5` 13/12.)

> The `distillation`/`cultivation` peaks above are **post-purge** (§11): the
> `_purge_redundant` `W_M` peel removes the dof each magic measurement consumes, so
> the all-magic blocks no longer carry measured-out residue — `distillation` 5→3,
> `cultivation_d3` 6→4, `cultivation_d5` 14→11 (peak `max_block`), turning
> `distillation` from a memory *loss* into a win (dense/NC 0.9×→2.2×) and
> `cultivation_d3` to parity. `cultivation_d5` improves 0.1×→0.5× but **remains a 2×
> loss** vs `clifft` (transient `2^11` > `2^10`; resident `2^10` only ties) — an honest
> limit case, not a win — with the sampled distribution unchanged.

> **Multi-round coherent (`coherent_d3_r3`): no systematic bias — matches `clifft`
> within sampling error.** An earlier note here claimed a "persistent ~1.5 % real
> systematic bias" on the per-measurement marginals. That claim was a **statistical
> artifact** and has been **retracted** after a rigorous re-test:
> - **Multi-seed spread (the decisive test).** Six independent seeds × 5 k shots:
>   `mean(NC−clifft)` over the previously-flagged positions = **−0.0007**, seed-to-seed
>   **std = 0.0042** — i.e. the per-seed mean fluctuates ±0.005 around zero. A stable
>   1.5 % bias would show `|mean| ≫ std` with a fixed sign; instead `|mean| ≪ std`.
>   The "NC 0.492 vs clifft 0.507, same sign" pattern was a **single-seed correlated
>   fluctuation** (the 33 syndrome marginals are correlated within a shot-set, so a
>   whole vector shifts coherently by ~0.015 per seed); the original "does not shrink
>   with N" reading came from a `max`-over-33-positions statistic (multiple-comparison
>   inflation) on one seed.
> - **Global high-N test.** NC 30 k vs `clifft` 120 k, **all 33** measurements:
>   `mean(NC−cl) = −0.0003`, `max|NC−cl| = 0.0068` (not the claimed 0.0149).
> - **Cross-backend control.** On the same noisy circuit the validated **TTN** backend
>   (which shares the `frame_layer` noise sampler and `core` U2/U4 de-fusion with NC) is
>   *equally* close to `clifft`: `mean|TTN−cl| = 0.0027` (8 k shots). NC at 30 k sits at
>   `mean(NC−cl) = −0.0003`, the same order. Both agree with `clifft` to within sampling
>   error — the bulk residual is not NC-specific glue.
>
> **Residual (minor, real, ~15× smaller than the retracted claim).** One ancilla's
> **rare** syndrome (`p ≈ 0.004`, measured in all three rounds → positions 4/12/20)
> shows NC reading **~+0.001 high** (≈25 % *relative*, NC `0.0046` vs clifft `0.0038`),
> reproducible across seeds (pooled `z ≈ 3–4`). This is a tiny rare-event effect
> (plausibly a numerical threshold such as `_compress_magic`'s `1e-10`, or rare-branch
> sampling) — *not* the structured multi-round bias previously described, and it leaves
> the bulk marginals exact to within sampling error. The §11 fixes (touched-qubit
> factoring, `W_M` purge) are bit-identical to the pre-purge trajectory and unrelated.
>
> **Validation note for d5_r5.** A tight end-to-end TVD is still not attainable because
> `clifft` itself is sample-starved there (~20 shots / 120 s) — so d5_r5 correctness
> rests on component checks (statevector fidelity 1.0), the all-magic / boundary-free
> circuits that PASS vs `clifft` (`cultivation_d3`, `distillation`, `coherent_d5_r1`),
> and the finest-factorization confirming the block representation is exact. The
> multi-round marginals are no longer flagged as biased.

---

## 10. Files

| file | what |
|---|---|
| `simulator.py` | `NearClifford` — tableau + dense magic register + pullback + both measurement paths. The verified core. |
| `lazy.py` | `LazyNearClifford` — deferral of rotations as physical-frame pending Paulis; anticommutation-core flush. |
| `block_magic.py` | `MagicRegister` / `BlockLazyNearClifford` — the tensor-product-of-blocks magic register and the factoring map. FLOP counters (`flop_mm`, `flop_norm`) for the compute study. |
| `backend.py` | `NearCliffordBackend` — the full clifft-bytecode backend: frame / noise / dormant / readout handling (shared `ttn_backend.frame_layer` helpers) + active state via the simulator. `block=True` selects the block register. ZXZ + U2/U4 de-fusion. |
| `scripts/verify_*.py`, `scripts/diag_peak_block.py` | verification + irreducibility diagnostics (§9). |

Compile inputs with the **default (fused)** pass manager (`clifft.compile(src)`) —
the shared noise/frame helpers are tuned for it; `bytecode_passes=None` renumbers the
noise-site pool and is *not* a valid input here.

---

## 11. Limitations — two formerly-open ones now addressed

* **Block factoring FLOP — cut by touched-qubit factoring (`factor(only=…)`).** The
  `norm`/`vdot` scan used to re-probe *every* qubit of *every* block after *every*
  operation, and dominated the backend's FLOP. But a rotation `exp(-iθP_S/2)` is a
  **local unitary on its support** `S` (`P_S = P'_S ⊗ I_rest`), so it cannot change
  the factorability of any qubit outside `S` — only `S` qubits can newly factor. So
  after a rotation we scan only `S`; only a measurement (a non-unitary projection,
  which *can* disentangle outside its support) still does a full scan. Bit-identical
  trajectory, same peaks; measured factoring FLOP **−25 % to −84 %**
  (`coherent_d5_r5` `flop_norm` 12.5 G → 2.0 G). All-magic circuits are still net more
  expensive than a dense sim, but by far less.
* **Measured-out magic — now purged (`_purge_redundant`, the `W_M` peel).** A
  magic-path projection on `P'` leaves the touched block in a `±1` eigenspace of `P'`
  — one qubit is redundant (the measurement consumed a dof). We reduce `P'` to a
  single-qubit `Z_r` with a block-local Clifford `W` (turn each support qubit's Pauli
  to `Z`, CNOT-collapse onto `r`), apply `W` to the block vector **and fold `W†` into
  the frame** (`U_C ← U_C W†`, an exact identity insertion via the right-multiply
  tableau primitives `right_h/right_s/right_cx`), so `r` becomes a product
  `Z`-eigenstate that `factor()` peels. Updating **both** the frame *and* register
  membership keeps the stabilizer/magic measurement-path decision consistent — so the
  trajectory is preserved *exactly* (verified bit-identical to the pre-purge code over
  200 random seeds; post-measurement statevector fidelity 1.0; `cultivation_d3`,
  `distillation`, `coherent_d5_r1` PASS vs `clifft`). It removes the dead-resident
  inflation: `cultivation_d3`'s 1-live+3-dead block → 1, and peak `max_block` drops
  `distillation` 5→3, `cultivation_d3` 6→4, `cultivation_d5` 14→**11 transient / 10
  resident** (8×–16× less magic memory/FLOP on the all-magic families) with no change
  to the sampled distribution. (Even at 11/10, `cultivation_d5` does not beat `clifft`'s
  `k=10` — see §8.4: transient is a 2× loss, resident only parity.)
* **Frame reduction at demotion (`decouple_demote`, DEFAULT ON) — no settled
  per-measurement memory loss.** Per-measurement, the near-Clifford state could exceed
  `clifft`'s active rank: a measured-out (demoted) magic qubit can stay entangled in a
  live block because the physical `Z_q` pulls back to a register Pauli **not supported on
  `q`** (traced: index 4's measurement has pullback support `{2,3}`), so the measurement's
  `_purge_redundant` consumes a *different* dof and `q` lingers as dead residue that later
  flushes re-promote. The frame reduction removes it in two parts, both **exact identity
  insertions** (state-exact ⇒ distribution-exact; no rng consumed):
  - **`q∈supp` (the easy case):** make the W-peel collapse onto the **demoted index
    itself** (`r=q`) instead of `supp[0]`, so the just-demoted qubit is the one peeled.
  - **`q∉supp` (the `cultivation_d5` case):** after the measurement, scan live blocks for
    dead qubits and peel any that are **parity-slaved** — a measured-out qubit usually has
    a stabiliser `Z_q ⊗ Z^{mz}` (its value = a parity `mz` of the rest; Schmidt rank 2,
    *stabiliser*-entangled not magic). Find `mz` by GF(2) elimination (`_gf2_solve`),
    numerically verify it stabilises the block, then reduce it to `Z_q` (CNOTs) and peel
    `q` via the same `_purge_redundant` machinery (`_reduce_dead` / `_find_z_stabilizer`).
    This is the near-Clifford analogue of `clifft`'s active-rank reduction.

  The cost is that folding the different `W†` changes later magic-vs-stabilizer branch
  decisions, so it is **not bit-identical** to the legacy path — it is a *different but
  equally correct sampler*. Default is ON; pass `decouple_demote=False` for the legacy
  **bit-identical** path (kept for the bit-identical regression check, §9).

  **Measured (gap + marginal harness):** with it on, **every circuit has `nc ≤ clifft` at
  the settled (resident) level at every measurement** — settled loss count → 0 everywhere,
  including `cultivation_d5` (`5→0`; its peak `max_block` drops `11→10` = `clifft` parity,
  exactly as predicted — *parity, not a win*, since the magic is irreducible). Transient:
  `distillation` (`−2→0`, `2→0`) and `cultivation_d3` (`−1→0`, `1→0`) fully fixed; the only
  residual is **one** transient `+1` spike left on `cultivation_d5` (`11→1` loss
  measurements). Distribution unchanged within sampling (`max|Δmarginal| ≤ 0.009`);
  `verify_block` ALL PASS (fidelity 1.0); runtime unchanged (the GF(2) peel is cheap).

  *Why the last spike survives (and what would remove it).* Traced: it is **not**
  intra-flush — it is the block **carried between two consecutive measurements**. `clifft`
  drops its active rank `k` at measurement *t*; the dead qubit it sheds is peeled from the
  near-Clifford block only when `_reduce_dead` next finds its decoupling stabiliser, which
  here is **cross-block** (it lives in the Clifford frame, not within the single dense
  block), so the block-local `Z_q⊗Z^{mz}` / `X_q⊗G` scan cannot reach it and the block lags
  `k` by one measurement. Adding X-type detection to the dense-block scan did *not* help
  (confirming the residue is genuinely cross-block). Removing this last spike needs the
  full **tableau-level** reduction (use the frame's stabiliser generators, not just the
  dense block) — `clifft`'s active reduction proper. Deferred; the **settled** no-loss
  guarantee already holds everywhere, and this is one momentary `+1` on the all-magic limit.
* **`coherent_d7_r7`** is not finalised (too slow to gather statistics), though the
  representation and code path are the same verified ones.
