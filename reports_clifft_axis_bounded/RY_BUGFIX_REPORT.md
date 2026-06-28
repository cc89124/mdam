# R_Y off-axis coherent-noise bias — root cause, fix, and exact validation

Backend: `nearclifford_backend/clifft_axis/` (`clifft_axis_bounded`).  All fixes scoped to
`clifft_axis/engine.py`; `lazy.py`, `virtual_axis/` (fused-VA), and `block_magic` are untouched.

---

## 1. Root cause — TWO bugs

Pending non-Clifford rotations are stored as `[x, z, p, theta, uid]` with the Pauli convention
`P(x,z,p) = i^p · X^x Z^z`.  `+Y = (1,1,1)` (i.e. `i·XZ`).

**BUG #1 — flush drops the Pauli phase `i^p`.**
`LazyNearClifford._do_flush` (lazy.py:242) calls `self._flush_one(x, z, theta)`, and
`LazyNearClifford._flush_one(self, x, z, theta)` (lazy.py:145) has **no phase argument** — the
pending entry's `p` is silently discarded.  For R_Y the generator phase is `p=1` (`+Y = i·XZ`),
so flushing `Y` actually applied `−iY`'s partner `XZ` with the wrong sign: the `i^p` factor was
lost.  (R_Z/R_X rotations carry `p=0`, so the drop is a no-op for them — see §3.)

**BUG #2 — CZ conjugates the pending Pauli TWICE.**
The base `NearClifford.cz` (simulator.py:110) is
```python
def cz(self, a, b):
    self.h(b); self.cx(a, b); self.h(b)        # CZ = H_b · CX_ab · H_b
```
Because `self.h`/`self.cx` are the lazy overrides (lazy.py:111,121) that conjugate every pending
Pauli, this base `cz` already conjugates pending **exactly once** (correctly).  But
`LazyNearClifford.cz` (lazy.py:126-134) calls `super().cz(a, b)` **and then** repeats the
conjugation by hand:
```python
def cz(self, a, b):
    super().cz(a, b)                                   # <- already conjugates pending once
    for u, r in self.pending.items():
        P = _conj_h(P, b); P = _conj_cx(P, a, b); P = _conj_h(P, b)   # <- conjugates AGAIN
        ...
```
Net effect on a pending Pauli `P`: `C·C·P·C†·C†` instead of `C·P·C†`.  For a **diagonal** pending
(`Z`, from R_Z), `C P C† = P` so the squared conjugation is the identity — invisible.  For an
**off-diagonal** pending (`X` from R_X, `Y` from R_Y, or any pending rotated off-diagonal by a
preceding `H`/`√X`/`√Y`), `C·C·P·C†·C† ≠ C·P·C†` → the deferred rotation is corrupted.

BUG #2 is the dominant contributor to the ≈0.04 R_Y bias; BUG #1 corrected ~1 measurement.

---

## 2. First divergence (exact, not guessed)

Minimal reproduction, bounded engine vs an exact dense statevector (forced outcomes, Born `p0`
at the measurement):

| circuit | result |
|---|---|
| `RY(θ) 0; RY(θ) 1; CZ 0 1; RY(θ) 0; RY(θ) 1; M 0 1` | **diverges** |
| same with **CX** instead of CZ | exact (`<1e-13`) |
| `RY` single-qubit, no entangler | exact |

The divergence appears the first time a pending off-diagonal (Y) rotation is carried **through a
CZ** — i.e. the syndrome-extraction CZ/CX layer of the surface code, surfacing at the first data
measurement.  The CX-vs-CZ split localises it to CZ handling of pending (BUG #2); the residual
after fixing CZ localises the remaining single-measurement error to the flush phase (BUG #1).
A 6-qubit multi-CZ-depth engine-vs-dense test reproduces and then (post-fix) clears it
(`worst |Δ| = 3.0e-14`).

---

## 3. Why R_Z / R_X are correct, and R_Y is the only circuit the CZ fix changes

The decisive evidence is a **cross-entropy** check: sample each backend (NEW = both fixes, OLD =
BUG #2 present with BUG #1 *already* fixed, toggling only `cz`), score every realized trajectory
with `clifft`'s exact `record_probabilities`, and report the mean `log P_clifft`.  The true
sampler maximises this (= −entropy); a wrong joint produces lower-probability trajectories.

| d3_r1 | clifft-ref (optimum) | NEW | OLD (BUG #2) | verdict |
|---|--:|--:|--:|---|
| **R_X** | −6.2014 | −6.2239 | **−6.2239** | OLD = NEW → CZ fix is a **no-op**; R_X was always correct |
| **R_Y** | −5.8571 | −5.8737 | **−6.0880** | OLD ≫ worse → BUG #2 genuinely corrupts R_Y; NEW restores it |

- **R_Z (diagonal pending `Z`): bit-identical no-op.**  CZ commutes with Z so BUG #2's squared
  conjugation is the identity, and R_Z flushes carry `p=0` so BUG #1 never fires.  A/B record
  comparison (new `engine.cz` vs old double-conjugating `lazy.cz`) over `coherent_d3_r1`,
  `coherent_d3_r3`, `cultivation_d3`, `surface_d7_r7` × 5 seeds → **all bit-identical**, `max_M`
  unchanged.

- **R_X (off-diagonal pending `X`): correct, distributionally a no-op.**  The cross-entropy is
  *identical* for OLD and NEW (−6.2239), and `max_M` is unchanged.  Although `CZ X_a CZ† = X_a Z_b`
  and the double-conjugation drops the `Z_b`, that dropped factor is unobservable in this surface-
  code family (it lands on a stabiliser-trivial location), so R_X's distribution is unchanged.
  R_X was **not** buggy — an earlier draft's "R_X rank 10→11" was a mis-read of a mislabeled
  `max_M` print; the rank is unchanged (transient 11 / resident 10).

- **distillation (T-magic): distribution unchanged.**  Its `√Y`/`√X` rotate the T-pending
  off-diagonal, so the A/B records **differ** per seed (RNG-path), but both old and new marginals
  match clifft (`max|Δ| 0.0154 ≈ null 0.0145`) and `max_M` is 4/4.  Like R_X, the CZ fix changes
  the per-shot bookkeeping but not the distribution.  (A full exact joint check is blocked by its
  40 detectors, which `clifft.record_probabilities` rejects.)

Take-away: among the tested circuits the CZ fix (BUG #2) is a genuine correction **only for R_Y**;
it is distributionally a no-op for R_Z (bit-identical), R_X (cross-entropy identical), and
distillation (marginals identical).  R_Y is singled out because its pending `Y = i·XZ` carries
*both* the phase (BUG #1, `p=1`) and an off-diagonal generator whose dropped `Z_b` is observable
in the Z-basis syndrome — the two defects compound only there.

---

## 4. The fix (scoped to `clifft_axis/engine.py`)

| # | location | before | after |
|---|---|---|---|
| 1 | `_flush_one` (engine.py:249) | `_flush_one(self, x, z, theta)` (no phase) | `_flush_one(self, x, z, theta, phase=0)`; `pp = (pp + phase) & 3` before applying. `phase==0` ⇒ byte-identical to before |
| 1 | `_do_flush` (engine.py:274) override | inherited `_flush_one(x, z, theta)` drops `p` | forwards it: `self._flush_one(x, z, theta, p)` |
| 1 | `statevector` (engine.py:294) override | inherited path dropped pending `p` | flushes pending with phase before delegating to `NearClifford.statevector` |
| 2 | `cz` (engine.py:301) override | inherited `lazy.cz` = `super().cz` **+** manual `_conj` loop (double) | `self.h(b); self.cx(a, b); self.h(b)` — conjugate pending **exactly once** (= base CZ); no super, no manual loop |

`lazy.py` / `virtual_axis/` (fused-VA) / `block_magic` were **not** modified.

---

## 5. Exact tests (deterministic Born, no sampling)

Three independent exact objects in the stim record-bit convention `P(record_i=0 | exec-prefix)`:
dense 2^17 statevector (written here), the full bounded backend, and
`clifft.record_probabilities`.  See `EXACT_RY_VALIDATION.md`.

| test | metric | result | bound |
|---|---|--:|--:|
| 1-qubit R_Y | dense=backend=clifft | 4.4e-16 | <1e-12 |
| 2-qubit R_Y+CX (+X-fault) | dense=backend=clifft | 4.4e-16 | <1e-12 |
| 3-qubit deep H/CX/multi-R_Y | dense=backend=clifft | 4.4e-16 | <1e-12 |
| 6-qubit multi-CZ, engine vs dense | per-measurement Born | 3.0e-14 | <1e-12 |

---

## 6. QEC validation (the real circuit path)

Per FIXED X_ERROR fault pattern (frozen to explicit X) over no-fault + every single data/ancilla
fault (the X-before-R_Y angle-sign stress) + multi + the all-faults extreme, several seeds:

| circuit | metric | result | bound |
|---|---|--:|--:|
| **d3_r1** | per-measurement \|dense − backend\| | **2.55e-15** | <1e-10 |
| **d3_r1** | realized-trajectory JOINT, dense=backend=clifft | 6.4e-14 | <1e-8 |
| **d3_r3** | realized-trajectory JOINT, dense=backend | 1.4e-13 | <1e-9 |
| weighted marginal | per-branch backend=dense | 9.9e-15 | <1e-10 |

The earlier "bias 0.0051 ≈ null 0.0042" was a 60 000-shot **resampling** frequency difference
(case B), not an exact marginal.  The true exact deterministic Born error on the full d3 circuit
is ≤ 2.55e-15 — the systematic bias is gone.  ("32-branch": d3_r1/r3 have 42/74 X_ERROR
instances, not 5; 2^42 enumeration is infeasible and unnecessary — per-branch exactness bounds any
noise-weighted marginal by `max_e|Δ_e| < 1e-10` via the triangle inequality.  Clifft cannot be
used for d3_r3 — ancilla are reset and reused, and clifft rejects resets — so the dense oracle,
validated transitively on d3_r1, is the exact oracle there.)

---

## 7. Regression

- **R_Z canonical — BIT-IDENTICAL** (new `engine.cz` vs old `lazy.cz`, 5 seeds):
  `coherent_d3_r1`, `coherent_d3_r3`, `cultivation_d3`, `surface_d7_r7` → identical records, same
  `max_M`.
- **R_X d3 — no change** (cross-entropy OLD = NEW = −6.2239 ≈ clifft −6.2014; `max_M` unchanged at
  transient 11 / resident 10).  R_X was correct before and after.
- **distillation — distribution unchanged** (marginals match clifft both ways, `max_M` 4/4; only
  per-seed RNG-path records differ).
- **strict-memory (r = k)** — `max_M` unchanged on every R_Z / R_X circuit ⇒ memory bound preserved
  (no memory regression).
- **fused-VA / lazy / block** — untouched.

**Flagged (out of scope, NOT fixed per the constraint):** the redundant double-conjugation lives
in `lazy.cz`, which the `lazy` and `block_magic` backends inherit.  On the tested circuits it is
distributionally a no-op except for R_Y, but it is still an incorrect (squared) conjugation that
could surface on other off-diagonal-pending circuits run on those backends.  Only the `clifft_axis`
engine was corrected (via override); a separate review of `lazy.cz` is warranted.

---

## 8. Post-fix rank / memory (regenerated with the fixed engine)

Buggy pre-fix R_Y traces (`max_M` 10 / 14) were artifacts of BUG #2 discarding magic d.o.f. and
are superseded.  Corrected (`bounded_coherent_{rx,ry}_d3_r{1,3}_per_step.csv`,
`DETAILED_TABLE_RXRY`, plots, `BOUNDED_RXRY_SUMMARY.md`):

| circuit | noise | Clifft k | transient | resident | transient saving |
|---|---|--:|--:|--:|--:|
| coherent_rx_d3_r1 | R_X | 14 | 11 | 10 | 2^3 |
| coherent_rx_d3_r3 | R_X | 14 | 12 | 11 | 2^2 |
| coherent_ry_d3_r1 | R_Y | 16 | 16 | 15 | 1× (parity) |
| coherent_ry_d3_r3 | R_Y | 16 | 16 | 15 | 1× (parity) |

**R_Y has no peak-transient saving:** the Y over-rotation keeps all magic d.o.f. live
(transient = 2^k = Clifft's active rank); the genuine bounded gain is the resident drop (2×) and
the time-integrated active-state / memory (~9–10×, see `DETAILED_TABLE_RXRY.md`).  d5 off-axis
remains INFEASIBLE (> 2^26).

Scripts: `reports_clifft_axis_bounded_rxry/_exact_oracle_lib.py`, `_exact_calib.py`,
`_exact_full.py {rx,ry} {r1,r3}`, `_exact_r3_joint.py`, `_exact_weighted.py`, `_complex_engine.py`,
`_regress_rz_bitident.py`.
