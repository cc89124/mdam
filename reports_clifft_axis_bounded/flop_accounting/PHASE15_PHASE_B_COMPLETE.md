# Phase B COMPLETE — clean measurement consume, p0 + post-state machine-exact

**The Phase B gate is closed.** The reduced data plane consumes clifft's active measurement
opcodes directly, with per-measurement Born p0 **and** post-collapse physical state matching the
dense physical oracle at **machine precision**, and bit-identical records/peak-rank/p0 vs the
authoritative path on cultivation_d3 **and** cultivation_d5. Default-off probe; a05843e / tag /
butterfly / localizer / Policy-3 default-off all preserved. Scripts: `/tmp/phaseB0_oracle.py`,
`/tmp/phaseB3_regression.py`, `/tmp/phaseB_parity_unit.py`.

## 0. The operator correction (root cause of the earlier 87-mismatch)

cultivation has **no `OP_MEAS_ACTIVE_DIAGONAL`** — every active measurement is
**`OP_MEAS_ACTIVE_INTERFERE`** (X-basis, dispatch = `nc.h(q)` then `measure_z(q)`, recorded with
`frame.zb`). The earlier clean `mz_clean` measured the array bit in the **Z-basis directly,
omitting the interference H-fold** → it measured Z instead of X (the wrong operator) → 87
mismatches. clifft's own `meas_interfere` (c=8) **is** the H-fold + Born, so the Hadamard here is
clifft's interference kernel, **not** a hidden relocalisation. The user's `x≠0` fail-fast applies to
DIAGONAL; INTERFERE is *meant* to interfere.

## 1. B0/B1/B2 dense oracle — cultivation_d3 (n=6), machine precision

For each active measurement, on the shared pre-measurement physical state (robust NaN-free dense
materialisation), **both** branches b=0,1 forced independently, compared to the dense projector
`(I ± X_q)/2`:

| check | worst over 20 seeds | covers |
|---|--:|---|
| **B0 operator+Born** `p0_array == p0_dense` | **4.4e-16** | det (p0∈{0,1}) AND random (p0=0.5) |
| **B1 projection** (keep axis, original basis) | **1.1e-16** | both branches |
| **B2 drop** (frame X-fold on keep=1, remaining-qubit state) | **2.2e-16** | both branches |

The `keep=1` drop folds X into the frame (`Zc[q]` negation) — this **is** the `Q_{t+1}` embedding
update; the remaining-qubit physical state is exact. (Earlier "B2 = 0.707" was an oracle bug:
slicing bit_q=0 after the frame put q in |1⟩; fixed by factoring q out at its actual post-state.)

## 2. measure_z is already the clean diagonal consumer (verified)

Budget-tag probe INSIDE `measure_z` on the reduced INTERFERE path (d3 & d5):
**only `sqnorm` / `normalize` / `drop` / `post-reduce`** — **zero** butterfly / localize / relocalize
/ pullback FP charges. `_pullback(Z_q)` and `_localize_to_Z` ARE called but are **FP-free**: the
H-fold Z-localises the array bit, so `xp=0` (no relocalising H) and the only work is the **FP-free
CNOT parity-fold** (`purge:cnot=0`). The 3.42× data-plane re-derivation was entirely the *rotation*
butterfly (fixed by eager diagonal `rrot`); the measurement was already clean once H-folded.

## 3. Explicit clean consumer + x≠0 fail-fast — bit-identical regression

An explicit consumer = B0 `_pullback` (FP-free frame pullback → reduced Z-parity) + **`xp≠0`
fail-fast** (refuses to relocalise; raises → Phase C/D) + B2 FP-free CNOT parity-contraction onto a
pivot + diagonal Born + `_drop_localized`:

| circuit | records_mis | peakrank_mis | p0_mis | FLOP clean | × clifft | parity-folds |
|---|--:|--:|--:|--:|--:|--:|
| cultivation_d3 | **0** | **0** | **0** | 2.18k | 1.21 | 0 |
| cultivation_d5 | **0** | **0** | **0** | 238.48k | **1.12** | 6 |

(60 seeds each.) The **`xp≠0` fail-fast never fired** across 1080+ d5 measurements → no cultivation
measurement needs a relocalising H/butterfly (the architecture's no-re-derivation property holds for
every measurement). 6 d5 measurements genuinely exercise the **multi-bit Z-parity fold** (the user's
B2 affine substitution) — and were exactly the 4-seed p0 divergences a naive single-bit consumer
produced.

## 4. B2 multi-bit parity contraction — independent unit test

`_localize_to_Z` on 40 random magic states with random 2–4 bit Z-parities, vs the dense
`(I ± Z^z)/2` projector: **worst Born error 2.78e-16, worst post-state 1.11e-16**. The contraction
`Z^z → sign·Z_r` (FP-free CNOT folds + exact frame fold) is Born- and state-exact — covering the
case d5 exercises but d3's circuit did not.

## 5. Status

| Phase | claim | evidence |
|---|---|---|
| **A** unitary | state-exact | d3 unitary-prefix statevector 2.2e-16 |
| **B0** operator | M_red = reduced Z-parity, xp=0 always | fail-fast never fires; p0 4.4e-16 |
| **B1** projection | post-projection state exact | d3 dense 1.1e-16, both branches |
| **B2** parity-contract + drop | Born + post-state exact | d3 dense 2.2e-16; 40-case parity unit 2.8e-16 |
| **B3** regression | records/rank/p0 bit-identical, FLOP ≤ authoritative | d3+d5, 60 seeds, 0 mismatch |

## 6. Free-born — MEASURED (the 0.96× projection is RETRACTED)

`/tmp/freeborn.py` (scripts/). Replace the born H-**butterfly** (`_h_axis`, `purge:h` 4/elem) with a
**copy-born** matching clifft's `expand` (write the new MSB block = a copy of the low block; the
1/√2 deferred to the measurement's `tot=s0+s1` renormalisation). Measured, cultivation_d5:

| variant | born FP-FLOP | **total × clifft** | copy traffic | peak words | tracemalloc | wall | bit-ident rec/rank/p0 |
|---|--:|--:|--:|--:|--:|--:|--:|
| current (butterfly) | 16.55k | 1.12× | 0 | 1024 | 8.81M | 76ms | — |
| free_scale (copy + 1/√2) | 8.28k | 1.08× | 4.14k | 1024 | 8.82M | 76ms | **0/0/0** |
| **free_defer (copy only = clifft)** | **0** | **1.04×** | 4.14k | 1024 | 8.82M | 76ms | **0/0/0** |

**The projected 0.96× was wrong on two counts:** (1) it subtracted the *full* 33.1k `purge:h`, but
only **half (16.55k) is the born** — the other half is the **meas-interfere interference H-fold**
(legitimately clifft's `meas_interfere`, not removable); (2) the copy is **not free** — it adds
4.14k-element (66 KB) memory traffic the in-place butterfly did not have. The deferred-norm born is
**0 arithmetic FLOP (= clifft) and bit-identical** (the 1/√2 is recovered by the Born `tot`), but
**peak workspace and wall are UNCHANGED** (the born allocates the same 2^{r+1} either way). So
free-born is real but modest: **1.12× → 1.04× MEASURED**. The residual 1.04× (≈9k FLOP) is the
meas-interfere realisation + `array_cz`/`array_s` coefficient deltas (the rotations already match
clifft *exactly*: `rot:diaghalf` 3·2^r ≡ clifft `array_t` cmul 6·2^{r-1}), a separate accounting.

CAVEAT (deferred-norm): the state is left un-normalised (scaled 2^{#born/2}); fine for sampling
(records, p0 ratios) and bit-identical on cultivation, but absolute-amplitude thresholds
(`1e-20` product test, `1e-24` normalise) could mis-fire on other circuits — to validate per family.

## 7. Phase C/D — ARRAY_CNOT/CZ gauge-vs-numerical, per-opcode PROVEN

`/tmp/phaseCD.py` (scripts/). clifft meter: `array_cnot` = **0 arithmetic** (pure amplitude
PERMUTATION; `processed` = traffic only), `array_cz` = `rcmul` (−1 on |11⟩), `array_swap` = 0.

**Verdict: on the MAGIC register these opcodes are NUMERICAL, not frame-gauge.** A CNOT/CZ between
two magic axes genuinely entangles/phases the dense array — it cannot be a frame relabel (the frame
carries only stabiliser/dormant qubits, routed via `o_cx`/`frame.cnot` = gauge, 0 numerical).

| check | result |
|---|---|
| **UNIT** `_cnot_axes` vs dense CNOT (60 random magic states) | **5.6e-17** (exact permutation, **0 arithmetic**) |
| **UNIT** CZ sub-block −1 vs dense CZ | **0.00e+00** (bit-exact phase) |
| **IN-CIRCUIT** d3 per-opcode `E_{t+1}=O·E_t`, CNOT numeric | n=46, worst **1.1e-16** |
| **IN-CIRCUIT** d3 per-opcode `E_{t+1}=O·E_t`, CZ numeric | n=4, worst **0.0e+00** |
| dormant-routed (gauge) CNOT/CZ in cultivation | **n=0** (all both-magic) |

Per-opcode cost vs clifft (r ≤ k ⟹ the invariant holds termwise):
- `ARRAY_CNOT`: reduced **0 arith** + 2^{r-1} swap traffic  =  clifft `array_cnot` **0 arith**, 2^{k-1} traffic.
- `ARRAY_CZ`: reduced −1 on 2^{r-2} (`rcmul`)  =  clifft `array_cz` `rcmul` on 2^{k-2}.

**COST-HONESTY FLAG:** clifft charges `array_cz` `rcmul=1104` (CONV 2 → 2208 FLOP on d5), but the
reduced BCOEF charges `reduce:cz=0` — the reduced **undercharges CZ**. Correcting `reduce:cz`→`rcmul`
(2/elem on the |11⟩ quarter-block) adds ~2.2k to the d5 reduced total (free_defer 221.9k→224.1k,
1.04×→1.05×). Minor (9 CZs) but must be fixed for an honest cost invariant. `reduce:cnot=0` is
correct (= clifft 0-arith permutation).

## 8. Next

d5_r5 (the genuine r<k quotient regime — cultivation has r=k so no quotient gain there; coherent_d5_r5
should show real r<k + a wall check), then the full cost invariant `Σ c_t^reduced·2^{r_t} ≤
Σ c_t^clifft·2^{k_t}` with the `reduce:cz` correction applied. The U_C-identity array routing remains
a cause-confirmation probe; the final source of truth is the clifft control plane (frame metadata-only).
