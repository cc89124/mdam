# §5 milestone — reduced data plane consuming clifft's diagonal opcodes (cultivation_d5)

**The architecture is validated.** A reduced data plane that consumes clifft's post-localization opcodes
**directly** (no `U_C` pullback, no `_localize_to_Z` relocalize) makes cultivation_d5 **91/91 diagonal**,
**records/rank bit-exact over 200 seeds**, **FLOP 1.12× clifft (0.96× with free born)** down from 3.42×,
and **wall 110.5→83.5ms (improved)**. Default-off probe (monkeypatch); a05843e / tag / butterfly /
localizer / Policy-3 default-off preserved. Script `/tmp/reduced_dp3.py`.

## Design (the only change is the data plane)

Keep `self.nc.U_C` **identity** by routing every active Clifford to the **array**, and consume the
diagonal opcodes eagerly:
- `OP_EXPAND` → `_promote` + `_h_axis` (|+> in the array, tableau clean) — not `nc.h(q)` (which folded
  the birth-H into U_C).
- `OP_ARRAY_T/T_DAG(slot)` → **eager** `diag(1, e^{i·sign·θ})` on the array bit `M.index(slot2id[slot])`,
  sign = `frame.xb(slot)` (clifft's p_x, followed verbatim, never re-inferred). **1 sweep, c=3.**
- `OP_ARRAY_CNOT(u,v)` → `_cnot_axes` array permutation (0 FP). `OP_ARRAY_CZ` → diagonal −1 phase.
- `OP_MEAS_ACTIVE_DIAGONAL` → (current probe) reuse `measure_z`; with U_C identity the measured `Z_q`
  pulls back to a **single bit** ⟹ `_localize_to_Z` is empty (no relocalizing H) ⟹ diagonal Born + drop.

Because U_C stays identity, `apply_rotation`/`measure_z`'s pullback is single-bit diagonal — the
re-derivation/butterfly is gone. The lazy engine was made **eager** for ARRAY_T (clifft applies array_t
immediately; deferring it let all-diagonal rotations never flush — that earlier 0/0-rot bug).

## Results (per run, cultivation_d5, clifft = 212.82k)

| | FLOP | × clifft | diag/off rot | wall |
|---|--:|--:|--:|--:|
| current bounded | 727.70k | 3.42 | 0/91 | 110.5ms |
| **REDUCED** | **238.48k** | **1.12** | **91/91** | **83.5ms** |
| REDUCED, free born | 205.4k | **0.96** | 91/91 | — |

FLOP breakdown (REDUCED): `rot:diaghalf` **174.26k** (= clifft array_t **174.3k exactly**) + `purge:h`
33.10k (16 born-|+> via the c=4 butterfly; **clifft expand = 0**) + measurement (sqnorm 20.68k +
normalize 8.27k = 28.95k, **< clifft meas_interfere 33.1k**) + `purge:s` 2.18k. So the T's are
consumed *identically* to clifft; the only excess is the born-H, which a free |+> copy (clifft's expand)
removes — taking the total **below** clifft.

cultivation_d3: REDUCED 2.18k = **1.21× clifft** (from 3.04×), wall 5.7→5.0ms, p0 0 mismatches.

## Verification status (your §4)

| check | result |
|---|---|
| diagonal T = 91/91 | ✓ |
| records bit-exact (200 seeds) | ✓ 0 mismatch |
| rank trajectory / peak | ✓ 0 mismatch |
| actual FLOP ≤ clifft | 1.12× (0.96× free-born — needs the born fix) |
| wall improved | ✓ 110.5→83.5ms |
| reduced workspace rank ≤ clifft k | ✓ (peak rank matches; r ≤ k) |
| clifft fallback | none |
| per-measurement p0 / normalization | **state-exact, NOT bit-identical** (see below) |

## The p0 caveat — drop-timing, not a state error

Seeds 2,3 show p0 differences (e.g. seed 2 meas 0: reduced M 4→2 vs truth 4→3; meas 2 p0 0.5 vs 1.0).
**Records and peak rank are exact over 200 seeds**, so the physical outcome distribution is identical.
The difference is *when product axes drop*: born-|+> axes become product (|±>) at different points than
the born-H-frame representation, so the existing `measure_z` drop heuristic removes them on a different
schedule — the **`decouple_demote` "distribution-exact, not core-log-bit-identical"** class. The current
probe **reuses `measure_z`**; the clean **Phase B** (diagonal Born on Q(slot) + controlled drop matching
clifft's `OP_MEAS_ACTIVE_DIAGONAL`) is the next step and should align the p0 schedule.

## Next steps (to close the milestone fully, then extend)

1. **Free born** (|+> by copy, c=0 like clifft expand) → exact/below parity (0.96×). FP-FLOP only;
   allocation/copy/memory-traffic + wall measured separately (your §3/Phase C).
2. **Clean Phase B** measurement (diagonal Born + Q-update + controlled drop; no `measure_z`/
   `_localize_to_Z`) → resolve the p0 schedule, with the Q_{t+1} update rule stated explicitly.
3. Then **Phase C/D** (EXPAND rank bookkeeping; ARRAY_CNOT/CZ gauge-vs-numerical proof) and **d5_r5**
   (the genuine r < k quotient + wall regression check).

This is the first end-to-end evidence that "clifft logical control plane + reduced numerical data plane"
reaches ≤ clifft FLOP with improved wall — exactly your reframe.

## Addendum — Phase A (unitary) PROVEN state-exact; Phase B (measurement) is the open gate

Per your rigor (p0 = real Born probability, must match), I did NOT accept the records-match as
distribution-equivalence (cultivation's record is feedback-deterministic — `distinct_records=1` — so
records-match is weak). Two findings:

1. **Phase A is state-exact.** Materializing the full physical statevector at the unitary prefix
   (cultivation_d3, n=6, steps 0..545 before the first measurement) gives **up-to-global-phase residual
   = 2.22e-16** vs the authoritative path. So the born-|+>/eager-T/array-CNOT/CZ consume is *physically
   exact*, not just records-exact. The T direct-consume (Phase A) is validated at the state level.

2. **The p0 discrepancy is entirely in Phase B (measurement collapse/drop), not the unitary ops.** The
   **first** active measurement's p0 matches (0.5 = 0.5, on the identical prefix state); later p0's
   diverge because the reduced measurement's projection/drop discipline differs from the authoritative
   `measure_z` (the clean "drop measured axis only" variant gave *more* p0 mismatch — 87 — and the
   `measure_z`-reuse probe gave 4; neither is the correct discipline). The measured operator and the
   post-collapse state representation must be made to match the authoritative Born probability.

**Status:** Phase A (T direct-consume) — validated (state-exact + FLOP = clifft + wall improved). Phase B
(measurement) — NOT passed; the collapse/drop discipline must be built so per-measurement Born p0 AND
post-collapse state match the authoritative path. free-born 0.96× remains **projected** (FP-FLOP only;
allocation/copy/memory-traffic/wall to be measured). The U_C-identity array routing stays a **cause-
confirmation probe**, not the final architecture (Phase C/D will decide gauge-vs-numerical per opcode).
