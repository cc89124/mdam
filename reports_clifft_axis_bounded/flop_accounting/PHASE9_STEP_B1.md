# Step B1 — Policy-3 persistent split (born-X promote): CORRECT but FLOP-neutral (honest)

**Verdict: the minimal persistent split is records+rank bit-exact and state-exact, but it does NOT win
on FLOP — it is a wash with the butterfly and still 3.4× clifft on cultivation.** The reason is
architectural and was foreseen in Phase 5: a single promote-basis diagonalizes only the rotations of
the *same measurement batch* as the promote; cross-measurement generators fall back to the butterfly.
No overclaim. Default-off flag `clifft_axis_policy3`; `bounded.py`/`engine.py` untouched; the new code is
isolated in `clifft_axis/policy3.py`. Scripts: `phase8_b1_verify.py`, `phase8_b1_flop.py`.

## 1. What was built (the user-approved design, point 1)

`CliftAxisPolicy3NearClifford(CliftAxisBoundedNearClifford)`: at `_promote`, materialize each
X-support axis in the **born-X basis** — `_h_axis(j)` on the fresh |0⟩ bit (= |+⟩ fill) folded by
`right_h(q)` into the frame. This pair is a **physical identity** (H on array · H in frame = I), so the
triggering X_q generator pulls back to **diagonal Z_q**. A rotation that lands diagonal (mx==0) uses the
existing c≈2–3 diagonal half-array (0 butterfly, 0 runtime H); one that does not falls back to the exact
parent butterfly/localizer.

## 2. Correctness — verified

| check | result |
|---|---|
| records bit-exact vs a05843e (9 circuits, all seeds) | **identical** |
| peak rank bit-exact | **identical** |
| per-measurement Born p0 bit-exact | **8/9 identical**; distillation differs (see §3) |
| 300-seed record/rank sweep (distillation, cult_d3/d5, d5_r5) | **0 mismatches** |

## 3. The distillation core-log difference (state-exact, not bit-identical)

distillation seed 7, measurement 4: the truth keeps a magic axis (M 1→0, p0=0.5); Policy-3 has already
dropped it (M 0→0, stabilizer, p0=None). Cause: an axis that is physically |±⟩ (X-eigenstate) is a
**product state in the born-X array**, so Policy-3 disentangles and drops it one measurement earlier.
**Physically exact** (the 300-seed records are identical), but it reclassifies that measurement
(magic→stabilizer), changing the core_log p0 schedule. This is the same "state-exact, NOT bit-identical"
class as `decouple_demote`. So Policy-3 is **records+rank bit-exact; core-log p0 schedule is state-exact**.

## 4. FLOP — the honest result (the win did NOT materialize)

| circuit | clifft | bounded(butterfly) | policy3 | p3/clifft | p3/bounded | diag/flush | bornH |
|---|--:|--:|--:|--:|--:|--:|--:|
| cultivation_d5 | 212.8k | 727.7k | 719.1k | **3.38** | **0.99** | 26/91 | 16 |
| cultivation_d3 | 1.80k | 5.47k | 4.98k | 2.77 | 0.91 | 11/29 | 6 |
| coherent_ry_d3_r1 | 12.29M | 12.85M | 12.04M | 0.98 | 0.94 | 19/57 | 17 |
| coherent_d5_r5 | 17.99G | 25.10M | 26.35M | 0.00 | 1.05 | 128/403 | 60 |
| distillation | 1.90k | 1.27k | 1.53k | 0.81 | 1.21 | 5/10 | 5 |

**cultivation_d5: 0.99× bounded — no win.** Only 26/91 (28%) rotations land diagonal; the other 65 fall
back to the butterfly (c=12, the dominant term), and the born-H adds 16·(c=4). The saves and the adds
cancel. (d5_r5 is a different regime — clifft is 18G full-statevector; bounded already wins 700×; the
FLOP-≤-clifft goal only bites in the small-rank cultivation regime.)

## 5. Why — the architectural wall (Phase 5, confirmed empirically)

clifft keeps the entangling Clifford **symbolic** (`virtual_frame`) and stores each magic axis in its own
eigenbasis, so the T generator is **always diagonal** there → c=3, 0 H. bounded **pulls the full Clifford
frame into the generator** at each `_pullback` → off-diagonal → butterfly/localize. The born-X fold only
absorbs the *prep*-H at promote; the **subsequent circuit Cliffords (CNOT/CZ/H) are re-pulled into the
next generator**, rotating it off-diagonal again. A single per-axis born basis cannot diagonalize
generators whose required basis changes across measurements — exactly the 72% that fall back.

This matches Phase 4/5 precisely: the realizable per-axis-basis result is **not** clifft parity. The
FLOP ladder for cultivation is:

| scheme | c (rotation) | × clifft | risk |
|---|--:|--:|---|
| butterfly (current default) | 12 | ~3.4–4.0 | shipped |
| per-rotation localize (existing localizer, ungated) | 7 | ~2.33 | low (verified `_loc_undo`) |
| per-batch simultaneous-diagonalize (Phase 4) | (4B+3q) | ~1.64 | medium (batch the localizer) |
| **born-X promote only (this Step B1)** | mixed 3/12 | **~3.4 (wash)** | done, but not useful |
| full symbolic-Clifford frame (clifft discipline) | 3 | **1.0 (parity)** | high (re-architecture) |

The born-X-promote-only scheme sits *worse* than per-rotation localize because its fallbacks are the
full butterfly, not the localizer. **Reaching FLOP ≤ clifft requires the full symbolic-Clifford frame
discipline** (keep CNOT/CZ/H symbolic, magic axis in its own eigenbasis, diagonal T always) — the
substantial re-architecture, which is the R_Y-frame-bug class.

## 6. Status of the artifact

`policy3.py` stays as a **default-off, correctness-validated building block**: it proves the born-basis
fold is a sound physical-identity primitive and bit-exact through measurements. It is NOT a FLOP win and
is not claimed as one. The Step-B0 result (diagonal dispatch exact per-T) stands independently.

## 7. Decision needed

The minimal split does not reach the goal. Three honest paths (cover message), in increasing
risk/reward: (a) ungate the existing localizer for the FLOP regime → 2.33× (still > clifft, low risk);
(b) per-batch simultaneous diagonalization → 1.64× (medium); (c) full symbolic-Clifford frame discipline
→ 1.0× parity (high risk, the only path to ≤ clifft). I recommend deciding (b) vs (c) before more code.
