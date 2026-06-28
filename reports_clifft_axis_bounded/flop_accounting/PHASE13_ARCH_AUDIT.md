# Architectural audit — bounded re-derives clifft's localization instead of consuming it

**Verdict: your diagnosis is exactly right, and confirmed in code + traces.** clifft's post-localization
opcode stream (`OP_ARRAY_T` on a specific active slot, `OP_MEAS_ACTIVE_DIAGONAL`, `OP_EXPAND_T`,
`OP_ARRAY_CNOT`) **reaches the backend**, but the bounded **data plane re-derives** every active
operation through `self.nc`'s own U_C tableau (physical pullback + butterfly/localizer + relocalizing
measurement), discarding clifft's "this is diagonal on slot" decision. The control plane (`self.frame`
= clifft's Pauli frame, the slot routing, the opcode stream) is already metadata-only and correct. So
the fix is to **quotient only the numerical state and consume clifft's diagonal opcodes directly**, not
to build a new C_sym / born-basis engine. S1/S2 were operating on the wrong layer.

## 1. The two pipelines (file/function level)

**clifft** (`clifft.compile(bytecode_passes=None)` → `clifft.sample`):
```
physical/HIR T,CNOT,measure
 → trace → HIR → lower → bytecode: OP_ARRAY_T(slot) / OP_MEAS_ACTIVE_DIAGONAL(slot) / OP_EXPAND_T(slot) / OP_ARRAY_CNOT(u,v)
 → svm execute: array_t = ONE diagonal phase on slot's array bit (c=3·2^k), meas_interfere (c=8), expand (|+>, 0 FLOP)
```
clifft never pulls a generator back; the slot IS the localized axis, the op IS diagonal.

**bounded** (`nearclifford_backend/backend.py` run_shot, lines 443-609):
```
same bytecode  OP_ARRAY_T(slot) ...
 → _rot(slot, angle)            [backend.py:324]  q = slot2id[slot]; theta = -angle if frame.xb(slot) else angle
 → self.nc.apply_rotation(0, 1<<q, theta)         [simulator.py:355]
 → self.nc._pullback(0,1<<q)    [simulator.py:276] = U_C^dag Z_q U_C through nc's OWN rebuilt tableau
 → off-diagonal generator → _flush_one butterfly (c=12) [engine.py:273 / bounded.py:66]
```
`self.nc`'s U_C is **rebuilt** by replaying clifft's active Cliffords into the engine's tableau:
`OP_ARRAY_CNOT → self.nc.cx` (backend.py:562), `OP_ARRAY_H → self.nc.h` (534), and crucially
`_birth → self.nc.h(q)` (backend.py:306) — the |+> birth is realized as an **H folded into nc's U_C**
(array stays |0>), not as |+> in the array. So `Z_q` pulled back through that H-laden U_C is
**off-diagonal**, and the measurement (`OP_MEAS_ACTIVE_DIAGONAL → measure_z → _localize_to_Z`,
bounded.py:385) **re-localizes by folding more H's into U_C**, re-polluting it.

## 2. Concrete event trace — one cultivation_d5 `OP_ARRAY_T`

| | clifft | current bounded |
|---|---|---|
| op | `OP_ARRAY_T` on slot | same opcode received |
| translation | `array_t` on slot's array bit | `_rot` → `apply_rotation(0, Z_q)` |
| generator | none (slot IS the axis) | `_pullback(Z_q)` = **xp=0x5d0, zp=0 → X on qubits {4,6,7,8,10}, weight-5 OFF-DIAGONAL** |
| kernel | 1 diagonal phase sweep, **c=3·2^k** | butterfly **c=12·2^r** (or localizer collapse+H+diag c=7) |
| array sweeps | **1** | 1 butterfly (off-diagonal pairs) (+H sweeps if localized) |

clifft's diagonal-on-slot becomes a weight-5 off-diagonal butterfly because bounded re-pulls `Z_q`
through its rebuilt, H-laden U_C. **This is the entire 3.42× FLOP gap** — not the bounded idea, but the
data-plane layer.

## 3. Prototype probe (birth fix only) — and what it reveals

Changing `_birth` to materialize |+> in the **array** (`nc._promote; nc._h_axis`, tableau left clean)
instead of `nc.h(q)`: **records/rank stay bit-exact (0 mismatch, 8 seeds)**, and **26/91 ARRAY_T become
diagonal** (vs 0/91 today). But FLOP does NOT improve (749.9k) because the other **65 are re-polluted by
the relocalizing measurement** (`_localize_to_Z` folds H into U_C; diagonal count falls as the
measurement count rises). So the re-derivation is in **both** the rotation path and the measurement path
— exactly the layering problem. The 26 that survive prove the clean-tableau pullback of `Z_q` **is** a
reduced Z-string (diagonal parity phase) when no relocalizing measurement has intervened.

## 4. Answers to your specific questions

1. **Why re-derive instead of consume?** `self.nc` (CliftAxisBoundedNearClifford) descends from the
   lazy/virtual-axis lineage — a **generic Pauli-rotation/measurement engine**: `apply_rotation` takes a
   *logical* Pauli and pulls it back; `measure_z` re-localizes. The reduction (quotient) is built on this
   physical-pullback interface (measurement localize-and-drop). clifft's opcodes are treated as physical
   ops to re-simulate, not as already-diagonal opcodes to consume.
2. **Inaccessible, or intentional?** The post-localization stream **is** accessible (the dispatch handles
   `OP_ARRAY_T/MEAS_ACTIVE_DIAGONAL/EXPAND_T/ARRAY_CNOT`) and the p_x is already read (`frame.xb(slot)`
   in `_rot`). The re-derivation is the **design of the data plane** (generic engine + measurement-core
   reduction), not a missing stream.
3. **Metadata-only frame + quotient just the state?** **Yes.** `self.frame` is already metadata-only.
   The data plane should consume `OP_ARRAY_T` as a diagonal parity phase on **Q(slot)** (the reduced
   Z-string), and `OP_MEAS_ACTIVE_DIAGONAL` as a diagonal Born on Q(slot) — no pullback, no relocalize.
4. **Obstruction if impossible?** None — it is possible (the 26/91 clean-tableau diagonals are the
   existence proof on bounded's own state; clifft is the full existence proof).
5. **Priority over S1/C_sym?** **Yes.** S1 born-basis and the independent C_sym were *re-implementing*
   clifft's localization. The correct architecture **keeps** clifft's localization and quotients only the
   numerical state. This supersedes them.
6. **Reduced-kernel translation:**
   - `OP_ARRAY_T/T_DAG(slot)` → diagonal parity phase on Q(slot) (1 sweep, c=3·2^r); T vs T† from `frame.xb(slot)`.
   - `OP_EXPAND_T(slot)` → birth a reduced axis in |+> (free copy, c=0 like clifft expand) + the diagonal T.
   - `OP_MEAS_ACTIVE_DIAGONAL(slot)` → diagonal Born on Q(slot) parity (branch sqnorm), collapse, drop slot, update Q (1 sweep, c=8).
   - `OP_ARRAY_CNOT(u,v)` → update Q (Z-preserving relabel; 0-FLOP permutation, never a relocalizing H).
   - demotion → drop the reduced axis (the existing measurement-drop is the quotient; keep it, but driven by the diagonal measurement, not `_localize_to_Z`).
7. **Does the |0>-fixing CNOT reduction send logical Z → reduced Z-string only?** **Yes** — the reduction
   is CNOT-based (Z-preserving), so Z → Z-string. With the tableau kept clean (no relocalizing H), every
   `OP_ARRAY_T` is a reduced Z-string ⟹ diagonal parity phase, **no basis recovery** — all cultivation
   T/T† diagonal. (The 26/91 confirm it where measurements haven't intervened; the measurement fix closes
   the rest.)
8. **Sweep-count ≤ clifft per opcode?** Yes: each `OP_ARRAY_T` is 1 reduced sweep (c=3·2^r) ≤ clifft's 1
   sweep (c=3·2^k) since r ≤ k. The invariant **n_t^reduced ≤ n_t^clifft** holds opcode-by-opcode.

## 5. The invariant we must restore (§4 of your note)

Target both, simultaneously:
- `r_t ≤ k_t` (rank — already enforced by the memory budget), and
- `n_t^reduced ≤ n_t^clifft` (sweep count per opcode — **currently broken**: bounded turns clifft's 1
  diagonal sweep into a butterfly + birth/relocalize H sweeps).

Then `Σ_t n_t^reduced·2^{r_t} ≤ Σ_t n_t^clifft·2^{k_t}` ⟹ **FLOP ≤ clifft**. The current implementation
secures the first inequality but violates the second — which is exactly why FLOP is 3.42× while rank is ≤.

## 6. Correct target structure (control plane / data plane)

```
Logical control plane (metadata only, = clifft, KEPT verbatim):
  self.frame (Pauli frame, p_x), slot routing, the OP_ARRAY_* / OP_MEAS_ACTIVE_DIAGONAL stream.
  No large complex state.
Reduced data plane (the only change):
  reduced array phi (2^r), quotient map Q: clifft slot -> reduced Z-string.
  consume each opcode O_t as Q(O_t) on phi: ARRAY_T -> parity phase, MEAS_ACTIVE_DIAGONAL -> diagonal Born+drop,
  EXPAND_T -> |+> birth + T, ARRAY_CNOT -> relabel Q. No apply_rotation pullback, no _localize_to_Z relocalize.
```
This requires NO independent physical engine, NO second measurement *rule* (clifft's measurement is
already diagonal — we consume it, not re-derive it), NO C_sym, NO generic relocalization.

## 7. Conclusion + the prototype to build next

The architecture is **feasible and correct**, and supersedes S1/C_sym. The minimal prototype (next step,
per your §5): take one cultivation_d5 `OP_ARRAY_T` whose slot has not been touched by a relocalizing
measurement, execute it as **one reduced Z-parity phase on Q(slot)**, and compare state/probability/FLOP
to clifft (`array_t`, c=3) and current bounded (butterfly, c=12). The birth-|+> probe already shows 26
such ARRAY_T are diagonal and records-exact; the prototype will pin one of them as a 1-sweep c=3 op
equal to clifft. The measurement path (`OP_MEAS_ACTIVE_DIAGONAL` consumed diagonally, no `_localize_to_Z`)
is the immediate follow-on that closes the remaining 65 and the full FLOP ≤ clifft.

Status: S2 C_sym work paused; no authoritative change; a05843e / tag / butterfly / localizer /
Policy-3 default-off all preserved. `_birth`-|+> is a throwaway probe (not committed).
