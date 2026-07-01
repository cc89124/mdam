# Wall-clock breakdown of the slow native-MDAM cases

**Question answered:** for each case where native MDAM is slower than Clifft *in wall-clock*, is the slowness
**dense arithmetic** or **control plane**? (FLOP/squeeze is closed; `flop_table_native.csv` dense FLOP is taken as
fixed. This is a *time* analysis, ns/shot per region.)

Method: clean wall (`native_mdam_vm.so`, no instrumentation) for native + Clifft, interleaved median of 11.
Region ns/shot from the **PROFILE build** (`-DMDAM_PROFILE`, `prof[17]`, faithful: record-mismatch 0 vs the
release build on seeds 1/7/42/123/999) scaled to the clean wall (the PROFILE build inflates the *total* by per-op
`now_ns`, so absolute prof ns are dropped; the **relative** per-region split — a clean per-phase partition — is
scaled onto the clean wall). Oracle path sub-split from `ORC_T` (rdtsc). Dense ns cross-checked against the
`dense_flop_*` counters (match `flop_table_native.csv`: cult_d5 517898≈517615, cult_d3 5128≈5129, distill 2317≈2318).
Ablation (`mc_skip`) used only for the *independent* regions (frame/noise/dormant); it is **not** a clean partition
for rot/engine-gate/boundary (skipping rot makes the boundary trivial → cult_d5 engine-gate ablation went **−75 %**),
so those come from PROFILE.

---

## 1. Slow-case one-line summaries

| bench | wall ratio | FLOP ratio | dense explains? | main bottleneck |
|---|---|---|---|---|
| **cultivation_d3** | **6.80×** (14757 / 2170 ns) | 1.35× | NO — dense ~10 % of wall | **pullback `F†PF` + inverse-frame maintenance** (control) |
| **cultivation_d5** | **2.43×** (198410 / 81789 ns) | 1.19× | PARTLY — dense ~45 % (≈ Clifft total alone) | **dense kernel (r=10) + pullback** (mixed) |
| **distillation** (authoritative) | **1.53×** (17113 / 11156 ns) | 0.48× | NO — dense ~2 % | **frame conjugation** — already SOLVED by `mc_fblock` → **0.79×** |

---

## 2. Wall breakdown tables (ns/shot, scaled to clean wall)

### Table A1 — cultivation_d3 (total 14757 ns, Clifft 2170, 6.80×)

| region | ns/shot | %total | dense? | compilable offline? |
|---|---:|---:|:--:|:--:|
| C inverse/pullback — **PLAN_PULLBACK `F†PF`** | **3813** | **25.8%** | no | **yes (StaticPlan: masks shot-static)** |
| C inverse — OP_ACTIVEGATE (engine cx/cz/s = tableau+inverse-frame conj) | 2125 | 14.4% | no | partly (inverse must stay live for pullback) |
| B frame symbolic update (h/s/cnot/cz/swap) | 1860 | 12.6% | no | **yes (fblock-style superinstruction)** |
| J RNG / noise bookkeeping | 1687 | 11.4% | no | no (runtime randomness) |
| A dispatch / interpreter (OP_OTHER) | 1165 | 7.9% | no | **yes (compiled opcode stream)** |
| I commit (MAGIC_COMMIT: drop + frame folds) | 956 | 6.5% | mixed | partly |
| H1 dense kernel (MAGIC_KERNEL, compiled cores) | 879 | 6.0% | **yes** | no (real arithmetic) |
| H2+I oracle measure (ORACLE: flush+born+drop, 1 call) | 851 | 5.8% | mixed | partly |
| ROT opcodes (defer only) | 722 | 4.9% | no | yes |
| dormant measurements | 691 | 4.7% | no | partly |
| **dense total (H1 + oracle-dense + commit-drop)** | **~1500** | **~10%** | — | — |
| **control total** | **~13250** | **~90%** | — | — |

### Table A2 — cultivation_d5 (total 198410 ns, Clifft 81789, 2.43×)

| region | ns/shot | %total | dense? | compilable offline? |
|---|---:|---:|:--:|:--:|
| H2 oracle measure (ORACLE, 10 calls) — of which: | 79760 | 40.2% | mixed | partly |
|   ↳ flush_core (rotations + per-rotation pullback) | 47218 | 23.8% | ~½ dense / ½ pullback | rotations no, pullback **yes** |
|   ↳ Pm_pullback `F†Z_qF` | 15314 | 7.7% | no | **yes** |
|   ↳ Born + drop | 15314 | 7.7% | **yes** | no |
| H1 dense kernel (MAGIC_KERNEL, 5 compiled cores) | 41306 | 20.8% | **yes** | no |
| C pullback (MAGIC_PLAN, ~all PLAN_PULLBACK) | 23254 | 11.7% | no | **yes (StaticPlan)** |
| C inverse — OP_ACTIVEGATE (engine conj) | 17609 | 8.9% | no | partly |
| B frame symbolic update | 9581 | 4.8% | no | **yes (fblock)** |
| J RNG / noise | 8597 | 4.3% | no | no |
| I commit (MAGIC_COMMIT) | 7271 | 3.7% | mixed | partly |
| A dispatch (OP_OTHER) | 5418 | 2.7% | no | **yes** |
| dormant + ROT opcodes | 5449 | 2.7% | no | yes |
| **dense total (H1 + oracle Born/drop/rotations)** | **~90000** | **~45%** | — | — |
| **control total** | **~108000** | **~55%** | — | — |
| of which **pullback (PLAN + Pm + flush)** | **~52500** | **~26%** | no | **yes** |

### Table A3 — distillation (authoritative, total 17113 ns, Clifft 11156, 1.53×) — SOLVED case

| region | ns/shot (ablation-clean) | %total | note |
|---|---:|---:|---|
| B frame symbolic update | ~4900 | ~22% (ablation) / 37% (prof, inflated) | **`mc_fblock` compiles this away → wall 0.79×** |
| boundary measure (85 deterministic, pullback+dispatch) | ~8900 | ~40% | all control; 0 dense applies |
| noise / dispatch / gate / dormant | rest | ~38% | control |
| **dense** | **~300** | **~2%** | 0 core rotations (deterministic stabilizer projections) |

---

## 3. Table B — dense floor vs wall gap

| bench | Clifft ns | MDAM ns | wall ratio | FLOP ratio | dense ns (≈) | control ns (≈) | control % | main cause |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| cultivation_d3 | 2170 | 14757 | 6.80× | 1.35× | ~1500 | ~13250 | **90%** | pullback + inverse-frame (control) |
| cultivation_d5 | 81789 | 198410 | 2.43× | 1.19× | ~90000 | ~108000 | **55%** | dense floor ≈ Clifft + pullback control |
| distillation | 11156 | 17113 | 1.53× | 0.48× | ~300 | ~16800 | **98%** | frame (SOLVED: fblock → 0.79×) |

**Reading:** cult_d3 and distillation are **control-plane bound** (dense < 10 %). cult_d5 is **mixed**: its dense work
alone (~90 µs) already ≈ Clifft's whole shot (82 µs), so even zero control gives ≈ parity — the r≈k dense floor is the
wall, and the +55 % control is what makes it 2.4×.

---

## 4. Table C — event counts (per shot)

| event | cultivation_d3 | cultivation_d5 | distillation |
|---|---:|---:|---:|
| opcode count | 322 | 1675 | 1995 |
| frame ops (h/s/cnot/cz/swap) | 122 | 798 | 1625 |
| RNG draws | 36 | 155 | 45 |
| compiled core calls | 4.0 | 5.0 | 4.0 |
| oracle core calls | 1.0 | 10.0 | 1.0 |
| core rotations (oracle, `core_apply`) | 1.04 | 33.07 | 0.00 |
| max core rank (peak r) | 4 | 10 | 4 |
| dense FLOP (rot/collapse/loc) | 3884/670/574 | 450506/30797/36596 | 1008/408/901 |
| dense FLOP total | 5128 | 517898 | 2317 |

---

## 5. Common vs case-specific bottlenecks

**Common (every slow case): the symbolic frame layer.**
- **pullback `F†PF`** (cult_d3 26 %, cult_d5 26 %) — the single largest *control* cost on the cultivations. It is a
  GF(2)/Pauli conjugation of the observable + each core rotation through the inverse frame, recomputed live per
  measurement. Its masks are **shot-static** (`StaticPlan`) — only the phase is dynamic.
- **inverse-frame maintenance** (OP_ACTIVEGATE: engine cx/cz/s, cult_d3 14 %, cult_d5 9 %) — keeps the invertible
  frame current so the pullback is O(weight). Coupled to the pullback.
- **frame symbolic update** (cult_d3 13 %, cult_d5 5 %, **distill 22–37 %**) — the dormant-Clifford bit XORs. This is
  exactly what `mc_fblock` already compiles away (distillation 1.53×→0.79×).
- **dispatch** (OP_OTHER, 3–8 %) — per-opcode interpreter tax.

**Case-specific:**
- **distillation**: ~100 % frame (1625 frame ops, 0 dense). The pure fblock case — solved.
- **cultivation_d5**: a **real dense kernel** (45 %, r=10, 2^10 butterflies, 33 oracle rotations) on top of control.
  This is the only slow case where dense is a first-order term.

---

## 6. Q1–Q10 per case

### cultivation_d3
- **Q1 dense % of wall:** ~10 %.
- **Q2 FLOP ratio vs wall ratio:** 1.35× vs 6.80× → ~5.5× gap is non-dense.
- **Q3 top-5 control regions:** pullback 26 %, inverse-frame gates 14 %, frame 13 %, noise 11 %, dispatch 8 %.
- **Q4 compile-removable:** pullback (static masks) ✔, frame (fblock) ✔, dispatch (compiled stream) ✔; inverse-frame partly; noise ✘.
- **Q5 must stay runtime:** RNG draws (36/shot), Born outcome, noise firing — randomness.
- **Q6 dispatch-only removal:** ~1165 ns (7.9 %).
- **Q7 static pullback/core/layout/localizer plan:** ~3900 ns (pullback 26 %) + part of inverse-frame → up to ~5000 ns.
- **Q8 direct projector:** localizer is ~0 here (574 FLOP, ~4 % LOC) → negligible.
- **Q9 expected after opts:** frame(fblock)+dispatch+static-pullback ≈ −6000 ns → ~8000 ns ≈ **3.7×**; full control removal floor ≈ dense ~1500 + irreducible RNG ≈ **~parity** (Clifft 2170 is the wall).
- **Q10 verdict:** **parity is the ceiling** — Clifft 2.17 µs is so small that even a fully-compiled MDAM lands near parity; no clean win.

### cultivation_d5
- **Q1 dense % of wall:** ~45 %.
- **Q2 FLOP ratio vs wall ratio:** 1.19× vs 2.43× → ~1.24× gap is non-dense control.
- **Q3 top-5 control regions:** pullback (plan+Pm+flush) 26 %, inverse-frame gates 9 %, frame 5 %, noise 4 %, dispatch 3 %.
- **Q4 compile-removable:** pullback ✔ (static masks), frame ✔ (fblock), dispatch ✔; dense kernel ✘.
- **Q5 must stay runtime:** dense butterflies (r=10), Born/drop, RNG (155/shot), noise.
- **Q6 dispatch-only removal:** ~5400 ns (2.7 %).
- **Q7 static pullback plan:** ~52 500 ns (26 %) is the prize — biggest single removable block.
- **Q8 direct projector:** removes localizer 36596 FLOP (~7 %) → modest wall (~few µs).
- **Q9 expected after opts:** remove pullback(26 %)+frame(5 %)+dispatch(3 %)+gates(part) ≈ −70 µs → ~128 µs ≈ **1.57×**; floor = dense ~90 µs ≈ **1.1×** (≈ Clifft, the FLOP-parity floor).
- **Q10 verdict:** **parity is the ceiling** — dense alone (90 µs) ≈ Clifft (82 µs); r≈k gives no dense advantage. Control removal moves 2.43×→~1.1×, not below.

### distillation
- **Q1 dense % of wall:** ~2 %.
- **Q2:** FLOP 0.48× vs wall 1.53× → all gap is control (frame).
- **Q3 top control:** frame 22–37 %, deterministic-measure pullback/dispatch ~40 %, noise/gate rest.
- **Q4–Q7:** frame fully compilable (fblock, done); deterministic-measure dispatch compilable.
- **Q8:** n/a (no localizer dense).
- **Q9:** **already 0.79×** with carry+fblock.
- **Q10 verdict:** **win — achieved.** Dense ~0, control was frame, compiled away.

---

## 7. Removal-possibility summary & expected targets

| lever | cult_d3 | cult_d5 | distillation |
|---|---|---|---|
| frame compile (fblock) | ~13 % | ~5 % | **~22–37 % (DONE → 0.79×)** |
| dispatch compile | ~8 % | ~3 % | done |
| **static pullback plan** | **~26 %** | **~26 %** | ~4 % |
| inverse-frame (keep live) | ~14 % (partial) | ~9 % (partial) | small |
| direct projector (drop localizer) | ~0 % | ~7 % FLOP | n/a |
| **current → after-control-removal → dense floor** | 6.80× → ~parity → 1.35× FLOP | 2.43× → ~1.1× → 1.10× FLOP | 1.53× → **0.79× (done)** |

---

## 8. Sanity check

- PROFILE build faithful: record mismatch **0** vs release (`sample_batch` both builds), seeds 1/7/42/123/999.
- dense FLOP counters match `flop_table_native.csv` (cult_d5 517898 vs 517615; cult_d3 5128 vs 5129; distill 2317 vs 2318).
- Region sum vs total: PROFILE Σregions = 77 % (cult_d3) / 91 % (cult_d5) of the *PROFILE* RUN; the residual is the
  per-op `now_ns` timer overhead (cult_d3 23 %, cult_d5 9 %), excluded by scaling onto the clean wall (not attributed).
- Ablation flagged as **non-partition** for rot/engine-gate/boundary (cult_d5 engine-gate ablation −75 %, coupled to dense);
  PROFILE used instead for those. Ablation used only for independent frame/noise/dormant.
- Timer overhead reported, not hidden: PROFILE inflation 1.70× (cult_d3) / 1.77× (cult_d5) / 4.89× (distillation,
  frame-dominated) — reason distillation uses ablation, not prof-scaling, for its frame number.

---

## 9. Final verdict — dense or control plane?

- **cultivation_d3 → CONTROL-PLANE** (dense ~10 %). Slow because of pullback + inverse-frame + frame + dispatch.
  But **parity is the ceiling**: Clifft 2.17 µs is too small for a win even at zero control.
- **cultivation_d5 → MIXED, control-dominated gap.** Dense is a real 45 % (r=10) and *alone* ≈ Clifft's whole shot;
  the +55 % control (pullback-led) is what makes it 2.43×. Removing control → ~1.1× = **parity ceiling** (r≈k, no dense edge).
- **distillation → CONTROL-PLANE, SOLVED.** Dense ~0; the frame control plane was the wall; `mc_fblock` → 0.79×.

**One line:** the slow cultivations are slow because of the **symbolic control plane (pullback `F†PF` + inverse-frame
+ frame), not dense FLOP** — but because they sit at **r≈k**, even fully removing that control only reaches **parity**,
not a win. The only win regime is `r ≪ k`, where the dense block is small and the same control compiles to a win
(distillation, proven).
