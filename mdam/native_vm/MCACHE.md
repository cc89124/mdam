# Authoritative-Edge Cache (`run_mcache`) — Gate M

A lightweight, **correctness-first** measurement-boundary edge cache for the MDAM near-Clifford
native VM. Built on the **authoritative** path (bit-exact vs Python), it caches the per-boundary
transition keyed by a small *semantic* key, so repeated boundary states reuse a stored edge instead
of recomputing the magic measurement.

## Ground truth — read this first

- **The only ground truth is the Python backend (`backend.sample`) or the native authoritative path
  (`nvm_mdam_run` / `nvm_mdam_sample_batch` on a VM with imem OFF, i.e. `va`).**
- **A `cmode`-vs-`cmode` comparison on the same warmed VM is NOT a correctness test.** Two fast paths
  that share the same (possibly wrong) cache/plan can agree with each other while both diverging from
  the authoritative result. The earlier *"distillation 2G/cmode5 bit-exact, beats Clifft 0.77×"* claim
  was exactly this — a **false positive** (warmed cmode vs warmed cmode). It is **retracted**. The
  real authoritative comparison showed those cmode paths were 0/100 on distillation: their edges came
  from the buggy F4/imem plan cache for the active-measurement dialect.
- `run_mcache` deliberately **does not touch F4/imem/old-cmode5 plan_cache**. Its edges are computed
  from the **authoritative `measure_z`** on a miss, so they are correct by construction.

## Design

- **Lightweight semantic key**, NOT a raw dynbit vector and NOT a full-state image:
  `key = FNV(mp, kind, sid, inv_sig, pend_sig, m_sig)` where
  - `sid` = interned id of the dense block (deduped in a state pool, signed-zero canonical),
  - `inv_sig` = hash of the inverse-frame phases, `pend_sig` = hash of live pending rotations,
  - `m_sig` = hash of the magic-axis set `M`, `mp`/`kind` = boundary index / type.
  Proven sufficient (0 conflicts) over 100k–300k boundary observations on distillation / cult_d3 /
  cult_d5 (see `verify_mcache_keys.py`). Generic minimal subset = `(sid, inv, pend, m)` (~16 B), the
  same for cult_d3 and cult_d5 — not benchmark-specific.
- **Record is not stored**: `rec = outcome ^ parity ^ i1` (parity = X-frame for diagonal, Z-frame for
  interfere/swap). Verified 0 violations.
- **MISS** → run the authoritative `measure_z` (mutates the engine correctly), store the edge
  `{p0, antis, per-outcome pool_id}`, dedup the post-boundary engine state into a pool.
- **HIT** → draw the Born rv, pick the outcome, restore the post-boundary state, set the record by the
  XOR rule. `measure_z` is **skipped**. `anti_s` (stabilizer ag_measure, an `idraw2` coin) stays live.
- **Default OFF.** The authoritative path is behaviorally unchanged (all shared-path edits are guarded
  by a default-false `bcap_on`; `run_mcache` is a separate method).

`mc_mode`: 0 off, 1 SHADOW (always live + build + verify, no skip), 2 FAST-snapshot (full hit skips
measure_z but does a full post-boundary engine restore), 3 FAST-carry (Phase 4 — see below).

## Phase 4 — lightweight carry hit path (`mc_mode == 3`)

Step 3-1 (cost decomposition, `verify_mcache_cost.py`) measured where the snapshot-FAST hit cost goes —
and it is **NOT** the snapshot restore the way one might assume:

| bench | hit-total | restore | **key-hash** |
|---|---|---|---|
| distillation | 991 cyc | 147 (14.9%) | **712 (71.9%)** |
| cultivation_d3 | 1315 | 131 (9.9%) | **1014 (77.1%)** |
| cultivation_d5 | 11055 | 1106 (10.0%) | **9547 (86.4%)** |

The bottleneck is **`mc_key` re-hashing the dense block (2^r amplitudes) every boundary** (72–86%), not the
restore (10–15%). So `mc_mode == 3` (a) **carries `cur_sid`** — the dense block changes only at a measure,
so `sid_in(B+1) == sid_out(B)`; the key reuses the carried id with **no dense re-hash**; and (b) **lazy
dense** — a hit restores only the frame (inverse/tableau/pending) and carries the dense by `sid` (no full
snapshot restore); the dense is re-materialised from the sid-pool only when a live (miss) boundary needs it.

After the carry, `key-hash` collapses (cult_d5 9547→~840 cyc) and the dense copy leaves the hit path
(cult_d5 restore 1106→~620 cyc). **Bit-exact** vs authoritative (multi-seed cold + warmed, SHADOW
fingerprint 0, `verify_mcache_multiseed.py`). The hit path is now: lookup → Born rv → carry `cur_sid` →
frame-restore (inv/tableau/pending) → record by XOR. No full-engine snapshot restore.

## Gate N — frame-block superinstruction (`mc_fblock`, default OFF)

The Phase-4 carry hit path removed the cache bookkeeping cost, but a **control-plane ablation** (timing-only,
`nvm_mcache_set_skip`, clean-wall delta — no per-op rdtsc overhead) showed where the *remaining* per-shot time
of distillation actually lives, and it is **not** the cache:

| category (distillation, 100%-hit carry) | ns/shot | share |
|---|---|---|
| **OUTER_FRAME** (pure Pauli-frame conjugation) | **~4300** | **30%** |
| VM-interpreter floor (dispatch + slot + reset + unablated ops) | ~6500 | 46% |
| NOISE (RNG draw + frame apply) | ~1370 | 10% |
| BOUNDARY hit path (the cache itself) | ~790 | 6% |

distillation is **81 % pure `MO_FRAME_*` opcodes** (1625 / 1995, in 90 runs, mean 18 / max 52 — `FRAME_CNOT`=833,
`FRAME_CZ`=770). Each one pays a full big-switch dispatch (6 array loads + a jump) for an ~8-cyc XOR body, so the
frame work *plus its dispatch* is ~45 % of the shot. `mc_fblock` (Gate N) **batches each maximal run of pure frame
opcodes into ONE dispatch + a tight inner loop** with `grow()` hoisted out — executing the **identical ops in the
identical order**, hence **bit-exact by construction** (no linear-map composition, which would *densify* short runs).

Result (carry mode 3, interleaved cold fresh-run, **bit-exact vs authoritative** — 16 seeds × 4000 shots × modes
1/2/3 = 0 mismatch):

| bench | carry OFF | **carry + fblock** | Clifft | fblock saves | ON/Clifft |
|---|---|---|---|---|---|
| **distillation** | ~14800 | **~8800** | ~11150 | **−41 %** | **0.79× (MDAM faster ~2300 ns)** |
| cultivation_d3 | ~8300 | ~7770 | ~2150 | −6.5 % | 3.62× (frame is only 38 %) |
| cultivation_d5 | ~196700 | ~195800 | ~81300 | −0.4 % | 2.41× (miss-bound, frame not the wall) |

**distillation is the first realistic case where the forced, bit-exact MDAM run is faster than Clifft** (carry
cache + frame-block, default-off features, 95 % hit on this state-repeating workload). Stable across T = 4k/8k/16k
(0.78–0.80×). Absolute ns/shot is system-load-dependent (the ratio is the reproducible metric; under load the
margin narrows to ~0.92× but stays < 1). cult_d3/d5 also get the *same generic* optimization (no benchmark
special-case) but gain little — they are not frame-dominated. The win is purely **structural to the workload**:
distillation's circuit is mostly the dormant Clifford frame, which compiles away; cult_d3 is Clifft-parity-hard
(its VM floor alone ≈ Clifft) and cult_d5 is dense-miss-bound.

`mc_fblock` is orthogonal to `mc_mode` (helps the frame ops in every mode). C API: `nvm_mcache_set_fblock(vm, 0|1)`.
Harness: `verify_mcache_fblock.py` (captured reference `expected_mcache_fblock.txt`).

## Expected output (correctness + no-regression)

All bit-exact vs the authoritative path (SHADOW + FAST warmed + FAST fresh + 4k batch, 0 mismatch):

- `verify_mcache_keys.py distillation,cultivation_d3,cultivation_d5 20000` → T1/T2/T3/T4 = 0 all.
- `verify_mcache.py distillation 3000 2000`  → SHADOW 3000/3000, FAST 2000/2000 + 2000/2000, batch 0.
- `verify_mcache.py cultivation_d3 4000 2000` → bit-exact, hit ~99%.
- `verify_mcache.py cultivation_d5 1500 800` → bit-exact, hit ~78%.
- `MDAM_BENCH=distillation verify_mdam_batch.py` → `ALL BATCH CHECKS PASS` (authoritative no-regression).

A captured reference run is in `expected_mcache_output.txt`.

## Performance (batch, warm steady-state; ns/shot)

Absolute ns/shot is system-load-dependent (the per-boundary cycle counts from `verify_mcache_cost.py`
are the stable metric); the ratios below are reproducible. The timed regime is **all-hit warm**
(same master seed replayed) — representative for distillation (saturates → any seed ~95% hit) but a
**best case for cult_d5** (fresh seeds are ~78% hit, so a real cult_d5 run sits between carry and auth).

| bench | authoritative | snapshot (mode 2) | **carry (mode 3)** | carry/snapshot | carry/Clifft | hit % | pool (4k batch) |
|---|---|---|---|---|---|---|---|
| distillation | ~17000 | ~13600 | ~13200 | 0.98× | ~1.19× | 95.6% | 58 (**saturates**) |
| cultivation_d3 | ~14200 | ~8400 | ~7400 | 0.89× | ~3.5× | 99.0% | ~1063 (slow growth) |
| cultivation_d5 | ~180000 | ~97000 | ~60600 | **0.62×** | ~0.74× | 78.5% (fresh) | ~35444 (grows ~4/shot) |

Per-boundary cyc/hit (stable, `verify_mcache_cost.py`): snapshot vs carry hit-total —
distillation 937→603, cult_d3 1323→902, cult_d5 **10697→1812** (key-hash 9207→834, restore 1123→623).

The table above is the **warm all-hit ceiling** (replay the same seed) — a best case, not a realistic run.

## Realistic fresh-run (cold cache, fresh seeds) — `verify_mcache_freshrun.py`

This is the honest end-to-end: a fresh VM, run T shots from an **empty** cache (it builds during the run),
so the **real miss rate** is paid. Same harness for all four. (auth / Clifft have no cache.)

| bench | hit % | authoritative | snapshot | **carry** | Clifft | carry/auth | carry/Clifft | cache mem (growth) |
|---|---|---|---|---|---|---|---|---|
| distillation | 95.4% | 16939 | 13683 | **13296** | 11091 | **0.78×** | 1.20× | **0.1 MB (saturates @51 sid)** |
| cultivation_d3 | 97.4% | 14470 | 8395 | **7882** | 2145 | **0.54×** | 3.67× | 1.6 MB @16k (slows: 0.20→0.07 sid/shot) |
| cultivation_d5 | **57.4%** | 189086 | 206060 | **188940** | 82614 | **1.00×** | 2.29× | **374 MB @8k (LINEAR, ~5 sid/shot)** |

cult_d5 three regimes (`carry`, 4000 shots): (a) warm all-hit **58.8 µs** [best case, replay] ·
(b) warm-then-fresh 69.7% hit **168.8 µs** · (c) **fresh cold 59.3% hit 182.4 µs [realistic]**.

### Speed win vs memory win — separated, honestly
- **distillation = real win.** Bounded memory (0.1 MB, saturates at 51 states), `carry/auth = 0.78×`
  (22% faster), 95% hit. Still **1.20× Clifft (slower)**.
- **cultivation_d3 = speed win, weak memory.** `carry/auth = 0.54×`, but **3.67× Clifft (slower)** — Clifft's
  parity regime is hard to approach. Memory grows but sublinearly (hit rate climbing → may approach bound).
- **cultivation_d5 = NOT a realistic win.** Fresh hit is only ~57–59% (not the 78% earlier estimate);
  `carry/auth = 1.00×` (**no speedup** — each miss runs the full `measure_z`, which dominates), **2.29×
  Clifft (slower)**, and memory **grows linearly (~5 sid/shot, 374 MB at 8k shots, unbounded)** — a memory
  **cost**, not a win. The warm all-hit 0.74×-Clifft figure was a pure artifact of replaying one seed.
- Note: for cult_d5 the *snapshot* mode is actually **slower than authoritative** (206 vs 189 µs) — its
  per-boundary overhead is a net loss at 57% hit; the carry path removes the dense re-hash and brings it
  back to **parity with authoritative** (no regression), but cannot turn a non-repeating workload into a win.

### Bottom line
The lightweight BoundaryKey + authoritative-edge cache + sid-carry is **bit-exact** and lightweight.
It is a genuine **bounded-memory + modest-speed win only for state-repeating workloads (distillation)**.
For non-repeating workloads (cult_d5) it is **neither a speed nor a memory win in realistic fresh runs**;
it only avoids regressing vs authoritative. No "beats Clifft" — the only sub-1× Clifft number (cult_d5
0.74×) exists solely in the unrealistic warm-all-hit replay regime.

## Reproduce

```bash
cd clifft-paper/mdam/native_vm && ./build.sh
PY="taskset -c 2 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /home/jung/clifft_env/bin/python"
$PY verify_mcache_keys.py distillation,cultivation_d3,cultivation_d5 20000   # key sufficiency
$PY verify_mcache.py distillation 3000 2000                                  # correctness + perf
$PY verify_mcache.py cultivation_d3 4000 2000
$PY verify_mcache.py cultivation_d5 1500 800
$PY verify_mcache_cost.py                                                     # Phase-4 hit-cost decomposition
$PY verify_mcache_carry.py                                                    # Phase-4 carry: bit-exact + timing
$PY verify_mcache_multiseed.py                                               # Phase-4 multi-seed + SHADOW fingerprint
$PY verify_mcache_fblock.py distillation 4000 8                              # Gate N frame-block: bit-exact + timing
MDAM_BENCH=distillation $PY verify_mdam_batch.py                              # authoritative no-regression
```

Captured reference outputs: `expected_mcache_output.txt` (Phase 0–3), `expected_mcache_phase4.txt`
(Phase 4 multi-seed + cost). `mc_mode` C API: `nvm_mcache_set_mode(vm, 0|1|2|3)`,
`nvm_mcache_batch`, `nvm_mcache_stats`, `nvm_mcache_set_time`/`nvm_mcache_cyc_get` (decomposition).
