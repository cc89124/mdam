# MDAM Native Batch VM — Results (consolidated)

One-page summary of the native MDAM (near-Clifford) sampler. The full per-gate development journey
(Gates A–K, 74 detail files) is archived at `/home/jung/mdam-vm-archive/prev-reports/`.

Source of truth: `mdam/native_vm/` (see its `README.md` and the repo-root
`PROJECT_STRUCTURE.md`). It verifies against the in-tree Python oracle (`mdam.frame` + `mdam.backend`)
+ `mdam.frame`), which is unchanged.
Benchmark: **cultivation_d3** (n=15 qubits, W=1, peak_rank=4, 322 opcodes/shot, 5 magic measures/shot,
21 measurements, ~504 noise sites/shot). Compiler g++ 11.4 `-O3 -march=native -std=c++17 -DNDEBUG`,
CPU i7-8700K @ 3.70 GHz, single-thread, `taskset -c 2`.

---

## What this is

A C++ batch VM that runs near-Clifford QEC magic-state sampling **end-to-end in native code** — one
`Python→C++` call per batch, zero per-shot Python callbacks — and is **bit-exact** to the authoritative
Python MDAM runtime (`run_shot` / `sample_batch`). The native path is **default OFF**; the authoritative
Python path is unchanged and remains the reference oracle.

The headline goal: **MDAM must never be meaningfully slower than Clifft, on any case.** cultivation_d3 is
the hardest case for MDAM — a structural LOSE regime (peak_rank 4 is too small for near-Clifford advantage;
see Gate G), so reaching parity *here* is the proof.

## Current result (Gate K, cultivation_d3)

| measure | value |
|---|---|
| native batch path | 1 Python→C++ call/batch, 0 per-shot callbacks |
| correctness | **bit-exact** vs authoritative (25/25 oneshot + 128 000-shot 0 mismatch, every gate) |
| **warm steady-state speed** | **~1.08–1.10× Clifft, FLAT across 1k→1M shots** (essentially parity) |
| cold-start (from-scratch) | 1.32× (1k) → 1.09× (128k); warmup amortizes away by ~tens of thousands of shots |
| RNG stream | identical (noise gap-sampler draws/fires unchanged by all optimizations) |

**MDAM does not strictly *beat* Clifft on d3 — it reaches parity.** The structural-WIN regime is high-rank
(d5_r5: Clifft active rank ~24, MDAM ~175× less memory) and is the next validation target.

### How parity was reached — the FAST ladder (all bit-exact)
```
  2G (BoundaryPlan)         ~6700 ns   3.0×     compiled magic-boundary plan
  + state_id key            4814       2.23×    intern resident state → integer id
  + carried-pp + oracle-fast 4060      1.85×    key on pre-fwd_map pp; Born oracle in FAST path
  + lazy survivor carry     3870       1.79×    drop eager survivor copy; carry only cur_sid
  + skip-to-next-fire       ~2344      ~1.09×    gap-sampler next_idx: skip 503/504 no-op noise sites
```
- **skip-to-next-fire** was the decisive lever: the noise sampler is a *gap sampler* (draws ~2 RNG/shot to
  find the next firing site), so 503.5/504 sites/shot are semantic no-ops. Visiting only blocks containing
  `next_idx` cut noise cost ~1610→~470 ns with **0 change to the RNG stream** (draws/fires identical).
- **shot-sweep audit** (fair: both sides store full output, rep-interleaved): `warm/Clifft` is flat at
  1.07/1.08/1.10/1.09/1.08 across 1k→1M → the 1.09× is a robust *warm steady-state*, not a 128k artifact.
  (A prior "0.92× at 1M, MDAM beats Clifft" claim was **retracted** — it came from an unfair output-handling
  asymmetry; with output matched MDAM sits at ~1.08–1.11× parity.)

## The journey, in one paragraph (Gates A–K)

**A–D**: built the native VM and full-batch sampler; proved the seed/RNG path bit-exact (PCG64 + SeedSequence
ported exactly); A/C ≈ 300× speedup from removing the Python control plane (4.70 ms → 15.7 µs/shot), leaving
MDAM ~7× slower than Clifft = pure numerical+symbolic core. **E**: removed all hot-loop heap allocation
(226→0/shot); boxed the result — MDAM's *dense arithmetic alone* (~1.24–1.47 µs) already **beats** Clifft
(~2.15 µs); the entire remaining gap is **symbolic control plane** (frame XOR, noise RNG, inverse-frame,
magic pullback), not arithmetic. **F/F-B/F5**: compiled the measurement-boundary symbolic region (M-keyed
static skeleton, snapshot+region-const frame fold, live inverse kept) → 4.7× → 3.7× gap. **G**: proved
cultivation_d3 is a *structural LOSE* case — even a FREE dense kernel loses 3.3× because peak_rank 4 is too
small; the win regime is high-rank. **I/J**: dissected the control-plane floor (operation-count bound, no hot
op); built the region-affine compiled-sampler feasibility (frame side fully parity-compilable, magic side
dense-coupled via amplitude-threshold rank reduction). **K**: the boundary-EDGE / compiled-region FAST path
(cmode5) that reuses d3's few transitions → **parity with Clifft, bit-exact**.

## Honest limits
- d3 result is **parity, not a strict win** — expected, since d3 is structurally unfavorable to near-Clifford.
- The native path is **default OFF**; correctness is always carried by the authoritative Python oracle.
- Frequency pinning needs sudo (unavailable), so multi-second large-N timings carry mild thermal noise; the
  cold-vs-warm-at-fixed-N comparison controls for it.

## Next
1. **High-rank d5_r5** — the structural MDAM-advantage regime (quantify the real win). *(recommended)*
2. (optional) non-noise floor lever to push d3 *strictly* below Clifft (diminishing returns; parity already met).

---
*Detailed per-gate reports, baselines, profiles, and timing JSONs: `/home/jung/mdam-vm-archive/prev-reports/`.*
