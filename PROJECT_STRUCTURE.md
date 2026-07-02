# clifft-paper — Project Structure

This repo is organized around **`mdam`**, the MDAM near-Clifford simulator. Everything that is part of the
implementation lives under `mdam/`; experiment inputs and results are separated out. The latest implementation
is the **native batch VM** (`mdam/native_vm/`), which verifies bit-exact against the in-tree Python oracle.

```
clifft-paper/
  mdam/                         # THE implementation (everything needed to run)
    frame/                      #   Pauli/Clifford frame layer   (was: ttn_backend)
    backend/                    #   near-Clifford backend     (was: nearclifford_backend)
      clifft_axis/cpp/          #     dense measurement-core kernel (mdm_core_executor.cpp)
    native_vm/                  #   C++ native batch VM — auth + lean + adaptive executor
    MDAM_auth_vs_lean.md        #   auth vs lean execution paths: principle · when each wins · decision rule
    MDAM_localized_computation.md #  localization write-up
  qec_bench/                    # experiment INPUT: benchmark circuits + BENCHMARKS.md (per-circuit doc)
  results/benchmark_comparison/ # wall_table.tsv/.md — auth + lean columns + best_path per bench
  distillation_scaling/         # scaling study (distillation-STYLE family; RESULTS.md has honest caveats)
  results/                      # experiment RESULTS: RESULTS.md (consolidated)
  PROJECT_STRUCTURE.md  README.md  LICENSE
```

## mdam/ — the implementation

Three layers, bottom-up (single clean dependency direction `frame → backend → native_vm`):

- **`mdam/frame/`** *(was `ttn_backend`)* — the Pauli/Clifford frame layer (the `U_C` in `|ψ⟩=U_C|χ⟩`). The
  native VM uses **only** `mdam.frame.frame_layer` (for verification). The package is the renamed old `ttn_backend`
  and still contains a legacy tree-tensor-network backend (`core.py`'s `TTNBackend`) from an earlier approach that
  the current near-Clifford path (auth/lean/adaptive) does **not** use.
- **`mdam/backend/`** *(was `nearclifford_backend`)* — the authoritative near-Clifford backend (the Python
  **oracle** the native VM is verified against, via `be.run_shot`). Modules: `backend`, `block_magic`,
  `lazy`, `simulator`, plus the magic-core engine `clifft_axis/` (Python: `bounded`, `engine`, `policy3`,
  `compiled_*`, …) and `virtual_axis/`. The native VM's C++ *production* path does not import these, but the
  verify harness (`verify_mdam_oneshot.py` / `verify_mdam_batch.py`) runs `run_shot` through the bounded
  engine as the reference, so they are required for verification.
  - **`mdam/backend/clifft_axis/cpp/`** — the dense measurement-core kernel. `native_vm/build.sh` links
    `mdm_core_executor.cpp` (`mdm_execute_core`); it is self-contained (stdlib only).
- **`mdam/native_vm/`** — **the latest implementation.** A C++ batch VM that runs the magic-core sampling
  end-to-end in native code, **bit-exact** to the Python oracle, native path default OFF. `./build.sh` rebuilds
  `native_mdam_vm.so`. See `mdam/native_vm/README.md` for the cmode table and per-file roles. Verification
  harness: `verify_mdam_oneshot.py`, `verify_mdam_batch.py`, `gate_k_*.py`, `verify_adaptive.py`.

## Execution paths (auth / lean / adaptive)

There are three runtime paths in the native VM; **`auth` and the current `lean` are behaviorally unchanged**,
the `adaptive` path is a new opt-in unifier (default OFF). Full write-up: `mdam/MDAM_auth_vs_lean.md`.

- **`auth` (authoritative)** — `nvm_mdam_sample_batch` / `run()`. No cache, exact per-shot, constant time. The
  bit-exact reference. Wins via `r≪k` localization (coherent benches: e.g. d7_r1 ~10⁴× vs the Clifft paper baseline).
- **`lean`** — `nvm_run_lean_fb_batch` / `run_lean` + miss-fallback to `run_mcache` (best-stack). Skips the engine
  gate-walk via a magic-core boundary **automaton built lazily at runtime** (no offline prefill; every shot is a
  real output). Wins iff the automaton **saturates** (distillation, cult_d3, rx_d3). Flags default OFF.
- **`adaptive`** — `nvm_run_lean_adapt_batch` (this session). **Exactly two production policies, `LEAN` and
  `AUTH`.** `run_mcache` is **not** a third policy — it is only the LEAN-miss recovery fallback (recover the shot +
  fill the cache). Lean-optimistic start; on judging the cache won't close, **sticky demote straight to `AUTH`
  (`run()`, == `sample_batch`)** — *not* `run_mcache` (which kept interning → OOM and is slower than auth on
  localization). On demote it stops shadow interning (`sg_shadow=0`) and **frees the lean tables + the dense-core
  mcache (`mc_pool`)**, so AUTH runs at constant memory. Demote fires on: (1) a **memory-budget check** (every 64
  shots, O(1) via a running dense-core byte counter): cache bytes > `ad_mem_cap` (512 MB) **and** still
  non-saturating. This is the **single OOM/size backstop** — it counts lean-table bytes too, so a node-table
  explosion trips it as well (the old node/edge count cap was removed as redundant). No fb gate: it catches both
  heavy-core all-miss (`coherent_d5_r5`, ~3.75 MB/shot → AUTH@191, RSS 1.7 GB vs prior SIGKILL) **and** light-core
  non-saturating caches (`cultivation_d5`, ~0.029 MB/shot → AUTH@~20k, peak RSS 4.5 GB → 0.86 GB, bounding a cache
  that otherwise balloons to multi-GB); (2) a **`!engine.magic_ever` early-localization demote** in the first window
  (never-materialized-magic + ~all-miss + growing node table) for maxM=0 pure localization (`d7_r1`/`d5_r1` →
  AUTH@4095, recovering most of auth 27371×/5.59×, up from run_mcache-bound 10145×/1.94×); or (3) the conservative
  perf path (past horizon **and** `node_rate` above floor **and** windowed lean cost > slow cost).
  Each gate is load-bearing: the memory gate's *non-saturating* condition spares small saturating winners
  (`distillation`, `cultivation_d3` — caches too small to reach the budget), `!magic_ever` separates localization
  from magic winners, and the fb gate within it keeps saturating pure-Clifford circuits (`surface_d7_r7`,
  `coherent_d3_r1`, fb=0) in LEAN. Bounding the cache keeps shot-parallel memory at `budget × workers` rather than
  unbounded. **Output is bit-identical to `lean`/auth**; policy only changes speed. Config: `nvm_adapt_config` (+ `ad_mem_cap`,
  `ad_fb_demote`); stats: `nvm_adapt_stats` (**14 doubles** — callers allocate ≥`D*14`). Verified by
  `verify_adaptive.py` (bit-exact across the switch + cult_d3/distillation/cult_d5 stay LEAN + demote fires) and
  `check_demote_auth.py` (d5_r5/d7_r1/d5_r1: no OOM, tables freed, recover auth). Segment-level mid-shot handoff /
  node-snapshot deopt is **deferred**; SIMD multi-shot is the separate speed axis for the saturating benches.

> The `wall_table` reports `best_path = argmin over {auth, lean, adapt}` as an **oracle** (measure all, pick best);
> the adaptive path is the runtime realization for the uniform-profile circuits in the current suite, and for a few
> rows (e.g. `coherent_d3_r3`) the demote-to-lazy-AUTH it finds is itself the best measured path. Clifft is an
> **external paper baseline only** — never referenced inside MDAM; speedups are reported as factual ratios, losses
> included, no "beats Clifft" framing.

> Naming: `ttn_backend` → `mdam.frame`, `nearclifford_backend` → `mdam.backend` (imports rewritten across the
> tree). Those old top-level package names no longer exist.

## qec_bench/ — experiment inputs

Benchmark circuits. The native VM reads `qec_bench/circuits/cultivation_d3.stim`. Not modified.

## results/ — experiment results

`RESULTS.md` — the consolidated one-page result (Gate A–K journey + the Gate K parity result + shot-sweep audit).

## Build & verify

```bash
cd mdam/native_vm
./build.sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  taskset -c 2 /home/jung/clifft_env/bin/python gate_k_shadow.py   # cmode4 == authoritative, 25/25 + 128k 0
```

## Cleanup note (recovery)

A large cleanup removed material unrelated to the MDAM native VM (other paper workspaces and accumulated
experiment reports). Everything was snapshotted first in git checkpoint commit **`3772f37`** and is fully
recoverable:
```
git checkout 3772f37 -- <path>      # restore a specific deleted file/dir
git reset --hard 3772f37            # restore the entire previous layout
```
Removed (recoverable from `3772f37`): `qv_bench/`, `magic_state_cultivation/`, `temporal_carving/`,
`benchmarks/`, `archive/`, `tests/`, `verify_data/`, `reports/`, `reports_archive/`,
`reports_clifft_axis_bounded/`, `reports_clifft_axis_bounded_rxry/`, `artifacts/` (its `RESULTS.md` was moved
to `results/`), and the experiment-script subtrees inside the packages (`mdam/frame/{scripts,docs,rasl,tests}`,
`mdam/backend/scripts`). The `clifft_axis` Python engine and `virtual_axis` were initially removed but
**restored** once verification showed they are part of the Python oracle (`run_shot` → bounded engine).
```
