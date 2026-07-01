# clifft-paper — Project Structure

This repo is organized around **`mdam`**, the MDAM near-Clifford simulator. Everything that is part of the
implementation lives under `mdam/`; experiment inputs and results are separated out. The latest implementation
is the **native batch VM** (`mdam/native_vm/`), which verifies bit-exact against the in-tree Python oracle.

```
clifft-paper/
  mdam/                         # THE implementation (everything needed to run)
    frame/                      #   frame layer / TTN base   (was: ttn_backend)
    backend/                    #   near-Clifford backend     (was: nearclifford_backend)
      clifft_axis/cpp/          #     dense measurement-core kernel (mdm_core_executor.cpp)
    native_vm/                  #   C++ native batch VM — the latest impl (Gate K parity result)
  qec_bench/                    # experiment INPUT: benchmark circuits (cultivation_d3.stim, …)
  results/                      # experiment RESULTS: RESULTS.md (consolidated)
  PROJECT_STRUCTURE.md  README.md  LICENSE
```

## mdam/ — the implementation

Three layers, bottom-up (single clean dependency direction `frame → backend → native_vm`):

- **`mdam/frame/`** *(was `ttn_backend`)* — the Pauli frame layer + TTN base. The native VM verifies against
  `mdam.frame.frame_layer`. Core modules: `frame_layer`, `clifford_frame`, `core`, `treewidth` (+ a few
  supporting modules).
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
  `native_mdam_vm.so` (byte-identical 387 944 bytes). See `mdam/native_vm/README.md` for the cmode table and
  per-file roles. Verification harness: `verify_mdam_oneshot.py`, `verify_mdam_batch.py`, `gate_k_*.py`.

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
