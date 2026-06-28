# mdam/native_vm — MDAM near-Clifford native batch VM

Location (source of truth): `/home/jung/clifft-paper/mdam/native_vm/`. See repo-root `PROJECT_STRUCTURE.md`.

A C++ batch VM that runs MDAM (near-Clifford magic-state) QEC sampling **end-to-end in native code**,
bit-exact to the authoritative Python runtime. The native path is **default OFF**; the Python path
(`run_shot` / `sample_batch`) is the reference oracle and is never modified by the native code.

Result on cultivation_d3: **~1.08–1.10× Clifft, bit-exact, robust across shot counts** (Gate K FAST path).
Full result write-up: [`../../artifacts/mdam_native_batch_vm/RESULTS.md`](../../artifacts/mdam_native_batch_vm/RESULTS.md).
Development journey (Gates A–K detail) is archived at `/home/jung/mdam-vm-archive/`.

## Dependencies (what this connects to)
- **Build** (C++ only, self-contained under `mdam/`): `native_mdam_vm.cpp` + the dense-core kernel
  `../clifft_axis/cpp/mdm_core_executor.cpp`. That `.cpp` is a **vendored copy** of the canonical
  `nearclifford_backend/clifft_axis/cpp/mdm_core_executor.cpp` (the live clifft_axis engine still builds
  its own `mdm_core_release.so` from the canonical file). The kernel includes only stdlib headers.
- **Verify / runtime oracle** (Python, live, unchanged): the verify scripts import the authoritative
  near-Clifford backend `nearclifford_backend.backend` (`_opname`, `count_idents`) and
  `ttn_backend.frame_layer` to `translate()` a circuit into the native program. These packages live at the
  repo root and are **NOT** part of `mdam/` — `mdam/native_vm` is the C++ *implementation* that ports and
  verifies against them. (Scripts resolve them via `sys.path` = repo root; `mdam/native_vm` sits exactly two
  levels under the root, so all `__file__`-relative paths are unchanged by the move from the old layout.)

## Build

```bash
./build.sh        # g++ -O3 -march=native -std=c++17 -DNDEBUG -shared -fPIC
                  # native_mdam_vm.cpp + ../clifft_axis/cpp/mdm_core_executor.cpp -> native_mdam_vm.so
```
Reference toolchain (g++ 11.4) produces a byte-identical 387 944-byte `native_mdam_vm.so`.

## Run / verify

All scripts assume single-thread + core pinning:
```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  taskset -c 2 /home/jung/clifft_env/bin/python <script>.py
```

| script | what it does |
|---|---|
| `verify_mdam_oneshot.py` | **core utility** — `translate`/`make_prog`/`pcg`/`BENCH`; oneshot bit-exact vs Python. Imported by all others. |
| `gate_k_fast.py` | **core utility** — `bind()` (ctypes binding of `native_mdam_vm.so`) + the Gate K FAST (cmode5) ladder check. |
| `gate_k_shadow.py` | Gate K SHADOW (cmode4): proves every boundary edge reproduces the live boundary bit-exact (25/25 + 128k 0). |
| `gate_k_noise_skip.py` | skip-to-next-fire noise scheduling: bit-exact + counters (site_calls 504→0.5) + wall. |
| `gate_k_shot_sweep.py` | fair warmup-amortization sweep (1k/8k/32k/128k/1M), cold vs warm vs Clifft, rep-interleaved. |
| `verify_mdam_batch.py` | full-batch bit-exact verification vs Clifft / authoritative. |

## Implementation files (the `native_mdam_vm.so` build)

Single TU `native_mdam_vm.cpp` (the C API: `nvm_*` create/run/batch/compile + cmode dispatch) +
the verified dense kernel `../clifft_axis/cpp/mdm_core_executor.cpp` (`mdm_execute_core`).
Header layers, bottom-up:

| header | role |
|---|---|
| `native_rng.hpp` | bit-exact numpy PCG64 Generator |
| `native_seed_expand.hpp` | numpy SeedSequence → PCG64 seeding |
| `native_frame.hpp` | dormant Clifford Pauli frame (`PauliFrame`) |
| `native_record.hpp` | measurement record |
| `native_noise.hpp` | Clifft noise gap-sampler + apply-site |
| `native_tableau.hpp` | U_C stabilizer tableau (Xc/Zc) right-folds |
| `native_pending.hpp` | pending ledger + packed Pauli conjugation |
| `native_inverse_frame.hpp` | incremental inverse-frame (O(weight) pullback) |
| `native_dense.hpp` | dense-state buffer for the measurement core |
| `native_magic_state.hpp` | composite near-Clifford dense-engine state |
| `native_magic_measure.hpp` | compiled magic measure (`try_compiled_measure`) |
| `native_oracle_measure.hpp` | oracle `measure_z` magic branch (incl. stabilizer ag_measure) |
| `native_mdam_shot.hpp` | full one-shot: opcode loop + magic core + Gate K edge cache (`MdamShot`) |
| `native_compiled_region.hpp` | Gate J/K region-compiled sampler: `run_jfast_2e` (cmode 1–5), noise skip, compiler+shadow |
| `native_instr.hpp` | optional control-plane instrumentation (`-DMDAM_INSTR`, default-off, profiling only) |

### cmode dispatch (in `run_jfast_2e`)
`0`=authoritative · `1`=SHADOW · `2`=FAST-2F · `3`=FAST-2G(BoundaryPlan) · `4`=Gate-K edge-cache SHADOW
· `5`=Gate-K edge-cache FAST (the parity result). Toggles: `nvm_mdam_vm_set_imem`, `nvm_mdam_vm_set_fb`,
`nvm_j2e_noise_skip` (default ON; affects only j-fast modes — authoritative path stays per-site).
