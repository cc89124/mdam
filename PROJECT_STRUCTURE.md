# clifft-paper Project Structure

This documents where the **current MDAM native batch VM implementation** lives and what it depends on.
The current implementation is organized under `mdam/`. The old `nearclifford_backend/native_vm/` layout no
longer holds the implementation (it was moved to `mdam/native_vm/`).

## mdam/

Source of truth for the current MDAM native batch VM implementation.

- **`mdam/native_vm/`** — the Gate K FAST-parity result, the current implementation.
  - `build.sh` rebuilds `native_mdam_vm.so` (byte-identical 387 944 bytes on the reference g++ 11.4).
  - `README.md` documents each file's role, the cmode dispatch table, and how to verify.
  - `native_mdam_vm.cpp` + `native_*.hpp` (15 headers) + the built `.so`; plus the verification scripts
    `verify_mdam_oneshot.py`, `verify_mdam_batch.py`, `gate_k_fast.py`, `gate_k_shadow.py`,
    `gate_k_noise_skip.py`, `gate_k_shot_sweep.py`.
- **`mdam/clifft_axis/cpp/mdm_core_executor.cpp`** — the dense measurement-core kernel the native VM build
  links (`mdm_execute_core`). The native VM build depends on this path:
  `mdam/native_vm/build.sh` → `../clifft_axis/cpp/mdm_core_executor.cpp`.
  It is a **vendored copy** of the canonical `nearclifford_backend/clifft_axis/cpp/mdm_core_executor.cpp`;
  the kernel is self-contained (stdlib only). The canonical file is kept in place because the live
  clifft_axis Python engine builds its own `mdm_core_release.so` from it.

### Runtime oracle (live, NOT under mdam/, NOT archived)

`mdam/native_vm` is the C++ *implementation* that ports and verifies against the authoritative Python
backend. Those Python packages are **live** and remain at the repo root:

- **`nearclifford_backend/`** — the authoritative near-Clifford backend (`backend.py`, `simulator.py`,
  `block_magic.py`, `lazy.py`, and the `clifft_axis/` engine). The verify scripts import
  `nearclifford_backend.backend` (`_opname`, `count_idents`). 129 files repo-wide depend on this package
  and the live clifft_axis engine builds the dense-core `.so` from it — so it is **not** archived or renamed.
- **`ttn_backend/`** — the frame layer (`ttn_backend.frame_layer`) used by `translate()`.

(Verify scripts resolve these via `sys.path` = repo root; `mdam/native_vm` sits exactly two levels under the
root, so the move from the old layout did not change any `__file__`-relative path.)

## artifacts/mdam_native_batch_vm/

Current-result summary. **`RESULTS.md`** only — the consolidated one-page result (Gate A–K journey in brief
+ the Gate K parity result + the shot-sweep audit). The detailed per-gate records were moved to
`/home/jung/mdam-vm-archive/prev-reports/`.

## qec_bench/

Benchmark circuits / data. **Not a cleanup target. Never moved or modified.**

## archive — `/home/jung/mdam-vm-archive/` (outside the repo)

Preserves the development history (moved, not deleted; recoverable):
- `prev-scripts/` — earlier experiment scripts (Gates D–J + transitional Gate K).
- `prev-builds/` — `.so` backups + old/auxiliary build TUs.
- `prev-tests/` — component unit tests + reference data + generators.
- `prev-reports/` — the detailed per-gate reports.

> Note: a separate, pre-existing `clifft-paper/archive/` (an older `nearclifford_backend` snapshot) is
> unrelated to this cleanup and was left untouched.

---

## Deviation from the original cleanup spec (surfaced honestly)

The original spec assumed `nearclifford_backend` ≈ the native VM folder and asked to (a) rewrite
`from nearclifford_backend …` → `from mdam …` and (b) move the whole `nearclifford_backend/` to archive.
Investigation showed `nearclifford_backend` is the **live authoritative backend** (129 dependent files; the
live clifft_axis engine builds the dense-core `.so` from it; `ttn_backend` is a separate live package).
Applying (a)/(b) would break the oracle, the dense-core build, and the broader research code. So:
**`nearclifford_backend` and `ttn_backend` were kept in place (live oracle), the native VM was moved to
`mdam/native_vm/`, and the dense core was vendor-copied into `mdam/clifft_axis/cpp/`.** Build + smoke test
pass from the new path.
