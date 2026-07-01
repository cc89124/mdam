# clifft-paper — MDAM near-Clifford simulator

This repo is organized around **`mdam`**, the MDAM near-Clifford simulator. The latest implementation is the
**native batch VM** at [`mdam/native_vm/`](mdam/native_vm/), which runs cultivation_d3 magic-core sampling
end-to-end in native C++, **bit-exact** to the in-tree Python oracle, reaching **~1.08–1.10× Clifft (parity)**.

See [`PROJECT_STRUCTURE.md`](PROJECT_STRUCTURE.md) for the full layout and [`results/RESULTS.md`](results/RESULTS.md)
for the result write-up.

## Quick start

```bash
cd mdam/native_vm
./build.sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  taskset -c 2 /home/jung/clifft_env/bin/python gate_k_shadow.py
```

## Layout (summary)

- `mdam/` — the implementation: `frame/` (frame layer, was ttn_backend), `backend/` (near-Clifford backend,
  was nearclifford_backend; holds the dense-core kernel `clifft_axis/cpp/`), `native_vm/` (the C++ native VM).
- `qec_bench/` — benchmark circuits (experiment input).
- `results/` — consolidated results (`RESULTS.md`).

> External dependency: `clifft` (the reference simulator, installed separately) is imported by the verify scripts.
