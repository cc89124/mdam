# QEC Benchmarks

Reproducible QEC benchmark scripts for the Clifft paper. This simulates Clifford and near-Clifford circuits.

These benchmarks were inspired by those in Haenel R, Luo X, Zhao C (2026) Tsim: Fast Universal Simulator for Quantum Error Correction. https://scirate.com/arxiv/2604.01059.

## Installation

From the `qec_bench/` directory:

```bash
uv sync
```

This installs [`clifft`](https://github.com/unitaryfoundation/clifft) from
PyPI (pinned to `v0.1.0`). The default PyPI wheel is built with
`CLIFFT_MAX_QUBITS=128`, which is sufficient for every circuit in this
workspace (the largest is the 118-qubit `d=7 r=7` surface code).

Install tsim (GPU machines only):

```bash
uv add "bloqade-tsim[cuda13]"
```

### Optional: build stim from source with AVX2

The default PyPI stim wheel ships only SSE2 (128-bit); the AVX2 variant is disabled upstream pending
[stim#432](https://github.com/quantumlib/Stim/issues/432).  On these
workloads the measured gap between SSE2 and a from-source AVX2 build
is ~2% after chunking (the `StimRunner` in `bench_common.py` already
chunks to keep the frame simulator's shot-major working set in L2),
so this is only needed if you want to eliminate SIMD width as a
variable.

```bash
# From a clone of https://github.com/quantumlib/Stim at the desired tag:
#   - uncomment the stim_avx2 Extension in setup.py
#   - uncomment the `if _tmp == 'avx2':` branch in
#     glue/python/src/stim/__init__.py
# then:
uv pip install "pybind11~=2.11.1"
uv pip install --no-build-isolation -e .
```

Verify the AVX2 variant is what loads at runtime:

```bash
python -c "import sys, stim; \
  print([m for m in sys.modules if m.startswith('stim._stim_')])"
# -> ['stim._stim_avx2']
```

## Workflow

### Step 1: Probe tsim (GPU machine)

Compile and probe-sample every tsim circuit, then save the best
strategy per circuit:

```bash
uv run python -m tsim_compile_check
```

This writes `configs/tsim_modes.json`. Commit it to the repo.

### Step 2: Run tsim benchmarks (GPU machine)

```bash
uv run python -m run_all tsim
```

### Step 3: Run clifft benchmarks (CPU machine)

```bash
uv run python -m run_all clifft
```

### Step 4: Run stim baseline (CPU machine)

```bash
uv run python -m run_all stim
```

Results are written to `results/<benchmark>.csv`.

`run_all` refuses to overwrite an existing non-empty results CSV
to protect the committed reference data.  Pass `--force` to allow
the overwrite, or `--results-dir <path>` to write elsewhere.

### Smoke testing

Use a scratch directory so the committed paper results stay intact:

```bash
uv run python -m run_all stim   --shots 1000 --repeats 1 --results-dir /tmp/smoke
uv run python -m run_all clifft --shots 1000 --repeats 1 --results-dir /tmp/smoke
```

## Paper matrix

Defined in `run_all.py` (`_build_matrix`). Shots are per-circuit
(calibrated to peak active rank), 3 repeats.

| Benchmark | Circuit | Shots | clifft | stim | tsim |
|-----------|---------|------:|--------|------|------|
| clifford_bench | d=7, r=7, p=1e-3 | 1M | yes | yes | yes |
| cultivation_bench | d=3, p=1e-3 | 1M | yes | | yes |
| cultivation_bench | d=5, p=1e-3 | 1M | yes | | yes |
| distillation_bench | prep=0.05 | 1M | yes | | yes |
| coherent_noise_bench | d=3, r=1, p=1e-3, rz=0.02 | 1M | yes | | yes |
| coherent_noise_bench | d=3, r=3, p=1e-3, rz=0.02 | 1M | yes | | yes |
| coherent_noise_bench | d=5, r=1, p=1e-3, rz=0.02 | 100k | yes | | yes |
| coherent_noise_bench | d=5, r=5, p=1e-3, rz=0.02 | 20 | yes | | yes |

## Layout

| Path | Purpose |
|------|---------|
| `run_all.py` | Circuit generation, benchmark matrix, and runner |
| `tsim_compile_check.py` | Probe tsim compilation, save `configs/tsim_modes.json` |
| `bench_common.py` | Shared runner classes and timing loop |
| `circuits/` | Vendored circuit templates (cultivation, distillation) |
| `configs/` | Saved tsim probe results (repo-tracked) |
| `results/` | Benchmark CSV outputs |

