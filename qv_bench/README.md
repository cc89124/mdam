# Quantum Volume Benchmark

Benchmarks Clifft against Qiskit-Aer, Qulacs, qsim, and Qrack on random
**Quantum Volume** circuits, scaling qubit count from 6 to 28. These
circuits are dense with non-Clifford gates, which is not the primary near-Clifford circuit target of Clifft.

Inspired by  Niekerk L van, Kumar D, Sharma AK, Meisel T, Paleico ML, Boehme C (2024) A comparison of HPC-based quantum computing simulators using Quantum Volume. https://doi.org/10.48550/arXiv.2412.20518.

## Dependencies

Install from the `qv_bench/` directory:

```bash
uv sync
uv sync --extra plot  # only needed for plot_qv.py
```

Baseline dependencies in this workspace include `qiskit`, `qiskit-aer`,
`qulacs`, `qsimcirq`, `ply`, `qiskit-qrack-provider`, and `pyqrack`.

Install `clifft` separately into the same `uv` environment before running
the `clifft` backend. This workspace builds from source (pinned to the
`v0.1.0` tag) so that `CLIFFT_MAX_QUBITS` can be set to 64, which is
faster than the 128-qubit PyPI wheel for the qubit range used here
(≤28 qubits):

```bash
uv pip install \
    --reinstall --no-cache --no-binary clifft \
    --config-settings "cmake.define.CLIFFT_MAX_QUBITS=64" \
    "git+https://github.com/unitaryfoundation/clifft.git@v0.1.0"
```

Verify the build picked up the new limit:

```bash
uv run python -c "import clifft; print(clifft.max_sim_qubits())"
# -> 64
```

If it prints `128`, the install reused a cached PyPI wheel; the
`--no-binary clifft --reinstall --no-cache` flags above force a
fresh build.  If you'd rather just use the 128-qubit PyPI wheel,
`uv pip install "clifft==0.1.0"` is functionally sufficient — the
source build with `CLIFFT_MAX_QUBITS=64` is only for a small
performance gain in the 6–28 qubit range.

## Running

All commands run from the `qv_bench/` directory.

```bash
# Quick test
uv run python -m qv_bench --min-q 6 --max-q 12 --repeats 1 --simulators clifft,qiskit

# Specific qubit counts
uv run python -m qv_bench --qubits 10,14,18,22

# Single simulator
uv run python -m qv_bench --simulators clifft

# Run for data in paper (16-threaded)
uv run python -m qv_bench \
    --min-q 6 --max-q 28 --step 2 \
    --mem-limit-gb 10 --timeout 600 \
    --threads 16 --repeats 3
```

Results are written to `results.csv` (or a custom path via `--output`).

Each `(simulator, N, seed)` combination runs in an isolated subprocess to
capture peak RSS memory and prevent GC drift between runs.

## Committed results

`results.csv`/`qv_scaling.pdf` is the 16-threaded run used for the paper figure, collected
on an AWS c8i.8xlarge (Intel Xeon 6, Granite Rapids).

## Plotting

```bash
uv run python -m qv_bench.plot_qv
```

Produces a log-scale execution time vs qubit count plot.

## Validation

```bash
uv run python -m qv_bench.validate_hop
```

Validates Clifft's statevector output against Qiskit-Aer using fidelity
checks and Heavy Output Probability (HOP) computation.

## CLI options (run_benchmark.py)

| Flag | Default | Description |
|------|---------|-------------|
| `--min-q` | `6` | Minimum qubit count |
| `--max-q` | `26` | Maximum qubit count |
| `--step` | `2` | Step size for qubit range |
| `--qubits` | — | Explicit comma-separated qubit counts (overrides min/max) |
| `--repeats` | `3` | Repetitions per configuration |
| `--simulators` | `clifft,qiskit,qulacs,qsim,qrack` | Comma-separated backends |
| `--mem-limit-gb` | `6.0` | Per-worker memory cap (RLIMIT_AS) |
| `--timeout` | `300` | Per-worker timeout in seconds |
| `--output` | `results.csv` | Output CSV path |
| `--seed` | `42` | Base RNG seed |
| `--threads` | `1` | Thread count exposed to simulator backends |

## Files

| File | Description |
|------|-------------|
| `qv_bench/run_benchmark.py` | Orchestrator — spawns subprocess workers, collects CSV |
| `qv_bench/worker.py` | Subprocess worker for a single (simulator, N, seed) run |
| `qv_bench/generator.py` | Random QV circuit generation (QASM 2.0) |
| `qv_bench/qasm_adapter.py` | Converts QASM to Clifft/stim, Qulacs, Cirq formats |
| `qv_bench/plot_qv.py` | Publication-ready scaling plot |
| `qv_bench/validate_hop.py` | Statevector fidelity + HOP validation |
