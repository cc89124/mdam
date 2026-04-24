# Magic State Cultivation: S-Proxy vs T-Gate Fidelity

Compares the S-gate proxy (Clifford) against the actual T-gate (non-Clifford)
for magic state cultivation, reproducing and extending results from Gidney et al.
["Magic state cultivation: growing T states as cheap as CNOT gates"](https://arxiv.org/abs/2409.17595).

The paper replaces every T gate with an S gate so the circuit can be simulated
efficiently in stim. This workspace simulates the actual T-gate circuit using
[Clifft](https://github.com/unitaryfoundation/clifft) (a non-Clifford stabilizer
simulator) and compares fidelity against the S-proxy baseline.

## Installation

From the `magic_state_cultivation/` directory:

```bash
uv sync
```

This installs the Python dependencies (including the vendored
`magic-state-cultivation` package at `vendor/src`).

Clifft must then be built from source with `CLIFFT_MAX_QUBITS=512` (the
default 128-qubit PyPI wheel does not fit the `d=5` inject+cultivate
circuit, nor any of the end-to-end circuits — the escape stage grows
into a `d=15` surface code with >450 qubits):

```bash
uv pip install \
    --reinstall --no-cache --no-binary clifft \
    --config-settings "cmake.define.CLIFFT_MAX_QUBITS=512" \
    "git+https://github.com/unitaryfoundation/clifft.git@v0.1.0"
```

Verify the build picked up the new limit before running production
simulations:

```bash
uv run python -c "import clifft; print(clifft.max_sim_qubits())"
# -> 512
```

If it prints `128` the build reused a cached wheel — add
`--no-binary clifft --reinstall --no-cache` (as above) and try again.

## Running Simulations

### Phase 1: Inject+Cultivate Error Rate

Uses importance sampling (`sample_k_survivors`) with tiered shot budgets
per stratum. Large strata are sub-chunked across workers for maximum
core utilization. One simulation run covers all noise levels via
Binomial PMF reweighting.

```bash
# Smoke test (~10s)
uv run python run_ic_tiered.py --smoke

# Production: both d=3 and d=5
uv run python run_ic_tiered.py --workers 24

# Production: d=3 or d=5 only
uv run python run_ic_tiered.py --d3-only --workers 24
uv run python run_ic_tiered.py --d5-only --workers 24
```

Per-stratum shot budgets are defined by `D3_TIERS` and `D5_TIERS` in
`run_ic_tiered.py`, set so that strata with larger Binomial weight at
the target noise levels get proportionally more shots (matching
Tuloup & Ayral's methodology). Aggregate totals for the committed
run in this repo are roughly:

| Config       | Strata | Total shots      |
|--------------|-------:|-----------------:|
| d=3 T-gate   |     13 | 1.3 × 10⁸        |
| d=3 S-proxy  |     13 | 1.3 × 10⁸        |
| d=5 T-gate   |     21 | 9.6 × 10¹⁰       |
| d=5 S-proxy  |     18 | 2.8 × 10¹¹       |

Results are saved as per-stratum JSON checkpoints in
`results/inject_cultivate/`. The script automatically resumes from
existing checkpoints.

### Phase 2: End-to-End Analytical Infidelity

Brute-force Monte Carlo at fixed noise levels. Each shot is simulated
via Clifft with `EXP_VAL` probes, decoded with the paper's desaturation
decoder, and frame-corrected. Gap thresholds are swept at plot time
to produce desaturation curves.

Target shot counts per (circuit, noise) configuration:

| Distance | Noise levels | Target shots | Configs | Rationale |
|----------|-------------|-------------|---------|-----------|
| d=3 | 0.0005, 0.001, 0.002 | 1B | 6 (2 circuits x 3 noise) | Resolves gap tails to ~60-65 |
| d=5 T-gate | 0.001 | 100B | 1 | Resolves gap tails to ~80 |
| d=5 T-gate | 0.002 | 289B | 1 | Matches Gidney's S-proxy sample size |
| d=5 S-proxy | 0.001, 0.002 | 25B | 2 | Validated against Gidney's data (<1% deviation) |

Our S-proxy results match Gidney et al.'s to within <1%, so additional
d=5 budget is allocated to T-gate only. Gidney's 1T (p=0.001) and 289B
(p=0.002) S-proxy datasets serve as the baseline for ratio comparisons.

```bash
# Smoke test (~30s, d=3 only)
uv run python run_end2end_bruteforce.py --smoke --noise 0.001

# d=3: 1B shots at each noise level
uv run python run_end2end_bruteforce.py \
    --noise 0.0005 0.001 0.002 \
    --total-shots 1e9 --d3-only --workers 24

# d=5 T-gate p=0.001: 100B shots
uv run python run_end2end_bruteforce.py \
    --noise 0.001 --total-shots 100e9 \
    --d5-only --t-gate-only --workers 48

# d=5 T-gate p=0.002: 289B shots
uv run python run_end2end_bruteforce.py \
    --noise 0.002 --total-shots 289e9 \
    --d5-only --t-gate-only --workers 48
```

Results are saved as one JSON file per (circuit, dcolor, noise) in
`results/end2end/`, containing gap-binned histogram counts (matching
Gidney's stats.csv format). To add more shots, increase `--total-shots`
and re-run — new histogram counts are merged into the existing file.

### Plotting

```bash
uv run python plot_results.py              # all plots
uv run python plot_results.py --ic-only    # IC only
uv run python plot_results.py --e2e-only   # E2E only
```

`plot_results.py` defaults to PNG output; pass `--format pdf` (which
uses `rsmf` for Quantum journal formatting) to produce the PDFs
committed to `figures/` and included in the paper. The PDF path needs
the `plot` optional dependency:

```bash
uv sync --extra plot
uv run python plot_results.py --format pdf
```

Reads saved data from `results/` and produces:
- **IC plots** (`figures/ic_comparison_d{3,5}.{png,pdf}`):
  Error rate vs expected attempts per kept shot for both the T-gate
  and S-proxy circuits, plus a T/S ratio panel with 95% Bayesian
  credible intervals. Separate plots per distance.
- **E2E combined plots** (`figures/e2e_combined_d{3,5}.{png,pdf}`):
  Two-panel figure per distance — desaturation (infidelity vs
  expected attempts per kept shot) on top, T/S infidelity ratio with
  Bayesian credible intervals on bottom, sharing the x-axis. A
  dashed I+C limit line shows the asymptote each E2E curve
  approaches as the gap threshold tightens.

No Clifft needed for re-plotting.

## How It Works

1. **Circuit conversion** (`convert_s_to_t.py`): Transforms stim S-gate proxy
   circuits into T-gate circuits for Clifft. Five transformation rules handle
   S/S_DAG substitution, d=5 errata flips, feedforward, detector healing, and
   EXP_VAL probe insertion.

2. **Importance sampling** (`lib/importance_sampling.py`): Stratified by fault
   count k using Clifft's `sample_k_survivors()`. Binomial PMF reweighting
   enables sweeping noise levels from a single simulation. Delta Method error
   bars for the ratio estimator.

3. **Decoder + frame tracking** (`lib/dual_decoder.py`): Wraps the paper's
   `CompiledDesaturationSampler` (PyMatching with gap confidence). Extracts
   Y_L Pauli frame tracking indices from `OBSERVABLE_INCLUDE` records to
   correct the random lattice surgery frame.

## Repository Structure

```
convert_s_to_t.py          # S-gate to T-gate circuit converter
run_ic_tiered.py           # Phase 1: IC driver (tiered budgets + sub-chunking)
run_end2end_bruteforce.py  # Phase 2: brute-force Monte Carlo driver
consolidate_ic_results.py  # Merge per-stratum IC checkpoints into analysis.json
plot_results.py            # Publication-quality plotting
lib/
    importance_sampling.py # Binomial PMF, ratio estimator, Delta Method
    dual_decoder.py        # Decoder wrapper + Pauli frame tracking
vendor/                    # Vendored code from Gidney et al. (see vendor/README.md)
    src/                   # Circuit generation + decoder
    tools/                 # CLI tools
```

## Vendored Code

The `vendor/` directory contains circuit generation and decoder code vendored
from [Strilanc/magic-state-cultivation](https://github.com/Strilanc/magic-state-cultivation)
(Gidney et al., commit `871e68f`). This code generates the S-gate proxy
circuits and implements the desaturation decoder used for both S-proxy
baseline comparison and T-gate frame correction. See `vendor/README.md`
for details.

## References

- Gidney, Shutty, and Jones, "Magic state cultivation: growing T states as cheap as CNOT gates" ([arXiv:2409.17595](https://arxiv.org/abs/2409.17595))
- Tuloup and Ayral, "Computing logical error thresholds with the Pauli Frame Sparse Representation" ([arXiv:2603.14670](https://arxiv.org/abs/2603.14670))
- Li et al., "SOFT: A High-Performance Simulator for Universal Fault-Tolerant Quantum Circuits" ([arXiv:2512.23037](https://arxiv.org/abs/2512.23037))
