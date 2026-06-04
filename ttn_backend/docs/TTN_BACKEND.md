# Clifft TTN Backend Implementation Notes

This document describes the current Python TTN backend in this repository. The
implementation is a prototype backend for validating Clifft bytecode sampling
against `clifft.sample`.

For the current end-to-end method, including tensor conventions, operation
dispatch, adjacent transport, actual memory metrics, and recent peak-compression
results, see `TTN_METHOD_DETAILED.md`.

## Directory Update

TTN 관련 구현/분석/검증 파일은 이제 루트에 흩어져 있지 않고
`ttn_backend/` package 아래에 정리되어 있다.

```text
ttn_backend/core.py              # TTNState, TTNBackend runtime
ttn_backend/backend_spec.py      # backend spec / homing / op classification
ttn_backend/frame_layer.py       # Pauli frame + noise helpers
ttn_backend/treewidth.py         # active graph / treewidth / JT construction
ttn_backend/layout_transform.py  # layout transforms
ttn_backend/rasl/                # RASL symplectic/candidate/cost modules
ttn_backend/scripts/             # reports and experiment drivers
ttn_backend/tests/               # unit tests
ttn_backend/docs/                # detailed notes
```

권장 실행 방식은 module 실행이다.

```bash
cd /home/jung/clifft-paper
/home/jung/clifft_env/bin/python -m ttn_backend.tests.test_ttn_transport
/home/jung/clifft_env/bin/python -m ttn_backend.scripts.metrics_report --help
```

기존 핵심 import는 package `__init__.py`에서 유지한다.

```python
from ttn_backend import TTNBackend, TTNState
```

## File Layout

- `ttn_backend/backend_spec.py`: replays Clifft bytecode structurally and exports a static
  junction-tree layout, active ident lifetimes, operation classes, and homing.
- `ttn_backend/frame_layer.py`: classical/Pauli frame support shared by the TTN backend.
  This replaces the old dense-state reference code.
- `ttn_backend/core.py`: tensor tree network state, local tensor updates, adjacent
  transport sweeps, and full bytecode dispatch.
- `ttn_backend/scripts/verify_ttn.py`: validation/benchmark driver.

The backend assumes a patched Clifft Python binding exposing constant-pool data
needed to match `clifft.sample`, especially `Program.pauli_masks`,
`Program.noise_sites`, and `Program.readout_noise`.

## Memory Metrics

`union["sum2"]` is a structural lower bound, not a runtime memory claim. It is
the sum of `2^|B|` over junction-tree bags and implicitly assumes every bond
dimension is 1.

The executable exact TTN memory model is bond-aware. A bag tensor has size
`2^|own(B)| * prod_e chi_e`, where `chi_e` is the observed bond dimension on
each incident edge. In the separator-saturated worst case, `chi_e <= 2^|sep_e|`,
so the static exact upper estimate is:

`M_separator_worst = sum_B 16 * 2^|own(B)| * prod_{e in N(B)} 2^|sep_e|`.

The runtime tracks:

- `peak_stored_bytes`: largest sum of all bag tensor byte sizes.
- `peak_pair_workspace_bytes`: largest adjacent two-bag transport workspace.
- `max_bond_dim_observed`: largest bond dimension observed.
- `top5_bag_sizes`: largest individual bag tensors at peak stored memory.
- `top5_pair_workspace`: largest transport workspaces.

## Tensor Size Formula

For a TTN bag `B` at time `t`, let:

- `p_B(t)`: number of active physical axes directly stored in `B`.
- `chi_e(t)`: actual bond dimension on incident edge `e`.
- `r_e(t) = log2 chi_e(t)`.

Then:

```text
N_B(t) = 2^p_B(t) * prod_{e~B} chi_e(t)
log2 N_B(t) = p_B(t) + sum_{e~B} r_e(t)
M_store(t) = 16 * sum_B N_B(t)
```

Workspace for an opened connected region `R` is:

```text
N_R(t) = 2^p_R(t) * prod_{e in boundary(R)} chi_e(t)
log2 N_R(t) = p_R(t) + sum_{e in boundary(R)} r_e(t)
```

The primary diagnosis is therefore to decompose the actual peak offender bag
into `p_B` and incident bond exponents. Use:

```bash
/home/jung/clifft_env/bin/python -m ttn_backend.scripts.memory_diagnosis_report \
  coherent_d5_r1 coherent_d5_r5 \
  --variants baseline \
  --runtime-timeout 60
```

This writes:

- `reports/ttn_memory_diagnosis_summary.csv`
- `reports/ttn_memory_diagnosis_edges.csv`
- `reports/ttn_memory_diagnosis.json`
- `reports/ttn_memory_diagnosis.md`

The edge table also compares allocated `chi_e` against a local adjacent two-bag
SVD rank when the matrix is small enough to factor exactly.
- `n_qr` and `n_transports`.

The gap between `M_static = union["sum2"] * 16` and `peak_stored_bytes` is the
bond-growth cost. High-degree hub bags can create multiplicative memory growth
even when the structural treewidth lower bound is small.

## Memory Risk Model

`memory_risk_report.py` computes the compile-time memory-risk exponents used to
identify the offending bags and edges:

```text
store_exp(B) = own_count(B) + sum(separator_bits(e) for e incident to B)
ws_exp(A,B) = own_count(A) + own_count(B)
            + sum(separator_bits(e) for e incident to A except A-B)
            + sum(separator_bits(e) for e incident to B except A-B)
R_store = max_B store_exp(B)
R_workspace = max_(A,B) ws_exp(A,B)
R_mem = max(R_store, R_workspace)
```

The report writes:

```bash
python3 memory_risk_report.py --variants baseline,hub3 --include-runtime \
  --out-csv reports/memory_risk.csv \
  --out-json reports/memory_risk.json \
  --out-md reports/memory_risk_summary.md
```

The JSON and Markdown include top-10 store offenders and top-10 workspace
offenders per circuit and layout variant. Runtime comparison uses observed
per-edge maximum bond dimensions to compute an observed store exponent upper
bound.

Current headline examples:

| Circuit | Variant | `R_store` | `R_workspace` | `R_mem` | runtime observed `R_store` |
| --- | --- | ---: | ---: | ---: | ---: |
| `coherent_d5_r5` | baseline | 120 | 158 | 158 | 23 |
| `coherent_d5_r5` | hub3 | 25 | 32 | 32 | 24 |
| `coherent_d7_r1` | baseline | 35 | 41 | 41 | 27 |
| `coherent_d7_r1` | hub3 | 34 | 41 | 41 | 24 |
| `coherent_d7_r7` | baseline | 412 | 538 | 538 | n/a |
| `coherent_d7_r7` | hub3 | 52 | 65 | 65 | 27 |

For `coherent_d5_r5` baseline, bag `B0` dominates store risk:
`own=9`, `degree=43`, `store_exp=120`. The largest workspace offender is edge
`B0-B4` with `ws_exp=158`. This confirms that the main problem is a small number
of high-degree hub products, not the structural `M_static` term.

`metrics_report.py` writes the paper data backbone:

```bash
python3 metrics_report.py --runtime-timeout 60 --variants baseline \
  --out-csv reports/baseline.csv --out-json reports/baseline.json

python3 metrics_report.py --runtime-timeout 60 --variants hub3 \
  --hub-degree-threshold 3 \
  --out-csv reports/hub3.csv --out-json reports/hub3.json
```

The CSV contains one row per circuit/layout variant. The JSON includes the same
row data plus `top5_bag_sizes`, `top5_pair_workspace`, and static top separator
bags.

`layout_transform.reduce_hub_degree()` provides a first conservative hub-degree
reduction baseline. It replaces a high-degree bag with a chain of copy bags that
preserve the original vertex set. This keeps operation coverage valid and lowers
degree, but it can increase separator-saturated bounds because internal copy
edges carry large separators. It is therefore a diagnostic baseline, not yet a
final bond-aware optimizer.

## Negative Result: Hub Degree Only

The naive `hub3` transform lowers graph degree but does not solve exact TTN
memory. On the current reports:

| Circuit | Variant | `D_max` | `M_static` MB | `M_sep_worst` MB | runtime peak MB | status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `coherent_d5_r5` | baseline | 43 | 0.068 | 2.127e31 | 134.224 | timeout |
| `coherent_d5_r5` | hub3 | 3 | 0.466 | 3883.909 | 101.496 | timeout |
| `coherent_d7_r1` | baseline | 27 | 0.157 | 550562.773 | 268.717 | complete |
| `coherent_d7_r1` | hub3 | 3 | 1.861 | 333398.554 | 272.912 | complete |
| `coherent_d7_r7` | baseline | 99 | 11.769 | 1.692e119 | n/a | rank error |
| `coherent_d7_r7` | hub3 | 3 | 542.398 | 1.275e11 | 2147.586 | timeout |

This is useful for the paper: degree reduction alone is insufficient when it
duplicates large separators. The next layout compiler needs a joint objective
over degree, separator size, and expected bond growth.

## Core Representation

`TTNState` stores one tensor per junction-tree bag. Each `TTNBag` has:

- `bag_id`: integer bag index.
- `neighbors`: sorted adjacent bag ids.
- `own_idents`: active logical identities currently hosted in this bag.
- `tensor`: dense complex tensor with axis order
  `[own_idents...] + [bond axes in neighbors order]`.

The bond graph is static. Every tree edge always has a bond axis; a trivial edge
has bond dimension 1.

The main invariants are:

- I1: each active ident appears in exactly one bag's `own_idents`.
- I2: each active ident appears in its assigned home bag.
- I3: both tensors adjacent to a tree edge have the same bond dimension.

`contract_into_one()` is a diagnostic path that contracts all bags into a dense
state tensor. The runtime sampler does not use it for normal operations.

## Backend Specification

`backend_spec.export_backend_spec(prog)` performs an instrumented bytecode replay:

1. `OP_EXPAND*` instructions create active identities.
2. Active two-axis operations add edges to a union interaction graph.
3. Measurements demote active identities.
4. SWAP-like instructions update the slot-to-ident map.

The union interaction graph is converted into a static junction tree. The
exported spec records which bag contains each operation and checks that every
two-axis operation is covered by some bag.

`assign_homes_and_classify(spec)` assigns each ident to one home bag and
classifies two-qubit operations:

- Class A: both active idents are in the same home bag.
- Class B/C: idents are in different bags and require a path contraction.

`TTNBackend` consumes this spec and keeps:

- `home_of`: ident to home bag map.
- `step_to_ident_expand`: bytecode step to newly promoted ident.
- `op_class_by_step_axes`: operation class lookup for two-ident gates.

## Per-Shot Execution

`TTNBackend.run_shot(prog, seed)` executes one sampled shot:

1. Initialize `TTNState` from the static bag tree.
2. Initialize `PauliFrame`, measurement record, and slot-to-ident map.
3. Create `ClifftNoiseSampler` for Clifft-style per-shot noise scheduling.
4. Iterate over Clifft bytecode in time order.
5. Dispatch frame operations, stochastic operations, active tensor operations,
   and measurements.
6. Return a sparse classical measurement record dictionary.

`TTNBackend.sample(prog, shots, seed, num_measurements)` derives per-shot seeds
from a NumPy master RNG and packs sparse records into a dense
`uint8[shots, num_measurements]` array.

## Frame Layer

`frame_layer.PauliFrame` stores X/Z parity bits per Clifft slot. It handles:

- frame Clifford gates: H, S, CNOT, CZ, SWAP;
- direct X/Z/Y frame flips;
- resetting a slot after measurement;
- frame corrections for active T/rotation and measurement paths.

`_apply_cp_mask()` applies `OP_APPLY_PAULI` masks from `Program.pauli_masks`.
Masks are interpreted as little-endian `uint64` X/Z word arrays when available.

`ClifftNoiseSampler` mirrors the `clifft.sample` noise schedule: it builds a
cumulative hazard array over `prog.noise_site_probabilities`, samples the next
firing site by exponential hazard gaps, and advances only when a scheduled noise
site fires. `_apply_noise_site()` then chooses the concrete Pauli mask within
that fired noise site.

## Tensor Operations

### Expand

`OP_EXPAND` promotes a new active ident at its home bag as `|+>`. The new
physical axis is inserted before all bond axes. `OP_EXPAND_T`,
`OP_EXPAND_T_DAG`, and `OP_EXPAND_ROT` expand as `|+>` and then apply a diagonal
phase, with frame-dependent conjugation where Clifft semantics require it.

### Single-Qubit Gates

Single-qubit active operations update the local tensor at the ident's home bag:

- H uses a 2x2 dense matrix.
- S/T/rotation use diagonal updates.

The corresponding Pauli frame update is also applied for Clifford gates.

### Canonical Center

`move_center(target)` moves the canonical center along the bag tree using QR
sweeps. For one edge `src -> dst`, the source tensor is QR-factorized across the
`src` side versus the bond to `dst`; Q stays in `src`, and R is absorbed into
`dst`.

### Two-Qubit Gates

Class A operations are local: both idents live in one bag, so the 4x4 unitary is
applied directly to two physical axes.

Class B/C operations use `apply_2q_class_B_path()` with adjacent transport:

1. Move the canonical center to the first bag on the path.
2. Transport the first ident one edge at a time until it is colocated with the
   second ident.
3. Apply the 4x4 gate locally in the destination bag.
4. Reverse the same one-edge transports to restore the static home invariant.

Each transport contracts only the two adjacent bags on the current edge and
QR-splits them immediately. The implementation records `peak_stored_bytes`,
`peak_pair_workspace_bytes`, `max_bond_dim`, `n_transports`, and `n_qr`.

### Measurements

`measure_z(ident, rng)` moves the center to the ident's home bag, computes the
local Z marginal, samples an outcome, projects the tensor, normalizes it, and
removes the ident axis from `own_idents`.

Active diagonal measurements use the Z measurement result combined with the X
frame bit. Active interfere measurements apply H first, then combine the
measured X-basis result with the Z frame bit.

Dormant measurements do not touch TTN tensors. Static dormant measurements read
from the frame; random dormant measurements draw a random bit and reset the
frame slot accordingly.

## Supported Bytecode Surface

The current dispatch covers the tested QEC benchmark subset:

- frame ops: H, S, S_DAG, CNOT, CZ, SWAP;
- stochastic/classical ops: APPLY_PAULI, NOISE, NOISE_BLOCK, READOUT_NOISE;
- dormant measurements: static and random variants;
- active lifecycle: EXPAND, EXPAND_T, EXPAND_T_DAG, EXPAND_ROT;
- active single-axis ops: ARRAY_H, ARRAY_S, ARRAY_S_DAG, ARRAY_T,
  ARRAY_T_DAG, ARRAY_ROT;
- active two-axis ops: ARRAY_CNOT, ARRAY_CZ;
- active multi-axis ops: ARRAY_MULTI_CNOT, ARRAY_MULTI_CZ;
- active measurements: diagonal, interfere, and swap-measure-interfere variants;
- ARRAY_SWAP.

Unsupported opcodes currently fall through. Detector, observable, postselect,
and expectation-value instructions are ignored because this backend returns only
measurement records.

## Validation Status

Recent 5000-shot comparisons against `clifft.sample` showed marginal agreement
near the Clifft self-sampling floor:

| Circuit | Clifft self marginal | TTN vs Clifft marginal | Status |
| --- | ---: | ---: | --- |
| `distillation` | 0.0310 | 0.0248 | pass |
| `cultivation_d3` | 0.0104 | 0.0074 | pass |
| `coherent_d3_r1` | 0.0162 | 0.0184 | pass |

Joint TVD remains large because the compared samplers use independent random
streams. The validation target is therefore marginal distribution agreement, not
shot-by-shot equality.

The metric collection run with 60-second per-shot timeouts produced:

| Circuit | Variant | `M_static` MB | runtime peak MB | pair workspace MB | timeout |
| --- | --- | ---: | ---: | ---: | --- |
| `distillation` | baseline | 0.000384 | 0.000384 | 0.000256 | false |
| `cultivation_d3` | baseline | 0.000768 | 0.000576 | 0.000256 | false |
| `coherent_d3_r1` | baseline | 0.000320 | 0.000464 | 0.000256 | false |
| `coherent_d5_r1` | baseline | 0.005248 | 0.129408 | 0.065536 | false |
| `coherent_d5_r5` | baseline | 0.068480 | 134.224 | 134.218 | true |
| `coherent_d7_r1` | baseline | 0.157440 | 268.717 | 268.435 | false |
| `coherent_d7_r7` | baseline | 11.769 | n/a | n/a | rank error |

## Known Limitations

- The implementation is a Python prototype using dense NumPy tensors per bag.
- No SVD truncation or bond-dimension cap is currently applied.
- Path contractions can become expensive when many bags or large separators are
  involved.
- Current hub-degree reduction is conservative and may reduce graph degree
  without improving exact runtime memory.
- The RNG stream is not expected to be bit-identical to Clifft; only the sampled
  distribution is targeted.
- Detector, observable, expectation-value, and postselection outputs are not
  produced.
- The backend depends on non-upstream Clifft binding additions for constant-pool
  inspection.
- Unsupported bytecode is not yet reported as a hard error.

## Basic Usage

```python
import clifft
from ttn_backend.backend_spec import export_backend_spec, assign_homes_and_classify
from ttn_backend import TTNBackend

prog = clifft.compile(open("qec_bench/circuits/distillation.stim").read())
spec = export_backend_spec(prog, strict=False)
homing = assign_homes_and_classify(spec)
backend = TTNBackend(spec, homing)
samples = backend.sample(prog, shots=100, seed=42)
```

For benchmark validation, run the repository's validation driver from
`/home/jung/clifft-paper`:

```bash
python3 verify_ttn.py
```
