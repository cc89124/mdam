# TTN Memory Diagnosis Plan

The next direction is layout and exact-rank diagnosis, not further RASL tuning.

RASL is currently treated as an auxiliary pass:

```text
RASL = preserve resident-memory proxy while reducing path/refactor work
```

It is not the main resident-memory optimization mechanism. The main question is
why the actual TTN tensors remain large.

## Core Formula

For bag `B` at time `t`:

```text
N_B(t) = 2^p_B(t) * prod_{e~B} chi_e(t)
E_B(t) = log2 N_B(t) = p_B(t) + sum_{e~B} log2 chi_e(t)
```

Total stored memory:

```text
M_store(t) = 16 * sum_B N_B(t)
M_store_peak = max_t M_store(t)
```

Workspace for an opened connected region `R`:

```text
N_R(t) = 2^p_R(t) * prod_{e in boundary(R)} chi_e(t)
```

## Diagnosis Cases

### Case A: physical-axis dominated

```text
p_B >> sum log2 chi_e
```

Interpretation:

```text
The layout put too many active axes into one bag.
```

Candidate remedies:

- bag split
- balanced layout
- pair-demand clustering

### Case B: bond-product dominated

```text
sum log2 chi_e >> p_B
```

Interpretation:

```text
The bag is a hub or receives too many large incident bonds.
```

Candidate remedies:

- degree-aware layout
- separator distribution
- hub split that does not inflate internal separators

### Case C: allocated chi larger than exact local rank

```text
allocated chi_e > numerical rank_e
```

Interpretation:

```text
The layout may be acceptable, but exact rank compression is missing or delayed.
```

Candidate remedies:

- exact SVD compression of zero singular values
- rank-revealing QR/SVD
- compression scheduling

## Implemented Report

Run:

```bash
cd /home/jung/clifft-paper
/home/jung/clifft_env/bin/python -m ttn_backend.scripts.memory_diagnosis_report \
  distillation cultivation_d3 coherent_d3_r1 coherent_d5_r1 coherent_d5_r5 \
  --variants baseline \
  --runtime-timeout 60
```

Outputs:

- `reports/ttn_memory_diagnosis_summary.csv`
- `reports/ttn_memory_diagnosis_edges.csv`
- `reports/ttn_memory_diagnosis.json`
- `reports/ttn_memory_diagnosis.md`

The summary includes:

- `peak_step`
- `peak_bag_id`
- `peak_bag_tensor_shape`
- `peak_bag_bytes`
- `p_B`
- `incident_edges`
- `incident_bond_dims`
- `sum_log2_bonds`
- `E_B = p_B + sum_log2_bonds`
- `total_stored_peak_bytes`
- `dense_bytes`
- `dense_over_total_stored`
- diagnosis label

The edge table includes:

- allocated current `chi`
- max observed allocated `chi`
- local adjacent two-bag SVD rank, when feasible
- allocated/rank ratio
- whether the edge is incident to the peak offender bag

## Layout Candidate Status

Currently implemented layout variants:

- `baseline`
- `hub3` via `reduce_hub_degree`

Not yet implemented as real layout generators:

- balanced binary layout
- pair-demand layout
- random layout ensemble

Do not report numbers for these unimplemented layouts. The next step is to add
candidate generators and evaluate them with the same actual diagnosis script.

## Time-Varying Graph Evolution

The next diagnostic layer records the actual TTN state after tensor-mutating
events and aggregates those events into a per-step live graph `G_t`.

Run:

```bash
cd /home/jung/clifft-paper
/home/jung/clifft_env/bin/python -m ttn_backend.scripts.time_graph_report \
  --runtime-timeout 60 --variants baseline
```

Outputs:

- `reports/time_graph_summary.csv`
- `reports/time_graph_steps.csv`
- `reports/time_graph_critical.csv`
- `reports/time_graph_b0_edges.csv`
- `reports/time_graph_b0_overlap.csv`
- `reports/time_graph_report.json`
- `reports/time_graph_report.md`

The step table contains:

- live axes
- live bags
- live TTN edges with `chi > 1`
- peak stored bytes for that step
- peak workspace bytes for that step
- peak offender bag and `E_B`
- B0 incident live bond load

The B0 analysis computes:

```text
union incident load     = sum_e max_t log2 chi_e(t)
max live incident load  = max_t sum_e log2 chi_e(t)
inactive contribution   = union incident load - max live incident load
```

Interpretation:

- If `inactive contribution` is large, a lazy/static-super-layout approach may
  help because many B0 incident bonds are not simultaneously live.
- If `max live incident load == union incident load`, lazy allocation alone
  cannot reduce the B0 bond product at the observed peak. The layout itself has
  to reduce the simultaneous hub load.

Current baseline result:

- `coherent_d5_r5` partial run: B0 union load `14`, max live load `14`.
  Lazy allocation does not help this observed B0 peak.
- `coherent_d7_r1`: B0 union load `13.46`, max live load `12.86`.
  Lazy allocation may help only slightly.
- `coherent_d7_r7`: the current backend cannot initialize the layout because
  B0 has degree `99`, exceeding numpy's maximum ndarray dimension of `64` even
  when all bond dimensions are initially one. This is a static layout
  representation failure, not a runtime rank-growth result.

## Static Peak-Bag TTN Compression

After the time-varying report showed that lazy allocation does not explain the
`coherent_d5_r5` peak B0 load, the next feasibility test is to take the fixed
peak B0 tensor and re-decompose it into a smaller numerical-rank TTN.

Run:

```bash
cd /home/jung/clifft-paper
/home/jung/clifft_env/bin/python -m ttn_backend.scripts.static_ttn_compression_experiment \
  --circuit coherent_d5_r5 \
  --step 977 \
  --bag B0 \
  --rank-rules rel energy \
  --tols 1e-8 1e-6 1e-4 \
  --mode depth1 recursive \
  --random-candidates 100 \
  --top-svd 6 \
  --max-depth 4 \
  --out-dir reports
```

Outputs:

- `reports/static_ttn_b0_compression_summary.csv`
- `reports/static_ttn_b0_compression_candidates.csv`
- `reports/static_ttn_b0_compression_tree_*.json`
- `reports/static_ttn_b0_compression_report.md`

Current `coherent_d5_r5` B0 result:

- old B0 tensor: `2^23` complex elements = `134,217,728` bytes.
- depth-1 split at `rel_tol=1e-8`: peak `2^20` elements, total `2^20.585`,
  relative reconstruction error `1.3e-12`.
- recursive split at `rel_tol=1e-8`: peak `2^19` elements, total `2^20.285`,
  relative reconstruction error `1.5e-12`.
- recursive split at `energy_tol=1e-4`: peak `2^16.948` elements, total
  `2^18.504`, relative reconstruction error `9.8e-5`.

Interpretation: the critical B0 peak tensor is compressible as a static
numerical-rank TTN. The current hub tensor is not an intrinsically dense object
at this snapshot; the current layout stores it in a bad tensor structure.
This is still a static feasibility result, not yet an executable runtime layout
optimizer.
