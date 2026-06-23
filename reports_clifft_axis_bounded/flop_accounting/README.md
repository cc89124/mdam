# FLOP / memory-traffic accounting — clifft_axis_bounded

Algorithmic FLOP and memory-traffic accounting for the bounded near-Clifford engine, measured via a
non-invasive hook on the engine's existing `DenseMemoryBudget.charge` calls. This folder contains
**only the FLOP/traffic results**; the rank-trace (`*_qubits.png`) and R_Y-fidelity reports live one
level up in `reports_clifft_axis_bounded/`.

## What is being claimed (and how strongly)
- **bounded FLOP = validated algorithmic FLOP** under a stated arithmetic convention. The hook coeffs
  were cross-checked 1:1 against a direct kernel-event meter (exact at r=1..6, on real circuits, and
  on unit-called kernels); turning the meter on/off leaves the record & max_M **bit-identical**.
- **clifft FLOP = modeled / estimated.** clifft's core is a compiled extension (`_clifft_core.abi3.so`)
  and cannot be instrumented; it is modeled as executing the *same shared events* as bounded but each
  at the full peak array `2^k` (no localize-and-drop). See `METHODOLOGY.md` §4 + §9 for the exact
  assumptions and why this is an upper bound.
- **rank / state-volume = trace-derived** (from the actual resident-rank sequence).

## Headline result
bounded's advantage = **two effects** that separate cleanly by whether peak rank `r_max < k`:
1. **peak-rank compression** (only when `r_max < k`) and
2. **dense-computation localization** (always — the rise/fall tail runs at `2^r ≪ 2^k`).

| regime | example | F_cl/F_bn | note |
|---|---|--:|---|
| `r_max ≪ k` | coherent_d5_r5 (R_Z, k24, peak13) | **3307×** | both effects; even the peak band wins 2048× |
| `r_max < k` | coherent_rx_d3_r1 (R_X, k14, peak11) | **33.9×** | both effects |
| `r_max = k` | coherent_ry_d3_r1 (R_Y, k16, peak16) | **4.3×** | localization only; peak band is a wash |
| magic-saturated | cultivation_d3 (T, k4, peak4) | **1.2×** | measurements probe magic directly |

⚠️ `F_cl/F_bn` uses the clifft flat-peak model; the trace-faithful `S_cl/S_bn` (state-volume) uses
clifft's real per-step rank. They are different models — see METHODOLOGY §7. FLOP↓ is **not** a
wall-clock claim (clifft runtime not yet measured; `ms` is bounded-only, Python-bound).

## Folder layout
```
flop_accounting/
├── README.md                     ← this file (index + headline + claims)
├── METHODOLOGY.md                ← EXACTLY how every number/graph is computed + assumptions to judge
├── PER_RANK_FLOP_MECHANISM.md    ← per-rank win decomposition (peak/shoulder/tail), full set
├── data/
│   ├── flop_all.csv              ← full 18-circuit table (FLOP, R/W words, bytes, S, inv, ms)
│   └── per_rank_<circ>.csv  (×9) ← FLOP bounded vs clifft-modeled per resident rank
├── figures/
│   ├── flop_rank_trace_<circ>.png (×9) ← rank mountain (top) + per-event & cumulative FLOP (bottom)
│   └── flop_by_rank_<circ>.png    (×9) ← FLOP histogram over rank, bounded vs clifft-modeled
└── scripts/
    ├── flop_all.py               ← full-suite table  -> data/flop_all.csv
    ├── flop_production.py        ← production summary + per-kernel breakdown (prints)
    ├── flop_per_measurement.py   ← generates figures/ + data/per_rank_*  (the mountain + histogram)
    ├── _flop_validate.py         ← the direct-vs-hook cross-validator (validation evidence)
    └── _SUPERSEDED_flop_count_v1.py  ← DEPRECATED first version (wrong offdiag=16, sqnorm=N); ignore
```

## Circuits covered by the figures (9, the validated set)
R_Y: coherent_ry_d3_r1, coherent_ry_d3_r3 · R_X: coherent_rx_d3_r1, coherent_rx_d3_r3 ·
R_Z: coherent_d3_r3, coherent_d5_r5 · T: cultivation_d3, cultivation_d5, distillation.
The full table (`flop_all.csv`) additionally covers the all-stabilizer (0-FLOP) and INFEASIBLE cases.

## Reproduce
Run from the repo root `/home/jung/clifft-paper` with the clifft venv:
```
cd /home/jung/clifft-paper
/home/jung/clifft_env/bin/python reports_clifft_axis_bounded/flop_accounting/scripts/flop_all.py
/home/jung/clifft_env/bin/python reports_clifft_axis_bounded/flop_accounting/scripts/flop_per_measurement.py
```
Inputs: circuits in `qec_bench/circuits/*.stim`; rank traces in `reports_clifft_axis_bounded/
bounded_<circ>_per_step.csv` (for the state-volume proxy). Outputs land back in this folder.

## Fidelity prerequisite (results below are all on validated circuits)
R_Y d3_r1/r3 EXACT Born (per-meas 2.55e-15, joint 1.4e-13); R_X EXACT Born + cross-entropy NEW=OLD;
R_Z + T via cz-fix no-op proof + prior trajectory-EXACT. Details: `../EXACT_RY_VALIDATION.md`,
`../RY_BUGFIX_REPORT.md`.
