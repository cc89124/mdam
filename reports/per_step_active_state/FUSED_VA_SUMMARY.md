# Per-step ACTIVE-STATE: Clifft baseline vs live fused virtual-axis

Peak active-state size (log2 dense-equivalent dimension, in qubits): **Clifft** (`2^k`, dense
active state) vs the **dense-free, single-frame live fused-VA backend**
(`nearclifford_backend/virtual_axis/fused_single_frame.py`, `FusedSingleFrame`). The fused
engine never builds clifft's `2^k` state and samples its own Born outcomes; **transient** =
peak fused workspace during a measurement-core contraction (= `fused_ws`), **resident** =
settled magic rank between measurements.

Per-step traces: `fused_va_<circuit>_per_step.csv` (columns: step, n_active,
fused_resident_qubits, fused_transient_qubits, fused_resident_dim, fused_transient_dim).
Reproduce: `clifft_env/bin/python reports/per_step_active_state/fused_va_generate.py`.

## Per-step graphs — Clifft 2^k vs live fused-VA

`fused_vs_clifft_<circuit>_qubits.png` (per circuit) and `fused_vs_clifft_ALL_qubits.png`
(8-up grid): y = active-state size in qubits (= log2 dense-equiv dimension); crimson = Clifft
`2^k` (the per-step active rank `n_active`), green = fused resident (solid) / transient
(dashed); the shaded band is the saving. Clifft cycles up to `k` every QEC round while the
fused line stays capped at `ws` — e.g. coherent_d5_r5: Clifft repeatedly hits 24, fused holds
12. (Early steps show fused at 0 because rotations are *deferred* in the pending ledger and the
dense magic only materialises at a measurement core flush.) Regenerate:
`clifft_env/bin/python reports/per_step_active_state/plot_fused_vs_clifft.py`.

## Peak — Clifft vs live fused-VA

| circuit | Clifft k | **fused transient** | **fused resident** | saving vs Clifft (transient) |
|---|--:|--:|--:|--:|
| coherent_d3_r1 | 5 | **1** | **0** | 2^4 |
| coherent_d3_r3 | 8 | **4** | **4** | 2^4 |
| coherent_d5_r1 | 13 | **1** | **0** | 2^12 |
| **coherent_d5_r5** | 24 | **12** | **12** | **2^12 (4096×)** |
| distillation | 5 | **3** | **3** | 2^2 |
| cultivation_d3 | 4 | **3** | **3** | 2^1 |
| cultivation_d5 | 10 | **9** | **9** | 2^1 |
| surface_d7_r7 | 0 | **0** | **0** | — |

Notes:
- **coherent_d5_r5**: Clifft holds `2^24`; live fused-VA peaks at `2^12` — a **4096×** smaller
  active state, dense-free.
- The fused map contracts the measured axis analytically (`2^(W−1)`), so transient = resident
  on the binding circuits (no +1 measurement spike).
- coherent_*_r1 have no persistent magic (resident 0); transient 1 = a single axis materialised
  then measured away. surface_d7_r7 is pure stabilizer (0).
- fused-VA is state-exact (final-state fidelity = 1.0 verified vs a dense statevector).
