# Reports — Clifft baseline vs live fused virtual-axis backend

Current work only: the **dense-free, single-frame live fused-VA backend**
(`nearclifford_backend/virtual_axis/fused_single_frame.py`) vs the **Clifft** `2^k` model.
No block backend, no TTN, no forced outcomes — the fused engine samples its own Born outcomes.

| folder | axis | main file |
|---|---|---|
| [`per_step_active_state/`](per_step_active_state/) | **memory** — peak active-state size (log2 dim, qubits) | `FUSED_VA_SUMMARY.md` + `fused_va_<circuit>_per_step.csv` |
| [`per_step_flops/`](per_step_flops/) | **compute** — runtime contraction FLOP | `FUSED_VA_FLOPS.md` + `fused_va_full_flops.csv` |

Reproduce: `clifft_env/bin/python reports/<folder>/fused_va_*generate.py`.

## Headline — coherent_d5_r5

| | Clifft | live fused-VA | reduction |
|---|--:|--:|--:|
| active state (memory) | `2^24` | **`2^12`** | **4096×** |
| runtime FLOP (full) | 209.3G | **367.4M** | **570×** |

Memory reduction > compute reduction because the fused contraction still touches every
amplitude of the `2^ws` workspace, and on **all-magic** circuits its Pauli-sum expansion can
do *more* FLOP than Clifft's analytic model (e.g. cultivation_d5 `0.42×`) — a memory↔compute
trade. The fused backend is state-exact (final-state fidelity = 1.0 verified).

Older block/TTN/3-way comparisons and per-measurement traces are archived in
[`../reports_archive/`](../reports_archive/).
