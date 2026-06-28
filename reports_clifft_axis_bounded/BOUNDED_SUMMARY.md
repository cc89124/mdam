# Per-step ACTIVE-STATE: clifft_axis_bounded LIVE backend

Peak active-state size of the canonical bounded near-Clifford engine (`CliftAxisBoundedNearClifford`, hard budget peak amplitude words <= 2^k_clifft). **transient** = peak materialized magic rank during a measurement (flush/promote peak before the localize-and-drop); **resident** = settled magic rank between measurements.  Per-step traces: `bounded_<circuit>_per_step.csv`.

| circuit | Clifft k | bounded transient (qubits) | bounded resident (qubits) | bounded transient dim | saving 2^(k-transient) |
|---|--:|--:|--:|--:|--:|
| coherent_d3_r1 | 5 | 0 | 0 | 2^0 | 2^5 |
| coherent_d3_r3 | 8 | 5 | 5 | 2^5 | 2^3 |
| coherent_d5_r1 | 13 | 0 | 0 | 2^0 | 2^13 |
| coherent_d5_r5 | 24 | 13 | 13 | 2^13 | 2^11 |
| cultivation_d3 | 4 | 4 | 4 | 2^4 | 2^0 |
| cultivation_d5 | 10 | 10 | 10 | 2^10 | 2^0 |
| distillation | 5 | 4 | 4 | 2^4 | 2^1 |
| surface_d7_r7 | 0 | 0 | 0 | 2^0 | 2^0 |
