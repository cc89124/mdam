## Per-step ACTIVE-STATE SIZE: Clifft vs TTN vs near-Clifford

Active-state size = log2(dense-equivalent dimension), in qubits. Clifft = #active idents k; TTN = log2(stored_bytes/16); near-Clifford = genuine magic-block qubits (0 for a pure stabilizer circuit). **near-Clifford MAIN = intra-step transient peak** (memory high-water mark); resident = settled step-boundary value. Linear PNGs: `<circuit>_per_step_qubits_linear.png`.

| circuit | Clifft k | TTN log2-dim | near-Clifford magic (transient) | near-Clifford (resident) |
|---|--:|--:|--:|--:|
| coherent_d3_r1 | 5 | 6.19 | 0 | 0 |
| coherent_d3_r3 | 8 | 9.48 | 5 | 4 |
| coherent_d5_r1 | 13 | 11.55 | 0 | 0 |
| coherent_d5_r5 | 24 | 23.76 | 13 | 12 |
| distillation | 5 | 5.36 | 3 | 2 |
| cultivation_d3 | 4 | 6.21 | 4 | 3 |
| cultivation_d5 | 10 | 11.17 | 10 | 9 |
| surface_d7_r7 | 0 | None | 0 | 0 |

## Per-step TOTAL memory: Clifft dense vs TTN vs near-Clifford

Linear-scale per-step PNGs are `<circuit>_per_step_linear.png`. This is the **TOTAL resident footprint** each backend holds (the point of this report; the state-only *dimension* view is in `per_step_active_state`). Clifft dense = `16*2^k`; **NC total = `16*2^block` (dense magic state) + metadata** (Clifford frame + unapplied pending), broken out in the next two columns. `dense/NC` is **total vs total**. NOTE: Clifft keeps an `O(n^2)` tableau too, but its `16*2^k` baseline omits it — so on tiny all-magic circuits NC's total can exceed Clifft's dense model (the metadata dominates the small `2^block`); the exponential state alone is parity-or-win (see `per_step_active_state`). **MAIN = transient high-water**; (resident) = settled. (memory only; correctness elsewhere)

### PEAK memory (max over steps)

| circuit | k | Clifft dense | TTN | NC TOTAL (transient) | NC TOTAL (resident) | – dense state | – metadata | dense/NC | TTN/NC |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| coherent_d3_r1 | 5 | 512.0 B | 1.1 KiB | 498.0 B | 498.0 B | 16.0 B | 482.0 B | 1.0× | 2.3× |
| coherent_d3_r3 | 8 | 4.0 KiB | 11.1 KiB | 1.4 KiB | 1.2 KiB | 512.0 B | 1.2 KiB | 2.9× | 8.0× |
| coherent_d5_r1 | 13 | 128.0 KiB | 46.8 KiB | 2.1 KiB | 2.1 KiB | 16.0 B | 2.1 KiB | 61.2× | 22.4× |
| coherent_d5_r5 | 24 | 256.0 MiB | 217.3 MiB | 135.0 KiB | 71.0 KiB | 128.0 KiB | 8.1 KiB | 1942.2× | 1648.2× |
| distillation | 5 | 512.0 B | 656.0 B | 230.0 B | 210.0 B | 128.0 B | 194.0 B | 2.2× | 2.9× |
| cultivation_d3 | 4 | 256.0 B | 1.2 KiB | 416.0 B | 416.0 B | 256.0 B | 288.0 B | 0.6× | 2.8× |
| cultivation_d5 | 10 | 16.0 KiB | 35.9 KiB | 16.2 KiB | 8.9 KiB | 16.0 KiB | 920.0 B | 1.0× | 2.2× |
| surface_d7_r7 | 0 | 16.0 B | n/a | 0.0 B | 0.0 B | 16.0 B | 0.0 B | n/a | n/a |

### SUM memory (area under the per-step curve)

| circuit | Clifft dense | TTN | NC TOTAL (transient) | NC TOTAL (resident) | – dense state | dense/NC | TTN/NC |
|---|--:|--:|--:|--:|--:|--:|--:|
| coherent_d3_r1 | 42.8 KiB | 149.4 KiB | 81.1 KiB | 81.1 KiB | 4.0 KiB | 0.5× | 1.8× |
| coherent_d3_r3 | 922.0 KiB | 2.3 MiB | 410.0 KiB | 393.8 KiB | 53.7 KiB | 2.2× | 5.9× |
| coherent_d5_r1 | 33.2 MiB | 12.8 MiB | 1.2 MiB | 1.2 MiB | 13.4 KiB | 27.0× | 10.4× |
| coherent_d5_r5 | 440.4 GiB | 59.4 GiB | 132.6 MiB | 129.7 MiB | 115.1 MiB | 3401.1× | 458.4× |
| distillation | 283.1 KiB | 946.9 KiB | 254.1 KiB | 215.9 KiB | 92.3 KiB | 1.1× | 3.7× |
| cultivation_d3 | 45.9 KiB | 161.1 KiB | 51.5 KiB | 49.6 KiB | 22.2 KiB | 0.9× | 3.1× |
| cultivation_d5 | 7.8 MiB | 8.1 MiB | 4.0 MiB | 3.9 MiB | 3.5 MiB | 2.0× | 2.0× |
| surface_d7_r7 | 43.0 KiB | n/a | 0.0 B | 0.0 B | 43.0 KiB | n/a | n/a |
