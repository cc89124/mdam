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

## Per-step memory: Clifft dense vs TTN vs near-Clifford block

Linear-scale per-step PNGs are `<circuit>_per_step_linear.png`. Clifft dense = `16*2^k` (dense active state). **The `dense/NC` ratio compares the EXPONENTIAL dense state only** — Clifft `16*2^k` vs near-Clifford `16*2^block` — so it is apples-to-apples (both omit their Clifford-frame metadata; Clifft has an `O(n^2)` tableau too). The **NC metadata** column (Clifford frame + unapplied pending) is the polynomial part Clifft's baseline omits — it is **not** in the ratio. **near-Clifford MAIN = intra-step transient high-water**; (resident) = settled step-boundary. (memory only; correctness covered elsewhere)

### PEAK memory (max over steps)

| circuit | k | Clifft dense | TTN | NC dense state (transient) | NC dense state (resident) | NC metadata | dense/NC | TTN/NC |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| coherent_d3_r1 | 5 | 512.0 B | 1.1 KiB | 16.0 B | 16.0 B | 482.0 B | 32.0× | 73.0× |
| coherent_d3_r3 | 8 | 4.0 KiB | 11.1 KiB | 512.0 B | 256.0 B | 1.2 KiB | 8.0× | 22.2× |
| coherent_d5_r1 | 13 | 128.0 KiB | 46.8 KiB | 16.0 B | 16.0 B | 2.1 KiB | 8192.0× | 2993.0× |
| coherent_d5_r5 | 24 | 256.0 MiB | 217.3 MiB | 128.0 KiB | 64.0 KiB | 8.1 KiB | 2048.0× | 1738.0× |
| distillation | 5 | 512.0 B | 656.0 B | 128.0 B | 64.0 B | 194.0 B | 4.0× | 5.1× |
| cultivation_d3 | 4 | 256.0 B | 1.2 KiB | 256.0 B | 128.0 B | 288.0 B | 1.0× | 4.6× |
| cultivation_d5 | 10 | 16.0 KiB | 35.9 KiB | 16.0 KiB | 8.0 KiB | 920.0 B | 1.0× | 2.2× |
| surface_d7_r7 | 0 | 16.0 B | n/a | 0.0 B | 0.0 B | n/a | n/a | n/a |

### SUM memory (area under the per-step curve)

| circuit | Clifft dense | TTN | NC dense state (transient) | NC dense state (resident) | dense/NC | TTN/NC |
|---|--:|--:|--:|--:|--:|--:|
| coherent_d3_r1 | 42.8 KiB | 149.4 KiB | 4.0 KiB | 4.0 KiB | 10.7× | 37.4× |
| coherent_d3_r3 | 922.0 KiB | 2.3 MiB | 53.7 KiB | 51.6 KiB | 17.2× | 44.7× |
| coherent_d5_r1 | 33.2 MiB | 12.8 MiB | 13.4 KiB | 13.4 KiB | 2534.9× | 975.6× |
| coherent_d5_r5 | 440.4 GiB | 59.4 GiB | 115.1 MiB | 112.5 MiB | 3917.5× | 528.0× |
| distillation | 283.1 KiB | 946.9 KiB | 92.3 KiB | 89.4 KiB | 3.1× | 10.3× |
| cultivation_d3 | 45.9 KiB | 161.1 KiB | 22.2 KiB | 21.0 KiB | 2.1× | 7.3× |
| cultivation_d5 | 7.8 MiB | 8.1 MiB | 3.5 MiB | 3.5 MiB | 2.2× | 2.3× |
| surface_d7_r7 | 43.0 KiB | n/a | 0.0 B | 0.0 B | n/a | n/a |
