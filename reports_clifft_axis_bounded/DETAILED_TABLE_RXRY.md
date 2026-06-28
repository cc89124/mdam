# DETAILED TABLE — Clifft baseline vs live clifft_axis_bounded backend

All bounded numbers are read from the per-step traces the `CliftAxisBoundedNearClifford` engine emits during its OWN run (`bounded_<circuit>_per_step.csv`) — no TTN, no block, no Clifft state, no forced outcomes.  **state size** = active-state dimension `2^b` (the dense magic register); **memory** = `16·2^b` bytes (complex128); the `O(n^2)`-bit CHP tableau is excluded (poly, same basis as Clifft's `2^k`).  `transient` = peak materialized magic rank during a measurement-core contraction (`2^(W-1)`); `resident` = settled magic rank between measurements.

## 1. Transient vs Resident (PEAK active-state)

| circuit | Clifft PEAK 2^k | bounded transient 2^b | transient vs Clifft | bounded resident 2^b | resident vs Clifft |
|---|---|---|---|---|---|
| coherent_rx_d3_r1 | 2^14 | 2^11 | 8x | 2^10 | 16x |
| coherent_rx_d3_r3 | 2^14 | 2^12 | 4x | 2^11 | 8x |
| coherent_ry_d3_r1 | 2^16 | 2^16 | parity | 2^15 | 2x |
| coherent_ry_d3_r3 | 2^16 | 2^16 | parity | 2^15 | 2x |

## 2. Active-State (2^x) — PEAK and integrated SUM

| circuit | Clifft PEAK | bounded tr. PEAK | tr x | bounded res. PEAK | res x | Clifft SUM | bounded tr. SUM | tr x | bounded res. SUM | res x |
|---|---|---|---|---|---|---|---|---|---|---|
| coherent_rx_d3_r1 | 2^14 | 2^11 | 8x | 2^10 | 16x | 2^19.31 | 2^14.99 | 19.9x | 2^14.85 | 22x |
| coherent_rx_d3_r3 | 2^14 | 2^12 | 4x | 2^11 | 8x | 2^21.04 | 2^18.06 | 7.88x | 2^17.96 | 8.45x |
| coherent_ry_d3_r1 | 2^16 | 2^16 | parity | 2^15 | 2x | 2^22.36 | 2^19.12 | 9.45x | 2^18.94 | 10.7x |
| coherent_ry_d3_r3 | 2^16 | 2^16 | parity | 2^15 | 2x | 2^23.94 | 2^20.85 | 8.53x | 2^20.69 | 9.52x |

## 3. Memory (bytes) — PEAK and integrated SUM

| circuit | Clifft PEAK | bounded tr. PEAK | tr x | bounded res. PEAK | res x | Clifft SUM | bounded tr. SUM | tr x | bounded res. SUM | res x |
|---|---|---|---|---|---|---|---|---|---|---|
| coherent_rx_d3_r1 | 256.0KiB | 32.0KiB | 8x | 16.0KiB | 16x | 9.9MiB | 509.8KiB | 19.9x | 460.8KiB | 22x |
| coherent_rx_d3_r3 | 256.0KiB | 64.0KiB | 4x | 32.0KiB | 8x | 32.8MiB | 4.2MiB | 7.88x | 3.9MiB | 8.45x |
| coherent_ry_d3_r1 | 1.0MiB | 1.0MiB | parity | 512.0KiB | 2x | 82.1MiB | 8.7MiB | 9.45x | 7.7MiB | 10.7x |
| coherent_ry_d3_r3 | 1.0MiB | 1.0MiB | parity | 512.0KiB | 2x | 245.4MiB | 28.8MiB | 8.53x | 25.8MiB | 9.52x |
