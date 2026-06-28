# DETAILED TABLE — Clifft baseline vs live clifft_axis_bounded backend

All bounded numbers are read from the per-step traces the `CliftAxisBoundedNearClifford` engine emits during its OWN run (`bounded_<circuit>_per_step.csv`) — no TTN, no block, no Clifft state, no forced outcomes.  **state size** = active-state dimension `2^b` (the dense magic register); **memory** = `16·2^b` bytes (complex128); the `O(n^2)`-bit CHP tableau is excluded (poly, same basis as Clifft's `2^k`).  `transient` = peak materialized magic rank during a measurement-core contraction (`2^(W-1)`); `resident` = settled magic rank between measurements.

## 1. Transient vs Resident (PEAK active-state)

| circuit | Clifft PEAK 2^k | bounded transient 2^b | transient vs Clifft | bounded resident 2^b | resident vs Clifft |
|---|---|---|---|---|---|
| coherent_d3_r1 | 2^5 | 2^0 | 32x | 2^0 | 32x |
| coherent_d3_r3 | 2^8 | 2^5 | 8x | 2^4 | 16x |
| coherent_d5_r1 | 2^13 | 2^0 | 8.19e+03x | 2^0 | 8.19e+03x |
| coherent_d5_r5 | 2^24 | 2^13 | 2.05e+03x | 2^12 | 4.1e+03x |
| distillation | 2^5 | 2^4 | 2x | 2^3 | 4x |
| cultivation_d3 | 2^4 | 2^4 | parity | 2^3 | 2x |
| cultivation_d5 | 2^10 | 2^10 | parity | 2^9 | 2x |
| surface_d7_r7 | 2^0 | 2^0 | parity | 2^0 | parity |

## 2. Active-State (2^x) — PEAK and integrated SUM

| circuit | Clifft PEAK | bounded tr. PEAK | tr x | bounded res. PEAK | res x | Clifft SUM | bounded tr. SUM | tr x | bounded res. SUM | res x |
|---|---|---|---|---|---|---|---|---|---|---|
| coherent_d3_r1 | 2^5 | 2^0 | 32x | 2^0 | 32x | 2^11.42 | 2^8.00 | 10.7x | 2^8.00 | 10.7x |
| coherent_d3_r3 | 2^8 | 2^5 | 8x | 2^4 | 16x | 2^15.85 | 2^11.74 | 17.2x | 2^11.69 | 17.9x |
| coherent_d5_r1 | 2^13 | 2^0 | 8.19e+03x | 2^0 | 8.19e+03x | 2^21.05 | 2^9.74 | 2.53e+03x | 2^9.74 | 2.53e+03x |
| coherent_d5_r5 | 2^24 | 2^13 | 2.05e+03x | 2^12 | 4.1e+03x | 2^34.78 | 2^22.85 | 3.92e+03x | 2^22.81 | 4.01e+03x |
| distillation | 2^5 | 2^4 | 2x | 2^3 | 4x | 2^14.15 | 2^13.04 | 2.15x | 2^13.04 | 2.15x |
| cultivation_d3 | 2^4 | 2^4 | parity | 2^3 | 2x | 2^11.52 | 2^10.47 | 2.07x | 2^10.44 | 2.11x |
| cultivation_d5 | 2^10 | 2^10 | parity | 2^9 | 2x | 2^18.97 | 2^17.80 | 2.25x | 2^17.79 | 2.27x |
| surface_d7_r7 | 2^0 | 2^0 | parity | 2^0 | parity | 2^11.43 | 2^11.43 | parity | 2^11.43 | parity |

## 3. Memory (bytes) — PEAK and integrated SUM

| circuit | Clifft PEAK | bounded tr. PEAK | tr x | bounded res. PEAK | res x | Clifft SUM | bounded tr. SUM | tr x | bounded res. SUM | res x |
|---|---|---|---|---|---|---|---|---|---|---|
| coherent_d3_r1 | 512B | 16B | 32x | 16B | 32x | 42.8KiB | 4.0KiB | 10.7x | 4.0KiB | 10.7x |
| coherent_d3_r3 | 4.0KiB | 512B | 8x | 256B | 16x | 922.0KiB | 53.6KiB | 17.2x | 51.6KiB | 17.9x |
| coherent_d5_r1 | 128.0KiB | 16B | 8.19e+03x | 16B | 8.19e+03x | 33.2MiB | 13.4KiB | 2.53e+03x | 13.4KiB | 2.53e+03x |
| coherent_d5_r5 | 256.0MiB | 128.0KiB | 2.05e+03x | 64.0KiB | 4.1e+03x | 440.4GiB | 115.1MiB | 3.92e+03x | 112.5MiB | 4.01e+03x |
| distillation | 512B | 256B | 2x | 128B | 4x | 283.1KiB | 131.9KiB | 2.15x | 131.6KiB | 2.15x |
| cultivation_d3 | 256B | 256B | parity | 128B | 2x | 45.9KiB | 22.2KiB | 2.07x | 21.7KiB | 2.11x |
| cultivation_d5 | 16.0KiB | 16.0KiB | parity | 8.0KiB | 2x | 7.8MiB | 3.5MiB | 2.25x | 3.5MiB | 2.27x |
| surface_d7_r7 | 16B | 16B | parity | 16B | parity | 43.0KiB | 43.0KiB | parity | 43.0KiB | parity |
