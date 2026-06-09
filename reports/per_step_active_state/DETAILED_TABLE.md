# Detailed per-step ACTIVE-STATE & MEMORY table

Baseline = **Clifft**. The `x` columns are the advantage = Clifft / backend (a bare multiple; >1x = backend is that many times smaller). PEAK and SUM are combined in one table per metric. Active-state size is written in `2^x` form (x = log2 of the dense-equivalent dimension). coherent_d7_r1/_d7_r7 excluded.

near-Clifford here is the **intra-step transient high-water mark** (the honest memory-feasibility peak: a measurement's anticommutation-core flush briefly forms a larger entangled block before its projector collapses it). The settled step-boundary **resident** value (lower; e.g. coherent_d5_r5 `2^12` resident vs `2^13` transient) is in `SUMMARY_TABLE.md`.

`†` coherent_d5_r5 is the full 3228-step circuit; its TTN line stops at step ~2289 (full-circuit TTN does not finish), so the TTN **SUM** is over that prefix and the TTN SUM ratio uses the Clifft sum over the same prefix. surface_d7_r7 is frame-only (no active idents); TTN fails to lay out -> '-'.

> **Read the `<1x` cells honestly — `cultivation_d5` is the all-magic limit, now at parity (frame reduction closed the last regression).** The headline metric is the intra-step **transient** `max_block`. With frame reduction ON (default), `cultivation_d5`'s transient peak is `2^10` = Clifft `2^10` — **parity, no longer a loss** (the earlier `2^11` 2x-loss was the *pre-reduction* number, now removed). The settled **resident** dips to `2^9` between measurements (a 2x *sub-peak* factorisation win: the 10 active idents no longer fit one block once measured-out qubits are decoupled), but the memory-provisioning peak is the transient `2^10`, so the honest headline is **parity, not a win** — the magic is irreducible. Block factoring guarantees the settled `max_block` never *exceeds* Clifft on any circuit; the only residual is a per-measurement, **sub-peak** transient `+1` at the lone measurement where Clifft's local rank dips (`cultivation_d5` meas 3: Clifft k=2 -> NC 3) — see the measurement-dependency report; it never reaches the global peak, so it does not change feasibility. **The MEMORY table below now compares the dense state only** (`16·2^k` vs `16·2^block`), so it matches the ACTIVE-STATE table and has **no spurious `<1x` cell**: NC's polynomial Clifford-frame metadata — the part Clifft's `16·2^k` baseline omits (Clifft keeps an `O(n^2)` tableau too) — is in its own table and never enters the ratio. (Earlier the byte ratio charged NC for that metadata while giving Clifft a metadata-free baseline, which made tiny all-magic circuits look like a loss even when the exponential state was parity-or-smaller, e.g. `coherent_d3_r1` had NC magic `2^0` yet a byte SUM `<1x`.)


## Transient & resident peak `max_block` vs Clifft (the honest no-regression picture)

> `transient` is the headline intra-step high-water (memory-provisioning peak); `resident` is the settled step-boundary block. With frame reduction ON, **no circuit's transient peak exceeds Clifft** — `cultivation_d5` is parity (`2^10 = 2^10`) and every other circuit is a win; the settled resident is parity-or-win everywhere (`cultivation_d5` settles to `2^9`).

| circuit | Clifft PEAK 2^k | NC transient 2^b | transient vs Clifft | NC resident 2^b | resident vs Clifft |
| --- | --: | --: | --: | --: | --: |
| coherent_d3_r1 | 2^5 | 2^0 | 32x win | 2^0 | 32x win |
| coherent_d3_r3 | 2^8 | 2^5 | 8x win | 2^4 | 16x win |
| coherent_d5_r1 | 2^13 | 2^0 | 8192x win | 2^0 | 8192x win |
| coherent_d5_r5 | 2^24 | 2^13 | 2048x win | 2^12 | 4096x win |
| distillation | 2^5 | 2^3 | 4x win | 2^2 | 8x win |
| cultivation_d3 | 2^4 | 2^4 | parity | 2^3 | 2x win |
| cultivation_d5 | 2^10 | 2^10 | parity | 2^9 | 2x win |
| surface_d7_r7 | 2^0 | 2^0 | parity | 2^0 | parity |

## ACTIVE-STATE SIZE  (dense-equivalent dimension, 2^x — NC = intra-step TRANSIENT peak)

| circuit | Clifft PEAK | TTN PEAK | TTN x | near-Clifford PEAK | NC x | Clifft SUM | TTN SUM | TTN x | near-Clifford SUM | NC x |
| --- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| coherent_d3_r1 | 2^5 | 2^6.19 | 0.44x | 2^0 | 32.0x | 2^11.42 | 2^13.22 | 0.29x | 2^8 | 10.7x |
| coherent_d3_r3 | 2^8 | 2^9.48 | 0.36x | 2^5 | 8.0x | 2^15.85 | 2^17.23 | 0.38x | 2^11.75 | 17.2x |
| coherent_d5_r1 | 2^13 | 2^11.55 | 2.7x | 2^0 | 8192x | 2^21.05 | 2^19.67 | 2.6x | 2^9.74 | 2535x |
| coherent_d5_r5 | 2^24 | 2^23.76 | 1.2x | 2^13 | 2048x | 2^34.78 | 2^31.89† | 5.5x | 2^22.85 | 3917x |
| distillation | 2^5 | 2^5.36 | 0.78x | 2^3 | 4.0x | 2^14.15 | 2^15.89 | 0.30x | 2^12.53 | 3.1x |
| cultivation_d3 | 2^4 | 2^6.21 | 0.22x | 2^4 | 1.0x | 2^11.52 | 2^13.33 | 0.28x | 2^10.47 | 2.1x |
| cultivation_d5 | 2^10 | 2^11.17 | 0.45x | 2^10 | 1.0x | 2^18.97 | 2^19.01 | 0.97x | 2^17.80 | 2.2x |
| surface_d7_r7 | 2^0 | - | - | 2^0 | 1.0x | 2^11.43 | - | - | 2^11.43 | 1.0x |

## MEMORY  (dense state bytes — Clifft `16·2^k` vs NC `16·2^block`, apples-to-apples)

| circuit | Clifft PEAK | TTN PEAK | TTN x | near-Clifford PEAK | NC x | Clifft SUM | TTN SUM | TTN x | near-Clifford SUM | NC x |
| --- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| coherent_d3_r1 | 512.0B | 1.1KiB | 0.44x | 16.0B | 32.0x | 42.8KiB | 149.4KiB | 0.29x | 4.0KiB | 10.7x |
| coherent_d3_r3 | 4.0KiB | 11.1KiB | 0.36x | 512.0B | 8.0x | 922.0KiB | 2.3MiB | 0.38x | 53.7KiB | 17.2x |
| coherent_d5_r1 | 128.0KiB | 46.8KiB | 2.7x | 16.0B | 8192x | 33.2MiB | 12.8MiB | 2.6x | 13.4KiB | 2535x |
| coherent_d5_r5 | 256.0MiB | 217.3MiB | 1.2x | 128.0KiB | 2048x | 440.4GiB | 59.4GiB† | 5.5x | 115.1MiB | 3917x |
| distillation | 512.0B | 656.0B | 0.78x | 128.0B | 4.0x | 283.1KiB | 946.9KiB | 0.30x | 92.3KiB | 3.1x |
| cultivation_d3 | 256.0B | 1.2KiB | 0.22x | 256.0B | 1.0x | 45.9KiB | 161.1KiB | 0.28x | 22.2KiB | 2.1x |
| cultivation_d5 | 16.0KiB | 35.9KiB | 0.45x | 16.0KiB | 1.0x | 7.8MiB | 8.1MiB | 0.97x | 3.5MiB | 2.2x |
| surface_d7_r7 | 16.0B | - | - | 16.0B | 1.0x | 43.0KiB | - | - | 43.0KiB | 1.0x |

### NC Clifford-frame metadata  (polynomial; NOT exponential, NOT in the ratio)

> The tableau + unapplied-pending bookkeeping NC carries. Clifft keeps an `O(n^2)` tableau of the same order but its `16·2^k` baseline omits it, so this column is shown for honesty only — it never enters `dense/NC`.

| circuit | NC metadata PEAK | NC metadata SUM |
| --- | --: | --: |
| coherent_d3_r1 | 482.0B | 77.1KiB |
| coherent_d3_r3 | 1.2KiB | 356.3KiB |
| coherent_d5_r1 | 2.1KiB | 1.2MiB |
| coherent_d5_r5 | 8.1KiB | 17.5MiB |
| distillation | 194.0B | 161.8KiB |
| cultivation_d3 | 288.0B | 29.3KiB |
| cultivation_d5 | 920.0B | 512.7KiB |
| surface_d7_r7 | 0.0B | 0.0B |
