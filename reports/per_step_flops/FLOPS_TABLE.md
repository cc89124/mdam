# Total compute by operation: Clifft (baseline) vs TTN vs near-Clifford

FLOP unit (complex mult=6, add=2, norm=4, vdot=8), SUM over the whole run. Per-operation columns + **TOTAL** (= the backend's FLOP ops + the shared Clifford bit-op floor) + advantage `x` = Clifft TOTAL / backend TOTAL. Clifft = analytic 2^k dense model (matmul only); TTN & near-Clifford = MEASURED. SVD = 0 (TTN exact mode). `†` coherent_d5_r5 TTN = executed prefix (~step 2289). surface_d7_r7 TTN fails to lay out -> '-'.

Clifford bit-op = polynomial GF(2) tableau/frame work (gate~n, meas~n^2, deferred rot~n); it is why near-Clifford's `0 FLOP` cases are bit-ops-only, not no-work.


> **Compute is a different axis from memory — and frame reduction helps both.** With frame reduction ON (default), peeling each measured-out qubit's dead residue keeps the magic blocks smaller, so the `norm`/`vdot` **factoring scan** runs over far fewer amplitudes: NC FLOP drops sharply vs the pre-reduction numbers (`cultivation_d5` `150M→12M`, `distillation` `41K→18K`, `coherent_d5_r5` norm `12.5G→2.0G`). This flips `distillation` to a **compute win (1.4x)** and lifts every coherent circuit (`d5_r5` 16x→**74x**). The remaining `NC x < 1x` rows (`cultivation_d3` 0.75x, `cultivation_d5` 0.29x, both up from 0.20x/0.02x) are *compute* losses, not memory losses: on **all-magic** circuits near-Clifford still pays a factoring scan on genuinely-irreducible magic that Clifft's analytic `2^k` model never spends, so it does more FLOP even when its memory is parity/better — the expected trade for the bounded block (§8.3/§8.4).


| circuit | Clifford bit-op | Clifft matmul | Clifft TOTAL | TTN contract | TTN QR | TTN TOTAL | TTN x | NC matmul | NC norm | NC TOTAL | NC x |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| coherent_d3_r1 | 3.5K | 17.3K | 20.8K | 12.1K | 11.3K | 26.9K | 0.77x | 0 | 0 | 3.5K | 5.9x |
| coherent_d3_r3 | 19.1K | 545.2K | 564.3K | 1.2M | 1.0M | 2.2M | 0.25x | 68.1K | 132.0K | 219.2K | 2.6x |
| coherent_d5_r1 | 70.6K | 12.7M | 12.8M | 7.9M | 7.6M | 15.6M | 0.82x | 0 | 0 | 70.6K | 181x |
| coherent_d5_r5 | 1.2M | 209.3G | 209.3G | 8.2T† | 8.9T | 17.1T† | 0.01x | 799.3M | 2.0G | 2.8G | 74.1x |
| distillation | 12.5K | 12.6K | 25.1K | 8.8K | 6.6K | 27.9K | 0.90x | 2.9K | 2.3K | 17.6K | 1.4x |
| cultivation_d3 | 2.8K | 23.6K | 26.4K | 75.9K | 58.1K | 136.8K | 0.19x | 11.5K | 20.9K | 35.1K | 0.75x |
| cultivation_d5 | 73.7K | 3.4M | 3.5M | 28.2M | 19.5M | 47.8M | 0.07x | 1.9M | 10.0M | 12.0M | 0.29x |
| surface_d7_r7 | 0 | 4.7K | 4.7K | - | - | - | - | 0 | 0 | 0 | ∞ |
