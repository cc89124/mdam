# Offline selector + targeted peel — results

Two compile-time devices that, together, make the near-Clifford (NC) backend **never
worse than the dense clifft model** while cutting its runtime overhead on the
circuits where NC actually wins.

1. **Offline selector** (`nearclifford_backend/selector.py`): under fixed Pauli noise
   the structural schedule (clifft active rank `k_t`, NC peak block `b_t`) is
   shot-invariant, so the backend is chosen *per circuit* before sampling. Deployed
   peak = `min(2^k, 2^b)` → never worse than clifft.
2. **Targeted peel** (`targeted_peel`, DEFAULT ON): the structure-once discovery
   pre-pass also records, per `factor()`-call, *which qubits actually peel*; at
   runtime `factor` probes only those (`O(s·2^b)`) instead of the whole block/support
   (`O(b·2^b)`). State-exact by construction — `factor(only=S)` never changes
   amplitudes — so it is **record-bit-identical**; a missed peel could only enlarge a
   block, never corrupt the trajectory (`structure_once_debug` re-scans and counts).

## Per-circuit (seed 42; cross-checked seed 7, 3-seed peel agreement)

| circuit         | k (clifft) | b (NC, transient) | selector | flop_mm (irreducible) | flop_norm base | flop_norm targeted | reduction | record bit-identical |
|-----------------|-----------:|------------------:|----------|----------------------:|---------------:|-------------------:|----------:|:--------------------:|
| distillation    |          5 |                 4 | **NC**   |              2.85e+03 |       2.296e+03|           4.240e+02|   **81.5%**|         yes          |
| cultivation_d3  |          4 |                 5 | clifft   |              1.15e+04 |       2.086e+04|           9.200e+02|     95.6% |         yes          |
| cultivation_d5  |         10 |                14 | clifft   |              1.94e+06 |       1.003e+07|           6.974e+05|     93.0% |         yes          |
| coherent_d5_r5  |         24 |                19 | **NC**   |              7.99e+08 |       2.024e+09|           1.892e+08|   **90.7%**|         yes          |

`flop_mm` (the O(2^b) Born/projection/rotation-apply arithmetic) is **unchanged** by
targeting — only `flop_norm` (the factor scan, NC's sole compute excess over clifft)
drops. b_max and all `flop_mm` are bit-for-bit invariant.

## Deployment (selector picks the backend; targeted peel applies on NC)

| circuit (NC-selected) | total compute base | total compute targeted | total reduction | dominant term after |
|-----------------------|-------------------:|-----------------------:|----------------:|---------------------|
| distillation          |        5.15e+03    |            3.28e+03    |     −36%        | flop_mm (arith)     |
| coherent_d5_r5        |        2.82e+09    |            9.88e+08    |     −65%        | flop_mm (arith)     |

After targeting, `flop_norm < flop_mm` on coherent_d5_r5 (1.89e8 < 7.99e8): the
factor scan is no longer the bottleneck; the irreducible `O(2^b)` dense arithmetic is
— the intended end state ("runtime cost = vector arithmetic on the precomputed core").

cultivation_d3 / cultivation_d5 have `b ≥ k`, so the selector routes them to clifft;
their (also large) flop_norm reduction is shown only to demonstrate targeting works —
in deployment that compute is never paid (clifft is used).

## Positioning

NC is **not** universally superior. The offline selector predicts when lazy NC
materialisation exposes a smaller measurement-visible magic core than clifft's active
rank and routes there; otherwise it falls back to clifft. cultivation is the negative
case the selector is meant to catch, not a failure. Targeted peel then removes NC's
factor-scan overhead on the circuits where NC is selected.

Reproduce: `python -m nearclifford_backend.scripts.targeted_peel_table` (fast circuits;
pass circuit names as args). Selector: `nearclifford_backend/selector.py`.
