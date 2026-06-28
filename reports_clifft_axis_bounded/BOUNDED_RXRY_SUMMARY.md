# Per-step ACTIVE-STATE: clifft_axis_bounded on OFF-AXIS (R_X / R_Y) noise

`coherent_d{d}_r{r}` with the coherent over-rotation on the X or Y axis (`R_X(0.02)` / `R_Y(0.02)`) instead of Z, compiled with **no bytecode fusion** (`compile_bounded`).

**transient** = peak materialised magic rank reached *during* a measurement (promote/flush before localize-and-drop); **resident** = settled magic rank *between* measurements. Per-step traces: `bounded_<circuit>_per_step.csv`.

> **Correctness (both off axes).** The R_Y CZ/flush phase bugs are fixed (`clifft_axis/engine.py`). Off-axis Born probabilities are now validated to machine precision against an independent dense 2^17 statevector and `clifft.record_probabilities` — d3_r1 per-measurement |Δ| ≤ 2.6e-15, joint trajectory ≤ 1.4e-13 (R_Y, all measurements active) and R_X marginals match clifft's exact marginals to sampling precision (R_X has stabilizer-correlated *dormant* measurements; the marginal test is the appropriate check there). See `EXACT_RY_VALIDATION.md`.

> **R_Y has NO peak-transient saving.** The off-axis R_Y over-rotation carries Y-support (x=1,z=1) on every data qubit, so the materialised magic rank equals Clifft's active rank k (transient = 2^k). The earlier '2^16→2^10, 64×' R_Y numbers were an artifact of the CZ double-conjugation bug (it discarded magic d.o.f.); they are superseded. The genuine bounded gain on R_Y is the **resident** drop (2×) and the **time-integrated** active-state/memory (~9–10×).

## Feasible (d=3)

| circuit | noise | Clifft k | transient | resident | transient dim | transient saving | resident saving |
|---|---|--:|--:|--:|--:|--:|--:|
| coherent_rx_d3_r1 | R_X | 14 | 11 | 10 | 2^11 | 2^3 | 2^4 |
| coherent_rx_d3_r3 | R_X | 14 | 12 | 11 | 2^12 | 2^2 | 2^3 |
| coherent_ry_d3_r1 | R_Y | 16 | 16 | 15 | 2^16 | 1× (parity) | 2^1 |
| coherent_ry_d3_r3 | R_Y | 16 | 16 | 15 | 2^16 | 1× (parity) | 2^1 |

## Infeasible (d=5): off-axis noise keeps the magic rank > 2^26 (1 GiB)

Unlike diagonal R_Z (d5_r5 transient = 2^13), off-axis noise carries X-support and keeps many magic d.o.f. simultaneously live, so localize-and-drop cannot bound the materialised register; it exceeds the 1-GiB ceiling.

| circuit | noise | Clifft k | bounded status |
|---|---|--:|---|
| coherent_rx_d5_r1 | R_X | 38 | INFEASIBLE>2^26 |
| coherent_rx_d5_r5 | R_X | 38 | INFEASIBLE>2^26 |
| coherent_ry_d5_r1 | R_Y | 47 | INFEASIBLE>2^26 |
| coherent_ry_d5_r5 | R_Y | 47 | INFEASIBLE>2^26 |
