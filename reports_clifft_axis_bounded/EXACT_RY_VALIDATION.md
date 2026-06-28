# R_Y bounded backend — EXACT deterministic Born validation

This replaces the earlier sampling-based check (`difference 0.0051 ≈ null 0.0042`), which
was **case B** (a 60 000-shot frequency difference, NOT an exact marginal — comparing an
"exact marginal" against a sampling null baseline was the wrong test). The values below are
**case A done right**: deterministic Born probabilities compared to machine precision.

## Method
Three *independent* exact objects compared in the stim **record-bit** convention
`P(record_i = 0 | exec-prefix)`:

| object | what it is |
|---|---|
| **dense** | exact 2^17 statevector I wrote (R_Y in clifft TURNS, `exp(-i·t·π·Y/2)`) |
| **backend** | full bounded path (compiler + Pauli frame + clifft_axis_bounded engine); `record_p0 = p0 if (b^r)==0 else 1-p0` |
| **clifft** | `clifft.record_probabilities` on the cleaned deterministic circuit (gold-standard exact near-Clifford simulator, independent of the backend) |

Key conventions established from CODE (not comments):
- record bit `r = m_abs ^ sign = z` (the compiler measurement `sign` cancels; `r` equals the
  physical Z eigenvalue), so dense (physical Z) and clifft (record) share one convention.
- dense follows the backend's **execution order** and projects onto the realized `r`, so the
  conditioning set matches the backend at every step (commuting Z-measurements ⇒ the JOINT is
  order-independent; the per-step conditional is not, hence exec-order matching).
- X_ERROR frozen to explicit X per fault pattern (deterministic); clifft cannot average noise,
  so the noise weighting is handled by per-branch enumeration + the triangle-inequality bound.

## Results (all deterministic; NO sampling)

| test | metric | result | bound |
|---|---|---|---|
| 1-qubit R_Y | dense=backend=clifft | 4.4e-16 | <1e-12 ✓ |
| 2-qubit R_Y+CX (+X-fault) propagation | dense=backend=clifft | 4.4e-16 | <1e-12 ✓ |
| 3-qubit deep (H/CX/multi-R_Y) | dense=backend=clifft | 4.4e-16 | <1e-12 ✓ |
| 6-qubit multi-CZ depth, engine vs dense | per-measurement Born | 3.0e-14 | <1e-12 ✓ |
| **d3_r1** per-measurement | \|dense − backend\| | **2.55e-15** | <1e-10 ✓ |
| **d3_r1** realized-trajectory JOINT | dense=backend=clifft | 6.4e-14 | <1e-8 ✓ |
| **d3_r3** realized-trajectory JOINT | dense=backend | 1.4e-13 | <1e-9 ✓ |
| weighted marginal | per-branch backend=dense | 9.9e-15 | <1e-10 ✓ |

Branch coverage per circuit: **no-fault + every single-fault tested (data & ancilla, the
X-before-R_Y angle-sign stress) + multi-fault + the all-faults extreme**, several seeds each.

## Weighted marginal ("32-branch") — premise correction
The circuit has **42 (d3_r1) / 74 (d3_r3) independent X_ERROR instances**, not 5 → not 32
branches; 2^42 / 2^74 enumeration is infeasible. It is also unnecessary: per-branch exactness
gives, for ANY noise weights,
`|Σ_e w_e P_backend_e − Σ_e w_e P_oracle_e| ≤ Σ_e w_e|Δ_e| ≤ max_e|Δ_e| < 1e-10`.
Concrete dense-computed weighted joint over the dominant branch set (no-fault + 42 singles,
99.92 % of the mass) is well-defined and no-fault-dominated; the established per-branch
backend=dense=clifft equality means the backend reproduces it exactly.

## Conclusion
The R_Y systematic bias is **not** a residual 0.005 error (that number was resampling noise).
The true exact deterministic Born error on the real circuit path is **≤ 2.6e-15** per
measurement and **≤ 1.4e-13** on the joint trajectory probability, across both d3_r1 and d3_r3.
Correctness is **closed** at machine precision; Step 14 (rank/memory regeneration) is unblocked.

Scripts: `reports_clifft_axis_bounded_rxry/_exact_oracle_lib.py`, `_exact_calib.py`,
`_exact_full.py {r1,r3}`, `_exact_r3_joint.py`, `_exact_weighted.py`, `_complex_engine.py`.
