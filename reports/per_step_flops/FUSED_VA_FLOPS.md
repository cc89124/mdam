# Compute (FLOP): Clifft baseline vs live fused virtual-axis

MEASURED runtime FLOP of the **dense-free single-frame live fused-VA backend**
(`nearclifford_backend/virtual_axis/fused_single_frame.py`), summed over one shot, against the
**Clifft** `2^k` dense model. Complex-arith convention: **complex mult/scale = 6 (or 2 for a
real scalar), add/sub = 2, vdot = 8, norm = 4** per element.

Only the **runtime state-vector contraction** is counted (compile, Pauli-frame update,
pullback and core extraction are polynomial **bit-ops**, excluded — same basis as Clifft's
model). Counting is via `flop_meter` (in-source instrumentation of every elementwise combine +
wrapped `_apply_pauli_local`/`_vec_*`/`kron`/`vdot`/`norm`); regression-verified that the
instrumentation does not change `fused_ws`.

Reproduce: `clifft_env/bin/python reports/per_step_flops/fused_va_full_generate.py`
(raw: `fused_va_full_flops.csv`).

## Three buckets

* **apply/kron** — Pauli applies (`_apply_pauli_local`, 6N) + `kron` (6N).
* **vdot/norm** — Born/normalisation scans (`vdot` 8N, `norm` 4N).
* **elementwise** — the axpy / scale / combine arithmetic *inside* the contraction kernels
  (`out += co·P|φ⟩`, `c0=½(φ0±Pp1)`, `out/‖out‖`, the `_vec_h/_vec_s` basis combines). **This is
  the term the earlier "floor" omitted — and it is ~as large as apply/kron**, so the floor
  under-counted by ≈2×.

## fused **full** FLOP vs Clifft model  ← main result

`full = apply/kron + vdot/norm + elementwise`. `reduction = clifft_total / full`.

| circuit | clifft_k | fused_ws | apply/kron | vdot/norm | elementwise | **fused FULL** | Clifft TOTAL | reduction |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| coherent_d3_r1 | 5 | 1 | 288 | 288 | 320 | **896** | 20.8K | 23× |
| coherent_d3_r3 | 8 | 4 | 50.8K | 7.0K | 71.5K | **129.3K** | 564.3K | 4.4× |
| coherent_d5_r1 | 13 | 1 | 864 | 864 | 960 | **2.7K** | 12.8M | 4762× |
| **coherent_d5_r5** | 24 | 12 | 142.0M | 26.1M | 199.3M | **367.4M** | 209.3G | **570×** |
| distillation | 5 | 3 | 2.8K | 0.9K | 4.0K | **7.6K** | 25.1K | 3.3× |
| cultivation_d3 | 4 | 3 | 5.1K | 1.6K | 5.6K | **12.3K** | 26.4K | 2.2× |
| cultivation_d5 | 10 | 9 | 3.4M | 0.25M | 4.6M | **8.3M** | 3.5M | 0.42× |
| surface_d7_r7 | 0 | 0 | 0 | 0 | 0 | **0** | 4.7K | ∞ |

- **coherent_d5_r5: fused FULL = 367.4M vs Clifft 209.3G → 570×** (memory side: `2^12` vs `2^24` =
  4096×). The compute reduction is smaller than the memory reduction because the fused
  contraction still touches every amplitude of the `2^ws` workspace.
- **cultivation_d5 = 0.42×** (and cultivation_d3 2.2× is thin): all-magic circuits are the
  expected *compute* trade — genuinely-irreducible magic still pays a contraction FLOP that
  Clifft's analytic `2^k` model never spends — while their *memory* is parity/better.
- The `2^ws` state contraction is **irreducibly runtime** (it needs the sampled amplitude
  vector); offline planning removes only the excluded bit-ops, never this FLOP.
- **surface_d7_r7 = 0 FLOP (`∞`), and this is correct, not a missing measurement.** It is a
  **pure-stabilizer** circuit — *no* non-Clifford rotation (`k=0`, `pending` is always empty), so
  no measurement core ever materialises a dense magic vector and the backend degenerates to a
  plain CHP tableau (Aaronson–Gottesman). The metric we count is the **dense state-vector
  contraction FLOP**, of which there is literally none → `0`. Clifft's `4.7K` is *its* number:
  Clifft is a dense `2^k` statevector engine, so even with `k=0` (active state = 1 amplitude) it
  still runs a floating-point gate matmul per gate over thousands of gates. The honest reading is
  **"no floating-point arithmetic at all in our backend," not "free"**: we still pay polynomial
  GF(2) tableau bit-ops (gate `~n`, measure `~n²`), which are excluded from the FLOP count on
  **both** sides. So `∞` here = "the near-Clifford backend spends zero FLOP on a magic-free
  circuit," the expected degenerate-to-CHP case — not a headline compute claim.

## floor vs full (the correction)

The earlier number counted only apply/kron + vdot/norm (= **floor**); `full` adds the
elementwise combines. The floor is **not** the fused total — it is a lower bound.

| circuit | floor (old) | **full (main)** | full / floor | reduction (floor) → (full) |
|---|--:|--:|--:|--:|
| coherent_d3_r3 | 57.8K | 129.3K | 2.24× | 9.8× → 4.4× |
| **coherent_d5_r5** | 168.1M | **367.4M** | 2.19× | 1245× → **570×** |
| cultivation_d5 | 3.7M | 8.3M | 2.24× | 0.95× → 0.42× |
| distillation | 3.7K | 7.6K | 2.08× | 6.8× → 3.3× |
| cultivation_d3 | 6.7K | 12.3K | 1.85× | 3.9× → 2.2× |

(elementwise ≈ apply/kron because each Pauli-apply term is followed by a scale-and-accumulate
`out += co·P|φ⟩` of 8N vs the apply's 6N. Use **full** for any headline claim.)
