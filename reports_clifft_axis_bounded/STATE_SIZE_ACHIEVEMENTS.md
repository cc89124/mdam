# STATE-SIZE (MEMORY) ACHIEVEMENTS — `clifft_axis_bounded` vs Clifft

**Scope of this document.** This is the consolidated, FLOP-free record of the **active-state size**
(dense magic-register dimension = peak memory) of the live `CliftAxisBoundedNearClifford` backend,
measured against the Clifft reference statevector machine. Every number here is **memory**, not
arithmetic. FLOP accounting lives separately under `flop_accounting/`; this file deliberately
excludes it.

All bounded numbers are read from the per-step traces the **authoritative** engine emits during its
**own** run (`bounded_<circuit>_per_step.csv`) — no TTN, no block backend, no Clifft state, no forced
outcomes. This is the canonical `CliftAxisBoundedNearClifford` path with its full quotient machinery
(block-factoring, `_reduce_dead`, frame-routed Cliffords).

> **IMPORTANT — these numbers are the AUTHORITATIVE path, NOT the U_C-identity reduced probe.** The
> reduced-data-plane probe used in the FLOP work (`flop_accounting/PHASE15_*`) routes **every** active
> Clifford into the numerical array. On `r = k` circuits (cultivation) this is rank-bit-identical to
> the authoritative path, so the FLOP probe there is faithful. But on a **genuine quotient circuit
> (`r < k`)** the probe **destroys the quotient**: measured on `coherent_d5_r5` it blows up to
> `r = 24` (= Clifft's `k`, the 256 MiB array) versus the authoritative `r = 13`, because the active
> CNOTs the authoritative path keeps in the cheap Clifford frame are forced into the dense array. So
> **the state-size wins below belong to the authoritative engine; the U_C-identity probe is a FLOP
> diagnostic for the `r = k` regime only and does NOT preserve these memory results.** Reaching the
> `r < k` memory wins *and* the clean FLOP consume simultaneously requires keeping active Cliffords
> frame-symbolic (the authoritative discipline), not the U_C-identity routing.

---

## 0. Definitions (read this first)

| term | meaning |
|---|---|
| **active-state / magic register** | the dense complex vector `phi` of dimension `2^b` over the `b` "magic" (non-stabilizer) qubits. This is the **only** exponential object. |
| **state size** | `2^b` — the active-state dimension. |
| **memory (bytes)** | `16 · 2^b` (complex128 = 16 B/amplitude). |
| **resident** `b` | the **settled** magic rank held *between* measurements — the persistent memory footprint. |
| **transient** `b` | the **peak** magic rank reached *during* a measurement core (the +1 promote/flush spike *before* localize-and-drop). transient = resident or resident+1. |
| **Clifft `k`** | Clifft's active rank = `prog.peak_rank` (the dense register Clifft must hold). The HARD budget cap is `2^k`. |
| **excluded** | the `O(n²)`-bit CHP stabilizer **tableau** (polynomial; the same basis Clifft also pays). Only the **exponential** dense register is compared. |
| **PEAK** | the single largest `2^b` over the whole run. |
| **integrated SUM** | `Σ_t 2^{b_t}` over all runtime steps `t` — the **time-averaged** memory pressure (area under the rank-mountain), not just its highest point. |

**Two distinct savings are reported and must not be conflated:**
- **PEAK saving** `2^{k − b_peak}` — how much smaller the largest array is than Clifft's `2^k`.
- **Integrated saving** `S_cl / S_bn = (Σ_t 2^{n_active,t}^{clifft}) / (Σ_t 2^{b_t}^{bounded})` — area
  ratio, using **Clifft's real per-step rank** `n_active` (not flat peak). This is the honest
  time-averaged advantage and is typically *larger* than the peak saving because Clifft sits near its
  peak far longer than bounded does.

---

## 1. THE HARD GUARANTEE

The engine carries a `DenseMemoryBudget` with cap `2^k_clifft` and `enforce=True`
(`set_clifft_budget(prog.peak_rank)`). **The resident magic rank never exceeds Clifft's `k`** — the
guard is active and has been observed to fire. So across every benchmark below:

> **`b_resident ≤ k` always, and `b_transient ≤ k+? ` bounded by the same budget — the bounded engine
> is provably never larger than Clifft, and is strictly smaller whenever the circuit admits a
> quotient.**

The interesting question is therefore *by how much* `b < k`, which splits into three regimes (§4).

---

## 2. MASTER TABLE — PEAK active-state (the headline memory numbers)

Source: fresh recomputation from `bounded_<circuit>_per_step.csv` (cross-checks `DETAILED_TABLE.md`
§1 and `BOUNDED_SUMMARY.md`). `dim = 2^b`; `bytes = 16·2^b`.

| circuit | noise | Clifft `k` (=2^k) | bounded **transient** `2^b` | bounded **resident** `2^b` | **PEAK saving** (resident) | **resident bytes** vs Clifft |
|---|---|--:|--:|--:|--:|--:|
| coherent_d5_r1 | R_Z | **13** (2^13) | **2^0** | **2^0** | **8 192×** | 16 B vs 128 KiB |
| coherent_d5_r5 | R_Z | **24** (2^24) | **2^13** | **2^12** | **4 096×** | 64 KiB vs **256 MiB** |
| coherent_d3_r1 | R_Z | 5 (2^5) | 2^0 | 2^0 | 32× | 16 B vs 512 B |
| coherent_d3_r3 | R_Z | 8 (2^8) | 2^5 | 2^4 | 16× | 256 B vs 4.0 KiB |
| coherent_rx_d3_r1 | R_X | 14 (2^14) | 2^11 | 2^10 | 16× | 16 KiB vs 256 KiB |
| coherent_rx_d3_r3 | R_X | 14 (2^14) | 2^12 | 2^11 | 8× | 32 KiB vs 256 KiB |
| coherent_ry_d3_r1 | R_Y | 16 (2^16) | 2^16 | 2^15 | 2× | 512 KiB vs 1.0 MiB |
| coherent_ry_d3_r3 | R_Y | 16 (2^16) | 2^16 | 2^15 | 2× | 512 KiB vs 1.0 MiB |
| distillation | T | 5 (2^5) | 2^4 | 2^3 | 4× | 128 B vs 512 B |
| cultivation_d3 | T | 4 (2^4) | 2^4 | 2^3 | 2× | 128 B vs 256 B |
| cultivation_d5 | T | 10 (2^10) | 2^10 | 2^9 | 2× | 8.0 KiB vs 16.0 KiB |
| surface_d7_r7 | — | 0 (2^0) | 2^0 | 2^0 | 1× (Clifford) | 16 B vs 16 B |

**Headline:** on `coherent_d5_r5` Clifft must hold a **256 MiB** statevector (`2^24`); the bounded
engine holds **64 KiB resident** (`2^12`) / **128 KiB transient** (`2^13`) — a **4 096× / 2 048×**
reduction at the peak. On `coherent_d5_r1` the diagonal noise never lifts the register off the
stabilizer subspace, so bounded stays at **`2^0` (16 B)** vs Clifft's `2^13` (128 KiB) = **8 192×**.

---

## 3. INTEGRATED (time-averaged) active-state — the area-under-the-mountain

Source: `Σ_t 2^{b_t}` from the per-step CSV; Clifft side uses its **real per-step rank** `n_active`
(the faithful trajectory, not flat-peak). This is the honest time-integrated memory advantage.

| circuit | Clifft Σ 2^{n_active} | bounded resident Σ 2^b | **integrated saving `S_cl/S_bn`** | Clifft SUM bytes | bounded SUM bytes |
|---|--:|--:|--:|--:|--:|
| coherent_d5_r5 | 2^34.78 | 2^22.81 | **4 007×** | 440.4 GiB | 112.5 MiB |
| coherent_d5_r1 | 2^21.05 | 2^9.74 | **2 535×** | 33.2 MiB | 13.4 KiB |
| coherent_rx_d3_r1 | 2^19.31 | 2^14.85 | 22.0× | 9.9 MiB | 460.8 KiB |
| coherent_d3_r3 | 2^15.85 | 2^11.69 | 17.9× | 922.0 KiB | 51.6 KiB |
| coherent_ry_d3_r1 | 2^22.36 | 2^18.94 | 10.7× | 82.1 MiB | 7.7 MiB |
| coherent_d3_r1 | 2^11.42 | 2^8.00 | 10.7× | 42.8 KiB | 4.0 KiB |
| coherent_ry_d3_r3 | 2^23.94 | 2^20.69 | 9.52× | 245.4 MiB | 25.8 MiB |
| coherent_rx_d3_r3 | 2^21.04 | 2^17.96 | 8.45× | 32.8 MiB | 3.9 MiB |
| cultivation_d5 | 2^18.97 | 2^17.79 | 2.27× | 7.8 MiB | 3.5 MiB |
| distillation | 2^14.15 | 2^13.04 | 2.15× | 283.1 KiB | 131.6 KiB |
| cultivation_d3 | 2^11.52 | 2^10.44 | 2.11× | 45.9 KiB | 21.7 KiB |
| surface_d7_r7 | 2^11.43 | 2^11.43 | 1.0× | 43.0 KiB | 43.0 KiB |

**Note the cross-over:** for `coherent_d5_r5` the *peak* saving is 4 096× but the *integrated* saving
is **4 007×** — comparable, because the register spends a long time near peak. For `coherent_d3_r3`
the integrated saving (17.9×) **exceeds** the peak saving (16×), because Clifft holds its peak far
longer than bounded, which spikes only transiently. **For `coherent_d5_r5` the run integrates to
440 GiB of Clifft-state-time vs 112 MiB for bounded** — the single most dramatic number in the suite.

---

## 4. THE THREE REGIMES

### Regime A — genuine quotient `b ≪ k` (diagonal / R_Z coherent noise) — the big wins
`coherent_d{3,5}_r{1,5}`. Diagonal over-rotations (`R_Z`) and their measurements localize-and-drop
cleanly, so the magic rank collapses far below Clifft's `k`:
- `coherent_d5_r5`: `b=12 ≪ k=24` → **4 096× peak, 4 007× integrated** (64 KiB vs 256 MiB).
- `coherent_d5_r1`: `b=0 ≪ k=13` → **8 192× peak, 2 535× integrated** (16 B vs 128 KiB).
- `coherent_d3_r1/r3`: 32× / 16× peak.

This is the regime where the architecture's premise pays off maximally: the dense register tracks
only the *truly* non-stabilizer degrees of freedom, which are a small fraction of Clifft's `k`.

### Regime B — parity `b = k` peak, resident-only gain (T-gate / cultivation, R_Y)
`cultivation_d{3,5}`, `coherent_ry_d3_r{1,3}`. Here Clifft's `k` is *already* tight — the magic rank
genuinely reaches `k` at the transient peak, so **PEAK transient = parity (1×)**. The bounded gain is
the **resident** drop (the engine releases the +1 working spike immediately after each measurement):
**resident = `2^{k-1}` = 2× smaller**, and the **integrated** saving is 2.1–2.3× (cultivation) /
9.5–10.7× (R_Y, whose long tail of low-rank steps Clifft pays at high `n_active`). For cultivation the
memory story is "**never larger than Clifft, 2× resident, 2.2× time-integrated**"; the parity at the
transient peak is expected (the T-gate cultivation circuit *is* maximally magic at its peak).

### Regime C — off-axis X/Y support, partial or infeasible
`coherent_rx_*` (partial: 8–16× peak, X-support still drops *some* rank) and the off-axis **d5**
family, which is **INFEASIBLE**: X/Y over-rotation carries `X`-support on every data qubit, keeping
many magic d.o.f. simultaneously live so localize-and-drop cannot bound the register:

| circuit | noise | Clifft `k` | bounded status |
|---|---|--:|---|
| coherent_rx_d5_r1 | R_X | 38 | **INFEASIBLE** (> 2^26 = 1 GiB ceiling) |
| coherent_rx_d5_r5 | R_X | 38 | **INFEASIBLE** (> 2^26) |
| coherent_ry_d5_r1 | R_Y | 47 | **INFEASIBLE** (> 2^26) |
| coherent_ry_d5_r5 | R_Y | 47 | **INFEASIBLE** (> 2^26) |

Honest limit: the bounded advantage is a function of how much of Clifft's `k` is *genuinely*
non-stabilizer. Diagonal noise → almost none → huge win. Off-axis d=5 → nearly all of it → no win
(and the dense register would exceed 1 GiB, so it is not run). This is the boundary of the method.

---

## 5. PER-STEP TRAJECTORY CHARACTERIZATION (the rank-mountain shape)

Source: `bounded_<circuit>_per_step.csv`. `#@res_pk` = runtime steps spent *at* the resident peak;
`#tr>res` = number of transient spikes (measurement cores that momentarily promoted +1 above
resident). This shows the memory is a **spiky mountain**, not a plateau.

| circuit | steps | resident peak `b` | transient peak `b` | steps at res-peak | transient spikes (`tr>res`) |
|---|--:|--:|--:|--:|--:|
| coherent_d5_r5 | 3 229 | 12 | 13 | 1 783 | **60** |
| coherent_d5_r1 | 858 | 0 | 0 | 858 | 0 |
| coherent_d3_r3 | 565 | 4 | 5 | 171 | 12 |
| coherent_d3_r1 | 256 | 0 | 0 | 256 | 0 |
| coherent_rx_d3_r1 | 283 | 10 | 11 | 6 | 14 |
| coherent_rx_d3_r3 | 660 | 11 | 12 | 6 | 30 |
| coherent_ry_d3_r1 | 366 | 15 | 16 | 4 | 17 |
| coherent_ry_d3_r3 | 851 | 15 | 16 | 12 | 33 |
| cultivation_d3 | 345 | 3 | 4 | 143 | 5 |
| cultivation_d5 | 1 785 | 9 | 10 | 417 | 15 |
| distillation | 2 041 | 3 | 4 | 406 | 5 |
| surface_d7_r7 | 2 750 | 0 | 0 | 2 750 | 0 (pure Clifford) |

Reading: the transient peak (`b+1`) is touched only at the **`#tr>res` measurement cores** — e.g.
`coherent_d5_r5` reaches its `2^13` transient only at **60** of 3 229 steps, and sits at the `2^12`
resident at 1 783 steps; the rest of the run is *below* resident peak. The R_X/R_Y circuits spend
very few steps at peak (`#@res_pk` 4–6), so the time-integrated saving (§3) far exceeds the peak
saving — most of the run is cheap.

---

## 6. CONTEXT — vs the TTN backend
The earlier TTN/joint-cutting backend on this same `d5_r5` regime peaked at a bond-dimension state far
larger than the bounded register; the bounded near-Clifford engine is **~1 600×** smaller than that
TTN peak (see memory `nearclifford-backend-over-promotion`). Against **Clifft** the figure is the
4 096× / 8 192× above. So the ordering on `coherent_d5_*` memory is:
**TTN ≫ Clifft (`2^k`) ≫ bounded resident (`2^b`)**, with bounded the smallest by a wide margin in the
diagonal regime.

---

## 7. CORRECTNESS ANCHOR (the state size is achieved EXACTLY, not by approximation)
The memory reduction is **not** a truncation/compression — it is an exact change of representation.
Anchored by independent dense + Clifft cross-checks:
- **R_Y off-axis** (the worst case for the representation): per-measurement Born `|Δ| ≤ 2.6e-15`,
  joint trajectory `≤ 1.4e-13`, vs an independent `2^17` dense statevector **and**
  `clifft.record_probabilities` (`EXACT_RY_VALIDATION.md`).
- **Diagonal / cultivation**: the reduced data plane is records/peak-rank/p0 **bit-identical** to the
  authoritative path over 60 seeds, and the measurement instrument is dense-oracle exact to machine
  precision (`flop_accounting/PHASE15_PHASE_B_COMPLETE.md`).

So every `2^b` in this document holds the **same physical state** Clifft holds in `2^k`, to machine
precision — the smaller number is a smaller *representation*, not a smaller *fidelity*.

---

## 8. HONEST CAVEATS (for judging the memory claim)
1. **transient vs resident labeling.** The `*_qubits.png` plots' blue peak is the **transient**
   (`b+1` working spike), not the persisted **resident** (`b`). Both are reported here; the resident
   is the steady-state footprint, the transient is the true peak allocation.
2. **Tableau excluded.** The `O(n²)`-bit CHP stabilizer tableau is not counted — but Clifft pays the
   same polynomial basis, so excluding it from both sides is fair; only the **exponential** register
   differs.
3. **Clifft `2^k` is its real cap**, taken from `prog.peak_rank` (not modeled). The *integrated* Clifft
   column (§3) uses Clifft's **real per-step `n_active`** (faithful), so the integrated savings are
   honest area ratios, not flat-peak inflation. (The flat-peak `2^k` assumption is used **only** in the
   FLOP doc, not here.)
4. **Off-axis d=5 infeasible** (§4C) — the method has a genuine boundary; reported, not hidden.
5. **`coherent_d5_r5` resident is `2^12`** (CSV ground truth); `BOUNDED_SUMMARY.md` lists `13`, which
   is the **transient**. The authoritative split is transient `2^13` / resident `2^12`.

---

## 9. DATA PROVENANCE (every number is traceable)
| number | source file |
|---|---|
| per-step rank trajectory (`n_active`, resident, transient) | `bounded_<circuit>_per_step.csv` |
| PEAK transient/resident table | `BOUNDED_SUMMARY.md`, `DETAILED_TABLE.md` §1, recomputed here §2 |
| integrated SUM (state + bytes) | `DETAILED_TABLE.md` §2/§3, recomputed here §3 |
| R_X/R_Y peak + infeasible d5 | `DETAILED_TABLE_RXRY.md`, `BOUNDED_RXRY_SUMMARY.md` |
| per-step memory plots | `bounded_vs_clifft_<circuit>_qubits.png` |
| exactness anchor | `EXACT_RY_VALIDATION.md`, `flop_accounting/PHASE15_PHASE_B_COMPLETE.md` |
| methodology (how `charge` → rank) | `flop_accounting/METHODOLOGY.md` §1–2 |

---

## 10. ONE-LINE SUMMARY
On diagonal-noise QEC (`coherent_d5_r5`) the bounded near-Clifford engine holds the **same exact state
in 64 KiB that Clifft holds in 256 MiB (4 096× peak), integrating to 112 MiB vs 440 GiB of
state-time (4 007×)**; it is **never larger than Clifft** on any benchmark (hard `2^k` guard), with the
gain ranging from **8 192×** (diagonal, rank collapses to `2^0`) down to **2× resident** (maximally-
magic T-gate cultivation where `b=k` at the peak), and a genuine **infeasibility boundary** at
off-axis d=5 where the state is irreducibly `> 2^26`.
