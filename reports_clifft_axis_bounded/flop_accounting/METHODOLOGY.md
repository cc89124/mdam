# METHODOLOGY — exactly how every FLOP number and graph is computed

Purpose: make the construction fully transparent so the representation and the bounded-vs-clifft
comparison can be judged. Read §9 first if you only want the assumptions.

---

## 1. Where every number comes from — the `budget.charge` hook
Every dense magic kernel in the engine already calls
```
DenseMemoryBudget.charge(resident, transient, where)
```
- `resident` = amplitudes the kernel actually touches = `phi.size` = `2^r`
- `where` = kernel label (`'rot:offdiag'`, `'sqnorm'`, `'purge:h'`, `'drop'`, …)

We monkeypatch `charge` to record one row per call and return the original result unchanged
(non-invasive: meter on/off → record & max_M **bit-identical**). **One event = one dense-kernel
call.** Raw data per event = `(N=resident, where)`. Stabilizer measurements are handled by the
Clifford frame and do **not** call `charge`, so they are absent (this is why the event axis ~285 is
shorter than runtime steps ~560).

## 2. The three quantities derived per event
| quantity | formula | note |
|---|---|---|
| rank `r` | `round(log2(N))` | N is a power of 2 → exact |
| bounded FLOP | `coeff[where] × N` | arithmetic on the actual array |
| clifft FLOP | `coeff[where] × 2^k`, shared events only | clifft modeled at peak (see §4) |
`2^k = cap = prog.peak_rank` (fixed per circuit).

## 3. Coefficient table `coeff[where]` (arithmetic ops per amplitude) — VALIDATED
```
offdiag (rot/collapse)  12   # 2x2 butterfly. alpha=cos is REAL → real×complex=2, not 6 (so 12N not 16N)
diag    (rot/collapse)   6   # one complex scalar mult / amplitude
meas / exp  (<P>)       10   # conj·gather·sum
sqnorm                   2   # |z|²=4 but N charged, N/2 processed → 4·(N/2)/N = 2
purge:h                  5
purge:s                  3
cnot / drop / promote / gf2scan   0   # permutation / copy / resize = traffic, NOT arithmetic
+ per collapse:  +6N    # norm+renorm, NOT charged by engine → added by hand (MODELED, §9.8)
```
Validation: a separate direct-event meter (counts real N / branch / passes inside each kernel) matches
this hook **exactly** at r=1..6 micro + real circuits + unit-calls. (The deprecated
`_SUPERSEDED_flop_count_v1.py` used offdiag=16 and sqnorm=N — both wrong; do not use.)

Memory traffic uses a parallel read/write-count table (`R words`, `W words`); `bytes = 16·(R+W)`
(complex128 = 16 B). Traffic is reported separately from FLOP.

## 4. The clifft "modeled" baseline — exact definition
For events whose `where` is **shared** (`rot`, `collapse`, `meas`, `sqnorm` — work clifft must also do):
```
clifft_FLOP += coeff[where] × 2^k        (cap = 2^peak_rank, constant)
```
bounded-only events (`purge`, `drop`, `reduce`, `promote`, `gf2scan`) are **not** charged to clifft.
Two assumptions, both unavoidable because the core is a compiled `.so`:
1. clifft runs the **same shared events, same count** as bounded;
2. clifft runs each at the **flat peak `2^k`** (rank held at peak throughout).
→ clifft FLOP is therefore an **upper-bound-flavoured estimate**. clifft's real active rank rises and
falls (see the `*_qubits.png` plots one level up), so a trace-faithful model would lower it.

## 5. Figure ① `flop_rank_trace_<circ>.png` (2 panels)
Arrays over events i=0..E-1: `rank[i]`, `fb[i]` (bounded FLOP), `fc[i]` (clifft-modeled FLOP).
- **Top — rank mountain:** blue step plot of `rank[i]` = the rank of the array bounded's i-th kernel
  ran on. ⚠️ at a rotation flush the array is momentarily +1-promoted, so this blue peaks at the
  **transient** value (e.g. 5), matching the `*_qubits.png` *transient* curve, not its *resident* (4).
  Red dashed = `k`, drawn **flat** (assumption §4.2 — not clifft's real trajectory).
- **Bottom — FLOP:** scatter of `fb[i]` vs i, **colored by `rank[i]`**; stems to 0; right twin-axis =
  cumulative `Σfb` (blue) vs `Σfc` (red dashed). Log axes if total ratio > 50.
- Read: blue cumulative **saturates** (no more large work) while red keeps rising linearly (`2^k` each).

## 6. Figure ② `flop_by_rank_<circ>.png` (grouped bars) — note the red bar's meaning
Bucket events by integer rank r:
```
bvals[r] = Σ_{events at rank r}  coeff×N              # bounded work done at rank r
cvals[r] = Σ_{events at rank r}  (shared?coeff:0)×2^k # the SAME events, costed at clifft's 2^k
```
Blue = `bvals[r]`, red = `cvals[r]` (log-y if ratio > 50). **The red bar is NOT "clifft's work at
rank r"** (clifft has no rank-r bucket) — it is "the operations bounded localized to rank r, costed as
if clifft did them at `2^k`." So red is large even at low r.

## 7. Tables (`data/flop_all.csv`) — column definitions
| column | definition |
|---|---|
| `bounded_FLOP` | `Σ_all events coeff×N` |
| `clifft_FLOP_modeled` | `Σ_shared events coeff×2^k` |
| `F_cl_over_F_bn` | ratio of the two |
| `R_words` / `W_words` / `bytes` | `Σ rc×N`, `Σ wc×N`, `16·(R+W)` |
| `S_cl_over_S_bn` | from per_step CSV: `Σ_t 2^{n_active_t}` / `Σ_t 2^{bounded_resident_t}` |
| `invocations` | number of dense-kernel events |
| `ms` | bounded 1-shot wall-clock (**bounded only, Python/scalar-bound**) |
Asymmetry to note: **`F_cl/F_bn` models clifft as flat peak `2^k`**, but **`S_cl/S_bn` uses clifft's
real per-step rank `n_active`** — two *different* clifft models, so the two ratios need not agree
(e.g. d3_r3: S 17.9× vs F 13.4×).

## 8. peak / shoulder / tail decomposition
Group the same events: **peak** = `r=r_max`, **shoulder** = `r=r_max−1`, **tail** = `r<r_max−1`.
Each band's `clifft/bounded` ratio is its local saving. When `r_max=k` (R_Y) the peak band has both
sides at `2^k` → ≈1× (wash); the advantage comes from shoulder (~2×) + tail (large).

## 9. Assumptions & limits — for judging validity
**Solid:**
1. bounded FLOP — validated, hook==direct, bit-identical, N exact (2^r).
2. rank = round(log2 N) — exact.
3. permutations = 0 FLOP, traffic separated — per convention.

**Assumptions / weak links (judge here):**
4. **clifft flat-peak (`2^k` constant)** — upper-bound flavour; clifft's real rank varies → current
   `F_cl/F_bn` may overcount clifft. A faithful model uses `n_active` (we already have it) and would
   lower the ratios.
5. **clifft event count = bounded's** — clifft may fuse/skip ops; not verifiable (compiled `.so`).
6. **"resident rank" shown is the transient (+1 working) size** — blue peak = transient (5), not the
   persisted resident (4). Labeling caveat.
7. **`S_cl/S_bn`** uses clifft's real rank but weights every runtime step equally (ignores per-kernel
   cost `c_i`) → differs from the FLOP ratio (R_Y: S 10.7× vs F 4.3×).
8. **`+6N` norm/renorm per collapse** — modeled by hand (engine does not charge it), not measured.
9. **R/W traffic coeffs** — derived from the same kernel analysis, not independently bit-counted like
   FLOP.
10. **`ms` is bounded-only, Python/scalar-bound** — not a wall-clock-speedup claim; clifft runtime not
    yet measured.

**Bottom line:** the bounded side (blue), ranks, and traffic are measured/validated; the clifft side
(red) rests on assumptions §9.4–9.5 (flat-peak + same events). The comparison's strength hinges on
those two; replacing flat-peak with the `n_active` trajectory is the cheap, defensible refinement.
