# Per-step TOTAL memory (bytes): Clifft dense vs TTN vs near-Clifford

Per runtime step, the **total resident memory in bytes** each backend holds — the whole
footprint, not just the dense state. This is the point of this report; the **state-only
(exponential dimension)** comparison is the companion `reports/per_step_active_state`.

## What is counted

| backend | total resident bytes |
|---|---|
| **Clifft** (crimson) | `16 · 2^k` (dense active state; k = active idents) |
| **TTN** (steelblue) | actual stored tensor bytes |
| **near-Clifford** (green solid) | **`16 · 2^block`  +  metadata** = dense magic state **+** Clifford frame (tableau) **+** unapplied pending rotations |

The green **dotted** line breaks out the dense magic state alone (`16·2^block`); the gap
up to the solid line is NC's polynomial metadata.

## Reading the `dense/NC` ratio honestly (total vs total)

`dense/NC` compares the two totals. On large/real circuits the exponential term
dominates and NC wins hugely (`coherent_d5_r5` total **135 KiB** vs Clifft **256 MiB**).

On **tiny all-magic** circuits the ratio can dip **below 1×** — NC's total exceeds
Clifft's. That is **not** an exponential regression; it is NC's `O(n²)` bookkeeping
dominating a tiny `2^block` state. Two things to keep in mind:

1. **It is conservative against NC.** Clifft is itself a near-Clifford simulator and
   keeps a Clifford tableau of the same `O(n²)` order — but its `16·2^k` baseline counts
   **only the dense state**, not that tableau. So the total comparison charges NC for its
   bookkeeping while giving Clifft a metadata-free baseline.
2. **The exponential state never loses.** Compare `16·2^block` vs `16·2^k` (the dotted
   line, = the `per_step_active_state` dimension table) and NC is **parity-or-win at
   every circuit**. Example: `coherent_d3_r1` has NC magic `2^0` (no magic at all), yet
   its *total* byte SUM is `<1×` purely from the tableau — the dense state is infinitely
   smaller than Clifft's `2^5`.

`pending` holds only **un-applied** rotations: a flushed rotation is removed from it
(`self.pending = keep` in `_flush_core`), so this metadata is never applied history.

## PNG lines

- **crimson** — Clifft dense active state `16·2^k`.
- **steelblue** — TTN backend, actual stored bytes (can exceed Clifft — that is the
  TTN/Clifft story, not NC).
- **green solid** — near-Clifford **TOTAL** footprint (dense magic state + metadata).
- **green dashed** — near-Clifford total, settled resident.
- **green dotted (faint)** — near-Clifford **dense magic state only** (`16·2^block`,
  metadata excluded); the gap to the solid line is the metadata.

## Tables

`SUMMARY_TABLE.md` — PEAK and SUM: `NC TOTAL`, broken out into `dense state` + `metadata`,
with the total-vs-total `dense/NC`. Full per-step data in the `*.csv`.

## Regenerate

```
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 /home/jung/clifft_env/bin/python \
  -m ttn_backend.scripts.per_step_memory_compare <circuit> --metric memory \
  --out-dir reports/per_step_memory_3way --reuse-ttn
/home/jung/clifft_env/bin/python -m ttn_backend.scripts.per_step_memory_table \
  reports/per_step_memory_3way
```
