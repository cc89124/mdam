# Per-step memory (bytes): Clifft dense vs TTN vs near-Clifford

Per runtime step, the **resident memory in bytes** each backend holds. Companion to
`reports/per_step_active_state` (which plots the active-state *dimension*).

## The comparison is the EXPONENTIAL dense state only (apples-to-apples)

The `dense/NC` ratio compares **only the dense state**:

| backend | dense-state bytes |
|---|---|
| **Clifft** (crimson) | `16 · 2^k` (k = active idents) |
| **near-Clifford** (green solid) | `16 · 2^block` (largest magic block) |
| settled resident (green dashed) | `16 · 2^block_resident` |

Both sides count **only the exponential state**, no Clifford-frame metadata. On this
metric near-Clifford is **parity-or-win at every circuit** (`cultivation_d3`,
`cultivation_d5` = parity; everything else a win), exactly matching the dimension table
in `per_step_active_state` — there is **no `<1x` cell**.

## NC's Clifford-frame metadata is shown separately, NEVER in the ratio

`near_clifft_bytes = 16·2^block  +  metadata`, where **metadata** = the Clifford tableau
(`2n` Pauli images) + the *unapplied* pending rotations (the lazy-deferral buffer; an
already-flushed rotation is removed from `pending`, so this is never applied history).
It is the **faint dotted line** in each PNG and the **NC metadata** column in
`SUMMARY_TABLE.md`.

This metadata is **polynomial** (`O(n²)`), not exponential. **Clifft keeps a tableau of
the same order**, but its `16·2^k` baseline omits it — so charging it to NC while giving
Clifft a metadata-free baseline is unfair to NC. That asymmetry (now removed from the
ratio) is exactly what used to make tiny all-magic circuits look like a memory *loss*
even when the exponential state was parity-or-smaller — e.g. `coherent_d3_r1` has NC
magic `2^0` (no magic at all) yet its byte **sum** showed `<1x` purely from counting NC's
tableau. The dotted line still shows the full footprint, honestly, for anyone who wants
it; it just does not enter the headline comparison.

## PNG lines

- **crimson** — Clifft dense active state `16·2^k`.
- **steelblue** — TTN backend, actual stored bytes (its own representation; can exceed
  Clifft — that is the TTN/Clifft story, not NC).
- **green solid** — near-Clifford dense magic state `16·2^block` (the comparison line).
- **green dashed** — near-Clifford settled resident.
- **green dotted (faint)** — near-Clifford + Clifford-frame metadata (Clifft's own
  tableau omitted from its line).

## Tables

`SUMMARY_TABLE.md` — PEAK and SUM, with `NC dense state`, a separate `NC metadata`
column, and the apples-to-apples `dense/NC` ratio. Full per-step data in the `*.csv`.

## Regenerate

```
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 /home/jung/clifft_env/bin/python \
  -m ttn_backend.scripts.per_step_memory_compare <circuit> --metric memory \
  --out-dir reports/per_step_memory_3way --reuse-ttn
/home/jung/clifft_env/bin/python -m ttn_backend.scripts.per_step_memory_table \
  reports/per_step_memory_3way
```
