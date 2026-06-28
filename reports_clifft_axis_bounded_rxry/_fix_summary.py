"""Rewrite BOUNDED_RXRY_SUMMARY.md from the (corrected, fixed-engine) per-step CSVs.
Fixes the prior mislabel (it reported max_M = the TRANSIENT peak in the 'resident' column)
and records that R_Y probabilities are now EXACTLY validated (see EXACT_RY_VALIDATION.md)."""
import csv
import sys
sys.path.insert(0, "/home/jung/clifft-paper")
import clifft

OUT = "reports_clifft_axis_bounded"
FEAS = ["coherent_rx_d3_r1", "coherent_rx_d3_r3", "coherent_ry_d3_r1", "coherent_ry_d3_r3"]
INFEAS = [("coherent_rx_d5_r1", 38), ("coherent_rx_d5_r5", 38),
          ("coherent_ry_d5_r1", 47), ("coherent_ry_d5_r5", 47)]


def k_of(c):
    return clifft.compile(open(f"qec_bench/circuits/{c}.stim").read()).peak_rank


def peaks(c):
    rows = list(csv.DictReader(open(f"{OUT}/bounded_{c}_per_step.csv")))
    tr = max(int(r["bounded_transient_qubits"]) for r in rows)
    res = max(int(r["bounded_resident_qubits"]) for r in rows)
    return tr, res


L = []
L.append("# Per-step ACTIVE-STATE: clifft_axis_bounded on OFF-AXIS (R_X / R_Y) noise\n")
L.append("`coherent_d{d}_r{r}` with the coherent over-rotation on the X or Y axis "
         "(`R_X(0.02)` / `R_Y(0.02)`) instead of Z, compiled with **no bytecode fusion** "
         "(`compile_bounded`).\n")
L.append("**transient** = peak materialised magic rank reached *during* a measurement "
         "(promote/flush before localize-and-drop); **resident** = settled magic rank "
         "*between* measurements. Per-step traces: `bounded_<circuit>_per_step.csv`.\n")
L.append("> **Correctness (both off axes).** The R_Y CZ/flush phase bugs are fixed "
         "(`clifft_axis/engine.py`). Off-axis Born probabilities are now validated to "
         "machine precision against an independent dense 2^17 statevector and "
         "`clifft.record_probabilities` — d3_r1 per-measurement |Δ| ≤ 2.6e-15, joint "
         "trajectory ≤ 1.4e-13 (R_Y, all measurements active) and R_X marginals match "
         "clifft's exact marginals to sampling precision (R_X has stabilizer-correlated "
         "*dormant* measurements; the marginal test is the appropriate check there). "
         "See `EXACT_RY_VALIDATION.md`.\n")
L.append("> **R_Y has NO peak-transient saving.** The off-axis R_Y over-rotation carries "
         "Y-support (x=1,z=1) on every data qubit, so the materialised magic rank equals "
         "Clifft's active rank k (transient = 2^k). The earlier '2^16→2^10, 64×' R_Y "
         "numbers were an artifact of the CZ double-conjugation bug (it discarded magic "
         "d.o.f.); they are superseded. The genuine bounded gain on R_Y is the **resident** "
         "drop (2×) and the **time-integrated** active-state/memory (~9–10×).\n")

L.append("## Feasible (d=3)\n")
L.append("| circuit | noise | Clifft k | transient | resident | transient dim | "
         "transient saving | resident saving |")
L.append("|---|---|--:|--:|--:|--:|--:|--:|")
for c in FEAS:
    k = k_of(c)
    tr, res = peaks(c)
    ax = "R_X" if "_rx_" in c else "R_Y"
    ts = f"2^{k - tr}" if k - tr else "1× (parity)"
    rs = f"2^{k - res}"
    L.append(f"| {c} | {ax} | {k} | {tr} | {res} | 2^{tr} | {ts} | {rs} |")

L.append("\n## Infeasible (d=5): off-axis noise keeps the magic rank > 2^26 (1 GiB)\n")
L.append("Unlike diagonal R_Z (d5_r5 transient = 2^13), off-axis noise carries X-support and "
         "keeps many magic d.o.f. simultaneously live, so localize-and-drop cannot bound the "
         "materialised register; it exceeds the 1-GiB ceiling.\n")
L.append("| circuit | noise | Clifft k | bounded status |")
L.append("|---|---|--:|---|")
for c, k in INFEAS:
    ax = "R_X" if "_rx_" in c else "R_Y"
    L.append(f"| {c} | {ax} | {k} | INFEASIBLE>2^26 |")

open(f"{OUT}/BOUNDED_RXRY_SUMMARY.md", "w").write("\n".join(L) + "\n")
print("WROTE", f"{OUT}/BOUNDED_RXRY_SUMMARY.md")
for c in FEAS:
    print(c, "k=", k_of(c), "transient/resident=", peaks(c))
