"""Aggregate the per-circuit *_summary.json from per_step_memory_compare into one
markdown table (clifft dense vs TTN vs near-Clifford block), PEAK and SUM."""
from __future__ import annotations
import json
import sys
from pathlib import Path

ORDER = ["coherent_d3_r1", "coherent_d3_r3", "coherent_d5_r1", "coherent_d5_r5",
         "distillation", "cultivation_d3", "cultivation_d5", "surface_d7_r7"]


def human(n):
    if n is None:
        return "n/a"
    n = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024 or unit == "TiB":
            return f"{n:.1f} {unit}"
        n /= 1024


def x(v):
    return f"{v:.1f}×" if v else "n/a"


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "reports/per_step_memory_3way")
    rows = []
    for circ in ORDER:
        p = out / f"{circ}_summary.json"
        if not p.exists():
            continue
        rows.append(json.loads(p.read_text()))
    # also pick up any extra summaries not in ORDER
    for p in sorted(out.glob("*_summary.json")):
        c = p.stem.replace("_summary", "")
        if c not in ORDER:
            rows.append(json.loads(p.read_text()))

    have_qubits = any("peak_nc_qubits" in r for r in rows)
    lines = []
    # The near-Clifford MAIN figure is the intra-step TRANSIENT high-water mark (the
    # honest memory-feasibility peak: a measurement's anticommutation-core flush
    # briefly forms a larger entangled block before its projector collapses it). The
    # settled step-boundary RESIDENT value is reported as a secondary column; it
    # under-reports the true peak (e.g. coherent_d5_r5: transient 13 vs resident 12).
    if have_qubits:
        lines.append("## Per-step ACTIVE-STATE SIZE: Clifft vs TTN vs near-Clifford\n")
        lines.append("Active-state size = log2(dense-equivalent dimension), in qubits. "
                     "Clifft = #active idents k; TTN = log2(stored_bytes/16); "
                     "near-Clifford = genuine magic-block qubits (0 for a pure "
                     "stabilizer circuit). **near-Clifford MAIN = intra-step transient "
                     "peak** (memory high-water mark); resident = settled "
                     "step-boundary value. Linear PNGs: "
                     "`<circuit>_per_step_qubits_linear.png`.\n")
        lines.append("| circuit | Clifft k | TTN log2-dim | "
                     "near-Clifford magic (transient) | near-Clifford (resident) |")
        lines.append("|---|--:|--:|--:|--:|")
        for r in rows:
            lines.append(
                f"| {r['circuit']} | {r.get('peak_clifft_qubits','?')} | "
                f"{r.get('peak_ttn_qubits')} | {r.get('peak_nc_qubits')} | "
                f"{r.get('peak_nc_qubits_resident')} |")
        lines.append("")
    lines.append("## Per-step memory: Clifft dense vs TTN vs near-Clifford block\n")
    lines.append("Linear-scale per-step PNGs are `<circuit>_per_step_linear.png`. "
                 "Clifft dense = `16*2^k` over concurrently-active idents; TTN = "
                 "resident bag bytes; near-Clifford = magic blocks + tableau. "
                 "**near-Clifford MAIN = intra-step transient high-water mark**; the "
                 "(resident) column is the settled step-boundary value. dense/NC and "
                 "TTN/NC ratios use the conservative transient peak. "
                 "(memory only; correctness covered elsewhere)\n")
    lines.append("### PEAK memory (max over steps)\n")
    lines.append("| circuit | k | Clifft dense | TTN | near-Clifford (transient) | "
                 "near-Clifford (resident) | dense/NC | TTN/NC |")
    lines.append("|---|--:|--:|--:|--:|--:|--:|--:|")
    for r in rows:
        lines.append(
            f"| {r['circuit']} | {r.get('max_active_idents','?')} | "
            f"{human(r['peak_clifft_bytes'])} | {human(r.get('peak_ttn_bytes'))} | "
            f"{human(r.get('peak_nc_bytes'))} | "
            f"{human(r.get('peak_nc_bytes_resident'))} | "
            f"{x(r.get('peak_dense_over_nc'))} | "
            f"{x(r.get('peak_ttn_over_nc'))} |")
    lines.append("\n### SUM memory (area under the per-step curve)\n")
    lines.append("| circuit | Clifft dense | TTN | near-Clifford (transient) | "
                 "near-Clifford (resident) | dense/NC | TTN/NC |")
    lines.append("|---|--:|--:|--:|--:|--:|--:|")
    for r in rows:
        lines.append(
            f"| {r['circuit']} | {human(r['sum_clifft_bytes'])} | "
            f"{human(r.get('sum_ttn_bytes'))} | {human(r.get('sum_nc_bytes'))} | "
            f"{human(r.get('sum_nc_bytes_resident'))} | "
            f"{x(r.get('sum_dense_over_nc'))} | {x(r.get('sum_ttn_over_nc'))} |")
    text = "\n".join(lines) + "\n"
    (out / "SUMMARY_TABLE.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
