"""Per-step memory: deferred-affine (A) representation vs clifft dense.

For each bytecode step we track the active-state size two ways:

  clifft dense   = 16 * 2^(active idents)              bytes  (complex128 vector)
  affine (A)     = ceil(a * S / 8) + ceil(S / 8)       bytes
                   a = # active idents (rows of the GF(2) map)
                   S = # live source bits (cols; referenced by some active row)
                   (+ source-value bitmask; the phase polynomial f is never built)

We report PEAK and SUM-over-steps for both and the ratios, mirroring the
§4.2 dense-vs-TTN table.  The affine representation is exact ONLY on the
boundary-free fragment; a non-diagonal op (H / U2 / U4 / X-basis INTERFERE
measurement) forces materialization, so circuits that contain one are reported
as REFUSED with the boundary step (the affine memory model does not apply to the
full run there).

Run:  python -m ttn_backend.scripts.affine_memory_compare [circuits...]
"""
from __future__ import annotations

import argparse
import clifft

from ttn_backend import treewidth as T_mod
from ttn_backend import frame_layer as ds_mod

# Must match AffineActiveBackend's boundary set.
BOUNDARY = {
    "OP_ARRAY_H", "OP_ARRAY_U2", "OP_ARRAY_U4",
    "OP_MEAS_ACTIVE_INTERFERE", "OP_MEAS_ACTIVE_INTERFERE_FORCED",
    "OP_SWAP_MEAS_INTERFERE", "OP_SWAP_MEAS_INTERFERE_FORCED",
}
EXPAND_OPS = {"OP_EXPAND", "OP_EXPAND_T", "OP_EXPAND_T_DAG", "OP_EXPAND_ROT"}
ZMEAS_OPS = {"OP_MEAS_ACTIVE_DIAGONAL", "OP_MEAS_ACTIVE_DIAGONAL_FORCED"}

DEFAULT = ["coherent_d3_r1", "coherent_d5_r1", "coherent_d7_r1",
           "cultivation_d3", "coherent_d5_r5", "distillation"]


def human(n):
    n = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB"):
        if n < 1024.0 or unit == "EiB":
            return f"{n:.2f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024.0


def analyze(circ):
    prog = clifft.compile(open(f"qec_bench/circuits/{circ}.stim").read())
    n = len(prog)
    bits = {}            # slot -> int bitmask over source-bit indices
    next_src = 0
    kmax = 0
    peak_dense = 0
    peak_aff = 0
    sum_dense = 0
    sum_aff = 0
    boundary_step = None

    for step in range(n):
        inst = prog[step]
        name = T_mod._opname(inst.opcode)
        a1 = int(inst.axis_1); a2 = int(inst.axis_2)

        if name in BOUNDARY:
            boundary_step = step
            break

        if name in EXPAND_OPS:
            bits[a1] = 1 << next_src
            next_src += 1
        elif name == "OP_ARRAY_CNOT":            # a1=control, a2=target
            if a1 in bits or a2 in bits:
                bits[a2] = bits.get(a2, 0) ^ bits.get(a1, 0)
        elif name == "OP_ARRAY_MULTI_CNOT":      # a1=target, mask=controls
            d = ds_mod._d(inst)
            for ctrl in ds_mod._bits(int(d["mask"])):
                if ctrl == a1:
                    continue
                bits[a1] = bits.get(a1, 0) ^ bits.get(ctrl, 0)
        elif name == "OP_ARRAY_SWAP":
            ba, bb = bits.pop(a1, None), bits.pop(a2, None)
            if bb is not None:
                bits[a1] = bb
            if ba is not None:
                bits[a2] = ba
        elif name in ZMEAS_OPS:
            bits.pop(a1, None)
        # diagonal phase (T/S/RZ/CZ/MULTI_CZ/PHASE_*) and frame/noise/dormant:
        # no change to the affine map -> memory unchanged.

        # --- memory after this step ---
        a = len(bits)
        kmax = max(kmax, a)
        dense = 16 * (1 << a)                      # 16 * 2^a bytes
        live = 0
        for m in bits.values():
            live |= m
        s = bin(live).count("1")                  # live source bits
        aff = (a * s + 7) // 8 + (s + 7) // 8      # packed map + source values
        peak_dense = max(peak_dense, dense)
        peak_aff = max(peak_aff, aff)
        sum_dense += dense
        sum_aff += aff

    return dict(circ=circ, n=n, kmax=kmax,
                peak_rank=int(prog.peak_rank),
                peak_dense=peak_dense, peak_aff=peak_aff,
                sum_dense=sum_dense, sum_aff=sum_aff,
                boundary_step=boundary_step)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("circuits", nargs="*", default=DEFAULT)
    args = ap.parse_args()
    circuits = args.circuits or DEFAULT

    print("=== affine (A) state memory vs clifft dense, per benchmark ===\n")
    hdr = (f"{'circuit':16s} {'kmax':>4s} {'PEAK dense':>12s} {'PEAK affine':>12s} "
           f"{'peak ratio':>11s} {'Σ dense':>13s} {'Σ affine':>12s} {'Σ ratio':>10s}")
    print(hdr)
    print("-" * len(hdr))
    refused = []
    for c in circuits:
        r = analyze(c)
        if r["boundary_step"] is not None:
            refused.append(r)
            print(f"{c:16s} {r['kmax']:4d} {'— REFUSED (boundary @ step ' + str(r['boundary_step']) + ')':>62s}")
            continue
        pr = r["peak_dense"] / r["peak_aff"] if r["peak_aff"] else float("inf")
        sr = r["sum_dense"] / r["sum_aff"] if r["sum_aff"] else float("inf")
        print(f"{c:16s} {r['kmax']:4d} {human(r['peak_dense']):>12s} "
              f"{human(r['peak_aff']):>12s} {pr:>10.3g}x {human(r['sum_dense']):>13s} "
              f"{human(r['sum_aff']):>12s} {sr:>9.3g}x")

    print("\nclifft dense = 16*2^(active idents);  affine = packed GF(2) map "
          "(a x S bits) + source values.")
    print("REFUSED = circuit has a non-diagonal boundary; affine model is exact only "
          "up to that step\n          (full run needs materialization there).")
    for r in refused:
        print(f"  {r['circ']:16s} boundary @ step {r['boundary_step']} of {r['n']} "
              f"(kmax before boundary = {r['kmax']}, clifft peak_rank = {r['peak_rank']})")


if __name__ == "__main__":
    main()
