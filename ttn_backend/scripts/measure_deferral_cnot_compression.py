"""Compute-side test of the deferred affine (A) idea.

The expensive compute in the TTN backend is the transport+QR from cross-bag
CNOT / MULTI_CNOT (basis-permutation ops). T/RZ/S are cheap diagonals; CZ is a
phase (would go into f). Deferring CNOTs into an affine map A only REDUCES
compute if, between two hard materialization boundaries (H / U2 / U4 / X-meas),
the raw CNOT sequence COMPRESSES: i.e. the net linear map A needs fewer CNOTs to
synthesize than were issued (cancellation / overlap). Otherwise materializing A
costs ~the same CNOT work, just relocated.

This replays the bytecode tracking the affine map over SLOTS (handling SWAP and
EXPAND as relabels), and per hard-boundary run reports:
  raw_cnot     = #CNOT + sum(#MULTI_CNOT controls)        (current backend work)
  net_offdiag  = off-diagonal nnz of the net A            (>=1 CNOT each to build)
A compression ratio raw/net >> 1 means deferral saves real CNOT/QR work; ~1 means
it only relocates the work to boundaries (no compute win).
"""
from __future__ import annotations

import argparse
import clifft
from ttn_backend import treewidth as T_mod
from ttn_backend import frame_layer as ds_mod

# Materialization boundaries = non-diagonal active ops. Must match
# AffineActiveBackend: besides H/U2/U4, the X-basis (interfere) measurements are
# boundaries -- INCLUDING the SWAP_MEAS variants and _FORCED forms. (An earlier
# version dropped OP_SWAP_MEAS_INTERFERE, so deferral windows in cultivation_d3 /
# distillation were not split at the X-measurement -> overstated net maps.)
BOUNDARY = {
    "OP_ARRAY_H", "OP_ARRAY_U2", "OP_ARRAY_U4",
    "OP_MEAS_ACTIVE_INTERFERE", "OP_MEAS_ACTIVE_INTERFERE_FORCED",
    "OP_SWAP_MEAS_INTERFERE", "OP_SWAP_MEAS_INTERFERE_FORCED",
}
# Z-basis active measurements. In the (A,f) frame these do NOT need to flush the
# affine map for *probabilities* (f-independent), but if the post-measurement
# collapse is materialized onto the tensor they become a flush point. split_zmeas
# models that "meas-flush" policy (a conservative full-barrier upper bound on the
# segmentation, i.e. lower bound on the achievable raw/GE).
ZMEAS = {"OP_MEAS_ACTIVE_DIAGONAL", "OP_MEAS_ACTIVE_DIAGONAL_FORCED"}


def analyze(circuit, split_zmeas=False):
    prog = clifft.compile(open(f"qec_bench/circuits/{circuit}.stim").read())
    A = {}                       # slot -> set of source tags (parity), lazily {slot}

    def row(s):
        if s not in A:
            A[s] = {("s", s)}
        return A[s]

    runs = []                    # (raw_cnot, net_offdiag, n_slots_touched)
    raw = 0
    fresh = 0                    # counter to give each EXPAND a unique source tag

    def ge_cnot_count():
        """Realizable CNOT count to synthesize the net linear map: count XOR
        row-operations in a GF(2) Gauss-Jordan reduction of the map matrix.
        This is an actual (upper-bound) synthesis cost, unlike off-diagonal nnz."""
        tags = set()
        for s, parity in A.items():
            tags |= parity
            tags.add(("s", s))
        # only keep slots that are an identity row {self} -> drop (no work);
        # build square matrix over tag index for the non-trivial part.
        idx = {t: i for i, t in enumerate(sorted(tags, key=lambda x: (x[0], x[1])))}
        rows = []
        for s in A:
            bits = 0
            for t in A[s]:
                bits |= 1 << idx[t]
            rows.append(bits)
        # add identity rows for tags that are pure sources never used as a slot key
        present_self = {("s", s) for s in A}
        for t in tags:
            if t not in present_self:
                rows.append(1 << idx[t])
        n = len(idx)
        # Gauss-Jordan over GF(2), counting row-additions (= CNOTs)
        mat = list(rows)
        cnots = 0
        r = 0
        for c in range(n):
            piv = next((k for k in range(r, len(mat)) if (mat[k] >> c) & 1), None)
            if piv is None:
                continue
            mat[r], mat[piv] = mat[piv], mat[r]
            for k in range(len(mat)):
                if k != r and ((mat[k] >> c) & 1):
                    mat[k] ^= mat[r]
                    cnots += 1
            r += 1
        return cnots

    def close_run():
        nonlocal raw
        off = 0
        touched = 0
        for s, parity in A.items():
            self_tag = ("s", s)
            off += len(parity - {self_tag})
            if parity != {self_tag}:
                touched += 1
        ge = ge_cnot_count() if touched else 0
        runs.append((raw, off, touched, ge))
        raw = 0
        A.clear()

    for i in range(len(prog)):
        inst = prog[i]
        name = T_mod._opname(inst.opcode)
        a1 = int(inst.axis_1); a2 = int(inst.axis_2)
        if name in BOUNDARY or (split_zmeas and name in ZMEAS):
            close_run()
            continue
        if name.startswith("OP_EXPAND"):
            A[a1] = {("s", a1)}          # fresh wire = identity row
            continue
        if name == "OP_ARRAY_SWAP":
            ra, rb = row(a1), row(a2)
            A[a1], A[a2] = rb, ra
            continue
        if name == "OP_ARRAY_CNOT":      # a1=control, a2=target
            A[a2] = row(a2) ^ row(a1)
            raw += 1
            continue
        if name == "OP_ARRAY_MULTI_CNOT":
            d = ds_mod._d(inst)
            tgt = a1
            for ctrl in ds_mod._bits(int(d["mask"])):
                if ctrl == tgt:
                    continue
                A[tgt] = row(tgt) ^ row(ctrl)
                raw += 1
            continue
        # everything else (T/RZ/S/ROT diagonal, CZ phase, Z-meas, frame, noise):
        # does not change the basis-permutation A -> ignore for this count.
    close_run()

    runs = [r for r in runs if r[0] > 0 or r[1] > 0]
    tot_raw = sum(r[0] for r in runs)
    tot_off = sum(r[1] for r in runs)
    tot_ge = sum(r[3] for r in runs)
    return dict(circuit=circuit, n_runs=len(runs), tot_raw=tot_raw,
                tot_off=tot_off, tot_ge=tot_ge,
                ratio_ge=(tot_raw / tot_ge) if tot_ge else float("inf"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("circuits", nargs="*", default=[
        "coherent_d5_r1", "coherent_d7_r1", "coherent_d5_r5", "distillation",
        "cultivation_d3"])
    args = ap.parse_args()
    print("(A,f) frame compute ceiling: raw CNOT vs Gauss-Jordan-minimal synthesis")
    print("  full-defer  = split only at hard boundaries (U2/U4/H/interfere); Z-meas deferred")
    print("  meas-flush  = ALSO split at Z-meas (post-measurement collapse materialized)\n")
    print(f"{'circuit':16s} {'rawCNOT':>8s} | {'full-defer GE':>13s} {'raw/GE':>7s} "
          f"| {'meas-flush GE':>13s} {'raw/GE':>7s}")
    for c in args.circuits:
        rf = analyze(c, split_zmeas=False)
        rm = analyze(c, split_zmeas=True)
        rrf = f"{rf['ratio_ge']:.2f}" if rf['ratio_ge'] != float("inf") else "inf"
        rrm = f"{rm['ratio_ge']:.2f}" if rm['ratio_ge'] != float("inf") else "inf"
        print(f"{c:16s} {rf['tot_raw']:8d} | {rf['tot_ge']:13d} {rrf:>7s} "
              f"| {rm['tot_ge']:13d} {rrm:>7s}")
    print("\nrawCNOT     = basis-permutation ops the current per-control backend issues")
    print("GE          = CNOTs to synthesize the NET map per segment (realizable)")
    print("raw/GE >> 1 => deferral+resynthesis cuts real CNOT/transport/QR work")
    print("full-defer needs Z-meas handled w/o materializing collapse (tensor stabilizer")
    print("read or deferred measurement); meas-flush is realizable with ordinary Z-meas.")


if __name__ == "__main__":
    main()
