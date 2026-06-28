"""Region-size audit of the U2/U4/H-like boundaries under a LAZY affine frame.

The earlier '~2x, 319 relevant phases' number came from a model that, at each
flush, realized (close to) the WHOLE (A,f). The user's objection: a U2 on qubit
j should only need j's own logical line localized + the phases that touch it --
NOT the whole state. This audits exactly that.

LAZY frame (never reset across boundaries):
  EXPAND s        -> row[s] = fresh stored var
  CNOT/MULTI_CNOT -> row[tgt] ^= row[ctrl]   (deferred, accumulates)
  SWAP            -> swap rows
  Z-meas s        -> localize row[s] (cost popcount-1), then REMOVE s (prunes the
                     map; ancillas that gathered parity are measured & gone)
  boundary on j   -> region = popcount(row[j]) ; localize it (cost popcount-1),
                     then rebase j to a fresh var (it carried on past the boundary)

For each boundary we report the region size = weight of l_j (how many stored vars
its value depends on = how big a parity must be made local). Small regions => the
boundary is cheap and 'convert whole state' is wrong. Plus f_touch: how many
still-deferred phases actually overlap the boundary's support.
"""
from __future__ import annotations

import argparse
from collections import Counter
import clifft
from ttn_backend import treewidth as T_mod
from ttn_backend import frame_layer as ds_mod

HARD_1Q = {"OP_ARRAY_H", "OP_ARRAY_U2"}
ZMEAS = {"OP_MEAS_ACTIVE_DIAGONAL", "OP_MEAS_ACTIVE_DIAGONAL_FORCED"}
HARDMEAS = {"OP_MEAS_ACTIVE_INTERFERE", "OP_MEAS_ACTIVE_INTERFERE_FORCED",
            "OP_SWAP_MEAS_INTERFERE", "OP_SWAP_MEAS_INTERFERE_FORCED"}
DIAG = {"OP_ARRAY_T", "OP_ARRAY_T_DAG", "OP_ARRAY_S", "OP_ARRAY_S_DAG",
        "OP_ARRAY_ROT", "OP_PHASE_T", "OP_PHASE_T_DAG", "OP_PHASE_ROT"}

DEFAULT = ["coherent_d5_r5", "coherent_d5_r1", "coherent_d7_r1", "distillation"]


def audit(circ):
    prog = clifft.compile(open(f"qec_bench/circuits/{circ}.stim").read())
    row = {}
    nvar = 0
    live_phases = []          # list of parity bitmasks still deferred (not yet materialized)
    boundary_regions = []     # (kind, region_size, f_touch_count)
    meas_local = []           # localization cost (popcount-1) of each Z-meas
    loc_cnots_boundary = 0
    loc_cnots_meas = 0

    def fresh():
        nonlocal nvar
        v = 1 << nvar; nvar += 1
        return v

    def get(s):
        if s not in row:
            row[s] = fresh()
        return row[s]

    def materialize_overlap(target_parity):
        # consume deferred phases overlapping target_parity's support; return count
        nonlocal live_phases
        keep = []; touched = 0
        for p in live_phases:
            if p & target_parity:
                touched += 1
            else:
                keep.append(p)
        live_phases = keep
        return touched

    for i in range(len(prog)):
        inst = prog[i]
        name = T_mod._opname(inst.opcode)
        a1 = int(inst.axis_1); a2 = int(inst.axis_2)
        if name.startswith("OP_EXPAND"):
            row[a1] = fresh()
        elif name == "OP_ARRAY_CNOT":
            get(a1); row[a2] = get(a2) ^ row[a1]
        elif name == "OP_ARRAY_MULTI_CNOT":
            d = ds_mod._d(inst); get(a1)
            for c in ds_mod._bits(int(d["mask"])):
                if c != a1:
                    row[a1] = row[a1] ^ get(c)
        elif name == "OP_ARRAY_SWAP":
            ra, rb = get(a1), get(a2); row[a1], row[a2] = rb, ra
        elif name in DIAG:
            live_phases.append(get(a1))
        elif name in HARD_1Q:
            ell = get(a1); w = bin(ell).count("1")
            ft = materialize_overlap(ell)
            boundary_regions.append(("U2/H", w, ft))
            loc_cnots_boundary += max(0, w - 1)
            row[a1] = fresh()                       # rebase, carries on
        elif name == "OP_ARRAY_U4":
            e1 = get(a1); e2 = get(a2)
            w = bin(e1 | e2).count("1")
            ft = materialize_overlap(e1 | e2)
            boundary_regions.append(("U4", w, ft))
            loc_cnots_boundary += max(0, w - 1)
            row[a1] = fresh(); row[a2] = fresh()
        elif name in ZMEAS:
            ell = get(a1); w = bin(ell).count("1")
            meas_local.append(w)
            loc_cnots_meas += max(0, w - 1)
            row.pop(a1, None)
        elif name in HARDMEAS:
            ell = get(a1); w = bin(ell).count("1")
            materialize_overlap(ell)
            boundary_regions.append(("Xmeas", w, 0))
            loc_cnots_boundary += max(0, w - 1)
            row.pop(a1, None)

    regs = [r[1] for r in boundary_regions]
    fts = [r[2] for r in boundary_regions]
    return dict(
        circ=circ,
        n_boundary=len(boundary_regions),
        region_hist=dict(sorted(Counter(regs).items())),
        mean_region=(sum(regs) / len(regs)) if regs else 0.0,
        max_region=max(regs) if regs else 0,
        ftouch_total=sum(fts),
        ftouch_mean=(sum(fts) / len(fts)) if fts else 0.0,
        loc_cnots_boundary=loc_cnots_boundary,
        loc_cnots_meas=loc_cnots_meas,
        n_meas=len(meas_local),
        phases_left_deferred=len(live_phases),
        total_phases=sum(1 for i in range(len(prog))
                         if T_mod._opname(prog[i].opcode) in DIAG),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("circuits", nargs="*", default=DEFAULT)
    args = ap.parse_args()
    for c in (args.circuits or DEFAULT):
        r = audit(c)
        print(f"=== {c} ===")
        print(f"  boundaries: {r['n_boundary']}   mean region (qubits to localize): "
              f"{r['mean_region']:.2f}   max: {r['max_region']}")
        print(f"  region-size histogram (weight of l_j): {r['region_hist']}")
        print(f"  f_touch: phases that actually overlap a boundary = {r['ftouch_total']} "
              f"(mean {r['ftouch_mean']:.2f}/boundary) of {r['total_phases']} total diag")
        print(f"  phases NEVER touching any boundary (droppable) = {r['phases_left_deferred']}")
        print(f"  localization CNOTs: boundaries={r['loc_cnots_boundary']}, "
              f"Z-meas={r['loc_cnots_meas']} (n_meas={r['n_meas']})")
        print(f"  => TOTAL lazy-frame materialize CNOTs ~ "
              f"{r['loc_cnots_boundary'] + r['loc_cnots_meas'] + r['ftouch_total']}")
        print()
    print("region = #stored vars the boundary qubit's value depends on (parity weight).")
    print("small region + small f_touch => boundary is LOCAL; whole-state convert is wrong.")
    print("f_touch counted once (consumed at first overlapping boundary); rest stay deferred.")


if __name__ == "__main__":
    main()
