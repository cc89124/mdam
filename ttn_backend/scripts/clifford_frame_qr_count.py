"""MEASURE the "push H (and all Cliffords) into the frame" idea.

If H is handed to a CLIFFORD frame (along with CNOT/CZ/S), then NO H boundary
ever hits the tensor. The only tensor operations are the non-Clifford rotations
RZ(theta) -- but each becomes a PAULI-STRING rotation e^{i theta P}, where
P = C^dag Z_q C and C is the accumulated Clifford. H mixes Z into X, so P is a
general Pauli string; its support's Steiner subtree on the tree = the QR cost to
apply it (rank-2 primitive works for any Pauli, not just Z).

We track the full symplectic frame: for each ident q, the tensor-space images
xcol[q], zcol[q] of true X_q, Z_q (Pauli strings over idents). A rotation RZ on q
costs Steiner(support(zcol[q])). We sum, and compare to:
  eager CNOT transport  (5283, the cost frame-H removes)
  active-H f_touch       (5405, the (A,f)-with-active-H diagonal-apply cost)

Weight-1 (single-ident) Pauli rotations are LOCAL (free, like eager). So the
real cost is the sum over rotations whose frame image is nonlocal.
NOTHING here is estimated -- every number is replayed on the real carving tree.
"""
from __future__ import annotations

import argparse
from collections import Counter

import clifft
from ttn_backend import treewidth as T_mod
from ttn_backend import frame_layer as ds_mod
from ttn_backend.scripts.diag_phase_ttn_cost import build_and_run
from ttn_backend.scripts.affine_diag_qr_count import steiner_edges

DIAG = {"OP_ARRAY_T", "OP_ARRAY_T_DAG", "OP_ARRAY_S", "OP_ARRAY_S_DAG", "OP_ARRAY_ROT"}
# ^ S/S_DAG are Clifford (go to frame); T/ROT are non-Clifford (rotations)
NONCLIFF_ROT = {"OP_ARRAY_T", "OP_ARRAY_T_DAG", "OP_ARRAY_ROT"}
CLIFF_1Q_S = {"OP_ARRAY_S", "OP_ARRAY_S_DAG"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("circuit", nargs="?", default="coherent_d5_r5")
    ap.add_argument("--chi-cache", default=None)
    args = ap.parse_args()
    R = build_and_run(args.circuit, chi_cache=args.chi_cache)
    prog, home, adj = R["prog"], R["home"], R["adj"]

    # symplectic frame columns over idents: image of true X_q, Z_q on the tensor.
    # represent each as (xmask, zmask) bitmask over idents.
    slot2id = {}; nextid = 0
    xcol = {}; zcol = {}   # ident -> (xmask, zmask)

    def new_id(slot):
        nonlocal nextid
        if slot in slot2id:
            return slot2id[slot]
        i = nextid; nextid += 1
        slot2id[slot] = i
        xcol[i] = (1 << i, 0)
        zcol[i] = (0, 1 << i)
        return i

    def pmul(a, b):
        return (a[0] ^ b[0], a[1] ^ b[1])   # support-only (ignore phase/sign)

    def cnot(a, b):       # control a, target b ; conjugation C^dag . C
        # Z_b -> Z_a Z_b ; X_a -> X_a X_b
        zcol[b] = pmul(zcol[a], zcol[b])
        xcol[a] = pmul(xcol[a], xcol[b])

    def hgate(q):
        xcol[q], zcol[q] = zcol[q], xcol[q]

    def sgate(q):
        xcol[q] = pmul(xcol[q], zcol[q])   # X->Y (gains Z support)

    def cz(a, b):
        xcol[a] = pmul(xcol[a], zcol[b])
        xcol[b] = pmul(xcol[b], zcol[a])

    def support_homes(xm, zm):
        s = xm | zm
        out = set(); i = 0
        while s:
            if s & 1:
                h = home.get(i)
                if h is not None:
                    out.add(h)
            s >>= 1; i += 1
        return out

    rot_steiner = []        # Steiner edges per non-Clifford rotation
    rot_weight = []         # support size (idents) per rotation
    meas_steiner = []       # Steiner edges per measurement (Pauli-string readout)
    nohome = 0

    for k in range(len(prog)):
        inst = prog[k]; name = T_mod._opname(inst.opcode)
        a1 = int(inst.axis_1); a2 = int(inst.axis_2)
        if name.startswith("OP_EXPAND"):
            i = new_id(a1)
            if name in ("OP_EXPAND_T", "OP_EXPAND_T_DAG", "OP_EXPAND_ROT"):
                # rotation on a fresh qubit -> image is Z_i (weight 1, local)
                rot_steiner.append(0); rot_weight.append(1)
        elif name == "OP_ARRAY_CNOT":
            new_id(a1); new_id(a2); cnot(a1, a2)
        elif name == "OP_ARRAY_MULTI_CNOT":
            d = ds_mod._d(inst); new_id(a1)
            for c in ds_mod._bits(int(d["mask"])):
                if c != a1:
                    new_id(c); cnot(c, a1)
        elif name == "OP_ARRAY_SWAP":
            slot2id[a1], slot2id[a2] = slot2id.get(a2), slot2id.get(a1)
        elif name == "OP_ARRAY_CZ":
            new_id(a1); new_id(a2); cz(a1, a2)
        elif name in CLIFF_1Q_S:
            i = slot2id.get(a1)
            if i is not None: sgate(i)
        elif name in NONCLIFF_ROT:
            i = slot2id.get(a1)
            if i is None:
                i = new_id(a1)
            xm, zm = zcol[i]
            sh = support_homes(xm, zm)
            w = bin(xm | zm).count("1")
            if len(sh) != w:
                nohome += 1
            rot_steiner.append(steiner_edges(adj, sh))
            rot_weight.append(w)
        elif name == "OP_ARRAY_H":
            i = slot2id.get(a1)
            if i is not None: hgate(i)
        elif name == "OP_ARRAY_U2":
            # de-fuse U2 = H * RZ(theta): RZ first (rotation on q), then H -> frame
            i = slot2id.get(a1)
            if i is not None:
                xm, zm = zcol[i]
                rot_steiner.append(steiner_edges(adj, support_homes(xm, zm)))
                rot_weight.append(bin(xm | zm).count("1"))
                hgate(i)
        elif name == "OP_ARRAY_U4":
            # de-fuse U4 = (H*RZ on lo) . CNOT(lo->hi): CNOT->frame, RZ rot, H->frame
            lo = slot2id.get(a1); hi = slot2id.get(a2)
            if lo is not None and hi is not None:
                cnot(lo, hi)
                xm, zm = zcol[lo]
                rot_steiner.append(steiner_edges(adj, support_homes(xm, zm)))
                rot_weight.append(bin(xm | zm).count("1"))
                hgate(lo)
        # measurements: ident leaves the frame; we don't apply rotation phases for
        # them here (cost of measurement handled separately). drop the slot.
        elif name in ("OP_MEAS_ACTIVE_DIAGONAL", "OP_MEAS_ACTIVE_DIAGONAL_FORCED",
                      "OP_MEAS_ACTIVE_INTERFERE", "OP_MEAS_ACTIVE_INTERFERE_FORCED"):
            slot2id.pop(a1, None)
        elif name in ("OP_SWAP_MEAS_INTERFERE", "OP_SWAP_MEAS_INTERFERE_FORCED"):
            slot2id[a1] = slot2id.get(a2); slot2id.pop(a2, None)

    n = len(rot_steiner)
    total = sum(rot_steiner)
    local = sum(1 for s in rot_steiner if s == 0)
    wh = dict(sorted(Counter(rot_weight).items()))
    sh = dict(sorted(Counter(rot_steiner).items()))
    print(f"=== {args.circuit}  FRAME-H (all Clifford in frame, real carving tree) ===")
    print(f"bags={len(adj)}  observed maxχ={R['max_bond']}")
    print(f"non-Clifford rotations applied as Pauli-string rotations = {n}")
    print(f"  weight (Pauli-string support, #idents) hist: {wh}")
    print(f"  Steiner-edges-per-rotation hist: {sh}")
    print(f"  weight-1 LOCAL (free) rotations = {local}/{n}")
    if nohome:
        print(f"  (note: {nohome} rotations referenced home=None idents; placed bits only)")
    print(f"  TOTAL frame-H tensor-apply QRs (Sum Steiner) = {total}")
    print(f"\nCOMPARE (all on the real tree, measured):")
    print(f"  eager CNOT transport            = 5283 QR")
    print(f"  active-H (A,f) f_touch diag      = 5405 QR")
    print(f"  frame-H all-rotations diag       = {total} QR")
    print(f"\nVERDICT: {'frame-H LOWER' if total < 5283 else 'frame-H NOT lower'} "
          f"than eager CNOT transport.")


if __name__ == "__main__":
    main()
