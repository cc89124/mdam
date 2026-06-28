"""Drive the verified NearClifford simulator from clifft bytecode and MEASURE the
live magic-register size |M| (= log2 of the genuinely-active dimension). All
Cliffords (CNOT/H/S/CZ/SWAP, and the de-fused U2/U4 Clifford parts) go into the
frame for free; only non-Clifford RZ become Pauli rotations that may promote a
qubit into the dense magic register. Measurements de-promote disentangled magic.

This is the ACTUAL simulator running (not a proxy): if |M| stays small, the
circuit is near-Clifford and the dense active dimension is 2^|M| << TTN chi.
"""
from __future__ import annotations
import argparse
import clifft
from ttn_backend import treewidth as T_mod
from ttn_backend import frame_layer as ds_mod
from ttn_backend.near_clifford import NearClifford

NONCLIFF = {"OP_ARRAY_T": 0.7853981633974483, "OP_ARRAY_T_DAG": -0.7853981633974483,
            "OP_ARRAY_ROT": 0.02}


def count_idents(prog):
    n = 0
    for k in range(len(prog)):
        if T_mod._opname(prog[k].opcode).startswith("OP_EXPAND"):
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("circuit", nargs="?", default="coherent_d5_r5")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--cap", type=int, default=22, help="abort if |M| exceeds this")
    args = ap.parse_args()
    prog = clifft.compile(open(f"qec_bench/circuits/{args.circuit}.stim").read())
    n = count_idents(prog)
    nc = NearClifford(n)
    import numpy as np
    nc.rng = np.random.default_rng(args.seed)

    slot2id = {}; nextid = 0
    maxM = 0; trajectory = []
    n_rot = n_meas = n_promote_events = 0

    def nid(slot):
        nonlocal nextid
        if slot in slot2id:
            return slot2id[slot]
        slot2id[slot] = nextid; nextid += 1
        return slot2id[slot]

    def rot(q, th):
        nonlocal n_rot
        nc.apply_rotation(0, 1 << q, th); n_rot += 1   # logical Z-rotation; frame pulls back

    for k in range(len(prog)):
        inst = prog[k]; name = T_mod._opname(inst.opcode)
        a1 = int(inst.axis_1); a2 = int(inst.axis_2)
        if name.startswith("OP_EXPAND"):
            q = nid(a1)
            if name in ("OP_EXPAND_T", "OP_EXPAND_T_DAG", "OP_EXPAND_ROT"):
                rot(q, 0.02)
        elif name == "OP_ARRAY_CNOT":
            nc.cx(nid(a1), nid(a2))
        elif name == "OP_ARRAY_MULTI_CNOT":
            d = ds_mod._d(inst); t = nid(a1)
            for c in ds_mod._bits(int(d["mask"])):
                if c != a1:
                    nc.cx(nid(c), t)
        elif name == "OP_ARRAY_SWAP":
            slot2id[a1], slot2id[a2] = slot2id.get(a2), slot2id.get(a1)
        elif name == "OP_ARRAY_CZ":
            nc.cz(nid(a1), nid(a2))
        elif name in ("OP_ARRAY_S", "OP_ARRAY_S_DAG"):
            if a1 in slot2id:
                nc.s(slot2id[a1], dag=(name == "OP_ARRAY_S_DAG"))
        elif name in NONCLIFF:
            if a1 in slot2id:
                rot(slot2id[a1], NONCLIFF[name])
        elif name == "OP_ARRAY_H":
            if a1 in slot2id:
                nc.h(slot2id[a1])
        elif name == "OP_ARRAY_U2":          # de-fuse H * RZ
            if a1 in slot2id:
                q = slot2id[a1]; rot(q, 0.02); nc.h(q)
        elif name == "OP_ARRAY_U4":          # de-fuse (H*RZ)_lo . CNOT(lo->hi)
            if a1 in slot2id and a2 in slot2id:
                lo = slot2id[a1]; hi = slot2id[a2]
                nc.cx(lo, hi); rot(lo, 0.02); nc.h(lo)
        elif name in ("OP_MEAS_ACTIVE_DIAGONAL", "OP_MEAS_ACTIVE_DIAGONAL_FORCED",
                      "OP_MEAS_ACTIVE_INTERFERE", "OP_MEAS_ACTIVE_INTERFERE_FORCED"):
            if a1 in slot2id:
                nc.measure_z(slot2id[a1]); n_meas += 1
                del slot2id[a1]
        elif name in ("OP_SWAP_MEAS_INTERFERE", "OP_SWAP_MEAS_INTERFERE_FORCED"):
            if a2 in slot2id:
                nc.measure_z(slot2id[a2]); n_meas += 1
            slot2id[a1] = slot2id.get(a2); slot2id.pop(a2, None)
        # OP_FRAME_* / noise / detector: dormant/frame layer, not the active state
        if len(nc.M) > maxM:
            maxM = len(nc.M)
            print(f"  step {k}: new max |M| = {maxM}  ({name})", flush=True)
        trajectory.append(len(nc.M))
        if len(nc.M) > args.cap:
            print(f"ABORT: |M| exceeded cap {args.cap} at step {k} -> magic too large.")
            break

    print(f"=== {args.circuit} : near-Clifford simulator run ===")
    print(f"idents(qubits)={n}  rotations applied={n_rot}  active measurements={n_meas}")
    print(f"*** MAX magic register |M| = {maxM}  -> active dimension 2^{maxM} = {2**maxM} ***")
    print(f"final |M| = {len(nc.M)}")
    # trajectory summary
    import numpy as np
    tr = np.array(trajectory)
    print(f"|M| trajectory: mean={tr.mean():.2f}  median={int(np.median(tr))}  "
          f"p90={int(np.percentile(tr,90))}  max={tr.max()}")
    print(f"\nCOMPARE: TTN observed chi = 2048 = 2^11 (d5_r5).")
    print(f"  near-Clifford active dim 2^{maxM} {'<' if maxM < 11 else '>='} TTN 2^11")


if __name__ == "__main__":
    main()
