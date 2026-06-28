"""Can the 244 boundary-touching (f_touch) phases of d5_r5 be synthesized with
fewer CNOTs? Breaks the cost down the way the user laid out:

  1. weight histogram of the f_touch parities
  2. how many are weight-1 LOCAL phases on the boundary qubit -> FUSABLE into the
     U2/U4 gate matrix (cost ~0), no CNOT at all
  3. unique parities after dedup (same parity -> merge angles -> one term)
  4. per-boundary phase-network (GraySynth-style, verified greedy) CNOT count for
     the remaining non-fusable parities + the boundary's own l_j localization

Lazy affine frame (defer across boundaries; Z-meas localizes+prunes; boundary
localizes its qubit + materializes the phases overlapping it, then rebases).
"""
from __future__ import annotations

from collections import Counter
import clifft
from ttn_backend import treewidth as T_mod
from ttn_backend import frame_layer as ds_mod
from ttn_backend.scripts.measure_phase_network_synthesis import synth_realize_set

HARD_1Q = {"OP_ARRAY_H", "OP_ARRAY_U2"}
ZMEAS = {"OP_MEAS_ACTIVE_DIAGONAL", "OP_MEAS_ACTIVE_DIAGONAL_FORCED"}
HARDMEAS = {"OP_MEAS_ACTIVE_INTERFERE", "OP_MEAS_ACTIVE_INTERFERE_FORCED",
            "OP_SWAP_MEAS_INTERFERE", "OP_SWAP_MEAS_INTERFERE_FORCED"}
DIAG = {"OP_ARRAY_T", "OP_ARRAY_T_DAG", "OP_ARRAY_S", "OP_ARRAY_S_DAG",
        "OP_ARRAY_ROT", "OP_PHASE_T", "OP_PHASE_T_DAG", "OP_PHASE_ROT"}


def reindex(parities, ell):
    """Map the union of support bits to dense indices 0..n-1; return (remapped
    parities, remapped ell list, n)."""
    bits = set()
    for p in parities:
        b = p
        while b:
            lb = b & -b; bits.add(lb.bit_length() - 1); b ^= lb
    for e in ell:
        b = e
        while b:
            lb = b & -b; bits.add(lb.bit_length() - 1); b ^= lb
    idx = {b: i for i, b in enumerate(sorted(bits))}

    def rm(p):
        out = 0
        b = p
        while b:
            lb = b & -b; out |= 1 << idx[lb.bit_length() - 1]; b ^= lb
        return out
    return [rm(p) for p in parities], [rm(e) for e in ell], len(idx)


def audit(circ):
    prog = clifft.compile(open(f"qec_bench/circuits/{circ}.stim").read())
    row = {}
    nvar = 0
    live = []                 # (parity, ) still-deferred phases
    ft_weights = []           # weight of each f_touch phase
    fusable = 0               # weight-1 phase exactly equal to the boundary l_j
    total_phase_net_cx = 0
    total_loc_cx = 0
    boundaries = 0
    all_ok = True
    uniq_parities = set()

    def fresh():
        nonlocal nvar
        v = 1 << nvar; nvar += 1
        return v

    def get(s):
        if s not in row:
            row[s] = fresh()
        return row[s]

    def do_boundary(ells):
        # ells: list of boundary l_j parities (1 for U2/H, 2 for U4)
        nonlocal live, total_phase_net_cx, total_loc_cx, fusable, all_ok, boundaries
        boundaries += 1
        support = 0
        for e in ells:
            support |= e
        touched = []; keep = []
        for p in live:
            if p & support:
                touched.append(p)
            else:
                keep.append(p)
        live = keep
        for p in touched:
            w = bin(p).count("1")
            ft_weights.append(w)
            uniq_parities.add(p)
        # fusable = weight-1 phase exactly equal to some boundary l_j (local on it)
        eset = set(ells)
        nonfuse = []
        for p in touched:
            if bin(p).count("1") == 1 and p in eset:
                fusable_local = True
            else:
                fusable_local = False
            if fusable_local:
                pass  # absorbed into the U2/U4 matrix, 0 CNOT
            else:
                nonfuse.append(p)
        # phase-network: realize boundary l_j's AND the non-fusable phases together
        realize = set(nonfuse) | set(ells)
        realize = {p for p in realize if p != 0}
        if realize:
            ps, _, n = reindex(list(realize), [])
            cx, ok = synth_realize_set(set(ps), n)
            if cx is None or not ok:
                all_ok = False
                cx = sum(bin(p).count("1") - 1 for p in realize)
            total_phase_net_cx += cx
        # count fusable
        return sum(1 for p in touched if bin(p).count("1") == 1 and p in eset)

    fus = 0
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
            live.append(get(a1))
        elif name in HARD_1Q:
            fus += do_boundary([get(a1)]); row[a1] = fresh()
        elif name == "OP_ARRAY_U4":
            fus += do_boundary([get(a1), get(a2)]); row[a1] = fresh(); row[a2] = fresh()
        elif name in ZMEAS:
            total_loc_cx += max(0, bin(get(a1)).count("1") - 1)
            row.pop(a1, None)
        elif name in HARDMEAS:
            fus += do_boundary([get(a1)]); row.pop(a1, None)

    return dict(circ=circ, boundaries=boundaries, n_ftouch=len(ft_weights),
                ft_weight_hist=dict(sorted(Counter(ft_weights).items())),
                fusable=fus, uniq=len(uniq_parities),
                phase_net_cx=total_phase_net_cx, meas_loc_cx=total_loc_cx,
                ok=all_ok)


def main():
    for c in ["coherent_d5_r5", "coherent_d5_r1", "coherent_d7_r1"]:
        r = audit(c)
        print(f"=== {c} ===")
        print(f"  boundaries={r['boundaries']}  f_touch phases={r['n_ftouch']}  "
              f"unique parities={r['uniq']}")
        print(f"  f_touch weight hist: {r['ft_weight_hist']}")
        print(f"  fusable (weight-1 == boundary l_j -> absorbed in U2/U4, 0 CNOT): {r['fusable']}")
        print(f"  phase-network CNOTs (boundaries' l_j + non-fusable phases, verified): "
              f"{r['phase_net_cx']}  [{'OK' if r['ok'] else 'MISMATCH'}]")
        print(f"  Z-meas localize CNOTs: {r['meas_loc_cx']}")
        print(f"  => hybrid materialize total ~ {r['phase_net_cx'] + r['meas_loc_cx']} CNOT")
        print()
    print("compare: d5_r5 raw per-control CNOT = 775")
    print("phase-network shares CNOTs across the f_touch parities; fusable ones cost 0.")
    print("NOTE: still raw CNOT count; TTN bond-weighted routing is a further (separate) factor.")


if __name__ == "__main__":
    main()
