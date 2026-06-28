"""Characterize whether a deferred affine+phase-polynomial (A, f) representation
can help on the real QEC active-op streams.

Idea under test: keep |psi> = sum_x e^{i f(x)} phi(x) |A x>, deferring
  CNOT / MULTI_CNOT / CZ            -> update A (XOR map)            [no tensor]
  T / T_dag / S / S_dag / RZ(ROT)   -> update f (phase polynomial)  [no tensor]
and only MATERIALIZE (absorb A,f into the tensor, paying QR/refactor) at a
"boundary": a non-diagonal op (H, general U2/U4) or a non-Z measurement.

The benefit is bounded by how SPARSE the boundaries are in the active-op stream:
long deferrable runs between boundaries => big windows where no bond growth /
no QR happens. Dense boundaries => frequent materialization => little gain.

This script replays the bytecode, tracks active idents (EXPAND promotes,
active measurement demotes), classifies each active op, and reports:
  - active-op class histogram
  - number of materialization boundaries
  - run-length distribution of deferrable ops between boundaries
  - whether there is a long final deferrable tail (=> 'never materialize' /
    sample-and-relabel shortcut applies for Z-basis sampling)
"""
from __future__ import annotations

import argparse
import statistics
import clifft
from ttn_backend import treewidth as T_mod
from ttn_backend import frame_layer as ds_mod

# active-op classification by opcode name
DEFERRABLE = {
    "OP_ARRAY_CNOT", "OP_ARRAY_MULTI_CNOT", "OP_ARRAY_CZ", "OP_ARRAY_MULTI_CZ",
    "OP_PHASE_T", "OP_PHASE_T_DAG", "OP_PHASE_ROT",
    "OP_ARRAY_T", "OP_ARRAY_T_DAG", "OP_ARRAY_S", "OP_ARRAY_S_DAG", "OP_ARRAY_ROT",
    "OP_EXPAND", "OP_EXPAND_T", "OP_EXPAND_T_DAG", "OP_EXPAND_ROT",
    "OP_ARRAY_SWAP",   # relabel of the affine map A -> deferrable (no materialize)
}
# Materialization boundaries = non-diagonal active ops (force absorbing A,f into a
# tensor). Must match AffineActiveBackend's boundary set: besides H/U2/U4, the
# X-basis (interfere) measurements are boundaries -- INCLUDING the SWAP_MEAS
# variants and the _FORCED forms. (An earlier version listed only
# OP_MEAS_ACTIVE_INTERFERE and silently dropped OP_SWAP_MEAS_INTERFERE, which
# misclassified cultivation_d3 / distillation as boundary-free.)
BOUNDARY = {
    "OP_ARRAY_H", "OP_ARRAY_U2", "OP_ARRAY_U4",
    "OP_MEAS_ACTIVE_INTERFERE", "OP_MEAS_ACTIVE_INTERFERE_FORCED",
    "OP_SWAP_MEAS_INTERFERE", "OP_SWAP_MEAS_INTERFERE_FORCED",
}
# active Z-measurement opcode names (soft boundary: sample-shortcut ok, collapse
# may need work). Names vary; matched by substring below.
ZMEAS_HINTS = ("MEAS_ACTIVE",)


def classify(name):
    if name in DEFERRABLE:
        return "defer"
    if name in BOUNDARY:
        return "boundary"
    if any(h in name for h in ZMEAS_HINTS):
        return "zmeas"
    return None  # frame/noise/dormant/detector -> not an active-state op


def analyze(circuit):
    prog = clifft.compile(open(f"qec_bench/circuits/{circuit}.stim").read())
    n = len(prog)
    hist = {}
    active_classes = []   # ordered list of ('defer'|'boundary'|'zmeas')
    unknown_active = {}
    for i in range(n):
        inst = prog[i]
        name = T_mod._opname(inst.opcode)
        c = classify(name)
        if c is None:
            # surface active-looking opcodes we didn't classify
            if name.startswith(("OP_ARRAY", "OP_PHASE", "OP_EXPAND", "OP_MEAS_ACTIVE")):
                unknown_active[name] = unknown_active.get(name, 0) + 1
            continue
        hist[name] = hist.get(name, 0) + 1
        active_classes.append(c)

    # materialization boundaries = 'boundary' ops (hard). zmeas counted separately.
    runs = []          # deferrable-op count between consecutive hard boundaries
    cur = 0
    n_boundary = 0
    n_defer = 0
    n_zmeas = 0
    for c in active_classes:
        if c == "defer":
            cur += 1
            n_defer += 1
        elif c == "zmeas":
            n_zmeas += 1
            # Z-meas does not force materialization for sampling; keep run going.
            cur += 1
        else:  # boundary
            runs.append(cur)
            cur = 0
            n_boundary += 1
    final_tail = cur  # deferrable ops after the last hard boundary
    runs_nonempty = [r for r in runs if r > 0]

    return dict(
        circuit=circuit,
        total_instr=n,
        active_ops=len(active_classes),
        n_defer=n_defer,
        n_zmeas=n_zmeas,
        n_boundary=n_boundary,
        boundary_density=(n_boundary / max(1, len(active_classes))),
        mean_run=(statistics.mean(runs_nonempty) if runs_nonempty else 0.0),
        median_run=(statistics.median(runs_nonempty) if runs_nonempty else 0.0),
        max_run=(max(runs_nonempty) if runs_nonempty else 0),
        final_tail=final_tail,
        hist=hist,
        unknown_active=unknown_active,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("circuits", nargs="*", default=[
        "distillation", "cultivation_d3", "coherent_d3_r1", "coherent_d5_r1",
        "coherent_d5_r5", "coherent_d7_r1"])
    args = ap.parse_args()

    print(f"{'circuit':16s} {'active':>7s} {'defer':>7s} {'bound':>6s} {'zmeas':>6s} "
          f"{'bnd_den':>8s} {'meanRun':>8s} {'maxRun':>7s} {'tail':>6s}")
    allhist = {}
    unknown = {}
    for c in args.circuits:
        r = analyze(c)
        print(f"{r['circuit']:16s} {r['active_ops']:7d} {r['n_defer']:7d} "
              f"{r['n_boundary']:6d} {r['n_zmeas']:6d} {r['boundary_density']:8.3f} "
              f"{r['mean_run']:8.1f} {r['max_run']:7d} {r['final_tail']:6d}")
        for k, v in r["hist"].items():
            allhist[k] = allhist.get(k, 0) + v
        for k, v in r["unknown_active"].items():
            unknown[k] = unknown.get(k, 0) + v
    print("\n=== boundary opcode breakdown (across all) ===")
    for k in sorted(allhist):
        tag = ("BOUNDARY" if k in BOUNDARY else
               "defer" if k in DEFERRABLE else
               "zmeas" if any(h in k for h in ZMEAS_HINTS) else "?")
        print(f"  {k:28s} {allhist[k]:8d}  [{tag}]")
    if unknown:
        print("\n=== UNCLASSIFIED active-looking opcodes (review) ===")
        for k in sorted(unknown):
            print(f"  {k:28s} {unknown[k]:8d}")


if __name__ == "__main__":
    main()
