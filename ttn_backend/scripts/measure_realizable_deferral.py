"""Realizable CNOT-batching window of a hybrid affine-TTN frame, under three
flush policies -- to separate what is a FUNDAMENTAL limit from an A-only-frame
artifact.

A flush of a logical qubit = the point where its deferred CNOTs must be
materialized onto the real TTN tensor, because some op needs its realized value.
Which ops force a flush depends on how complete the frame is:

  policy "A-only"          : diagonal (ROT/T/S/PHASE) and CZ ALSO flush
                             (no phase polynomial f -> a diagonal on a deferred
                             qubit can't be absorbed).  [the earlier, too-
                             conservative model]
  policy "(A,f) meas-flush": diagonal/CZ are absorbed into f (NO flush); only
                             Z-measurement (collapse materialized) and hard
                             non-diagonal boundaries flush.
  policy "(A,f) full-defer": even mid-circuit Z-measurement is deferred (handled
                             as a Z-parity/stabilizer read or deferred-measurement);
                             ONLY hard non-diagonal boundaries (U2/U4/H/interfere)
                             flush.  [the true (A,f) ceiling]

Window = #raw CNOTs accumulated into a target since its last flush (counted
directly, so no XOR-set re-expansion artifact). meanWin/maxWin show whether the
window is long enough for Gauss-Jordan resynthesis to cut transports.

The GE-minimal compute ratio for the full-defer policy (split only at hard
boundaries) is reported by scripts/measure_deferral_cnot_compression.py
(d5_r5: raw 775 -> GE 93 = 8.33x). That 8.33x is the (A,f) ceiling; the question
here is how the WINDOW (hence realizability) changes with the measurement policy.
"""
from __future__ import annotations

import argparse
import clifft
from ttn_backend import treewidth as T_mod
from ttn_backend import frame_layer as ds_mod

DIAG_1Q = {"OP_ARRAY_T", "OP_ARRAY_T_DAG", "OP_ARRAY_S", "OP_ARRAY_S_DAG",
           "OP_ARRAY_ROT", "OP_PHASE_T", "OP_PHASE_T_DAG", "OP_PHASE_ROT"}
HARD_1Q = {"OP_ARRAY_H", "OP_ARRAY_U2"}
ZMEAS = {"OP_MEAS_ACTIVE_DIAGONAL", "OP_MEAS_ACTIVE_DIAGONAL_FORCED"}
HARD_MEAS = {"OP_MEAS_ACTIVE_INTERFERE", "OP_MEAS_ACTIVE_INTERFERE_FORCED"}

DEFAULT = ["coherent_d5_r1", "coherent_d7_r1", "coherent_d5_r5",
           "cultivation_d3", "distillation"]

POLICIES = ("A-only", "(A,f) meas-flush", "(A,f) full-defer")


def analyze(circ, policy):
    diag_flush = (policy == "A-only")          # diagonal/CZ force a flush?
    zmeas_flush = (policy != "(A,f) full-defer")  # Z-meas forces a flush?
    prog = clifft.compile(open(f"qec_bench/circuits/{circ}.stim").read())
    cnt = {}            # slot -> raw CNOTs accumulated into it since last flush
    raw = 0
    windows = []

    def flush(s):
        w = cnt.get(s, 0)
        if w:
            windows.append(w)
        cnt[s] = 0

    for i in range(len(prog)):
        inst = prog[i]
        name = T_mod._opname(inst.opcode)
        a1 = int(inst.axis_1); a2 = int(inst.axis_2)

        if name.startswith("OP_EXPAND"):
            cnt[a1] = 0
        elif name == "OP_ARRAY_CNOT":
            cnt[a2] = cnt.get(a2, 0) + 1
            raw += 1
        elif name == "OP_ARRAY_MULTI_CNOT":
            d = ds_mod._d(inst); tgt = a1
            for ctrl in ds_mod._bits(int(d["mask"])):
                if ctrl == tgt:
                    continue
                cnt[tgt] = cnt.get(tgt, 0) + 1
                raw += 1
        elif name == "OP_ARRAY_SWAP":
            cnt[a1], cnt[a2] = cnt.get(a2, 0), cnt.get(a1, 0)
        elif name in DIAG_1Q:
            if diag_flush:
                flush(a1)
        elif name in ("OP_ARRAY_CZ",):
            if diag_flush:
                flush(a1); flush(a2)
        elif name == "OP_ARRAY_MULTI_CZ":
            if diag_flush:
                d = ds_mod._d(inst)
                flush(a1)
                for t in ds_mod._bits(int(d["mask"])):
                    if t != a1:
                        flush(t)
        elif name in HARD_1Q:                  # U2 / H: always hard flush
            flush(a1)
        elif name == "OP_ARRAY_U4":            # general U4: hard flush both
            flush(a1); flush(a2)
        elif name in ZMEAS:
            if zmeas_flush:
                flush(a1)
            cnt.pop(a1, None)
        elif name in HARD_MEAS:                # X/Y-basis: always hard flush
            flush(a1); cnt.pop(a1, None)
        elif name in ("OP_SWAP_MEAS_INTERFERE", "OP_SWAP_MEAS_INTERFERE_FORCED"):
            cnt[a1], cnt[a2] = cnt.get(a2, 0), cnt.get(a1, 0)
            flush(a2); cnt.pop(a2, None)

    for s in list(cnt):
        flush(s)

    mean_w = (sum(windows) / len(windows)) if windows else 0.0
    return dict(circ=circ, raw=raw, n_flush=len(windows),
                mean_w=mean_w, max_w=(max(windows) if windows else 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("circuits", nargs="*", default=DEFAULT)
    args = ap.parse_args()
    circuits = args.circuits or DEFAULT
    print("CNOT-batching window of the hybrid affine-TTN frame, by flush policy")
    print("window = #raw CNOTs accumulated into a target before it must flush\n")
    print(f"{'circuit':15s} {'policy':18s} {'#flush':>7s} {'meanWin':>8s} {'maxWin':>7s}")
    for c in circuits:
        for p in POLICIES:
            r = analyze(c, p)
            print(f"{c:15s} {p:18s} {r['n_flush']:7d} {r['mean_w']:8.2f} {r['max_w']:7d}")
        print()
    print("A-only           : diagonal ROT/CZ flush too (no f)  -> shortest window")
    print("(A,f) meas-flush : ROT/CZ absorbed in f; Z-meas still flushes (collapse)")
    print("(A,f) full-defer : even Z-meas deferred; only U2/U4/H/interfere flush")
    print("\nGE-minimal compute ratio at full-defer (split only at hard boundaries):")
    print("  see measure_deferral_cnot_compression.py -- d5_r5 raw 775 -> GE 93 = 8.33x")


if __name__ == "__main__":
    main()
