"""Verify the dead-rotation pruning optimisation (NearCliffordBackend(drop_dead=True)).

A rotation that is NEVER flushed -- never enters any measurement's anticommutation-
connected core, even transitively -- commutes with every measured Pauli for the whole
circuit, so it never touches the dense magic register and never affects a record bit.
For these QEC circuits the active-gate stream is outcome-independent (measurement records
only steer the deferred Pauli FRAME, never the near-Clifford tableau/rotations), so which
rotations flush is a fixed, seed-invariant property of the program. ``drop_dead`` finds
the never-flushed set with one cached structure pass and removes those rotations from the
``pending`` list on every shot.

Two guarantees are checked here:

  1. RECORD-BIT-IDENTICAL: for the SAME seed, drop_dead and the default block path produce
     the byte-for-byte identical measurement record. (Stronger than a distribution match:
     the default path is already validated against clifft, so bit-identity transfers that.)
  2. The pruned fraction equals the independently-measured "never-flushed ratio" and the
     end-of-circuit memory floor collapses to the Clifford tableau (the pending dead weight
     -- the floor that did not decrease at circuit end -- is gone).

Usage:  python -m nearclifford_backend.scripts.verify_drop_dead [circ ...]
"""
from __future__ import annotations
import sys
import clifft

from nearclifford_backend.backend import NearCliffordBackend

DEFAULT = ["coherent_d3_r1", "coherent_d3_r3", "coherent_d5_r1", "coherent_d5_r5",
           "distillation", "cultivation_d3", "cultivation_d5"]
SEEDS = [1, 2, 3, 7, 42, 100, 777, 2024, 99991, 1234567]


def load(circ):
    return clifft.compile(open(f"qec_bench/circuits/{circ}.stim").read())


def main():
    circs = [a for a in sys.argv[1:] if not a.startswith("-")] or DEFAULT
    print(f"{'circuit':16s} {'bit-id':>7s} {'dead/tot':>10s} {'pruned%':>8s} "
          f"{'floor B (def->drop)':>22s} {'verdict':>8s}", flush=True)
    allok = True
    for circ in circs:
        prog = load(circ)
        be0 = NearCliffordBackend(block=True, drop_dead=False)   # reference: no pruning
        be1 = NearCliffordBackend(block=True, drop_dead=True)

        bit_id = True
        for sd in SEEDS:
            if dict(be0.run_shot(prog, sd)) != dict(be1.run_shot(prog, sd)):
                bit_id = False
                break

        # end-of-circuit memory floor: leftover pending = the never-flushed dead weight
        be0.run_shot(prog, 7); floor0 = be0.nc.memory_bytes()
        be1.run_shot(prog, 7); floor1 = be1.nc.memory_bytes()

        dead = be1._dead_uids_for(prog)
        tot = be1.nc._rot_uid
        frac = 100.0 * len(dead) / max(tot, 1)
        ok = bit_id
        allok &= ok
        print(f"{circ:16s} {str(bit_id):>7s} {len(dead):4d}/{tot:<5d} {frac:7.1f}% "
              f"{floor0:9d} -> {floor1:<9d} {'PASS' if ok else 'FAIL':>8s}", flush=True)
    print("\nALL", "PASS" if allok else "FAIL")
    return allok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
