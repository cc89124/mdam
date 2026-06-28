"""Experiment table for offline-selector + targeted-peel.

Per circuit:
  k_max        clifft active rank (dense 2^k model)            = max len(slot2id)
  b_max        near-Clifford peak block (true in-merge transient) = selector's metric
  flop_mm      irreducible O(2^b) dense arithmetic (Born/projection/rotation apply)
  flop_norm    factoring scan -- baseline (full) vs targeted (recorded peelers only)
  selector     NC iff b_max < k_max  (else clifft: NC cannot shrink the state)

The deployed (selector) path is min(2^k, 2^b) -> never worse than clifft; on the
NC-selected circuits targeted-peel cuts flop_norm (the only compute excess over
clifft) by ~80-95%, after which the irreducible flop_mm dominates."""
import sys, os
os.chdir("/home/jung/clifft-paper"); sys.path.insert(0, "/home/jung/clifft-paper")
sys.setrecursionlimit(50000)
import numpy as np, clifft
from nearclifford_backend.backend import NearCliffordBackend
from nearclifford_backend.block_magic import MagicRegister

# transient (in-merge) block peak -- the honest high-water, selector's b_max
_TRANS = [0]
_orig_merge = MagicRegister._merge
def _merge_w(self, support):
    b = _orig_merge(self, support)
    _TRANS[0] = max(_TRANS[0], len(self.blocks[b][0]))
    return b
MagicRegister._merge = _merge_w

def measure(circ, seed=42):
    prog = clifft.compile(open(f"qec_bench/circuits/{circ}.stim").read())
    out = {}
    for targeted in (False, True):
        _TRANS[0] = 0
        be = NearCliffordBackend(block=True, targeted_peel=targeted)
        kmax = [0]
        def rec(s, bk, _k=kmax): _k[0] = max(_k[0], len(bk.slot2id))
        r = be.run_shot(prog, seed, step_recorder=rec)
        out["targeted" if targeted else "base"] = dict(
            record=dict(r), flop_mm=be.nc.mag.flop_mm, flop_norm=be.nc.mag.flop_norm,
            k_max=kmax[0], b_max=_TRANS[0])
    return out

CIRCS = sys.argv[1:] or ["distillation", "cultivation_d3", "cultivation_d5"]
rows = []
for circ in CIRCS:
    m = measure(circ)
    b, t = m["base"], m["targeted"]
    pick = "NC" if t["b_max"] < t["k_max"] else "clifft"
    red = 100 * (1 - t["flop_norm"] / max(b["flop_norm"], 1))
    rows.append(dict(circ=circ, k=t["k_max"], b=t["b_max"], pick=pick,
                     fmm=t["flop_mm"], fnb=b["flop_norm"], fnt=t["flop_norm"],
                     red=red, identical=(b["record"] == t["record"])))

print("\n### Per-circuit: selector decision + targeted-peel compute\n")
print(f"| {'circuit':16} | k(clifft) | b(NC) | selector | flop_mm | flop_norm base | "
      f"flop_norm targ | reduce | rec-identical |")
print("|" + "-"*18 + "|" + "-"*11 + "|" + "-"*7 + "|" + "-"*10 + "|" + "-"*9 + "|"
      + "-"*16 + "|" + "-"*16 + "|" + "-"*8 + "|" + "-"*15 + "|")
for r in rows:
    print(f"| {r['circ']:16} | {r['k']:>9} | {r['b']:>5} | {r['pick']:>8} | "
          f"{r['fmm']:>7.2e} | {r['fnb']:>14.3e} | {r['fnt']:>14.3e} | {r['red']:>5.1f}% | "
          f"{str(r['identical']):>13} |")
print("\nDONE")
