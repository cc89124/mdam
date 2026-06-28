"""Full-benchmark memory + speed: virtual-axis vs physical block NC vs clifft (2^k
dense model). Excludes coherent_d7_*. Memory = peak dense exponent (clifft k; block
in-merge transient B; VA resident |M| at step boundaries AND in-flush transient).
Speed = wall-clock/shot (block, VA; clifft is the non-runnable 2^k reference)."""
import sys, os, time
os.chdir("/home/jung/clifft-paper"); sys.path.insert(0, "/home/jung/clifft-paper")
sys.setrecursionlimit(50000)
import numpy as np, clifft
from nearclifford_backend.backend import NearCliffordBackend
from nearclifford_backend.block_magic import MagicRegister
from nearclifford_backend.virtual_axis.virtual_axis_runtime import VirtualAxisNearClifford

_BLK = [0]
_om = MagicRegister._merge
def _mw(self, s):
    b = _om(self, s); _BLK[0] = max(_BLK[0], len(self.blocks[b][0])); return b
MagicRegister._merge = _mw

_VAT = [0]
_of = VirtualAxisNearClifford._flush_one
def _fw(self, x, z, t):
    r = _of(self, x, z, t); _VAT[0] = max(_VAT[0], len(self.M)); return r
VirtualAxisNearClifford._flush_one = _fw

CIRCS = ["coherent_d3_r1", "coherent_d3_r3", "coherent_d5_r1", "coherent_d5_r5",
         "distillation", "cultivation_d3", "cultivation_d5", "surface_d7_r7"]


def measure(circ):
    prog = clifft.compile(open(f"qec_bench/circuits/{circ}.stim").read())
    # memory (single structural shot)
    _BLK[0] = 0; _VAT[0] = 0
    beB = NearCliffordBackend(block=True); k = [0]; bres = [0]
    def rB(s, bk): k[0] = max(k[0], len(bk.slot2id)); bres[0] = max(bres[0], bk.nc.mag.max_block())
    beB.run_shot(prog, 42, step_recorder=rB)
    beV = NearCliffordBackend(virtual_axis=True); vres = [0]
    def rV(s, bk): vres[0] = max(vres[0], len(bk.nc.M))
    beV.run_shot(prog, 42, step_recorder=rV)
    res = dict(circ=circ, k=k[0], blkB=_BLK[0], blk_res=bres[0],
               va_res=vres[0], va_trans=max(_VAT[0], vres[0]))
    # speed: warm (pre-pass) then time
    beB.run_shot(prog, 7); beV.run_shot(prog, 7)
    t0 = time.perf_counter(); nB = 0
    while time.perf_counter() - t0 < 2.0 and nB < 60:
        beB.run_shot(prog, 100 + nB); nB += 1
    res["blk_ms"] = (time.perf_counter() - t0) / nB * 1e3
    t0 = time.perf_counter(); nV = 0
    while time.perf_counter() - t0 < 2.0 and nV < 60:
        beV.run_shot(prog, 100 + nV); nV += 1
    res["va_ms"] = (time.perf_counter() - t0) / nV * 1e3
    return res


if __name__ == "__main__":
    circs = sys.argv[1:] or CIRCS
    print(f"{'circuit':16} | {'clifft_k':>8} {'blockB':>6} {'VA_res':>6} {'VA_tr':>5} | "
          f"{'VA_res vs clifft':>16} | {'block ms':>9} {'VA ms':>8} {'VA/blk':>7}")
    for c in circs:
        try:
            r = measure(c)
        except Exception as e:
            print(f"{c:16} | ERROR {type(e).__name__}: {e}"); continue
        cmp = ("=" if r['va_res'] == r['k'] else ("<" if r['va_res'] < r['k'] else ">")) + f" clifft"
        print(f"{r['circ']:16} | {r['k']:>8} {r['blkB']:>6} {r['va_res']:>6} {r['va_trans']:>5} | "
              f"{cmp:>16} | {r['blk_ms']:>8.2f} {r['va_ms']:>7.2f} "
              f"{r['va_ms']/max(r['blk_ms'],1e-9):>6.2f}x")
    print("DONE")
