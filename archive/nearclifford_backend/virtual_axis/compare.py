"""Virtual-axis vs physical-block NC vs clifft: exactness + memory + speed.

Exactness: per-measurement-bit marginals over N shots must match the block backend
(distribution-exact; virtual-axis is not bit-identical because the reduction reorders
the lazy frame/RNG, like decouple_demote). Memory: peak dense exponent (block's true
in-merge transient B; virtual-axis |M|; clifft active rank k). Speed: wall-clock/shot."""
import sys, os, time
os.chdir("/home/jung/clifft-paper"); sys.path.insert(0, "/home/jung/clifft-paper")
sys.setrecursionlimit(50000)
import numpy as np, clifft
from nearclifford_backend.backend import NearCliffordBackend
from nearclifford_backend.block_magic import MagicRegister
from nearclifford_backend.virtual_axis.virtual_axis_runtime import VirtualAxisNearClifford

# --- block backend true in-merge transient peak (samples inside the kron merge) ---
_BLK_TRANS = [0]
_orig_merge = MagicRegister._merge
def _merge_w(self, support):
    b = _orig_merge(self, support)
    _BLK_TRANS[0] = max(_BLK_TRANS[0], len(self.blocks[b][0]))
    return b
MagicRegister._merge = _merge_w

# --- virtual-axis true |M| transient (samples after every rotation flush + reduction) ---
_VA_TRANS = [0]
_orig_flush_one = VirtualAxisNearClifford._flush_one
def _va_flush(self, x, z, theta):
    r = _orig_flush_one(self, x, z, theta)
    _VA_TRANS[0] = max(_VA_TRANS[0], len(self.M))
    return r
VirtualAxisNearClifford._flush_one = _va_flush


def marginals(be, prog, seeds):
    """mean measurement-bit vector over shots (per classical index)."""
    acc = {}
    for sd in seeds:
        rec = be.run_shot(prog, sd)
        for k, v in rec.items():
            acc[k] = acc.get(k, 0) + int(v)
    keys = sorted(acc)
    return np.array([acc[k] / len(seeds) for k in keys]), keys


def compare(circ, n_exact=3000, n_time=40):
    prog = clifft.compile(open(f"qec_bench/circuits/{circ}.stim").read())
    seeds = list(range(1, n_exact + 1))
    beB = NearCliffordBackend(block=True)
    beV = NearCliffordBackend(virtual_axis=True)
    # exactness: marginals
    mB, kB = marginals(beB, prog, seeds)
    mV, kV = marginals(beV, prog, seeds)
    tvd = float(np.max(np.abs(mB - mV))) if kB == kV else 9.99
    # memory: peak exponents (single representative shot, structural -> shot-invariant)
    _BLK_TRANS[0] = 0; _VA_TRANS[0] = 0
    kmax = [0]; vares = [0]
    def recB(s, bk): kmax[0] = max(kmax[0], len(bk.slot2id))
    def recV(s, bk): vares[0] = max(vares[0], len(bk.nc.M))
    beB.run_shot(prog, 42, step_recorder=recB)
    beV.run_shot(prog, 42, step_recorder=recV)
    blk_trans = _BLK_TRANS[0]; va_trans = max(_VA_TRANS[0], vares[0]); clk = kmax[0]
    # speed: wall-clock per shot (exclude the one-off structure pre-pass: warm first)
    beB.run_shot(prog, 7); beV.run_shot(prog, 7)
    t0 = time.perf_counter()
    for sd in range(100, 100 + n_time): beB.run_shot(prog, sd)
    tB = (time.perf_counter() - t0) / n_time
    t0 = time.perf_counter()
    for sd in range(100, 100 + n_time): beV.run_shot(prog, sd)
    tV = (time.perf_counter() - t0) / n_time
    return dict(circ=circ, clk=clk, blk_trans=blk_trans, va_trans=va_trans,
                tvd=tvd, tB=tB * 1e3, tV=tV * 1e3)


if __name__ == "__main__":
    circs = sys.argv[1:] or ["distillation", "cultivation_d3", "cultivation_d5"]
    print(f"{'circuit':16} | {'clifft_k':>8} {'block_B':>7} {'VA_|M|':>6} | "
          f"{'maxTVD':>7} | {'block ms':>9} {'VA ms':>8} {'VA/blk':>7}")
    for c in circs:
        r = compare(c)
        print(f"{r['circ']:16} | {r['clk']:>8} {r['blk_trans']:>7} {r['va_trans']:>6} | "
              f"{r['tvd']:>7.4f} | {r['tB']:>8.2f} {r['tV']:>7.2f} {r['tV']/max(r['tB'],1e-9):>6.2f}x")
    print("DONE")
