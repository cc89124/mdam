"""Frame-only core capture: record the pulled-back measurement cores WITHOUT building
clifft's dense 2^k magic state.  Uses the lazy backend's `resource_only` mode (it sizes the
core support via the Clifford frame only -- never materialises phi), so the cores for a
circuit whose magic rank is too large for the dense capture (coherent_d5_r5: 2^24) are still
extractable.  The EV it returns is byte-for-byte the same format as test_c3.capture_stream.
"""
import os
import sys

os.chdir("/home/jung/clifft-paper")
sys.path.insert(0, "/home/jung/clifft-paper")
sys.setrecursionlimit(200000)

import clifft

from nearclifford_backend.backend import NearCliffordBackend, count_idents
from nearclifford_backend.lazy import LazyNearClifford


def capture_stream_resource(circ, seed=1):
    """(n, EV) with EV = [(Pm, [(P,theta)...]) ...], frame-pulled-back, magic NEVER built."""
    EV = []
    o_fc = LazyNearClifford._flush_core
    o_df = LazyNearClifford._do_flush

    def fc(self, qx, qz):
        EV.append((self._pullback(qx, qz), []))
        return o_fc(self, qx, qz)

    def df(self, qx, qz, flush):
        if EV and getattr(self, "resource_only", False):
            for (x, z, p, theta, uid) in flush:          # increasing-uid order preserved
                EV[-1][1].append((self._pullback(x, z), theta))
        return o_df(self, qx, qz, flush)

    LazyNearClifford._flush_core = fc
    LazyNearClifford._do_flush = df
    try:
        prog = clifft.compile(open(f"qec_bench/circuits/{circ}.stim").read())
        n = count_idents(prog)
        be = NearCliffordBackend(lazy=True, resource_only=True)
        be.run_shot(prog, seed)
    finally:
        LazyNearClifford._flush_core = o_fc
        LazyNearClifford._do_flush = o_df
    return n, EV


if __name__ == "__main__":
    # validation: resource-only capture must equal the dense capture_stream on small circuits
    from nearclifford_backend.virtual_axis.test_c3 import capture_stream
    for circ in (sys.argv[1:] or ["cultivation_d3", "coherent_d3_r3", "distillation"]):
        nD, evD = capture_stream(circ)
        nR, evR = capture_stream_resource(circ)
        same = (nD == nR and len(evD) == len(evR)
                and all(a == b for a, b in zip(evD, evR)))
        print(f"{circ:16} dense:(n={nD},cores={len(evD)})  resource:(n={nR},cores={len(evR)})  "
              f"identical={same}")
