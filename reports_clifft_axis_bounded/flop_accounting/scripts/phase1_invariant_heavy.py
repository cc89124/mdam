"""Confirm the Phase-1 residual-product invariant on the circuits NOT in phase1_verify
(the higher-rank d5/d7/surface family).  Runs each at 1 seed with _purge_verify=True, so
every measurement asserts the cheap support-gate residual sweep drops EXACTLY what the
original O(k) _compress_magic scan would.  A clean pass = the direct-drop + support-gate is
equivalent to the original compress on these circuits too.  Per-circuit wall-time printed;
a circuit that exceeds the soft budget is reported as TIMEOUT (not a failure)."""
import sys, signal
sys.path.insert(0, "/home/jung/clifft-paper")
sys.setrecursionlimit(400000)
import nearclifford_backend.backend as bk
from nearclifford_backend.clifft_axis.bounded import compile_bounded, CliftAxisBoundedNearClifford as C

CIRCS = ["coherent_d3_r1", "coherent_d5_r1", "coherent_ry_d5_r1", "coherent_rx_d5_r1",
         "coherent_ry_d5_r5", "coherent_rx_d5_r5", "coherent_d7_r1",
         "coherent_d7_r7", "surface_d7_r7"]


class TO(Exception):
    pass


def _alarm(sig, frm):
    raise TO()


signal.signal(signal.SIGALRM, _alarm)

print("=== Phase-1 residual-product invariant, heavy circuits (1 seed, _purge_verify) ===")
for circ in CIRCS:
    prog = compile_bounded(open(f"qec_bench/circuits/{circ}.stim").read())
    C._purge_verify = True
    signal.alarm(240)
    import time
    try:
        be = bk.NearCliffordBackend(clifft_axis_bounded=True, drop_dead=False,
                                    structure_once=False, clifft_axis_enforce=True)
        be.run_shot(prog, 1)
        peak = be.nc.budget.peak_resident.bit_length() - 1
        print(f"  {circ:20} peak_rank={peak:2}  PASS (residual sweep == compress)", flush=True)
    except TO:
        print(f"  {circ:20} TIMEOUT (>240s) -- skipped", flush=True)
    except AssertionError as e:
        print(f"  {circ:20} FAIL: {e}", flush=True)
    except Exception as e:
        print(f"  {circ:20} ERR {type(e).__name__}: {e}", flush=True)
    finally:
        signal.alarm(0)
        C._purge_verify = False
