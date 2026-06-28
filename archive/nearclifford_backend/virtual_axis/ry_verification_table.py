"""Verification table for R_Y (and R_Z/R_X) coherent noise on the fused single-frame backend.

Emits, per (circuit, noise_axis):
  circuit, noise_axis, status, clifft_k, fused_ws, resident, max_core_rot, max_pauli_sum_terms,
  error_if_any

For axis-aligned noise (R_Z) and the small / feasible cases a FULL fused run is done.  Off-axis
R_Y (and large R_X) blow the magic rank up to ~clifft_k (both syndrome sectors anticommute), so
their fused workspace 2^ws is RAM-infeasible -- those rows are a CAPPED dry-run: the trajectory
advances until the first measurement core whose workspace would exceed 2^CAP, then reports
clifft_k (exact, from compile) + that core's W as the estimated workspace + the OOM reason.

Run: clifft_env/bin/python -m nearclifford_backend.virtual_axis.ry_verification_table
"""
import copy
import sys

sys.path.insert(0, "/home/jung/clifft-paper")

import clifft
import nearclifford_backend.backend as bk
import nearclifford_backend.virtual_axis.fused_integrate as fi
from nearclifford_backend.virtual_axis.fused_single_frame import FusedSingleFrame, compile_circuit

CAP = 22                                       # 2^22 complex = 64 MiB/vector workspace ceiling.
                                               # (The COMPUTE wall -- Pauli-sum 2^L for off-axis
                                               # R_Y -- is enforced inside flush_core_virtual by
                                               # `_term_guard`/LargeCoreNeedsProjectedTN, C-7.1.)


class _OverCap(Exception):
    """MEMORY wall only: a measurement core's workspace 2^(W-1) would exceed 2^CAP."""
    def __init__(self, W, L, reason="mem"):
        self.W, self.L, self.reason = W, L, reason


def _probe_W(eng, rots, Pm):
    """Tableau-only work-basis size for this core (no phi materialised)."""
    pe = copy.deepcopy(eng); pe.phi = None
    for (P, th) in rots:
        pe._mask_for(P)
    pe._mask_for(Pm)
    return len(pe.magic)


def measure(circ_text, seed=1, cap=CAP):
    """Run the backend; instrument core/pauli-sum stats. Capped: abort the FIRST core whose
    workspace would exceed 2^cap. Returns dict of stats (+ possibly an OOM reason)."""
    prog = compile_circuit(circ_text)
    k = prog.peak_rank
    stats = {"clifft_k": k, "max_core_rot": 0, "max_pauli_terms": 0,
             "fused_ws": None, "resident": None, "error": ""}

    orig_flush = fi.flush_core_virtual
    orig_psum = fi._pauli_sum

    def psum_hook(masks):
        out = orig_psum(masks)
        stats["max_pauli_terms"] = max(stats["max_pauli_terms"], len(out))
        return out

    def flush_hook(eng, rots, Pm, forced=None, rng=None):
        L = len(rots)
        stats["max_core_rot"] = max(stats["max_core_rot"], L)
        # COMPUTE wall is now enforced INSIDE flush_core_virtual (`_term_guard` ->
        # LargeCoreNeedsProjectedTN, caught in measure()); here we only guard the MEMORY wall
        # (workspace 2^W).  W <= r_in + L + 1; skip the tableau probe when the bound is in cap.
        if len(eng.magic) + L <= cap:
            return orig_flush(eng, rots, Pm, forced=forced, rng=rng)
        W = _probe_W(eng, rots, Pm)
        if W - 1 > cap:
            raise _OverCap(W, L, "mem")
        return orig_flush(eng, rots, Pm, forced=forced, rng=rng)

    fi.flush_core_virtual = flush_hook
    fi._pauli_sum = psum_hook
    # the engine imports flush_core_virtual by name -> patch the binding it actually calls
    import nearclifford_backend.virtual_axis.fused_single_frame as fsf
    orig_fsf_flush = fsf.flush_core_virtual
    fsf.flush_core_virtual = flush_hook

    orig_lazy = bk.LazyNearClifford
    bk.LazyNearClifford = FusedSingleFrame
    try:
        be = bk.NearCliffordBackend(lazy=True, drop_dead=False, structure_once=False)
        be.run_shot(prog, seed)
        stats["fused_ws"] = be.nc.max_fused_ws
        stats["resident"] = be.nc.max_M
        stats["status"] = "full-run"
    except fi.LargeCoreNeedsProjectedTN as e:                 # COMPUTE wall (engine guard, C-7.1)
        et = e.estimated_terms.bit_length() - 1
        stats["fused_ws"] = e.W - 1                           # workspace is memory-fine
        stats["resident"] = None
        stats["max_core_rot"] = max(stats["max_core_rot"], e.L)
        stats["status"] = "projected-tn-required"
        stats["error"] = (f"off-axis: core L={e.L} -> fused Pauli-sum ~2^{et} terms (> TERM_CAP); "
                          f"workspace 2^{e.W-1} is memory-OK, COMPUTE blows up; needs projected-TN")
    except _OverCap as e:                                     # MEMORY wall (workspace 2^W)
        stats["fused_ws"] = e.W - 1
        stats["resident"] = None
        stats["status"] = "capped-dry-run"
        stats["error"] = (f"OOM: a measurement core needs workspace 2^{e.W-1} "
                          f"(> cap 2^{cap}); {e.L} rotations; fused run skipped")
    finally:
        fi.flush_core_virtual = orig_flush
        fi._pauli_sum = orig_psum
        fsf.flush_core_virtual = orig_fsf_flush
        bk.LazyNearClifford = orig_lazy
    return stats


def main():
    base3 = open("qec_bench/circuits/coherent_d3_r3.stim").read()
    base5 = open("qec_bench/circuits/coherent_d5_r5.stim").read()
    hdr = ("circuit", "noise_axis", "status", "clifft_k", "fused_ws", "resident",
           "max_core_rot", "max_pauli_sum_terms", "error_if_any")
    print(",".join(hdr), flush=True)
    for cname, base in [("coherent_d3_r3", base3), ("coherent_d5_r5", base5)]:
        for ax in ("R_Z", "R_X", "R_Y"):
            txt = base.replace("R_Z(0.02)", f"{ax}(0.02)")
            s = measure(txt)
            print(",".join(str(x) for x in (
                cname, ax, s["status"], s["clifft_k"],
                s["fused_ws"], s["resident"] if s["resident"] is not None else "-",
                s["max_core_rot"], s["max_pauli_terms"], s["error"] or "-")), flush=True)


if __name__ == "__main__":
    main()
