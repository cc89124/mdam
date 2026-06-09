"""Measurement-level non-Clifford dependency trace for the near-Clifford backend.

For each circuit we run the BLOCK backend and record, FOR EACH MEASUREMENT, how
many of the currently-pending non-Clifford rotations that measurement actually
depends on (its anticommutation-connected core), versus how many pending
rotations it does NOT touch. The headline message:

    each measurement sees a SMALL non-Clifford dependency core, so the
    near-Clifford state size stays far below Clifft's active dense rank.

Outputs (reports/measurement_dependency_trace/):
  measurement_dependency_trace_<circuit>.csv   one row per measurement
  measurement_dependency_summary.csv           one row per circuit
  measurement_dependency_<circuit>.png         2-panel plot: (1) rotations each
                                               measurement depends on (pending vs
                                               core); (2) state size needed at each
                                               measurement (clifft vs nc), no shading

NOTE the distinctions the columns keep separate (do not conflate):
  * never_flushed  -- a WHOLE-CIRCUIT property (rotations no measurement ever
                      depends on); summary-only.
  * not_needed_this_meas = pending_rot_before - nc_core_rot -- a PER-measurement
                      property (pending rotations THIS measurement does not touch).
  * Clifft is compared on active rank / state size, NOT on rotation count.

cultivation_d5 is the all-magic limit: with frame reduction (default on) its RESIDENT
state is <= clifft at every measurement (max_block 11->10 = parity, not a win); one
intra-flush TRANSIENT +1 spike remains, shown honestly. r1 are diagnostic/appendix.
"""
from __future__ import annotations

import csv
import os

import clifft
from ttn_backend import frame_layer as ds_mod
from nearclifford_backend.backend import NearCliffordBackend, _opname
from nearclifford_backend.block_magic import BlockLazyNearClifford, _support

OUT = "reports/measurement_dependency_trace"
SEED = 12345

# headline (thesis holds: NC state < Clifft active dense), then appendix r1.
# cultivation_d5 is the all-magic limit: with frame reduction (default on) its
# RESIDENT state is now <= clifft at every measurement (11->10 = parity, not a win);
# one intra-flush TRANSIENT +1 spike remains, shown honestly.
HEADLINE = ["coherent_d5_r5", "coherent_d3_r3", "distillation", "cultivation_d3",
            "cultivation_d5"]
APPENDIX = ["coherent_d3_r1", "coherent_d5_r1"]                 # diagnostic only
CIRCUITS = HEADLINE + APPENDIX

TRACE = {}                       # shared per-run state (single-threaded)

ROW_COLS = ["circuit", "shot_id", "meas_idx", "step_id", "record_id", "meas_qubit",
            "pending_rot_before", "nc_core_rot", "not_needed_this_meas",
            "nc_core_qubits", "nc_state_log2_transient", "nc_state_log2_resident",
            "clifft_active_rank_at_meas", "state_gap_log2",
            "core_rot_ids", "core_qubit_ids"]

SUMMARY_COLS = ["circuit", "total_rotations", "ever_flushed_rotations",
                "never_flushed_rotations", "never_ratio", "max_pending_rot_before",
                "max_nc_core_rot", "max_nc_core_qubits", "max_clifft_active_rank",
                "max_nc_state_log2_transient", "max_nc_state_log2_resident",
                "max_state_gap_log2", "avg_state_gap_log2", "sum_2_nc_core_qubits"]


# --------------------------------------------------------------- instrumentation
_orig_apply = BlockLazyNearClifford.apply_rotation
_orig_flush_core = BlockLazyNearClifford._flush_core
_orig_measure_z = BlockLazyNearClifford.measure_z


def _apply_rotation(self, x, z, theta):
    if not hasattr(self, "_rot_ids"):
        self._rot_ids = []
        self._next_rot_id = 0
    _orig_apply(self, x, z, theta)
    self._rot_ids.append(self._next_rot_id)        # parallel id, order-aligned
    self._next_rot_id += 1


def _flush_core(self, qx, qz):
    ids = getattr(self, "_rot_ids", [])
    pending_before = len(self.pending)
    in_core = self._core_indices(qx, qz)
    ncore = sum(1 for c in in_core if c)
    core_ids = []
    core_suppmask = 0                                # union pullback support of core
    if ncore:
        aligned = len(ids) == len(in_core)
        if aligned:
            core_ids = [ids[j] for j in range(len(in_core)) if in_core[j]]
        for j in range(len(in_core)):
            if in_core[j]:
                r = self.pending[j]
                xp, zp, _ = self._pullback(r[0], r[1])      # physical -> pre-frame
                core_suppmask |= xp | zp
        if aligned:                                  # keep ids aligned to keep-set
            self._rot_ids = [ids[j] for j in range(len(in_core)) if not in_core[j]]
    _orig_flush_core(self, qx, qz)
    # nc_core_qubits = the FACTORED entangled BLOCK the core materialises (the actual
    # dense cost), NOT the raw union footprint -- the block-factoring is exactly why
    # the state stays small. Take the largest magic block that holds any core-support
    # qubit, measured just after the flush (before the measured-qubit merge / purge).
    core_blk_qubits = []
    if ncore:
        seen = set()
        for t in _support(core_suppmask, 0):
            if self.mag.has(t):
                bi = self.mag.q2b[t]
                if bi not in seen:
                    seen.add(bi)
                    qs = self.mag.blocks[bi][0]
                    if len(qs) > len(core_blk_qubits):
                        core_blk_qubits = sorted(qs)
    self._last_flush = dict(pending_before=pending_before, core_rot=ncore,
                            core_ids=core_ids, core_qubits=core_blk_qubits)


def _measure_z(self, q):
    self.take_step_peak()                            # rearm to settled
    out = _orig_measure_z(self, q)
    t_blk, _ = self.take_step_peak()                 # transient peak of THIS measure
    be = TRACE["backend"]
    active = set(be.slot2id.values()) - {q}          # post-measurement active set
    r_blk, _ = self.mag.live_stats(active)
    lf = getattr(self, "_last_flush", {}) or {}
    pb = int(lf.get("pending_before", 0)); cr = int(lf.get("core_rot", 0))
    arank = int(TRACE["active_rank"])
    cq = lf.get("core_qubits", [])
    TRACE["rows"].append({
        "circuit": TRACE["circuit"], "shot_id": TRACE["shot"],
        "meas_idx": TRACE["meas_ctr"], "step_id": TRACE["step"],
        "record_id": TRACE.get("record_id", -1), "meas_qubit": q,
        "pending_rot_before": pb, "nc_core_rot": cr, "not_needed_this_meas": pb - cr,
        "nc_core_qubits": len(cq), "nc_state_log2_transient": int(t_blk),
        "nc_state_log2_resident": int(r_blk), "clifft_active_rank_at_meas": arank,
        "state_gap_log2": arank - int(t_blk),
        "core_rot_ids": " ".join(str(i) for i in lf.get("core_ids", [])),
        "core_qubit_ids": " ".join(str(i) for i in cq),
    })
    TRACE["meas_ctr"] += 1
    return out


BlockLazyNearClifford.apply_rotation = _apply_rotation
BlockLazyNearClifford._flush_core = _flush_core
BlockLazyNearClifford.measure_z = _measure_z

_MEAS_OPS = ("OP_MEAS_ACTIVE_DIAGONAL", "OP_MEAS_ACTIVE_INTERFERE",
             "OP_SWAP_MEAS_INTERFERE")


def _recorder(step, backend):
    prog = TRACE["prog"]
    if step >= len(prog):
        return
    TRACE["step"] = step
    TRACE["active_rank"] = len(backend.slot2id)       # = Clifft active rank at step
    name = _opname(prog[step].opcode)
    if any(name.startswith(m) for m in _MEAS_OPS):
        d = ds_mod._d(prog[step])
        TRACE["record_id"] = int(d.get("classical_idx", -1))


# ---------------------------------------------------------------------- run
def trace_circuit(circuit):
    src = open(f"qec_bench/circuits/{circuit}.stim").read()
    prog = clifft.compile(src)
    be = NearCliffordBackend(block=True)
    TRACE.clear()
    TRACE.update(prog=prog, backend=be, circuit=circuit, shot=0, meas_ctr=0,
                 step=0, active_rank=0, record_id=-1, rows=[])
    be.run_shot(prog, SEED, step_recorder=_recorder)
    rows = TRACE["rows"]
    total = int(getattr(be.nc, "_next_rot_id", 0))
    never = len(getattr(be.nc, "pending", []))        # leftover = never flushed
    return rows, total, never


def summarize(circuit, rows, total, never):
    ever = total - never
    gaps = [r["state_gap_log2"] for r in rows] or [0]
    return {
        "circuit": circuit, "total_rotations": total,
        "ever_flushed_rotations": ever, "never_flushed_rotations": never,
        "never_ratio": round(never / total, 4) if total else 0.0,
        "max_pending_rot_before": max((r["pending_rot_before"] for r in rows), default=0),
        "max_nc_core_rot": max((r["nc_core_rot"] for r in rows), default=0),
        "max_nc_core_qubits": max((r["nc_core_qubits"] for r in rows), default=0),
        "max_clifft_active_rank": max((r["clifft_active_rank_at_meas"] for r in rows), default=0),
        "max_nc_state_log2_transient": max((r["nc_state_log2_transient"] for r in rows), default=0),
        "max_nc_state_log2_resident": max((r["nc_state_log2_resident"] for r in rows), default=0),
        "max_state_gap_log2": max(gaps), "avg_state_gap_log2": round(sum(gaps) / len(gaps), 3),
        "sum_2_nc_core_qubits": sum(2 ** r["nc_core_qubits"] for r in rows),
    }


# ---------------------------------------------------------------------- plot
def plot_circuit(circuit, rows, summ, headline):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [r["meas_idx"] for r in rows]
    if not xs:
        return
    pend = [r["pending_rot_before"] for r in rows]
    core = [r["nc_core_rot"] for r in rows]
    cl = [r["clifft_active_rank_at_meas"] for r in rows]
    nc = [r["nc_state_log2_transient"] for r in rows]      # size NEEDED at the measure

    fig, ax = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True)
    tag = "HEADLINE" if headline else "appendix (r1)"
    fig.suptitle(f"Measurement-level non-Clifford dependency  —  {circuit}  [{tag}]"
                 f"   ({len(xs)} measurements)",
                 fontsize=13, fontweight="bold")

    # (1) rotation dependency: pending vs the core this measurement depends on
    ax[0].plot(xs, pend, color="#9aa0a6", lw=1.6, label="pending_rot_before")
    ax[0].plot(xs, core, color="#c0392b", lw=2.0, label="nc_core_rot")
    ax[0].set_ylabel("rotation count")
    ax[0].set_title("(1) rotations each measurement depends on", fontsize=10)
    ax[0].legend(fontsize=9, loc="upper left"); ax[0].grid(alpha=0.3)

    # (2) state size needed at each measurement: clifft vs nc
    ax[1].plot(xs, cl, color="#1a2a3a", lw=2.1, label="clifft")
    ax[1].plot(xs, nc, color="#c0392b", lw=2.0, label="nc")
    ax[1].set_ylabel("log2 state dimension")
    ax[1].set_xlabel("meas_idx")
    ax[1].set_title("(2) state size needed at each measurement: clifft vs nc", fontsize=10)
    ax[1].legend(fontsize=9, loc="upper right"); ax[1].grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(OUT, f"measurement_dependency_{circuit}.png")
    fig.savefig(path, dpi=130); plt.close(fig)
    return path


# ---------------------------------------------------------------------- main
def main():
    os.makedirs(OUT, exist_ok=True)
    summaries = []
    for c in CIRCUITS:
        print(f"[trace] {c} ...", flush=True)
        rows, total, never = trace_circuit(c)
        # per-circuit trace CSV
        with open(os.path.join(OUT, f"measurement_dependency_trace_{c}.csv"), "w",
                  newline="") as f:
            w = csv.DictWriter(f, fieldnames=ROW_COLS); w.writeheader()
            for r in rows:
                w.writerow(r)
        summ = summarize(c, rows, total, never)
        summaries.append(summ)
        p = plot_circuit(c, rows, summ, headline=(c in HEADLINE))
        print(f"   {len(rows):4d} measurements, total_rot={total}, never={never} "
              f"({summ['never_ratio']:.1%}), max_gap={summ['max_state_gap_log2']} -> {p}")
    # combined summary CSV
    with open(os.path.join(OUT, "measurement_dependency_summary.csv"), "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLS); w.writeheader()
        for s in summaries:
            w.writerow(s)
    print(f"\nsummary -> {os.path.join(OUT, 'measurement_dependency_summary.csv')}")
    return summaries


if __name__ == "__main__":
    main()
