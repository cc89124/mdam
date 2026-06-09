"""Per-step memory: Clifft dense active-state vs the TTN backend.

For each runtime step we record two numbers:

  Clifft (dense active-state) = 16 * 2^(#active idents at that step)
      This is the dense state vector over the currently-active (promoted)
      identities -- the representation whose 2^k growth is Clifft's memory wall
      (e.g. k=24 -> 256 MiB). It is the baseline the TTN backend is compared
      against throughout this repo.

  TTN backend (actual stored)  = sum_B 16 * numel(bag_B)
      The real resident memory of the tensor-tree network at that step
      (self._stored_bytes()), captured via the trace recorder.

Outputs a per-step CSV, a PNG plot (log-y), and a summary comparing both the
per-step PEAK and the SUM over all steps (area under the curve).

Run:
  OPENBLAS_NUM_THREADS=4 /home/jung/clifft_env/bin/python \
    -m ttn_backend.scripts.per_step_memory_compare coherent_d5_r1 \
    --max-steps 0 --out-dir reports/per_step_memory
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

# the carving seeder (temporal_carving/seed.py) recurses once per axis on a
# degenerate mincut split; large stabilizer circuits (surface_d7_r7 ~ 1000 axes)
# blow the default 1000-frame limit. Raise it (depth stays well under a few k).
sys.setrecursionlimit(50000)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import clifft
from ttn_backend.backend_spec import export_backend_spec, assign_homes_and_classify
from ttn_backend import TTNBackend
from nearclifford_backend.backend import NearCliffordBackend

POLICY = dict(
    TTN_FUSE_MULTICNOT="1",
    TTN_PERSISTENT_MULTICNOT="1",
    TTN_PERSISTENT_MULTICNOT_MIN_MULTIS="2",
    TTN_DESTRUCTIVE_OPEN="1",
    TTN_FUSE_MULTICNOT_BATCH="1",
    TTN_FUSE_MULTICNOT_CAP_BYTES=str(64 * 1024 * 1024),
    TTN_FUSE_MULTICNOT_TOTAL_CAP_BYTES=str(64 * 1024 * 1024),
    TTN_PREFISSION_TRANSPORT_CAP_BYTES=str(64 * 1024 * 1024),
    TTN_PREFISSION_MIN_GAIN="1.01",
)


def _build_spec_homing(prog, layout):
    base_spec = export_backend_spec(prog, strict=False)
    if layout == "union":
        return base_spec, assign_homes_and_classify(base_spec)
    # carving_leaf layout (the best exact layout used in the benchmarks)
    from temporal_carving.pipeline import run as run_pipeline
    from ttn_backend.scripts.qec_temporal_carving_report import trace_from_program
    from ttn_backend.scripts.qec_temporal_carving_runtime import (
        build_carving_executable_spec,
    )
    trace = trace_from_program(prog, strict=False)
    carving = run_pipeline(trace, seeder="recursive_balanced_mincut",
                           refine_moves=("nni",), seed=0,
                           partitioner="networkx", exact=False)
    return build_carving_executable_spec(base_spec, carving["tree"])


def run_circuit(circuit, max_steps, rewrite, seed, timeout, layout):
    saved = {}
    env = dict(POLICY)
    env["TTN_MULTICNOT_PARITY_REWRITE"] = "1" if rewrite else "0"
    for k, v in env.items():
        saved[k] = os.environ.get(k)
        os.environ[k] = v
    try:
        with open(f"qec_bench/circuits/{circuit}.stim") as h:
            src = h.read()
        prog = clifft.compile(src)
        spec, homing = _build_spec_homing(prog, layout)

        per_event = {}  # step_id -> dict(ttn_bytes max, n_active max, op_kind)

        def rec(row):
            s = row.get("step_id")
            if s is None:
                return
            b = int(row["stored_bytes"])
            na = len(row["live_axes"])
            cur = per_event.get(s)
            if cur is None:
                per_event[s] = dict(ttn_bytes=b, n_active=na,
                                    op_kind=row.get("op_kind"))
            else:
                cur["ttn_bytes"] = max(cur["ttn_bytes"], b)
                cur["n_active"] = max(cur["n_active"], na)

        backend = TTNBackend(spec, homing, trace_recorder=rec)
        ms = None if not max_steps else int(max_steps)
        backend.run_shot(prog, seed, runtime_timeout=timeout, max_steps=ms)
        total_steps = len(prog)
        m = backend.state.metrics
        info = dict(
            n_qr=int(m.get("n_qr", 0)),
            n_transports=int(m.get("n_transports", 0)),
            rewrite_windows=int(m.get("multicnot_parity_rewrite_windows", 0)),
        )
        return per_event, total_steps, info
    finally:
        for k, old in saved.items():
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old


def run_circuit_nc(circuit, max_steps, seed, timeout):
    """Per-step resident memory of the near-Clifford block backend (one shot).
    Returns {step: peak memory_bytes during/entering that step}, or None if the
    one shot exceeds `timeout` wall-seconds (so a heavy circuit just drops the NC
    line instead of crashing the whole comparison)."""
    import time as _time
    with open(f"qec_bench/circuits/{circuit}.stim") as h:
        src = h.read()
    prog = clifft.compile(src)
    be = NearCliffordBackend(block=True)
    per_event = {}
    t0 = _time.perf_counter()

    def rec(step, backend):
        if _time.perf_counter() - t0 > timeout:
            raise TimeoutError(f"near-Clifford shot exceeded {timeout}s")
        nc = backend.nc
        # MAIN metric = the intra-step TRANSIENT high-water mark (the honest memory
        # feasibility figure): a measurement's anticommutation-core flush briefly
        # forms a larger entangled block (all pending rotations applied + factored)
        # just BEFORE the measurement projector collapses it. take_step_peak()
        # returns that (max_block, memory_bytes) peak over the interval since the
        # last record and rearms -- it mutates only bookkeeping, never the
        # trajectory. The settled step-boundary value under-reports the true peak
        # (e.g. coherent_d5_r5: transient 13 / ~133 KB vs resident 12 / ~72 KB).
        t_blk, t_mem = nc.take_step_peak()
        # SECONDARY metric = settled, step-boundary RESIDENT of the genuine ACTIVE
        # resource: exclude blocks whose qubits are all measured-out (dead tensor
        # factors the measure path leaves resident -- not safe to drop mid-run, but
        # they are not active state).
        active = set(backend.slot2id.values())
        r_blk, r_magic = nc.mag.live_stats(active)
        r_mem = r_magic + nc.overhead_bytes()
        na = len(backend.slot2id)            # concurrent active idents (= clifft k)
        prev = per_event.get(step)
        if prev is None:
            per_event[step] = dict(bytes=t_mem, n_active=na, max_block=t_blk,
                                   bytes_resident=r_mem, max_block_resident=r_blk)
        else:
            prev["bytes"] = max(prev["bytes"], t_mem)
            prev["n_active"] = max(prev["n_active"], na)
            prev["max_block"] = max(prev["max_block"], t_blk)
            prev["bytes_resident"] = max(prev["bytes_resident"], r_mem)
            prev["max_block_resident"] = max(prev["max_block_resident"], r_blk)

    ms = None if not max_steps else int(max_steps)
    n_run = len(prog) if not max_steps else min(len(prog), int(max_steps))
    try:
        be.run_shot(prog, seed, max_steps=ms, step_recorder=rec)
    except TimeoutError as e:
        print(f"  [near-Clifford] {e}; dropping NC line for {circuit}")
        return None, n_run
    return per_event, n_run + 1


def load_ttn_from_csv(path, n_steps):
    """Load the ttn_stored_bytes column (keyed by step) from an existing per-step
    CSV, so a fresh near-Clifford line can be spliced in WITHOUT re-running the
    expensive (and on d5_r5 deliberately-truncated) TTN backend. Returns a list of
    len n_steps with int bytes or None (blank = unexecuted/truncated tail)."""
    ttn = [None] * n_steps
    with open(path) as f:
        header = next(f).rstrip("\n").split(",")
        si = header.index("step")
        ti = header.index("ttn_stored_bytes")
        for line in f:
            p = line.rstrip("\n").split(",")
            s = int(p[si])
            if 0 <= s < n_steps and p[ti] != "":
                ttn[s] = int(p[ti])
    return ttn


def densify_field(per_event, n_steps, field, init):
    """Forward-fill one field of a {step: dict} per-event map over 0..n_steps-1."""
    out = []
    last = init
    for s in range(n_steps):
        ev = per_event.get(s)
        if ev is not None:
            last = ev[field]
        out.append(last)
    return out


def densify(per_event, n_steps):
    """Forward-fill steps with no active op (memory unchanged) into dense series."""
    steps, clifft_b, ttn_b, nact = [], [], [], []
    last_ttn = 16          # scalar init
    last_na = 0
    for s in range(n_steps):
        ev = per_event.get(s)
        if ev is not None:
            last_ttn = ev["ttn_bytes"]
            last_na = ev["n_active"]
        steps.append(s)
        nact.append(last_na)
        clifft_b.append(16 * (2 ** last_na))
        ttn_b.append(last_ttn)
    return steps, clifft_b, ttn_b, nact


def human(n):
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024 or unit == "TiB":
            return f"{n:.2f} {unit}"
        n /= 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("circuit")
    ap.add_argument("--max-steps", type=int, default=0, help="0 = full circuit")
    ap.add_argument("--rewrite", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--layout", choices=["carving", "union"], default="carving")
    ap.add_argument("--yscale", choices=["linear", "log"], default="linear")
    ap.add_argument("--metric", choices=["memory", "qubits", "active_dim"],
                    default="memory",
                    help="memory = resident bytes; qubits = log2 dense-equiv dim "
                         "(clifft=k, TTN=log2(stored/16), near-Clifford=max magic "
                         "block); active_dim = the dense-equivalent DIMENSION "
                         "(#amplitudes) on a LINEAR axis (clifft=2^k, TTN=stored/16, "
                         "near-Clifford=2^max_block)")
    ap.add_argument("--from-csv", action="store_true",
                    help="skip simulation; re-plot from the existing per-step CSV")
    ap.add_argument("--no-nc", action="store_true",
                    help="skip the near-Clifford block-backend line")
    ap.add_argument("--reuse-ttn", action="store_true",
                    help="do NOT re-run the (expensive) TTN backend; load its "
                         "ttn_stored_bytes column from the existing per-step CSV "
                         "and refresh only the near-Clifford line. Use when the NC "
                         "metric definition changed but the TTN/clifft lines did "
                         "not (e.g. transient vs resident NC accounting).")
    ap.add_argument("--nc-timeout", type=float, default=600.0,
                    help="wall-seconds budget for the single near-Clifford shot")
    ap.add_argument("--out-dir", default="reports/per_step_memory")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # MAIN near-Clifford series = intra-step TRANSIENT high-water mark; the
    # *_res series = settled step-boundary RESIDENT (secondary).
    nc_b = None
    nc_blk = None
    nc_b_res = None
    nc_blk_res = None
    if args.from_csv:
        steps, clifft_b, nact = [], [], []
        ttn_tmp, nc_tmp, blk_tmp, ncr_tmp, blkr_tmp = [], [], [], [], []
        with open(out / f"{args.circuit}_per_step.csv") as f:
            header = next(f).rstrip("\n").split(",")
            idx = {name: i for i, name in enumerate(header)}
            has_nc = "near_clifft_bytes" in idx
            has_blk = "near_clifft_qubits" in idx
            has_ncr = "near_clifft_bytes_resident" in idx
            has_blkr = "near_clifft_qubits_resident" in idx

            def cell(p, name):
                i = idx.get(name, -1)
                return p[i] if 0 <= i < len(p) else ""
            for line in f:
                p = line.rstrip("\n").split(",")
                steps.append(int(p[0])); nact.append(int(p[1]))
                clifft_b.append(int(p[2]))
                ttn_tmp.append(int(p[3]) if p[3] != "" else None)
                if has_nc and cell(p, "near_clifft_bytes") != "":
                    nc_tmp.append(int(cell(p, "near_clifft_bytes")))
                if has_blk and cell(p, "near_clifft_qubits") != "":
                    blk_tmp.append(int(cell(p, "near_clifft_qubits")))
                if has_ncr and cell(p, "near_clifft_bytes_resident") != "":
                    ncr_tmp.append(int(cell(p, "near_clifft_bytes_resident")))
                if has_blkr and cell(p, "near_clifft_qubits_resident") != "":
                    blkr_tmp.append(int(cell(p, "near_clifft_qubits_resident")))
        n_steps = len(steps)
        n_prog = n_steps
        info = {}
        ttn_b = ttn_tmp if any(v is not None for v in ttn_tmp) else None
        if has_nc and len(nc_tmp) == n_steps:
            nc_b = nc_tmp
        if has_blk and len(blk_tmp) == n_steps:
            nc_blk = blk_tmp
        if has_ncr and len(ncr_tmp) == n_steps:
            nc_b_res = ncr_tmp
        if has_blkr and len(blkr_tmp) == n_steps:
            nc_blk_res = blkr_tmp
    else:
        info = {}
        # near-Clifford run is the AUTHORITATIVE source of the active-ident count
        # (= clifft's dense k) and of the near-Clifford memory line itself.
        nc_event, n_steps_nc = (None, 0)
        if not args.no_nc:
            nc_event, n_steps_nc = run_circuit_nc(
                args.circuit, args.max_steps, args.seed, args.nc_timeout)
        # TTN line: re-run the backend, OR (with --reuse-ttn) splice the existing
        # CSV's TTN column so only the NC line is refreshed (avoids re-running the
        # expensive / deliberately-truncated TTN backend).
        per_event, n_prog = None, None
        if not args.reuse_ttn:
            try:
                per_event, n_prog, info = run_circuit(
                    args.circuit, args.max_steps, bool(args.rewrite), args.seed,
                    args.timeout, args.layout)
            except Exception as e:
                print(f"  [TTN] failed ({type(e).__name__}: {e}); dropping TTN line")
        # step range to plot
        if per_event:
            cap = (min(n_prog, args.max_steps) if args.max_steps else n_prog)
            n_steps = max(min(cap, max(per_event) + 1), n_steps_nc)
        else:
            n_steps = n_steps_nc
        if n_prog is None:
            n_prog = n_steps
        # clifft dense baseline from NC active count (authoritative); else from TTN
        if nc_event is not None:
            nact = densify_field(nc_event, n_steps, "n_active", 0)
            nc_b = densify_field(nc_event, n_steps, "bytes", 16)
            nc_blk = densify_field(nc_event, n_steps, "max_block", 0)
            nc_b_res = densify_field(nc_event, n_steps, "bytes_resident", 16)
            nc_blk_res = densify_field(nc_event, n_steps, "max_block_resident", 0)
        elif per_event:
            _, _, _, nact = densify(per_event, n_steps)
        else:
            nact = [0] * n_steps
        clifft_b = [16 * (2 ** na) for na in nact]
        steps = list(range(n_steps))
        if args.reuse_ttn:
            csv0 = out / f"{args.circuit}_per_step.csv"
            ttn_b = load_ttn_from_csv(csv0, n_steps) if csv0.exists() else None
            if ttn_b is not None and not any(v is not None for v in ttn_b):
                ttn_b = None
            if ttn_b is None:
                print(f"  [TTN] --reuse-ttn: no usable TTN column in {csv0}; "
                      f"dropping TTN line")
        else:
            ttn_b = densify_field(per_event, n_steps, "ttn_bytes", 16) if per_event else None
            # if TTN stopped early (timeout) before the last plotted step, do NOT
            # forward-fill its last value across the unexecuted tail -- that would
            # falsely show the memory "staying" elevated. Mark the tail as missing.
            if ttn_b is not None and per_event:
                ttn_last = max(per_event)
                # only a genuine early stop (timeout) leaves a big unexecuted tail; a
                # 1-step gap is just the near-Clifford run's extra final-state record.
                if ttn_last < 0.98 * (n_steps - 1):
                    for s in range(ttn_last + 1, n_steps):
                        ttn_b[s] = None
                    print(f"  [TTN] executed only steps 0..{ttn_last} of {n_steps - 1} "
                          f"(stopped early); tail left blank, not forward-filled")

    ttn_real = [v for v in ttn_b if v is not None] if ttn_b else []
    sum_clifft = sum(clifft_b)
    peak_clifft = max(clifft_b)
    sum_ttn = sum(ttn_real) if ttn_real else None
    peak_ttn = max(ttn_real) if ttn_real else None
    # near-Clifford MAIN = intra-step TRANSIENT high-water mark; *_res = settled
    # step-boundary RESIDENT (secondary).
    sum_nc = sum(nc_b) if nc_b else None
    peak_nc = max(nc_b) if nc_b else None
    sum_nc_res = sum(nc_b_res) if nc_b_res else None
    peak_nc_res = max(nc_b_res) if nc_b_res else None
    # exponential-STATE-only NC bytes (16*2^block): the apples-to-apples figure vs
    # Clifft's 16*2^k -- BOTH count only the dense state, no metadata. The Clifford
    # frame + unapplied-pending overhead (= nc_b - nc_state_b) is the polynomial part
    # Clifft's own (un-counted) tableau mirrors; reported separately, never in dense/NC.
    nc_state_b = [16 * (1 << b) for b in nc_blk] if nc_blk else None
    nc_state_b_res = [16 * (1 << b) for b in nc_blk_res] if nc_blk_res else None
    sum_nc_state = sum(nc_state_b) if nc_state_b else None
    peak_nc_state = max(nc_state_b) if nc_state_b else None
    sum_nc_state_res = sum(nc_state_b_res) if nc_state_b_res else None
    peak_nc_state_res = max(nc_state_b_res) if nc_state_b_res else None
    peak_nc_overhead = (max(a - b for a, b in zip(nc_b, nc_state_b))
                        if nc_b and nc_state_b else None)

    # CSV (skip rewrite in replot mode)
    csv_path = out / f"{args.circuit}_per_step.csv"
    if not args.from_csv:
        with open(csv_path, "w") as f:
            f.write("step,n_active,clifft_dense_bytes,ttn_stored_bytes,"
                    "near_clifft_bytes,near_clifft_qubits,"
                    "near_clifft_bytes_resident,near_clifft_qubits_resident,"
                    "dense_over_ttn,dense_over_nc\n")
            for i, (s, c, na) in enumerate(zip(steps, clifft_b, nact)):
                t = ttn_b[i] if (ttn_b and ttn_b[i] is not None) else ""
                nv = nc_b[i] if nc_b else ""
                bq = nc_blk[i] if nc_blk else ""
                nvr = nc_b_res[i] if nc_b_res else ""
                bqr = nc_blk_res[i] if nc_blk_res else ""
                f.write(f"{s},{na},{c},{t},{nv},{bq},{nvr},{bqr},"
                        f"{(c/t) if t else ''},"
                        f"{(c/nc_b[i]) if nc_b and nc_b[i] else ''}\n")

    # ---- choose plotted series + y-axis by metric ----
    NAN = float("nan")  # unexecuted TTN tail (stopped early) -> gap, not a value
    nc_res_plot = None  # secondary near-Clifford line: settled step-boundary RESIDENT
    nc_res_label = None
    nc_meta_plot = None  # memory metric only: NC incl. Clifford-frame metadata (faint)
    if args.metric == "qubits":
        # active-state size = log2(dense-equivalent dimension), in qubits.
        # clifft = k active idents; TTN = log2(stored_bytes/16); near-Clifford =
        # genuine magic-block qubit count (0 for a pure stabilizer circuit).
        clifft_plot = [float(na) for na in nact]
        ttn_plot = ([NAN if b is None else (math.log2(b / 16) if b > 16 else 0.0)
                     for b in ttn_b] if ttn_b else None)
        nc_plot = [float(v) for v in nc_blk] if nc_blk is not None else None
        nc_res_plot = ([float(v) for v in nc_blk_res]
                       if nc_blk_res is not None else None)
        ylabel = ("active-state size  (qubits = log2 dense-equiv. dim)"
                  f"{', log scale' if args.yscale == 'log' else ''}")
        title = f"Per-step active-state size: {args.circuit}  (steps 0..{n_steps-1})"
        nc_label = "near-Clifford block (transient peak magic qubits)"
        nc_res_label = "near-Clifford (step-boundary resident)"
        clifft_label = "Clifft (dense active idents k)"
        ttn_label = "TTN backend (log2 stored dim)"
    elif args.metric == "active_dim":
        # active-state size = the dense-equivalent DIMENSION (number of complex
        # amplitudes), plotted on a LINEAR axis. clifft = 2^k; TTN = stored/16;
        # near-Clifford = 2^max_block (magic dimension, tableau excluded).
        clifft_amp = [2.0 ** na for na in nact]
        ttn_amp = ([NAN if b is None else b / 16.0 for b in ttn_b]
                   if ttn_b else None)
        nc_amp = [2.0 ** v for v in nc_blk] if nc_blk is not None else None
        peak_amp = max([max(clifft_amp)]
                       + ([t for t in (ttn_amp or []) if t == t] or [0])
                       + [max(nc_amp or [0])])
        uname, udiv = next((u, d) for u, d in (("G", 1e9), ("M", 1e6),
                                               ("K", 1e3), ("", 1))
                           if peak_amp >= d or u == "")
        clifft_plot = [a / udiv for a in clifft_amp]
        ttn_plot = [a / udiv for a in ttn_amp] if ttn_amp is not None else None
        nc_plot = [a / udiv for a in nc_amp] if nc_amp is not None else None
        ylabel = f"active-state size ({uname}amplitudes, linear)"
        title = f"Per-step active-state size: {args.circuit}  (steps 0..{n_steps-1})"
        nc_label = "near-Clifford block (2^magic-qubits)"
        clifft_label = "Clifft (dense active-state, 2^k)"
        ttn_label = "TTN backend (stored / 16)"
    else:
        peak_any = max(peak_clifft, peak_ttn or 0, peak_nc or 0)
        unit, div = next((u, d) for u, d in (("GiB", 2**30), ("MiB", 2**20),
                                             ("KiB", 2**10), ("B", 1))
                         if peak_any >= d or u == "B")
        clifft_plot = [b / div for b in clifft_b]
        ttn_plot = ([NAN if b is None else b / div for b in ttn_b]
                    if ttn_b else None)
        # MAIN NC line = exponential dense-state ONLY (16*2^block), apples-to-apples
        # with Clifft's 16*2^k (both count only the dense state). The faint dotted line
        # adds NC's Clifford-frame metadata -- the polynomial part Clifft's own tableau
        # mirrors but the 2^k baseline omits, shown for honesty, never in the ratio.
        nc_plot = [b / div for b in nc_state_b] if nc_state_b else None
        nc_res_plot = [b / div for b in nc_state_b_res] if nc_state_b_res else None
        nc_meta_plot = [b / div for b in nc_b] if nc_b else None
        ylabel = f"memory ({unit}{', log scale' if args.yscale == 'log' else ', linear'})"
        title = f"Per-step memory: {args.circuit}  (steps 0..{n_steps-1})"
        nc_label = "near-Clifford dense magic state (16*2^block)"
        nc_res_label = "near-Clifford (settled resident, 16*2^block)"
        clifft_label = "Clifft (dense active-state, 16*2^k)"
        ttn_label = "TTN backend (actual stored)"

    # plot
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(steps, clifft_plot, label=clifft_label, color="crimson", lw=1.4)
    if ttn_plot is not None:
        ax.plot(steps, ttn_plot, label=ttn_label, color="steelblue", lw=1.4)
    if nc_plot is not None:
        ax.plot(steps, nc_plot, label=nc_label, color="seagreen", lw=1.6)
    if nc_res_plot is not None and args.metric in ("qubits", "memory"):
        ax.plot(steps, nc_res_plot, label=nc_res_label, color="seagreen",
                lw=1.0, ls="--", alpha=0.65)
    if nc_meta_plot is not None:
        ax.plot(steps, nc_meta_plot, color="seagreen", lw=0.8, ls=":", alpha=0.5,
                label="NC + Clifford-frame metadata (Clifft's own omitted from its line)")
    ax.set_yscale(args.yscale)
    ax.set_xlabel("runtime step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.grid(True, which="both", alpha=0.3)
    if args.yscale == "linear":
        ax.set_ylim(bottom=0)
    mtag = {"qubits": "_qubits", "active_dim": "_dim"}.get(args.metric, "")
    suffix = "" if args.yscale == "log" else "_linear"
    png_path = out / f"{args.circuit}_per_step{mtag}{suffix}.png"
    fig.tight_layout()
    fig.savefig(png_path, dpi=130)

    # near-Clifford MAIN = intra-step TRANSIENT high-water mark (the memory
    # feasibility figure: a measurement core-flush briefly forms a larger block
    # before its projector collapses it). *_resident = settled step-boundary value.
    peak_nc_qubits = max(nc_blk) if nc_blk else None
    peak_nc_qubits_res = max(nc_blk_res) if nc_blk_res else None
    peak_ttn_qubits = (max(math.log2(b / 16) if b > 16 else 0 for b in ttn_real)
                       if ttn_real else None)
    summary = dict(
        circuit=args.circuit,
        steps_compared=n_steps,
        program_steps=n_prog,
        peak_clifft_bytes=peak_clifft,
        peak_ttn_bytes=peak_ttn,
        peak_nc_bytes=peak_nc,
        peak_nc_bytes_resident=peak_nc_res,
        # exponential dense-state only (16*2^block) -- apples-to-apples vs Clifft 16*2^k
        peak_nc_state_bytes=peak_nc_state,
        peak_nc_state_bytes_resident=peak_nc_state_res,
        peak_nc_overhead_bytes=peak_nc_overhead,
        peak_dense_over_ttn=peak_clifft / peak_ttn if peak_ttn else None,
        peak_dense_over_nc=peak_clifft / peak_nc_state if peak_nc_state else None,
        peak_dense_over_nc_full=peak_clifft / peak_nc if peak_nc else None,
        peak_ttn_over_nc=peak_ttn / peak_nc_state if peak_nc_state else None,
        sum_clifft_bytes=sum_clifft,
        sum_ttn_bytes=sum_ttn,
        sum_nc_bytes=sum_nc,
        sum_nc_bytes_resident=sum_nc_res,
        sum_nc_state_bytes=sum_nc_state,
        sum_nc_state_bytes_resident=sum_nc_state_res,
        sum_dense_over_ttn=sum_clifft / sum_ttn if sum_ttn else None,
        sum_dense_over_nc=sum_clifft / sum_nc_state if sum_nc_state else None,
        sum_dense_over_nc_full=sum_clifft / sum_nc if sum_nc else None,
        sum_ttn_over_nc=sum_ttn / sum_nc_state if sum_nc_state else None,
        max_active_idents=max(nact) if nact else 0,
        peak_clifft_qubits=max(nact) if nact else 0,
        peak_ttn_qubits=round(peak_ttn_qubits, 2) if peak_ttn_qubits else None,
        peak_nc_qubits=peak_nc_qubits,
        peak_nc_qubits_resident=peak_nc_qubits_res,
        **info,
    )
    (out / f"{args.circuit}_summary.json").write_text(json.dumps(summary, indent=2))

    def _x(v):
        return f"{v:.2f}x" if v else "n/a"

    print(f"=== {args.circuit}  (steps 0..{n_steps-1} of {n_prog}) ===")
    print(f"max active idents (k)     : {max(nact) if nact else 0}")
    if args.metric == "qubits":
        print(f"PEAK active-state qubits  : Clifft {summary['peak_clifft_qubits']}  "
              f"TTN {summary['peak_ttn_qubits']}  "
              f"near-Clifford {summary['peak_nc_qubits']} (transient) / "
              f"{summary['peak_nc_qubits_resident']} (resident)")
    print(f"PEAK  Clifft dense        : {human(peak_clifft)}")
    print(f"PEAK  TTN backend         : {human(peak_ttn) if peak_ttn else 'n/a'}")
    if peak_nc:
        print(f"PEAK  near-Clifford       : {human(peak_nc)} (transient)"
              + (f" / {human(peak_nc_res)} (resident)" if peak_nc_res else ""))
    print(f"PEAK  dense/TTN           : {_x(summary['peak_dense_over_ttn'])}")
    if peak_nc:
        print(f"PEAK  dense/NC            : {_x(summary['peak_dense_over_nc'])}")
        print(f"PEAK  TTN/NC              : {_x(summary['peak_ttn_over_nc'])}")
    print(f"SUM   Clifft dense (all steps): {human(sum_clifft)}")
    print(f"SUM   TTN backend  (all steps): {human(sum_ttn) if sum_ttn else 'n/a'}")
    if sum_nc:
        print(f"SUM   near-Clifford(all steps): {human(sum_nc)}")
    print(f"SUM   dense/TTN           : {_x(summary['sum_dense_over_ttn'])}")
    if sum_nc:
        print(f"SUM   dense/NC            : {_x(summary['sum_dense_over_nc'])}")
        print(f"SUM   TTN/NC              : {_x(summary['sum_ttn_over_nc'])}")
    print(f"wrote {csv_path}")
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
