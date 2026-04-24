"""Shared benchmark infrastructure.

Provides runner classes (StimRunner, ClifftRunner, TsimRunner) and a
generic timing loop that individual benchmarks can reuse.
"""

from __future__ import annotations

import os
import time
import traceback
import multiprocessing
from pathlib import Path
from typing import TypeAlias

# Pin each process to 1 OpenMP thread — parallelism is achieved by
# running multiple processes, not by multi-threading within a process.
os.environ["OMP_NUM_THREADS"] = "1"

import stim

# Type alias: circuits may be stim.Circuit objects or raw program text
# (e.g. for non-Clifford circuits that stim cannot parse).
CircuitLike: TypeAlias = stim.Circuit | str

_HERE = Path(__file__).resolve().parent
RESULTS_DIR = _HERE / "results"

_WARMUP_SHOTS = 64


# ---------------------------------------------------------------------------
# Per-simulator runners
# ---------------------------------------------------------------------------


class StimRunner:
    """Stim detector-sampler benchmark runner."""

    # Stim's Python `sample()` allocates the frame simulator with
    # batch_size == num_shots, so the shot-major x/z tables and
    # measurement record scale linearly with the requested shots and
    # spill out of L2/L3 cache at the shot counts used in this paper.
    # Chunking the call keeps the working set L2-resident; 1024 matches
    # the default picked by stim's own CLI path
    # (sample_batch_detection_events_writing_results_to_disk).
    _CHUNK_SHOTS = 1024

    def compile(self, circuit: CircuitLike, shots: int) -> None:
        if isinstance(circuit, str):
            circuit = stim.Circuit(circuit)
        self._sampler = circuit.compile_detector_sampler()

    def compile_metadata(self) -> dict[str, object]:
        return {}

    def sample(self, shots: int) -> None:
        done = 0
        while done < shots:
            n = min(self._CHUNK_SHOTS, shots - done)
            self._sampler.sample(n, separate_observables=True)
            done += n


class ClifftRunner:
    """Clifft benchmark runner."""

    def compile(self, circuit: CircuitLike, shots: int) -> None:
        import clifft

        self._prog = clifft.compile(
            circuit if isinstance(circuit, str) else str(circuit),
            hir_passes=clifft.default_hir_pass_manager(),
            bytecode_passes=clifft.default_bytecode_pass_manager(),
        )
        self._clifft = clifft

    def compile_metadata(self) -> dict[str, object]:
        k_hist = list(self._prog.active_k_history)
        return {"peak_active_k": max(k_hist) if k_hist else 0}

    def sample(self, shots: int) -> None:
        self._clifft.sample(self._prog, shots)


class TsimRunner:
    """Tsim benchmark runner."""

    def __init__(self, strategy: str = "default") -> None:
        self._strategy = strategy

    def compile(self, circuit: CircuitLike, shots: int) -> None:
        import tsim

        tc = tsim.Circuit(circuit if isinstance(circuit, str) else str(circuit))
        self._sampler = tc.compile_detector_sampler(strategy=self._strategy)
        # Warmup with the full shot count so JAX JIT-compiles all GPU
        # kernels here rather than on the first timed sample call.
        self._sampler.sample(shots, separate_observables=True)

    def compile_metadata(self) -> dict[str, object]:
        total = sum(
            csg.num_graphs
            for comp in self._sampler._program.components
            for csg in comp.compiled_scalar_graphs
        )
        return {"tsim_num_graphs": total}

    def sample(self, shots: int) -> None:
        self._sampler.sample(shots, separate_observables=True)


RUNNERS: dict[str, type] = {
    "stim": StimRunner,
    "clifft": ClifftRunner,
    "tsim": TsimRunner,
}


# ---------------------------------------------------------------------------
# Parallel sampling worker
# ---------------------------------------------------------------------------


def _worker_cpu_assignments(threads: int) -> list[int | None]:
    """Pick a distinct CPU for each worker, or None if pinning is unavailable.

    Workers migrating across cores cause L1/L2 thrash that inflates
    sample_s by a factor that depends on how busy the rest of the
    machine is.  Pinning removes that noise source.  If the process's
    affinity mask exposes fewer CPUs than requested threads (over-
    subscription), we still issue a warning and fall back to unpinned
    workers so the benchmark runs.
    """
    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        return [None] * threads
    available = sorted(os.sched_getaffinity(0))
    if len(available) < threads:
        print(
            f"  WARNING: threads={threads} exceeds available CPUs "
            f"({len(available)}); workers will NOT be pinned."
        )
        return [None] * threads
    return [available[i] for i in range(threads)]


def _sample_worker(args: tuple) -> float:
    """Compile and sample in an isolated process, return sample_s."""
    sim_name, circuit_str, shots, strategy, cpu = args

    if cpu is not None and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {cpu})

    factory = RUNNERS[sim_name]
    if factory is TsimRunner:
        runner = factory(strategy=strategy)
    else:
        runner = factory()

    runner.compile(circuit_str, shots)

    t0 = time.perf_counter()
    runner.sample(shots)
    return time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Generic benchmark loop
# ---------------------------------------------------------------------------


def run_benchmark_loop(
    *,
    circuits: list[tuple[dict[str, object], CircuitLike]],
    simulators: list[str],
    repeats: int,
    output_csv: Path,
    tsim_strategy: str | dict[str, str] = "default",
    label_key: str = "circuit",
    threads: int = 1,
) -> list[dict[str, object]]:
    """Run the compile-once / sample-many timing loop.

    Each entry in *circuits* is a ``(metadata, circuit)`` pair.  The
    metadata dict must include a ``"shots"`` key.  The full metadata
    dict is included verbatim in every output row for that circuit.

    When *threads* > 1, each repeat spawns *threads* parallel worker
    processes via ``multiprocessing`` (spawn context), each
    independently compiling and sampling *shots*.  The reported
    ``sample_s`` is the wall time of the pool call (includes
    per-worker compile overhead, which is negligible for Clifft/Stim).

    Results are written incrementally to *output_csv* so that partial
    data survives interruptions.
    """
    import pandas as pd

    results: list[dict[str, object]] = []
    total = len(circuits) * len(simulators) * repeats
    done = 0

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    def _append_row(row: dict[str, object]) -> None:
        results.append(row)
        tmp = output_csv.with_suffix(".csv.tmp")
        pd.DataFrame(results).to_csv(tmp, index=False)
        os.replace(tmp, output_csv)

    for metadata, circuit in circuits:
        label = str(metadata.get(label_key, ""))
        shots = metadata["shots"]
        circuit_str = circuit if isinstance(circuit, str) else str(circuit)

        for sim in simulators:
            factory = RUNNERS.get(sim)
            if factory is None:
                print(f"  Unknown simulator '{sim}', skipping.")
                continue

            if factory is TsimRunner:
                if isinstance(tsim_strategy, dict):
                    strat = tsim_strategy.get(label, "default")
                else:
                    strat = tsim_strategy
            else:
                strat = "default"

            # Compile in main process for metadata and compile_s.
            runner = factory(strategy=strat) if factory is TsimRunner else factory()
            header = f"{label} {sim}"
            print(f"  {header}: compiling ...", end="", flush=True)
            try:
                t0 = time.perf_counter()
                runner.compile(circuit_str, shots)
                compile_s = time.perf_counter() - t0
            except Exception as exc:  # noqa: BLE001
                tb = traceback.format_exc()
                print(f" ERROR ({type(exc).__name__}: {exc})")
                for rep in range(repeats):
                    done += 1
                    _append_row(
                        {
                            **metadata,
                            "threads": threads,
                            "simulator": sim,
                            "repeat": rep,
                            "status": "ERROR",
                            "compile_s": "",
                            "sample_s": "",
                            "effective_shots_per_s": "",
                            "error_detail": tb,
                        }
                    )
                continue

            compile_meta = runner.compile_metadata()
            meta_info = "".join(f", {k}={v}" for k, v in compile_meta.items())
            print(f" {compile_s * 1e3:.1f}ms{meta_info}")

            for rep in range(repeats):
                done += 1
                tag = f"[{done}/{total}] {header} (rep {rep + 1}/{repeats})"
                print(f"  {tag} -> ", end="", flush=True)

                row: dict[str, object] = {
                    **metadata,
                    **compile_meta,
                    "threads": threads,
                    "simulator": sim,
                    "repeat": rep,
                }

                try:
                    if threads == 1:
                        t1 = time.perf_counter()
                        runner.sample(shots)
                        sample_s = time.perf_counter() - t1
                    else:
                        cpus = _worker_cpu_assignments(threads)
                        worker_args = [
                            (sim, circuit_str, shots, strat, cpus[i])
                            for i in range(threads)
                        ]
                        ctx = multiprocessing.get_context("spawn")
                        with ctx.Pool(threads) as pool:
                            t1 = time.perf_counter()
                            pool.map(_sample_worker, worker_args)
                            sample_s = time.perf_counter() - t1

                    total_shots = threads * shots
                    eff_shots_per_s = round(total_shots / sample_s, 1)

                    row["status"] = "SUCCESS"
                    row["compile_s"] = round(compile_s, 6)
                    row["sample_s"] = round(sample_s, 6)
                    row["effective_shots_per_s"] = eff_shots_per_s
                    print(f"SUCCESS ({sample_s:.3f}s, {eff_shots_per_s:.0f} shots/s)")
                except Exception as exc:  # noqa: BLE001
                    row["status"] = "ERROR"
                    row["compile_s"] = round(compile_s, 6)
                    row["sample_s"] = ""
                    row["effective_shots_per_s"] = ""
                    row["error_detail"] = traceback.format_exc()
                    print(f"ERROR ({type(exc).__name__}: {exc})")

                _append_row(row)

    return results
