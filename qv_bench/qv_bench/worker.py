"""Isolated subprocess worker for benchmarking a single (simulator, num_qubits, seed) combination.

Spawned by the orchestrator in a fresh process to prevent GC drift.

Usage
-----
    python worker.py <simulator> <num_qubits> <seed>

where *simulator* is one of: ``clifft``, ``qiskit``, ``qulacs``, ``qsim``, ``qrack``.
"""

from __future__ import annotations

import json
import os
import resource
import sys
import time

# ---- pin every backend to a configured thread count BEFORE heavy imports --
_DEFAULT_THREADS = os.environ.get("CLIFFT_BENCH_THREADS", "1")
os.environ["OMP_NUM_THREADS"] = _DEFAULT_THREADS
os.environ["MKL_NUM_THREADS"] = _DEFAULT_THREADS
os.environ["OPENBLAS_NUM_THREADS"] = _DEFAULT_THREADS
os.environ["QRACK_DISABLE_OPENCL"] = "1"  # Qrack CPU-only; suppress native "No platforms" msg

VALID_SIMULATORS: set[str] = {"clifft", "qiskit", "qulacs", "qsim", "qrack"}


def _thread_count() -> int:
    """Return the configured backend thread count."""
    try:
        return int(os.environ.get("CLIFFT_BENCH_THREADS", _DEFAULT_THREADS))
    except ValueError:
        return 1


def _apply_mem_limit() -> None:
    """Apply ``RLIMIT_AS`` if *CLIFFT_BENCH_MEM_LIMIT_GB* is set.

    On platforms where the current hard limit is lower than the requested
    limit, clamp the soft limit instead of failing the worker before the
    benchmark starts.
    """
    limit_gb_str: str | None = os.environ.get("CLIFFT_BENCH_MEM_LIMIT_GB")
    if limit_gb_str is None:
        return
    try:
        limit_bytes: int = int(float(limit_gb_str) * (1 << 30))
    except ValueError:
        return
    try:
        _soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        if hard == resource.RLIM_INFINITY:
            soft = limit_bytes
        else:
            soft = min(limit_bytes, hard)
        resource.setrlimit(resource.RLIMIT_AS, (soft, hard))
    except (OSError, ValueError):
        # Some platforms do not support RLIMIT_AS reliably for Python worker
        # processes. In that case keep running without the cap.
        return


def _peak_mb() -> float:
    """Return peak RSS in megabytes.

    ``ru_maxrss`` is reported in kilobytes on Linux and bytes on macOS.
    """
    usage = resource.getrusage(resource.RUSAGE_SELF)
    if sys.platform == "darwin":
        return usage.ru_maxrss / (1024.0 * 1024.0)
    return usage.ru_maxrss / 1024.0


def _result_json(
    *,
    status: str,
    threads: int,
    exec_s: float = 0.0,
    compile_s: float = 0.0,
    sample_s: float = 0.0,
    peak_mb: float = 0.0,
    error: str | None = None,
) -> str:
    """Build the single-line JSON result."""
    payload: dict[str, object] = {
        "status": status,
        "threads": threads,
        "exec_s": round(exec_s, 6),
        "compile_s": round(compile_s, 6),
        "sample_s": round(sample_s, 6),
        "peak_mb": round(peak_mb, 1),
    }
    if error is not None:
        payload["error"] = error
    return json.dumps(payload)


# ---- per-simulator runners -----------------------------------------------


def _run_clifft(qasm: str) -> dict[str, float | int]:
    from qv_bench.qasm_adapter import to_clifft_stim

    import clifft  # heavy import inside branch

    clifft.set_num_threads(_thread_count())
    active_threads = clifft.get_num_threads()
    stim = to_clifft_stim(qasm)

    t0 = time.perf_counter()
    hir_pm = clifft.default_hir_pass_manager()
    byt_pm = clifft.default_bytecode_pass_manager()
    prog = clifft.compile(stim, hir_passes=hir_pm, bytecode_passes=byt_pm)
    t_compile = time.perf_counter() - t0

    t1 = time.perf_counter()
    clifft.sample(prog, shots=1)
    t_sample = time.perf_counter() - t1

    total = t_compile + t_sample
    return {
        "exec_s": total,
        "compile_s": t_compile,
        "sample_s": t_sample,
        "threads": int(active_threads),
    }


def _run_qiskit(qasm: str) -> dict[str, float]:
    from qiskit.circuit import QuantumCircuit
    from qiskit.compiler import transpile as qk_transpile
    from qiskit_aer import AerSimulator

    qkc = QuantumCircuit.from_qasm_str(qasm)
    sim = AerSimulator(method="statevector", max_parallel_threads=_thread_count())
    qkc_compiled = qk_transpile(qkc, sim)  # NOT timed

    t0 = time.perf_counter()
    sim.run(qkc_compiled, shots=1).result()
    exec_s = time.perf_counter() - t0

    return {"exec_s": exec_s, "compile_s": 0.0, "sample_s": 0.0}


def _run_qulacs(qasm: str) -> dict[str, float]:
    from qulacs import QuantumState
    from qv_bench.qasm_adapter import to_qulacs_circuit

    qc, nq = to_qulacs_circuit(qasm)
    state = QuantumState(nq)

    t0 = time.perf_counter()
    qc.update_quantum_state(state)
    exec_s = time.perf_counter() - t0

    return {"exec_s": exec_s, "compile_s": 0.0, "sample_s": 0.0}


def _run_qsim(qasm: str) -> dict[str, float]:
    import qsimcirq
    from qv_bench.qasm_adapter import to_cirq_circuit

    circuit = to_cirq_circuit(qasm)
    sim = qsimcirq.QSimSimulator(qsimcirq.QSimOptions(cpu_threads=_thread_count()))

    t0 = time.perf_counter()
    sim.run(circuit, repetitions=1)
    exec_s = time.perf_counter() - t0

    return {"exec_s": exec_s, "compile_s": 0.0, "sample_s": 0.0}


def _run_qrack(qasm: str) -> dict[str, float]:
    from qiskit.circuit import QuantumCircuit
    from qiskit.providers.qrack import QasmSimulator

    qkc = QuantumCircuit.from_qasm_str(qasm)
    sim = QasmSimulator(shots=1)
    # Skip transpile — input QASM is already in cx+u3 basis, and the
    # qrack provider's target property is incompatible with Qiskit 2.x transpile.

    t0 = time.perf_counter()
    sim.run(qkc, shots=1).result()
    exec_s = time.perf_counter() - t0

    return {"exec_s": exec_s, "compile_s": 0.0, "sample_s": 0.0}


_RUNNERS: dict[str, object] = {
    "clifft": _run_clifft,
    "qiskit": _run_qiskit,
    "qulacs": _run_qulacs,
    "qsim": _run_qsim,
    "qrack": _run_qrack,
}


def main() -> None:
    if len(sys.argv) != 4:
        print(
            f"Usage: {sys.argv[0]} <simulator> <num_qubits> <seed>",
            file=sys.stderr,
        )
        sys.exit(2)

    simulator: str = sys.argv[1]
    num_qubits: int = int(sys.argv[2])
    seed: int = int(sys.argv[3])

    if simulator not in VALID_SIMULATORS:
        print(
            f"Unknown simulator '{simulator}'. Choose from {sorted(VALID_SIMULATORS)}.",
            file=sys.stderr,
        )
        sys.exit(2)

    try:
        # -- resource limits ---------------------------------------------------
        _apply_mem_limit()

        # -- generate QV circuit (not timed) -----------------------------------
        from qv_bench.generator import generate_qv_qasm

        qasm: str = generate_qv_qasm(num_qubits, seed=seed)

        # -- run the selected simulator ----------------------------------------
        runner = _RUNNERS[simulator]
        timings: dict[str, float | int] = runner(qasm)  # type: ignore[operator]

        # -- report results ----------------------------------------------------
        print(
            _result_json(
                status="SUCCESS",
                threads=(
                    int(timings["threads"])
                    if "threads" in timings
                    else _thread_count()
                ),
                exec_s=timings["exec_s"],
                compile_s=timings.get("compile_s", 0.0),
                sample_s=timings.get("sample_s", 0.0),
                peak_mb=_peak_mb(),
            )
        )
    except Exception as exc:  # noqa: BLE001
        print(
            _result_json(
                status="ERROR",
                threads=_thread_count(),
                error=f"{type(exc).__name__}: {exc}",
            )
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
