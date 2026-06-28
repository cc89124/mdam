# Archived from nearclifford_backend/ (2026-06-22)

Safe-separation backup: tests / diagnostics / standalone scripts NOT reachable from the
import closure of the **bounded** (`clifft_axis/`) or **fused-va** (`virtual_axis/` impl) backends.
Shared infrastructure (backend.py, lazy.py, simulator.py, block_magic.py) and both backend
implementations (incl. fused-va deps flop_meter.py, bench_memory.py) were KEPT.

## 32 files moved

- clifft_axis/bench_state_size.py
- clifft_axis/test_memory_bound.py
- clifft_axis/verify.py
- scripts/diag_peak_block.py
- scripts/measurement_dependency_trace.py
- scripts/targeted_peel_table.py
- scripts/verify_backend.py
- scripts/verify_block.py
- scripts/verify_drop_dead.py
- scripts/verify_lazy.py
- scripts/verify_simulator.py
- scripts/verify_structure_once.py
- selector.py
- virtual_axis/benchmark_all.py
- virtual_axis/capture_block.py
- virtual_axis/capture_block_core.py
- virtual_axis/capture_resource.py
- virtual_axis/compare.py
- virtual_axis/debug_core.py
- virtual_axis/forced_capture.py
- virtual_axis/probe_cores.py
- virtual_axis/ry_verification_table.py
- virtual_axis/test_c1.py
- virtual_axis/test_c2.py
- virtual_axis/test_c4.py
- virtual_axis/test_c5.py
- virtual_axis/test_fused.py
- virtual_axis/test_fused_core.py
- virtual_axis/test_localize.py
- virtual_axis/test_reduce_exact.py
- virtual_axis/test_ry_rotation.py
- virtual_axis/test_synth.py
