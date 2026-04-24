# Clifft Benchmark Scripts

Scripts for simulations and benchmarks from the paper introducing [Clifft](https://github.com/unitaryfoundation/clifft), a fast simulator for near-Clifford quantum circuits.

This repository houses each category of simulation in its own subflorder, with its own `pyproject.toml`, `uv.lock`, and installation flow.

## Workspaces

| Workspace | Purpose | Notes |
|-----------|---------|-------|
| `qec_bench/` | QEC and paper-style benchmarks | Contains `clifford_bench`, `cultivation_bench`, `distillation_bench`, `coherent_noise_bench`, plus shared `bench_common.py` and `tsim_compile_check.py` |
| `qv_bench/` | Quantum Volume benchmark | Separate environment for Qiskit, Qulacs, qsim, and Qrack dependencies |
| `magic_state_cultivation/` | S-proxy vs T-gate magic state cultivation fidelity comparison | Reproduces and extends Gidney et al. (arXiv:2409.17595); requires a 512-qubit Clifft source build |
