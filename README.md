
# MDAM: Measurement-Driven Active-State Materialization for near-Clifford QEC Sampling

**MDAM**은 near-Clifford 양자오류정정(QEC) 회로를 빠르게 샘플링하기 위한 시뮬레이터이다.

## Research Overview

MDAM은 near-Clifford QEC 회로를 측정 기준으로 필요한 만큼만 물질화하는 샘플러이다.

기존 near-Clifford 시뮬레이션은 회로 전체의 magic rank `k`를 기준으로 dense state 비용을 낸다. 그러나 QEC 회로에서는 gate가 누적한 전체 non-Clifford 자유도와 실제 측정이 요구하는 자유도가 다를 수 있다.

MDAM은 회로 실행을 **gate 구간**과 **measurement 구간**으로 분리한다. Gate 구간에서는 Clifford frame과 pending non-Clifford 정보를 symbolic하게 유지하고, measurement 구간에서만 Born probability 계산에 필요한 active magic state를 물질화한다.

`maxM`은 전체 실행 중 측정이 실제로 요구한 최대 active magic rank이다. 따라서 MDAM의 dense 비용은 전체 `k`가 아니라 `maxM`에 의해 결정된다.

```text
conventional cost:  2^k
MDAM cost:          2^maxM
````

반복적인 syndrome measurement 구조에서는 같은 measurement-boundary context가 여러 shot에서 재등장한다. MDAM은 이를 `BoundaryKey → BoundaryResult` transition reuse로 처리해 반복 gate-walk 비용을 줄인다.

MDAM은 회로별로 AUTH와 LEAN 중 유리한 경로를 선택한다. AUTH는 measurement-driven materialization을 직접 수행하고, LEAN은 반복되는 boundary transition을 재사용한다. 초기 probe에서 active-state 비용과 BoundaryKey reuse 효과를 실측한 뒤 실행 경로를 결정한다.

전체 알고리즘과 비용 모델은 [`mdam/mdam_full_algorithm.md`](mdam/mdam_full_algorithm.md)를 참고한다.

---

## Highlights

* Measurement-driven active-state materialization
* Exact near-Clifford QEC sampling
* Gate/measurement phase separation
* BoundaryKey-based transition reuse
* Adaptive AUTH/LEAN execution
* Bit-exact agreement with the authoritative reference path
* C++ native batch VM with optional compiled walk execution

---

## Benchmark Results

**Baseline.** 비교 기준은 near-Clifford 시뮬레이터 **clifft**이며, 기본 컴파일 설정을 그대로 사용한다
— 기본값에 squeeze 게이트-스케줄 최적화가 포함되어있다.

**Environment.** Intel Core i7-8700K (3.7 GHz), 32 GB RAM, Linux. 두 시뮬레이터 모두 단일 코어에
고정(taskset)하고 single-thread BLAS로 실행한다. 각 수치는 cold-start 배치 하나의 총 실행시간을
shot 수로 나눈 값(cold-amortized wall-clock)이며, `speedup = clifft_ns / mdam_ns`.

| Benchmark         |  k | maxM | route          |    Adaptive | AUTH-only | LEAN-only |
| ----------------- | -: | ---: | -------------- | ----------: | --------: | --------: |
| coherent_rx_d3_r1 | 14 |   10 | LEAN(compiled) |  **56.95×** |     0.79× |    56.53× |
| coherent_rx_d3_r3 | 14 |   11 | LEAN(interp)   |   **3.17×** |     0.40× |     3.30× |
| coherent_d3_r3    |  8 |    4 | LEAN(compiled) |   **3.41×** |     0.30× |     3.38× |
| distillation      |  5 |    3 | LEAN(interp)   |   **2.00×** |     0.66× |     1.95× |
| cultivation_d3    |  4 |    3 | LEAN(compiled) |   **1.59×** |     0.16× |     1.49× |
| coherent_d3_r1    |  5 |    0 | LEAN(compiled) |   **1.48×** |     0.31× |     1.46× |
| surface_d7_r7     |  0 |    0 | LEAN(compiled) |   **1.00×** |     0.43× |     1.02× |
| cultivation_d5    | 10 |    9 | LEAN(interp)   |   **0.74×** |     0.43× |     0.75× |
| coherent_d5_r1    | 13 |    0 | AUTH           |   **6.68×** |     6.85× |     1.99× |
| coherent_d7_r1    | 25 |    0 | AUTH           | **34,841×** |   35,566× |   10,318× |
| coherent_d5_r5    | 24 |   12 | AUTH           |    **815×** |      820× |      652× |

Full benchmark data and analysis outputs are in:

```text
results/benchmark_comparison/
```

---

## Result Summary

MDAM의 결과는 세 regime으로 나뉜다.

### BoundaryKey saturation regime

반복되는 measurement-boundary context가 많아 LEAN이 유리한 회로이다.

```text
coherent_rx_d3_r1
coherent_rx_d3_r3
coherent_d3_r3
distillation
cultivation_d3
```

### Localization-dominant regime

측정이 요구하는 active magic rank `maxM`이 작아 AUTH가 유리한 회로이다.

```text
coherent_d5_r1
coherent_d7_r1
coherent_d5_r5
```

### Non-saturating regime

새로운 BoundaryKey가 계속 등장해 LEAN reuse가 충분히 포화되지 않는 회로이다.

```text
cultivation_d5
```

---

## Repository Structure

```text
clifft-paper/
  mdam/
    MDAM implementation and native sampler

  qec_bench/
    QEC benchmark circuits and benchmark metadata

  results/
    Benchmark tables, adaptive execution logs, and analysis figures

  docs/
    Algorithm notes and cost-model documentation
```

---

## Build

```bash
cd mdam/native_vm
./build.sh
```

---

## Run

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
taskset -c 2 python mdam_run.py coherent_d3_r3 1000000
```

Execution flow:

```text
probe
→ adaptive AUTH/LEAN routing
→ optional LEAN walk compilation
→ production sampling
```

---

## Correctness

All reported results are checked against the authoritative reference path.

For the same seed, MDAM must produce bit-exact measurement records across execution modes.

```text
AUTH
LEAN(interp)
LEAN(compiled)
```
