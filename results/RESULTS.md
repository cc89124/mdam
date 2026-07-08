# MDAM Results

near-Clifford QEC 회로 11종에 대한 MDAM 샘플러의 성능·정확성 요약. 상세 데이터와 그림은
[`benchmark_comparison/`](benchmark_comparison/)에 있다.

## 측정 조건

- **Baseline**: clifft, 기본 컴파일 설정(squeeze 게이트-스케줄 최적화 포함). 비교 기준으로만 사용.
- **Environment**: Intel Core i7-8700K (3.7 GHz), 32 GB RAM, Linux, 단일 코어 고정(taskset),
  single-thread BLAS.
- **지표**: cold-start 배치 하나의 총 실행시간 / shot 수 (cold-amortized wall-clock),
  `speedup = clifft_ns / mdam_ns`.
- **정확성**: 표의 모든 값은 authoritative 기준 경로와 shot별 측정 기록이 **bit-exact**함을
  확인한 실행에서 나왔다 (AUTH / LEAN(interp) / LEAN(compiled) 모든 경로 동일 record).

## 결과

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

## 해석

**두 성능 축이 서로 다른 회로군을 이긴다.**

- **Localization** (`maxM ≪ k`): 측정이 요구하는 active magic rank가 작아, dense 비용이 `2^k`가
  아니라 `2^maxM`으로 떨어진다. `coherent_d7_r1`(k=25, maxM=0)의 34,841×, `coherent_d5_r5`
  (k=24, maxM=12)의 815×가 이 축의 결과이며, adaptive는 이 회로들을 AUTH로 라우팅한다.
- **BoundaryKey 재사용** (포화 회로): 반복 syndrome 측정에서 같은 measurement-boundary context가
  재등장하므로, boundary transition 캐시가 gate-walk 비용을 대체한다. off-axis rotation 회로
  (`coherent_rx_d3_r1` 56.95×)부터 magic 회로(distillation 2.00×, cultivation_d3 1.59×)까지가
  이 축의 결과이며, adaptive는 LEAN으로 라우팅하고 반복량이 충분하면 walk를 회로 전용
  바이너리로 컴파일한다(LEAN(compiled)).

**Ablation 열이 각 구성요소의 기여를 분리해 보여준다.**

- `AUTH-only` (캐시 층 없음): LEAN 회로군에서 0.16×–0.79×로 떨어진다 — BoundaryKey 캐시의 기여.
- `LEAN-only` (AUTH 라우팅 없음): localization 회로군에서 1.99×/10,318×/652×로 떨어진다 —
  AUTH 옵션의 기여. LEAN 회로군에서는 Adaptive ≈ LEAN-only로, 라우팅 자체의 오버헤드는 측정
  오차 수준이다.

**메모리.** LEAN의 캐시는 BoundaryKey→transition 자동자 테이블뿐이며(노드당 수십 바이트),
전 벤치 실측 peak는 63 MB 이하다. AUTH는 상수 메모리로 동작한다.

**한계 regime.** `cultivation_d5`(0.74×)는 새로운 BoundaryKey가 계속 등장하는 비포화 회로로,
localization도 재사용도 충분히 성립하지 않는 유일한 손실 사례다.
