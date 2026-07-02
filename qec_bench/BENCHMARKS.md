# QEC Benchmark Suite — 회로 상세 설명

이 문서는 `qec_bench/circuits/`의 각 벤치마크 회로가 **무엇이고, 어떤 구조이며, 왜 그 회로인지**를 매우 자세히 설명한다. 이 스위트의 목적은 **near-Clifford 시뮬레이션(MDAM)을 서로 다른 물리 regime에서 시험**하는 것이다.

> 문서는 회로 설명 전용이며 **어떤 벤치마크 정의도 바꾸지 않는다** (신규 md만 추가). 성능/알고리즘 논의는 [`../mdam/MDAM_auth_vs_lean.md`](../mdam/MDAM_auth_vs_lean.md), [`../results/benchmark_comparison/wall_table.tsv`](../results/benchmark_comparison/wall_table.tsv) 참조.

---

## 0. 한눈에 보기

핵심 축 두 개:
- **`k` (peak_rank)** = 회로가 요구하는 **명목 magic 차원** = 정확 시뮬레이터(Clifft)가 강제로 들고 가는 dense 크기 `2^k`.
- **`maxM`** = MDAM이 측정 투영 localization 이후 **실제로 물질화하는 magic 차원** (`2^maxM`). `maxM ≪ k`면 강한 localization.

| bench | family | qubits | k | maxM | nmeas | magic 원천 | regime |
|---|---|--:|--:|--:|--:|---|---|
| surface_d7_r7 | Clifford | 118 | 0 | 0 | 385 | 없음 (depolarizing만) | 순수 Clifford |
| coherent_d3_r1 | 회전노이즈 | 26 | 5 | 0 | 17 | R_Z(0.02) | 소규모 localization |
| coherent_d3_r3 | 회전노이즈 | 26 | 8 | 4 | 33 | R_Z(0.02) | 소규모 control-plane |
| coherent_d5_r1 | 회전노이즈 | 64 | 13 | 0 | 49 | R_Z(0.02) | **r≪k localization** |
| coherent_d5_r5 | 회전노이즈 | 64 | 24 | 12 | 145 | R_Z(0.02) | r<k localization |
| coherent_d7_r1 | 회전노이즈 | 118 | 25 | ~0 | 97 | R_Z(0.02) | **r≪k localization** |
| coherent_rx_d3_r1 | 회전노이즈 | 26 | 14 | 10 | 17 | R_X(0.02) | off-axis (약한 localization) |
| coherent_rx_d3_r3 | 회전노이즈 | 26 | 14 | 11 | 33 | R_X(0.02) | off-axis |
| coherent_ry_d3_r1 | 회전노이즈 | 26 | 16 | — | 17 | R_Y(0.02) | off-axis (최대 magic) |
| cultivation_d3 | MSC | 15 | 4 | 3 | 21 | T / T_DAG | magic-saturated |
| cultivation_d5 | MSC | 42 | 10 | 9 | 112 | T / T_DAG | magic-saturated (비포화 캐시) |
| distillation | color-code distill | 85 | 5 | 3 | 85 | T_DAG ×5 | magic, 구조적(포화) |

(`ry`, `rx_d5_r*` 등 일부 변형은 wall_table에 없어 maxM 미기재.)

---

## 1. `surface_d7_r7` — 순수 Clifford 기준선

**정체:** stim의 `surface_code:rotated_memory_z` — **rotated surface code로 논리 1큐빗을 거리 d=7, 라운드 r=7 동안 보존(memory)** 하는 표준 QEC 메모리 실험. 자기공명(magic) 게이트가 **하나도 없다**.

**구조:**
- 118 물리큐빗 (거리-7 rotated surface code: data + ancilla).
- 매 라운드: data 큐빗 준비 → syndrome ancilla와 CX 얽힘 → ancilla 측정(X/Z stabilizer). 7라운드 반복 후 data 논리 측정.
- 노이즈: `DEPOLARIZE1/2`, `X_ERROR`, measure-flip — 전부 **확률적 Pauli 노이즈(Clifford)**.

**왜 이 회로인가:** magic이 전혀 없는(`k=0`) **degenerate 기준선**. Clifft·MDAM 둘 다 dense를 안 쓰고 순수 tableau/frame 부기만 함 → 순수 **dispatch/제어평면 오버헤드**를 재는 대조군. 여기서 MDAM이 Clifft에 못 이기면(0.43×) 그 격차는 100% 제어평면 차이지 수학 차이가 아니다.

**생성:** `_clifford_circuit(distance=7, rounds=7, phys_error_rate=1e-3)` (`run_all.py`).

---

## 2. `coherent_*` — rotated surface code + **코히런트 회전 노이즈**

**정체:** `surface_d7_r7`와 **동일한 rotated surface code 메모리**인데, 확률적 depolarizing 노이즈를 **결정론적 작은 회전(coherent error)** 으로 바꾼 것. 즉 "**논리 1큐빗을 surface code로 인코딩하고, 매 라운드 물리큐빗마다 non-Clifford 회전을 노이즈로 넣은**" 회로.

**생성 (`_coherent_noise_circuit`):**
1. `surface_code:rotated_memory_z` (거리 d, 라운드 r) 생성.
2. 모든 `DEPOLARIZE1/2` 줄을 → **`R_Z(0.02) <같은 타깃>`** 으로 치환. (변형: `rx`=`R_X(0.02)`, `ry`=`R_Y(0.02)`.)

**파라미터:**
- **`d`** = surface code 거리 (3/5/7) — 물리큐빗 수와 syndrome 구조 결정.
- **`r`** = QEC 라운드 수 — **회전 레이어가 r번 누적** → 유효 magic(k) 증가. (예: d3에서 r1→k5, r3→k8.)

**핵심 물리 — 왜 localization이 되나:** 작은 회전이 magic을 조금씩 만들지만, **매 라운드 syndrome 측정(=투영)이 그 magic을 다시 붕괴**시킨다. outcome이 정해지면 회전이 결정론적이 되어 `maxM`이 작아짐. 그래서 `d5_r1`, `d7_r1`은 **maxM=0** (순수 Clifford로 붕괴) → **`r≪k` localization**. 이 regime에서 MDAM authoritative가 압도적으로 빠르다 (Clifft는 2^k dense를 강제로 들지만 MDAM은 2^0로 접음 → d7_r1에서 ~3만 배).

### 2.1 On-axis (`R_Z`) — 강한 localization
`R_Z`는 Z-basis 측정과 **교환**하므로 magic이 syndrome에 잘 흡수됨 → maxM 매우 작음. `coherent_d{3,5,7}_r*`.

### 2.2 Off-axis (`R_X`, `R_Y`) — 약한 localization
`R_X`/`R_Y`는 Z-basis 측정과 **교환하지 않아** 회전이 잘 안 붕괴됨 → 같은 `d,r`에서 **훨씬 큰 k**: `rz d3_r1 k=5` vs `rx d3_r1 k=14` vs `ry d3_r1 k=16`. maxM도 큼(rx maxM≈10-11). off-axis는 MDAM localization이 약해 이득이 작은 스트레스 케이스.

---

## 3. `cultivation_*` — **Magic State Cultivation (MSC)**

**정체:** **Magic State Cultivation** (Gidney, Shutty, Jones, *"Magic state cultivation: growing T states as cheap as CNOT gates"*, 2024). distillation의 대안으로, **작은 코드에 T를 주입 → cultivation 체크로 검증/정제 → 거리를 키워(escape) 고품질 T 상태를 배양**하는 프로토콜.

**구조 (회로에서 확인됨):**
- 작은 코드 준비 (`RX`/`R` init + CX 사다리 인코딩).
- **`T` / `T_DAG` magic 주입** (cult_d3: 각 T·T_DAG 몇 개, k=4).
- **`MPP` (multi-Pauli product 측정) + `MX` + DETECTOR** = cultivation 체크(모핑 측정)로 magic 품질 검증.
- **거리 성장:** d3(15 qubit, k=4) → d5(42 qubit, k=10)로 코드가 커지며 magic 차원도 증가.

**파라미터:** `d` = cultivation 목표 거리 (3, 5). 큰 d일수록 더 많은 magic·측정.

**핵심 물리 — 왜 magic-saturated인가:** coherent와 달리 **magic이 실재하며(주입된 T) 측정으로 잘 안 붕괴됨** → `maxM ≈ k` (cult_d3 maxM=3/k=4, cult_d5 maxM=9/k=10). localization이 거의 없어 MDAM도 거의 full dense를 들어야 함. **cult_d3**는 magic 상태공간이 작아(automaton ~314 phase-canon 노드) **캐시가 포화** → lean 이득. **cult_d5**는 상태공간이 조합적으로 커서(노드 선형 증가) **포화 실패** → 캐시도 localization도 안 통하는 유일한 "양쪽 다 지는" 케이스.

**생성:** `_cultivation_circuit(distance, phys_error_rate)` = `cultivation_d{d}.stim` 템플릿 읽어 노이즈만 치환.

> 주: 이 벤치 변형은 `has_postselection=False`(모든 결과를 측정·샘플). 원 MSC는 체크 실패 shot을 post-select로 버리지만, 성능 벤치에서는 전 trajectory를 돌린다.

---

## 4. `distillation` — **[[17,1,5]] color code magic state distillation**

**정체:** **논리 magic state distillation** (Rodriguez et al., *"Experimental demonstration of logical magic state distillation"*, *Nature* 2025). **[[17,1,5]] 2D color code** 5개 패치에 노이즈 magic을 준비하고, 색코드의 **transversal 구조**로 **5→1 정제**하는 실제 실험 회로.

**구조 (회로에서 확인됨):**
- 85 물리큐빗 = **5 블록 × 17 qubit** ([[17,1,5]] color code 5패치).
- 각 블록에 **`R_X(-0.304)` + `T_DAG` magic 주입** (블록당 1개, 총 5개 → **k=5**).
- `SQRT_Y`, `SQRT_X`, `CZ` 로 color code 인코딩 + transversal 정제 Clifford.
- DETECTOR 40개 + OBSERVABLE 로 정제 성공 판정, `M` 로 논리 측정.

**핵심 물리 — 왜 구조적(포화)인가:** magic이 **대각(diagonal, T_DAG)** 이고 주입 패턴·color code stabilizer 측정이 **매우 규칙적** → 방문하는 magic-core 상태가 **소수의 구조화된 집합**(automaton 29 노드)으로 한정 → **캐시가 즉시 포화** → **lean 최고 승리(1.99×, fb 0%)**. 단 k=5가 작아 Clifft의 2^5 dense도 싸서 auth는 오히려 짐(near-Clifford 부기 오버헤드 > 2^5 dense).

**생성:** `_distillation_circuit(prep_noise)` = 고정 `distillation.stim` 템플릿(tsim `ColorEncoder5`로 사전 생성) 읽어 노이즈만 치환. **크기 파라미터 없음** (단일 인스턴스).

---

## 5. 종합: `k` vs `maxM`, 그리고 regime이 MDAM 경로를 정한다

`k`(Clifft가 강제로 드는 dense)와 `maxM`(MDAM이 실제로 드는 dense)의 관계가 성능 regime을 결정한다:

| 관계 | 예 | 의미 | MDAM 최적 경로 |
|---|---|---|---|
| `maxM ≪ k` | coherent d5_r1/d7_r1 (maxM 0, k 13/25) | 측정 투영이 magic 붕괴 → dense 작음 | **auth** (localization, 최대 승) |
| `maxM ≈ k` + 상태공간 유한 | cult_d3, distillation | dense 크지만 구조적 반복 | **lean** (캐시 포화) |
| `maxM ≈ k` + 상태공간 폭발 | cult_d5 | dense 크고 다양 | 양쪽 다 짐 (lean이 덜 나쁨) |
| off-axis, 중간 | coherent_rx/ry | 회전이 안 붕괴 → k 큼, maxM 중간 | 경우별 |

**요약:**
- **coherent** = "노이즈가 magic을 만들지만 QEC가 도로 지운다" → **localization** regime → auth.
- **cultivation(MSC) / distillation** = "진짜 magic을 넣는다" → **saturation** regime → 구조적이면(distillation, cult_d3) lean, 다양하면(cult_d5) 양쪽 실패.
- **surface** = magic 없음 → 순수 제어평면 대조군.

자세한 auth/lean 원리·측정치는 [`../mdam/MDAM_auth_vs_lean.md`](../mdam/MDAM_auth_vs_lean.md) 참조.
