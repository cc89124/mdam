# MDAM full-shot: Python-overhead removal (Stage A = S1+S2) + numerical-kernel C++ port (Stage B)

> 기존 MDAM full-shot의 93.1–99.9%를 차지하던 "Python control plane"은 측정 결과 단일 덩어리가
> 아니라 (1) opcode-dispatch 인터프리터 오버헤드, (2) Clifford frame + pending-ledger 켤레, (3)
> Python scalar dense numerical kernel의 셋으로 갈렸다(Phase 1, MDAM_python_runtime_semantics.md).
> 이번 작업은 사용자 결정에 따라 **full C++ batch VM이 아니라** (1)(2)를 Stage A(S1+S2), (3)을
> Stage B로 제거했다. 세 단계 모두 feature-flag default OFF이고 authoritative Python path는 보존했으며,
> RNG·measurement 알고리즘·rotation ordering·state rank·branch-pair 수학은 변경하지 않았다. 정확성
> 검증 결과는 **9개 벤치 × 5 seed에서 record bit-identical(전 단계, 단독 및 결합)**, 그리고 numerical
> 커널 단위테스트 800케이스 max|diff| = 6.3e-16이다.

기준 구성 = compiled_core=True, fused OFF, clifft_axis_bounded, taskset -c 2, threads=1, warmed,
outer public-API(`backend.sample`) median. 산출물: `artifacts/mdam_fullshot_cpp_runtime/`.

---

## 1. 무엇을 구현했나 (전부 default OFF, authoritative 보존)

### S1 — offline precompiled dispatch (Python, C++ 불필요)
`backend.py`: `NearCliffordBackend.compiled_dispatch` 플래그 + `_precompile_dispatch`(prog→정수
dispatch-id + 사전 디코드 인자/각도, 1회 캐시) + `_run_shot_compiled`(정수 dispatch 이벤트 루프).
hot loop에서 `_opname`(enum→str), `inst.as_dict()`, `if name==...` 문자열 체인, per-step
`cmath.phase`를 제거. authoritative `run_shot`은 3줄 guard 아래 그대로 유지.

### S2 — C++ frame conjugation + in-place pending (`bounded.py`, `compiled_frame.py`, `cpp/mdm_frame_kernel.cpp`)
`CliftAxisBoundedNearClifford._compiled_frame` 플래그. 핵심 관찰: incremental inverse-frame가 켜져
있으면 hot pullback은 Ax/Az를 쓰고, **측정과 측정 사이에는 Xc/Zc를 읽는 코드가 없다**. 따라서
h/s/cx의 per-gate 테이블 켤레(`_apply_clifford_to_all`)를 `_tab_gates`에 **deferral**하고, measure_z
진입 시 한 측정-구간 전체를 **단 한 번의 C++ `clifford_conj_seq` 호출**로 flush한다(pack→C++→unpack은
gate마다가 아니라 측정마다 = 899회→72회/shot). Pauli mask는 n>64(d5_r5 n=72)를 위해 W=⌈n/64⌉
uint64 워드로 저장. pending 켤레는 dict 재생성을 없애고 **in-place 변형**으로 대체. C++ 켤레 규칙은
simulator.{h,s,cx} fn-closure / lazy._conj_{h,s,cx}와 bit-identical(검증). RNG·numerical 미접촉.

### Stage B — C++ dense Pauli linear combination (`engine.py`, `compiled_numerical.py`, `cpp/mdm_lincomb_kernel.cpp`)
`CliftAxisNearClifford._compiled_numerical` 플래그. `_pauli_lincomb_inplace`의 **full-formula 분기**
(off-diagonal butterfly + non-diaghalf diagonal — ry/rx의 scalar `bit_count` hot path)를 C++로 라우팅.
Step-1 "diaghalf" global-phase fast path와 mz==0 global scalar는 Python에 남겨 state까지 bit-identical
유지. mx/mz는 magic-register 비트라 uint64로 충분. 수학(general-Pauli recurrence, rotation ordering,
state/work rank, branch-pair)은 불변.

---

## 2. 정확성 검증 (FAIL = 0)

| 검증 | 결과 |
|---|---|
| S1: record bit-identity (9 benches × 5 seeds × {default fused, baseline compiled_core}) | **ALL PASS** |
| S2 단독 & S1+S2: record bit-identity (9 × 5) | **ALL PASS** |
| Stage B: lincomb C++ vs Python 단위테스트 (800 cases, off-diag+diag, rotation+collapse 계수) | max\|diff\| **6.3e-16** |
| Stage B 단독 & S1+S2+B: record bit-identity (9 × 5) | **ALL PASS** |

seeds = {7,8,9,42,123}; d5_r5/ry는 shots=6, cult_d5 shots=10, 나머지 shots=24.

---

## 3. 단계별 full-sample wall-time (ms/shot)

outer `backend.sample` median, warmed, taskset -c 2, threads=1. (`all_stage_wall.csv`)

| benchmark | baseline | after S1 | after S1+S2 | after S1+S2+B | total speedup | Clifft | 최종/Clifft |
|---|--:|--:|--:|--:|--:|--:|--:|
| distillation | 11.878 | 9.173 | 9.064 | 9.121 | 1.30× | 0.011 | 807× 느림 |
| cultivation_d3 | 6.044 | 4.509 | 4.305 | 4.371 | 1.38× | 0.0022 | 1987× 느림 |
| cultivation_d5 | 70.984 | 60.772 | 55.676 | 50.024 | 1.42× | 0.082 | 608× 느림 |
| coherent_d3_r1 | 1.348 | 0.798 | 0.883 | 0.891 | 1.51× | 0.0013 | 685× 느림 |
| coherent_d3_r3 | 6.160 | 4.809 | 4.101 | 4.175 | 1.48× | 0.014 | 296× 느림 |
| coherent_d5_r1 | 5.680 | 3.914 | 3.340 | 3.232 | 1.76× | 0.207 | 15.6× 느림 |
| **coherent_d5_r5** | 132.51 | 120.54 | 81.28 | **81.13** | **1.63×** | 8618.9 | **106× 빠름** |
| coherent_rx_d3_r1 | 7.064 | 6.238 | 5.756 | 4.962 | 1.42× | 0.131 | 37.9× 느림 |
| **coherent_ry_d3_r3** | 468.06 | 464.98 | 450.94 | **68.29** | **6.85×** | 5.016 | 13.6× 느림 |

읽는 법:
- **S1**(dispatch)이 큰 곳: 저-rank/stabilizer-heavy(d3_r1 1.69×_S1만, d5_r1 1.45×, d3_r3 1.28×, distill 1.29×).
- **S2**(frame C++)가 큰 곳: **d5_r5** S1 120.5 → S1+S2 81.3(conjugation-bound 유일 승리 벤치), cult_d5 60.8→55.7, d5_r1 3.91→3.34.
- **Stage B**(numerical C++)가 큰 곳: **ry** S1+S2 450.9 → S1+S2+B **68.3**(6.6×), rx 5.76→4.96, cult_d5 55.7→50.0.
- **d5_r5는 Clifft 대비 66× 빠름(직전 fair-walltime, 130ms) → 106× 빠름(81ms)** 으로 개선.

---

## 4. 8-category cProfile breakdown (ms/shot): baseline → S1+S2+B

cProfile은 **상대 attribution**용(총합이 RELEASE wall보다 부풀려짐: per-call instrumentation).
헤드라인 wall은 §3. 카테고리화는 함수명 기반 근사(특히 `unknown`에는 dispatch 관련 getattr/builtin
잔여가 섞임 — `dispatch`+`unknown`을 합쳐 "인터프리터/dispatch 오버헤드"로 읽는 것이 안전).
(`category_breakdown.json`)

| category | d5_r5 base | d5_r5 S1S2B | ry base | ry S1S2B | distill base | distill S1S2B | d3_r3 base | d3_r3 S1S2B |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| dispatch | 34.4 | 17.9 | 8.5 | 4.0 | 18.5 | 11.6 | 5.8 | 2.9 |
| frame-ledger | 91.9 | 75.4 | 43.5 | 26.3 | 0.5 | 0.4 | 3.1 | 2.6 |
| **py-numerical** | 3.2 | 3.2 | **559.9** | **36.9** | 0.1 | 0.1 | 0.3 | 0.3 |
| cpp-numerical | 13.6 | 13.7 | 1.9 | 6.1 | 0.3 | 0.3 | 0.9 | 0.9 |
| stabilizer-gk | 2.5 | 2.7 | 0.4 | 0.5 | 0.01 | 0.01 | 0.2 | 0.2 |
| rng-noise | 0.6 | 0.6 | 0.06 | 0.06 | 1.6 | 1.6 | 0.06 | 0.06 |
| record-output | 0.1 | 0.1 | 0.04 | 0.04 | 0.04 | 0.03 | 0.02 | 0.02 |
| unknown(+dispatch잔여) | 58.6 | 15.2 | 24.1 | 19.5 | 4.0 | 3.4 | 2.9 | 2.3 |
| **cProfile total** | 204.9 | 128.7 | 638.5 | 93.4 | 25.1 | 17.5 | 13.3 | 9.1 |

읽는 법:
- **dispatch + unknown**(인터프리터 오버헤드)이 S1으로 d5_r5 93→33, distill 22.5→15, d3_r3 8.7→5.2로 급감.
- **frame-ledger**는 S2로 d5_r5 91.9→75.4, ry 43.5→26.3 감소(tableau 켤레는 C++로 빠져 cProfile에서
  사라짐; Python 잔여는 `_flush_tableau` pack/unpack + pending in-place `_conj_cx`).
- **py-numerical**은 Stage B로 ry **559.9→36.9**(C++로 이동; cpp-numerical 1.9→6.1). d5_r5는 측정-core가
  이미 C++(compiled_core.execute=cpp-numerical 13.6)라 py-numerical은 원래 작음.

---

## 5. 사용자 질문 5개 답변

1. **d5_r5 frame/ledger ~115ms이 S2 후 얼마?** Phase-1 추정 ~115ms(= tableau 켤레 ~73 + pending ~48,
   cProfile)에서, **S2가 tableau 켤레를 C++로** 옮겨 cProfile에서 사라졌다(Python 잔여 `_flush_tableau`
   pack/unpack ~19ms). pending `_conj_cx`는 dict-재생성 제거(in-place)로 줄었으나 Python에 남아
   **이제 frame 비용의 #1 잔여(~26ms)**다. 이 run cProfile frame-ledger: **91.9 → 75.4 ms**. **실측
   full-sample 기준 d5_r5는 S1 120.5 → S1+S2 81.3 ms(−39ms)**.
2. **ry_d3_r3 Python scalar numerical 444ms이 Stage B 후 얼마?** cProfile py-numerical **559.9 → 36.9 ms**
   (off-diagonal/non-diaghalf diagonal을 C++로; cpp-numerical 1.9→6.1). **실측 full-sample ry
   450.9 → 68.3 ms(Stage B 단독 6.6×, 전체 6.85×)**.
3. **각 단계 full-sample speedup?** §3 표. 요약: S1=1.05–1.69×(dispatch-bound), S2=d5_r5 1.48×·cult_d5/
   d5_r1, Stage B=ry 6.6×·rx·cult_d5. **전체 total: ry 6.85×, d5_r1 1.76×, d5_r5 1.63×, d3_r1 1.51×,
   d3_r3 1.48×, cult_d5 1.42×, rx 1.42×, cult_d3 1.38×, distill 1.30×**.
4. **correctness와 rotation UID/order 완전 유지?** **예.** 9 benches × 5 seeds에서 **record
   bit-identical(전 단계, 단독 및 결합)**; numerical 단위테스트 800 cases **max\|diff\| 6.3e-16**.
   S1은 동일 op/RNG draw를 동일 순서로 replay; S2/B는 rotation UID 부여·`_flush_core` 순서·branch-pair
   수학·state/work rank를 변경하지 않음.
5. **software overhead 제거 후 남는 MDAM/Clifft FLOP 차이?** Stage B는 numerical **실행**을 C++로
   옮겼을 뿐 **FLOP 수는 줄이지 않았다**. 직전 측정의 같은-rank FLOP 열세(cult_d5 ~19×, ry ~12× vs
   Clifft localized diagonal)는 그대로다. 그 결과 §3 최종/Clifft 열: MDAM은 **d5_r5만 승(106× 빠름,
   직전 66×에서 개선)**, 나머지는 여전히 느림(ry 13.6×, d5_r1 15.6×, d3_r3 296× 등). mid/low-rank가
   software-overhead 제거 후에도 지는 직접 원인 = (a) direct general-Pauli butterfly의 amplitude당
   FLOP가 Clifft localized보다 많음(같은-rank FLOP 열세) + (b) 잔여 Python(pending 켤레, 측정 제어) +
   (c) tiny-array ctypes 고정비. 이 FLOP 열세를 줄이는 measurement-core fusion이 다음 단계다.

---

## 6. 남은 것 / 다음 단계

- **S2 pending 켤레의 완전 C++화(잔여).** S2는 더 큰 frame 비용인 tableau 켤레를 C++로 옮겼고
  (d5_r5의 지배 비용), pending은 dict-재생성 제거(in-place)까지만 했다. post-S2 측정에서 pending
  `_conj_cx`가 d5_r5 frame 비용의 #1 잔여다. 완전 C++화는 per-entry segment-offset 배칭 또는 엔트리
  표현 변경이 필요하고 bit-identity 위험이 있어 별도 단계로 분리.
- **같은-rank FLOP 열세(=Stage B가 바꾸지 않은 것).** 이번 작업은 Python software overhead와 numerical
  커널의 *실행*만 C++로 옮겼다. MDAM direct general-Pauli executor의 같은-rank FLOP 열세(직전 측정:
  cult_d5 ~19×, ry ~12× vs Clifft localized diagonal)는 그대로다. 이를 줄이는 measurement-core
  fusion은 별도 다음 단계다.

> 이번 작업은 Python software overhead(S1 dispatch, S2 frame conjugation)와 numerical 커널 실행
> (Stage B)을 C++/precompile로 옮긴 것이며, current MDAM direct general-Pauli executor의 같은-rank
> FLOP 열세는 변경하지 않았다. 이 FLOP 열세를 줄이는 measurement-core fusion은 별도 다음 단계다.
