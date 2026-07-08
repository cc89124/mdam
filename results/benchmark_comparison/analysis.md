# MDAM vs Clifft — 분석 통합본

wall_table.csv(최종 표)를 뒷받침하는 분석·진단의 핵심만 모은 문서. 개별 과정 리포트 11개를 이 문서로
통합하고 폐기함 (2026-07-07). 측정 규약: taskset -c 2, single-thread, cold-amortized total_wall/N.

---

## 1. 지는 회로의 비용 구조: 병목은 산술이 아니라 제어

AUTH 경로(run()) 해부 (rdtsc, N=4000): **dense FLOP 커널은 전체의 4–16%뿐**, 나머지는 심볼릭
제어+기계(pullback, oracle, plan, dispatch).

| bench | auth ns/shot | dense kernel 비중 | 제어+기계 |
|---|---:|---:|---:|
| cultivation_d3 (k=4) | 26,715 | 4.3% | 95.7% |
| coherent_d3_r3 (k=8) | 69,054 | 7.7% | 92.3% |
| cultivation_d5 (k=10) | 344,799 | 16.0% | 84.0% |

warm 해석 walk의 분해(fblock OFF): frame 21–45% + residual(dispatch/DORM/feedback) 48–62%
= **76–90%가 codegen으로 공격 가능**; 제거 불가능한 바닥 = noise-RNG + Born draw + hash probe + dorm coin.

oracle 내부(세부): d3_r3는 rebuild_inverse_frame 1.8%+subst 3.5% 수준(작음), cult_d5는 flush_core
18.3%가 최대 항목 — cult_d5의 손실은 제어가 아니라 dense 자체가 큰 비포화 회로라는 뜻.

**결론(비용 모델 2×2)**: 승리 조건은 (i) 2^k 절대 dense work가 커서 r≪k localization이 먹히거나
(AUTH 승리: d7_r1, d5_r5, d5_r1), (ii) boundary 통계가 포화해 walk가 성립하거나 (LEAN 승리: 나머지).
둘 다 아닌 코너가 손실 regime (현재 cultivation_d5 하나).

## 2. coherent_d3_r3: "구조적 패배"는 두 개의 구현 버그였다

라벨 0.42× LOSS의 실체 (수리 후 3.16× WIN):

1. **run_lean이 MO_ARRAY_U2 opcode를 미지원** → 모든 shot이 walk를 완주 못 함(fb=100%) → "비포화"로
   오판. 규명 사슬: bcap key 실측은 84–94% 재사용(포화) ↔ walk fb=100% 모순 → same-seed 재현
   2000/2000 실패(=버그) → opcode 히스토그램에서 U2×8 발견. 수리: sg_u2_sign(U2의 frame 분기를
   회전 부호처럼 edge key에 fold, build/walk 대칭) + walk에서 frame-only U2 실행. 25 seeds bit-exact.
2. **fallback-restore pool(mc_pool)이 메모리 예산을 오염** → 잘못된 강등. v2 판정식의
   pool-first eviction으로 자동 해결 (아래 wall_table.tsv 헤더 참조).

같은 pool 문제가 coherent_rx_d3_r3에도 있었음 (0.42× → v2 자동 3.36×).

## 3. 포화 진단 (saturation_curve.png / lean_warm_long.png)

방법: bcap으로 가장 세밀한 boundary key의 (a) marginal miss m(N) vs 문턱 m\*=1/B, (b) distinct key
성장 D(N)~N^β, (c) 투영 E[T]=H·walk+(1−H)·fallback. **β가 곧 판별량**: d3_r3 β=0.52(포화, E[T]가
baseline 관통), cult_d5 β=0.80(비포화, m이 m\*의 4배 위에서 정체, 깊은 경계 mp7–11 ~44% 신규율).
이 β와 비용 앵커가 v2 adaptive 판정식(적분 기준)의 입력이 됨 — mdam_full_algorithm_math.md §13.

**sign-parametric cache 여지(측정만 해둠, 미구현)**: 부호를 뺀 구조 key K_struct의 재사용률은
d3_r3 86%/cult_d3 99%/**cult_d5 85%**(K_full은 46%)이고 구조-결정론 위반 0 — 구조 플랜은 shot-불변,
부호 의존은 p0 수치뿐. cult_d5의 비포화가 "부호 엔트로피" 때문이라면 이 방향이 남은 카드.

## 4. codegen(walk 컴파일) 계층의 정직한 회계

- **perfect-warm 상한** (자동자 완성 가정, dispatch 제거만): fragmented-op 회로에서 해석 대비
  cult_d3 1.49× / d3_r1 1.49× / rx_d3_r1 1.39×; distillation 0.92× **패배** (긴 pure-frame run은
  해석기의 fblock superinstruction이 unrolled 본문의 I-cache 압박을 이김) → race가 필요한 이유.
- **cold 회계 (break-even N\*)**: 고정비 = g++ ~2–6s + 자동자 warm. 첫 실행 기준(N\*): rx_d3_r1
  **27k** / distillation 532k / cult_d3 3.2M / d3_r1 6.2M. .so 캐시 재실행 기준: 13k / 41k / 476k /
  639k. **결론: 컴파일은 고정 회로의 대량 shot 재실행 자산이며, 저랭크 회로의 첫 소량 실행에서는
  게이트가 거절하는 것이 옳다** (gate의 COMPILE_EST=7s 보수치가 이를 보장; 3s 과소치는 marginal
  engage로 순손실 1.5s를 냈던 실측 있음).
- **tier 통합 검증**: 빈 캐시에서 게이트는 전부 거절하고 오버헤드 0.98–1.01×(≈공짜); 캐시 히트+엔게이지
  시 판정 포함 순이득 (최종 수치는 wall_table.csv의 mdam vs nocompile 열). race의 후회 비용 ≈ 컴파일
  1회 + chunk 1개 (distillation 실측 ~0.4%).

## 5. 커버리지 갭 (pre-existing)

`reduce_full would fire`: oracle 측정이 full basis reduction을 요구하는 희귀 shot(~1/2M,
cultivation_d3)에서 native 엔진이 hard-error. **모든 native 경로(authoritative 포함)에서 동일 발화**
— codegen/tier와 무관한 엔진 커버리지 한계. cult_d3의 10M 단일 스트림을 막음 (표는 N=1e6 + seed
재시도 공개로 측정). 엔진이 이를 커버하기 전까지 10M급 cult_d3 데모 불가.

## 6. State-pool 제거 (2026-07-07, 원리 결정 + ablation)

**결정: MDAM의 유일한 캐시는 BoundaryKey→transition이다.** dense state 스냅샷 풀(mc_pool,
fallback-restore 가속용)은 "state는 측정 시점에만 물질화한다"는 MDAM 원리에 위배 → 알고리즘에서
제거(컴파일 기본값 `mc_pool_off=1`; `nvm_mc_pool_off(vm,0)`은 A/B 전용).

**Ablation (11벤치 × pool ON/OFF, v2 adaptive cold 단일호출, bit-exact 22/22, pool_ablate.tsv):**
pool OFF는 9/11 벤치에서 승리 또는 동률 — d3_r3 2.76×→2.99×, cult_d5 0.69×→0.71×(evict@20415 소멸,
526MB→0), cult_d3/d3_r1 소폭 개선, AUTH 3종 동률. 유일한 비용 = off-axis rx 두 개(rx_d3_r1 −12%,
rx_d3_r3 −6%; miss 시 dense core 재구축이 비싼 회로 — pool이 실제로 일하던 유일한 regime).
**메모리: pool peak 296–533MB 전부 소멸, 남는 캐시는 자동자 테이블뿐(최대 63MB, cult_d5) →
메모리 버짓 이벤트가 전 벤치에서 사라짐**(v2 사다리는 freeze-vs-AUTH만 남음; cap은 알고리즘
요소가 아니라 선택적 리소스 가드). 추가 발견: **lean 강제 d5_r5의 OOM은 전적으로 pool이 원인**
(pool-free에서 711×로 완주, 30s 캡), cult_d5 lean도 0.72×→0.80× 개선(pool 삽입 비용 제거).
tier(전체 알고리즘) 재측정: d3_r3 3.16×→3.41×, cult_d3 1.55×→1.59×, cult_d5 0.72×→0.74×,
rx_d3_r1 64.08×→56.95×, rx_d3_r3 3.30×→3.17×.

## 7. 계보

wall 표의 확정판은 wall_table.csv(+tsv 문서 주석), 판정식 유도는 mdam_full_algorithm_math.md §11–13.
이 문서로 통합·폐기된 리포트: wall_table.md(Gate-K 구판), d3r3_lean_u2_fix.md, codegen_cold.md,
codegen_lean.md, codegen_exec_wall{,2,3}.md, ctrl_profile.md, auth_prof_losers.md, oracle_dissect.md,
structkey_reuse.md.
