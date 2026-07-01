# MDAM 실행 경로: `authoritative` vs `lean` — 원리 · 방법 · 선택 기준

이 문서는 MDAM 샘플러의 **두 가지 실행 경로**를 매우 자세히 설명한다.

- **`auth` (authoritative)** — 캐시 없는 정공법. 매 shot을 처음부터 정확히 계산.
- **`lean` (lean-walk, path-3)** — magic-core automaton으로 엔진 gate-walk를 건너뛰는 캐시 경로. miss 시 best-stack으로 fallback.

두 경로는 **비트 단위로 동일한 결과(bit-exact)** 를 낸다. 차이는 오직 **속도**이며, 어느 쪽이 빠른지는 회로의 구조에 따라 갈린다. 이 문서는 그 원리와 판단 기준을 정리한다.

> **핵심 한 줄:** `best_path = min(auth_ns, lean_ns)`. `fb(fallback_pct)`는 판단 기준이 아니라 **왜 그런지 설명하는 진단 지표**다. 물리적 예측: **localization 되는 회로(r≪k) → auth**, **magic core가 작고 반복적(automaton 포화) → lean**.

---

## 0. 사전 지식: MDAM near-Clifford 상태 표현

MDAM은 상태를 다음 형태로 유지한다:

```
|ψ⟩ = U_C |χ⟩
```

- **`U_C` = Clifford frame** — 세 겹으로 구성:
  - 외부 `NativeFrame` (qubit별 x/z parity 비트)
  - `inverse_frame` (O(weight) pullback)
  - tableau `Xc/Zc` 생성자(generator)
- **`|χ⟩` = magic core** — `2^r` 크기의 dense 상태 벡터 + **deferred pending rotations**(아직 적용 안 한 회전).

여기서 `k`는 회로가 건드리는 논리 qubit 폭 수준의 상한, `r`은 **실제로 물질화(materialize)된 magic 차원**이다. **`r ≤ k`이며, `r ≪ k`인 회로가 "localization이 잘 되는 회로"** 다. dense는 `2^r`이므로 `r`이 작으면 상태가 통째로 작다.

### 게이트가 무엇을 건드리는가 (코드 검증됨)

측정 경계(measurement boundary) **사이**에서:

| 연산 | 실제로 건드리는 것 | 비용 |
|---|---|---|
| `engine.cx/cz/s/h` | `tableau.fwd_cx` + `inverse_frame.fwd_cx` **뿐** (모든 n개 생성자 순회 = 게이트당 2n pconj) | 순수 F2 심볼릭 부기 |
| `sampler.apply_site` (noise) | **외부 NativeFrame** x/z parity만 | 부기 + RNG |
| `apply_mask` (feedback, `MO_APPLY_PAULI`) | **외부 NativeFrame만** | 부기 |
| dense `2^r` | **경계 사이엔 안 건드림** — `sid`로 carry, 측정에서만 물질화 | — |

⇒ 경계 사이 opcode 루프(전체의 ~85%)는 **100% 심볼릭 F2 부기**다. 여기엔 dense 연산도 RNG도 없다(엔진 층은 RNG-free로 검증됨). RNG는 오직 **noise 샘플링**과 **측정 Born 추첨**에서만 발생한다.

이 사실이 lean의 존재 근거다: **경계 사이 gate-walk는 결정론적 심볼릭 변환이므로, 그 누적 효과를 하나의 transition으로 접을 수 있다.**

---

## 1. Path A — `authoritative` (정공법)

### 1.1 방법

`run()` / `nvm_mdam_sample_batch`. 캐시 없이 매 shot을:

1. opcode를 순서대로 실행 (engine gate-walk = tableau + inverse-frame conjugation)
2. noise 사이트마다 `should_fire` → `apply_site`
3. 측정 경계마다 dense `2^r` 물질화 → Born 확률 계산 → 추첨 → 붕괴
4. feedback mask 적용

**매 shot이 독립적, 완전 정확, 상수 시간.** 이것이 **정답 기준(ground truth)** 이며 lean의 bit-exact 검증 대상이다.

### 1.2 비용 모델

```
auth_ns ≈ (게이트 수) × 2n(F2 부기) + (측정 수) × 2^r(dense 물질화) + noise RNG
```

- **`r ≪ k`이면 `2^r`이 작아 dense 물질화가 거의 공짜** → auth가 **매우 빠름**.
- **`r ≈ k`(magic 포화)이면 `2^r`이 커서** 측정마다 무거운 dense → auth가 **느림**.

### 1.3 auth가 유리한 경우

**localization이 잘 되는 회로 (r ≪ k):**

- dense core가 작아 매-shot 계산이 이미 싸다.
- 동시에 **그 큰 k 때문에 방문하는 상태공간이 조합적으로 폭발** → 캐시(lean)가 절대 포화하지 못함.
- 예: `coherent_d5_r1` (auth 6.85×), `coherent_d7_r1` (auth 35566.63×), `coherent_d5_r5` (auth 819.68×).

이 회로들에서 auth는 lean이 절대 못 따라오는 **압도적 우위**를 가진다. lean은 여기서 매 shot 캐시를 miss하고 fallback으로 되돌아가므로 순수 손해다.

---

## 2. Path B — `lean` (lean-walk, path-3)

### 2.1 원리: 경계 사이를 하나의 automaton transition으로 접기

측정 경계 사이의 gate-walk가 순수 심볼릭 F2 변환이라면, 경계에서 경계로 넘어가는 것은 **결정론적 상태 기계(Mealy machine)** 로 표현할 수 있다. 두 개의 전제를 증명했다.

#### PREMISE-1 (separability, 증명 완료)

경계 signature 수열은 결정론적 automaton이다:

```
key(mp+1) == f( mp, key(mp), Born outcome, in-segment rotation-sign bits )
```

- **signs OFF** (Born outcome만): cult_d3 10489 viol, cult_d5 34957 viol → **coupled** (부족).
- **signs ON** (+ rot()이 읽는 `frame.xb` sign 비트, order-hash): cult_d3 **0 viol** (3895 edges / 476105 checks = 99.2% 재사용), cult_d5 **0 viol** (114351 edges), distillation 0/0. 전부 bit-exact.
- **noise가 automaton에 결합하는 유일한 통로는 `rot()`이 회전 부호를 위해 읽는 `frame.xb(slot)` 비트뿐**이다. noise/feedback이 그 비트를 뒤집는다. lean은 segment noise를 그리는 동안 이 부호 비트를 **공짜로** 얻는다.

#### PREMISE-2 (automaton 완전성, 증명 완료)

automaton은 **완전한 결정론적 확률 Mealy machine**이다 (seeds 1/7/42/123 × 20k/60k/140k 전 행렬, 전부 bit-exact + 0 viol):

- `node → p0` (Born threshold) 결정론적 (cult_d3 2.39M checks 0 viol)
- `node → antis` (anti-commuting = 50/50 여부) 결정론적
- `(node, outcome, signs) → next` 결정론적

⇒ lean은 **엔진 measure_z 없이** cached `p0` + transition으로 Born outcome을 추첨할 수 있다.

### 2.2 방법: `run_lean` / `nvm_run_lean_fb_batch`

lean walk는 **프레임 층 + automaton**만 돌린다:

- **프레임 층 유지:** `frame.cnot/cz/h/s` + noise + feedback + dormant + rotation-sign 누적 (`sg_sign_acc`).
- **automaton으로 경계 처리:** cached `p0` → Born 추첨, `node→antis` → `idraw2()`(50/50), `edge→next-node`.
- **건너뛰는 것 (전체 opcode 루프의 ~85%):** engine gate-walk 전체 = tableau / inverse-frame / pending / dense / measure_z.

RNG가 정확히 유지되는 이유: 엔진 층이 RNG-free이고 (검증됨), 경계당 정확히 1회 추첨하기 때문. 그래서 lean은 authoritative와 **동일한 난수 소비 순서**를 갖는다.

#### miss 시 fallback

lean이 **캐시에 없는 edge를 만나면(`ln_incomplete`)** 그 shot을 버리고 **best-stack(`run_mcache` mode-3)** 으로 다시 돌린다. fallback이 정확성을 보장하므로 lean은 **항상 bit-exact**다. `ln_fb_count`로 fallback 횟수를 센다.

> **중요:** fallback 대상은 auth가 **아니라 best-stack**이다. 이 때문에 fb가 100%여도 lean이 auth보다 나을 수 있다(best-stack이 auth보다 빠른 회로에서). 그래서 판단 기준은 fb가 아니라 실측 `min(auth_ns, lean_ns)`이다.

### 2.3 적용한 최적화 (전부 bit-exact 유지)

cult_d3를 best-stack 3.14× → lean ~0.98×까지 끌어내린 4+1개 최적화:

1. **lean walk** — automaton으로 engine gate-walk 스킵 (핵심).
2. **fblock in `run_lean`** — `MO_FRAME_*` + `MO_ARRAY_{CNOT,CZ,S,H}`(lean에선 순수 프레임 연산)를 슈퍼인스트럭션으로 배칭.
3. **noise `should_fire()` inline-guard** — 발화 안 하는 사이트는 `apply_site` 호출 자체를 스킵. **1763→533ns, 단일 최대 레버.**
4. **int-node-id automaton** — 경계당 hash 1 + array read 2 (기존 `unordered_map.find` 3회 대신). 430→231ns.
5. **lazy per-case operand loads** — dispatch에서 필요한 피연산자만 로드.

### 2.4 lean이 유리한 경우: **포화(saturation)**

lean은 **automaton 테이블이 더 이상 안 커질 때(포화)** 이긴다. 즉 회로가 방문하는 **magic-core 상태 공간이 유한(bounded)** 이어야 한다.

- **distillation:** 29 nodes — 완전 유한, 즉시 포화 → 강한 승.
- **cult_d3:** ~314 phase-canon nodes — 유한, ~100k shot에서 포화 → 승/parity.
- **rx_d3:** 포화 → 큰 승 (52×/12×).
- **cult_d5:** 138850 → 605055, shot당 ~3 노드 **선형 증가 = 포화 안 함** → miss-bound.
- **localization 회로 (d5_r1/d7_r1/d5_r5, r≪k):** 상태공간 조합적 폭발 → **절대 포화 안 함** → fb ~100%.

#### 포화는 압축으로 못 만든다 (진단 완료)

`nvm_diag_compress`로 물리적으로 타당한 병합(부동소수 반올림 + global phase 정규화)을 적용해도:

- cult_d3: exact 3688 → rounded 776 → phase-canon 314 (8.5%) — 대부분 **중복** → 이미 포화.
- cult_d5: exact 183043 → rounded 168417 → phase-canon 137961 (**75.4%**) — 25%만 global-phase 중복이고 **나머지는 물리적으로 구별되는 magic-core 상태**. `|amp|²`로 접으면 28%까지 줄지만 **미래 간섭에서 위상이 중요하므로 부당(unsound)**.

⇒ **cult_d5는 압축으로 포화시킬 수 없다** — 진짜 상태 다양성이지 중복이 아니다. 캐시/automaton 접근은 **본질적으로 유한한 magic-core 상태 공간을 요구**한다.

---

## 3. `best-stack` (lean의 fallback 대상, 참고)

`run_mcache` mode-3 = **boundary-edge 캐시** + fblock + rb_static. auth에 측정 경계 전이 캐시를 더한 중간 경로다. lean의 miss fallback이 이 경로로 간다. 단독으로도 auth보다 빠른 회로가 있어(예: coherent_d3_r3), 이 경우 fb=100%여도 lean_ns < auth_ns가 된다.

---

## 4. 판단 기준 정리

### 4.1 결정 규칙

```
best_path = argmin( auth_ns, lean_ns )   # 실측 per-shot 시간이 작은 쪽
```

- **`fb`는 판단 기준이 아니라 진단 지표.** fallback이 best-stack이라 fb만으로 auth/lean 우열이 안 갈린다.
- **로버스트한 자동 선택:** 워밍업 probe(수천 shot)로 `lean_ns`와 `auth_ns`를 둘 다 재고 작은 쪽 채택.
  ```
  if fb_probe 높음 and auth_ns < lean_ns:  use auth
  else:                                     use lean
  ```

### 4.2 물리적 예측 (사전 판단용)

| 조건 | 최선 경로 | 이유 |
|---|---|---|
| **localizable (r≪k)** — 큰 k인데 auth_ns 작고 fb~100% | **auth** | dense 작아 auth 이미 쌈 + 큰 k로 캐시 폭발 |
| **bounded magic core** — fb→0으로 포화 | **lean** | 대부분 shot이 gate-walk 스킵 |
| **경계 케이스** — fb 높지만 auth도 느림 | **lean** (덜 나쁨) | best-stack fallback이 느린 auth보다 나음 |

---

## 5. 전체 측정 결과 (cold-amortized, 10M target, bit-exact)

`taskset -c 2`, single-thread BLAS. `speedup = clifft_ns / path_ns` (>1이면 Clifft 기준선보다 빠름). 값은 `wall_table.tsv`와 동일.

| bench | k | nmeas | auth speedup | lean speedup | fb% | **best_path** | **best_speedup** |
|---|---|---|---|---|---|---|---|
| coherent_rx_d3_r1 | 14 | 17 | 0.79× | **52.29×** | 0.2 | lean | **52.29×** |
| coherent_rx_d3_r3 | 14 | 33 | 0.40× | **11.96×** | 7.2 | lean | **11.96×** |
| distillation | 5 | 85 | 0.66× | **1.99×** | 0.0 | lean | **1.99×** |
| cultivation_d3 | 4 | 21 | 0.16× | **1.10×** | 0.1 | lean | **1.10×** |
| coherent_d3_r1 | 5 | 17 | 0.31× | **1.03×** | 0.0 | lean | **1.03×** |
| surface_d7_r7 | 0 | 385 | 0.43× | **1.00×** | 0.0 | lean | **1.00×** |
| cultivation_d5 | 10 | 112 | 0.43× | **0.72×** | 34.2 | lean | **0.72×** |
| coherent_d3_r3 | 8 | 33 | 0.30× | **0.36×** | 100.0 | lean | **0.36×** |
| coherent_d5_r1 | 13 | 49 | **6.85×** | 1.91× | 98.4 | auth | **6.85×** |
| coherent_d7_r1 | 25 | 97 | **35566.63×** | 9928.75× | 100.0 | auth | **35566.63×** |
| coherent_d5_r5 | 24 | 145 | **819.68×** | OOM | 100.0 | auth | **819.68×** |

**해석:**
- **lean 최선 8개** — magic core 포화. rx 계열(52×, 12×)이 특히 큼(자체 포화, fb~0).
- **auth 최선 3개** — r≪k localization. d5_r5(k=24)는 상태공간이 천문학적이라 lean이 **OOM**(캐시 자체 불가), auth localization 819.68×가 유일한 길.
- cult_d5·d3_r3은 둘 다 Clifft에 지지만(<1×) lean이 auth보다 **덜 나쁜** 쪽이라 best_path=lean.

---

## 6. 정확성 보증

- **premise-1/2 증명 완료** — separability + automaton 완전성이 seed/shot 전 행렬에서 0 violation.
- **reduced-execution(run_lean) bit-exact** — **unseen 테스트 seed**(warm seed와 다름, replay 아님)에서 authoritative sample_batch와 비교: distillation 60000/60000 complete **0 record viol**, cult_d3 59816/60000 (0.3% uncached) **0 viol**, cult_d5 10837/16000 (32.3% uncached) **0 viol**. "incomplete/uncached" = fallback 대상이지 오류가 아님.
- **fallback이 정확성을 보장** — 캐시에 없으면 full 경로로 재실행하므로 lean은 언제나 bit-exact.
- **wall_table** 값은 벤치당 2 seed × 2000 shot으로 authoritative 대비 spot-check.

---

## 7. 실행 방법 · 플래그 · 파일

### 7.1 플래그 (모두 default OFF — authoritative `run()`/`sample_batch`는 불변)

| 플래그 | 용도 |
|---|---|
| `nvm_sg_shadow` / `nvm_sg_signs` | separability shadow (검증용, sign 비트 포함) |
| `nvm_sg_stats` / `nvm_sg_reset` | shadow 통계(edges/viol/nodes/p0/antis) / 리셋 |
| `nvm_run_lean_batch` | lean walk (fallback 없음, 검증용) |
| `nvm_run_lean_fb_batch` | **lean + miss-fallback (프로덕션 경로)** |
| `nvm_lean_reset_counts` / `nvm_lean_stats` | lean 카운터 (out[2]=`ln_fb_count`) |
| `nvm_diag_compress` | automaton 압축 진단 (exact/rounded/phase-canon/\|amp\|²) |
| best-stack: `nvm_mcache_set_mode(vm,3)`, `nvm_mcache_set_fblock`, `nvm_rb_static` | fallback 경로 구성 |

### 7.2 빌드

```bash
cd /home/jung/clifft-paper/mdam/native_vm
g++ -O3 -march=native -std=c++17 -DNDEBUG -shared -fPIC \
    native_mdam_vm.cpp ../backend/clifft_axis/cpp/mdm_core_executor.cpp \
    -o native_mdam_vm.so
```

### 7.3 핵심 파일

- **`native_vm/native_mdam_shot.hpp`** — sg shadow 블록, `lean_measure()`(int-keyed), `run_lean()`, lean-fblock(`lfb_build/exec/action`), noise should_fire inline-guard.
- **`native_vm/native_mdam_vm.cpp`** — `nvm_*` export.
- **`results/benchmark_comparison/wall_table.tsv`** — 두 경로(auth/lean) + best_path 병기 결과표.
- scratchpad 하니스: `sg_shadow.py`, `sg_mealy.py`, `sg_lean.py`, `sg_time.py`, `sg_abl.py`, `cold_amort.py`, `diag_compress.py`, `uniform_wall.py`.

---

## 8. 정직성 캐비엇

- **warm-only 수치는 테이블 빌드 비용을 빼서 과대평가한다.** 포화 의존 주장은 **항상 cold-amortized**(빈 테이블 시작, 총 wall/N)로 보고.
  - warm-only로는 cult_d5가 0.41× "승"처럼 보였으나, cold-amortized로는 **1.49× 손해**(포화 안 함) — 신기루였음.
- **lean은 cold로 못 돈다** — 엔진 없이 pre-built 테이블을 읽는다. 빈 테이블에서 fallback이 점진적으로 채운다. best-stack도 warm 캐시다. **cold/cache-free MDAM = authoritative.**
- MDAM 경쟁력은 본질적으로 **반복 샘플링(warm) 속성**이다.
- Clifft는 **외부 논문 기준선**이다. 이 문서는 비율을 **사실로만** 보고하며, 손해(<1×)도 숨기지 않는다.
