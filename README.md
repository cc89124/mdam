# MDAM — near-Clifford 양자회로 샘플러

**MDAM**은 *거의* Clifford인 양자오류정정(QEC) 회로를 샘플링하는 시뮬레이터다. 큰 stabilizer 골격에
소량의 비-Clifford("magic") 성분 — T 게이트, coherent `R_Z`/`R_X`/`R_Y` 회전, magic state cultivation·
distillation — 이 붙은 회로가 대상이다. 핵심 목표는 **실제로 필요하지도 않은 magic을 위해 비용을 내지 않는 것**
이다.

`k` 차원의 magic을 가진 상태를 정확히 시뮬레이션하면 dense `2^k` 벡터가 필요하다. MDAM은 **실제로
물질화(materialize)된 magic만** 유지하며(`maxM`), 실제 QEC 회로에서 이 값은 `k`보다 훨씬 작다(`maxM ≪ k`).
아래 벤치마크에서 `maxM/k`는 `0/25`부터 `12/24`까지로, MDAM은 `2^24`–`2^25` 대신 `2^0`–`2^12` 크기의
코어만 다룬다.

모든 것은 in-tree 기준 경로("authoritative")와 **비트 단위로 동일(bit-exact)** 하다. 여러 실행 전략은
오직 *속도*만 바꿀 뿐, shot별 측정 기록은 절대 바꾸지 않는다.

---

## 저장소 구조

```
clifft-paper/
  mdam/                         # 구현체 (의존 방향: frame -> backend -> native_vm)
    frame/                      #   Pauli/Clifford frame 층 (U_C: NativeFrame·inverse_frame·tableau)
    backend/                    #   near-Clifford 백엔드 = 검증 기준 Python oracle
      clifft_axis/cpp/          #     dense 측정-코어 커널 (mdm_core_executor.cpp, 2^r 물질화)
    native_vm/                  #   ★ 실제 구현: C++ native 배치 VM (auth + lean + adaptive)
      native_mdam_shot.hpp      #     엔진 + lean automaton + mc_pool + adaptive 실행기
      native_mdam_vm.cpp        #     nvm_* C 익스포트 (ctypes 진입점)
      verify_adaptive.py        #     adaptive bit-exact / 보호 / demote 검증
    MDAM_auth_vs_lean.md        #   auth vs lean 실행 경로 상세 설명
  qec_bench/                    # 벤치마크 회로(.stim) + BENCHMARKS.md (회로별 문서)
  results/benchmark_comparison/ # wall_table.tsv (auth / lean / adapt / best_path 결과표)
  PROJECT_STRUCTURE.md          # 전체 레이아웃 + 실행 경로 레퍼런스
  README.md
```

- **`mdam/frame`** — Pauli frame 층. `|ψ⟩=U_C|χ⟩`의 Clifford frame(외부 `NativeFrame`·`inverse_frame`·tableau).
- **`mdam/backend`** — near-Clifford 백엔드. native VM이 bit-exact를 검증받는 **Python oracle**(`be.run_shot`).
  `clifft_axis/cpp`의 dense 커널이 측정에서 `2^r` 코어를 물질화한다.
- **`mdam/native_vm`** — **실제로 돌리는 구현.** `auth`/`lean`/`adaptive` 세 경로가 모두 여기 C++에 있고,
  Python oracle에 bit-exact다.
- 모듈별 세부는 [`PROJECT_STRUCTURE.md`](PROJECT_STRUCTURE.md), 각 벤치마크 회로가 무엇인지(surface-code
  메모리, Magic State Cultivation, `[[17,1,5]]` color-code distillation, coherent-noise 변형)는
  [`qec_bench/BENCHMARKS.md`](qec_bench/BENCHMARKS.md) 참조.

---

## 원리

MDAM은 상태를 다음과 같이 분해해서 유지한다.

```
|ψ⟩ = U_C |χ⟩
```

- **`U_C` — Clifford frame**, 세 겹: 외부 qubit별 `NativeFrame`(x/z parity 비트), `inverse_frame`
  (O(weight) pullback), tableau `Xc/Zc` 생성자.
- **`|χ⟩` — magic core**: `2^r` dense 벡터 *+* 아직 적용 안 한 pending rotation. `r ≤ k`이며 `r ≪ k`인
  회로가 "localization이 잘 되는" 회로다.

코드로 검증된 핵심 구조적 사실: **두 측정 경계 *사이*의 모든 게이트는 순수 심볼릭 F2 부기**다 —
`tableau.fwd_cx` + `inverse_frame.fwd_cx`뿐이고 dense 연산도 난수도 없다. dense `2^r` 코어는 *측정에서만*
물질화되고, RNG는 noise와 Born 추첨에서만 발생한다. 따라서 경계 사이 opcode 루프(전체의 ~85%)는 결정론적
심볼릭 변환이며, 이것이 바로 캐시로 접을 수 있는 이유다.

### MDAM이 빠른 세 축

성능은 세 아이디어가 겹쳐서 나온다. **① 측정에 필요한 magic만 만들고, ② 경계 사이는 캐시로 접고,
③ 회로마다 둘 중 자동으로 고른다.**

**① 측정에 필요한 state만 만든다 (localization).**
dense `2^r` 코어는 측정 경계에서만, 그것도 회로가 *실제로 건드린* magic 차원 `maxM`개만 물질화한다. 경계
사이 게이트는 심볼릭 F2 부기라 dense를 안 건드린다. `maxM ≪ k`면 매 shot이 통째로 싸다(`2^maxM` vs
`2^k`). 이것만으로 localization 회로(예: `coherent_d7_r1`, `maxM=0`)는 캐시 없이도(auth) 압도적으로 빠르다.

**② 경계 사이를 캐시로 접는다 (cache 전략).**
경계 사이 gate-walk가 결정론적 심볼릭 변환이므로, 경계→경계 전이를 하나의 **magic-core automaton
transition**으로 접을 수 있다(separability·automaton 완전성 두 전제를 증명). `lean` 경로는 런타임에 이
automaton을 lazy하게 쌓아 engine gate-walk(~85%)를 통째로 건너뛴다. 단, 이득은 automaton이 **포화**
(도달 상태공간이 유한)할 때만 난다. 캐시에 없는 전이를 만나면 `run_mcache`로 그 shot을 복구하고 캐시를
채운다 — 그래서 lean은 miss가 나도 **항상 bit-exact**다. (`run_mcache`는 이 복구 fallback일 뿐, 별도
경로가 아니다.)

**③ 회로마다 둘 중 자동으로 고른다 (adaptive).**
①(auth)이 이길지 ②(lean)가 이길지는 회로 구조에 달렸고 미리 알기 어렵다 — `maxM ≪ k`면 localization,
magic-core 상태공간이 유한하면 캐시. `adaptive`는 **LEAN으로 낙관적으로 시작해, 캐시가 안 닫힌다고
판단하면 AUTH로 sticky 전환**하며 그 순간 캐시를 전부 해제한다(아래 상세). 이 두 축을 하나의 프로덕션
경로로 통합한 것이 **현재 시점의 최종 결과**다.

### 세 가지 실행 경로 (전부 bit-exact)

| 경로 | 하는 일 | 유리한 경우 |
|---|---|---|
| **`auth`** (authoritative) | 캐시 없이 매 shot 측정마다 `2^r` 물질화, 상수 시간 | `r ≪ k` **localization** — 코어가 작아 매 shot이 이미 싸고, 큰 `k` 때문에 어떤 캐시든 폭발 |
| **`lean`** | **magic-core 경계 automaton**으로 gate-walk를 건너뜀; 캐시 *miss* 시 `run_mcache`로 fallback해 shot 복구 + 캐시 채움 | automaton이 **포화(saturate)** — 도달 가능한 magic-core 상태공간이 유한(distillation, cultivation_d3, off-axis rx) |
| **`adaptive`** | 프로덕션 경로: LEAN으로 낙관적으로 시작하다가 캐시가 안 닫힌다고 판단하면 **AUTH로 sticky 전환** | 항상 — 회로마다 올바른 경로를 자동 선택 |

`auth`와 `lean`은 동작이 고정돼 있고, 실제로 돌리는 것은 `adaptive`다.

### 무엇을, 어떤 형태로 캐시하나 (두 층)

캐시는 두 층이다. **(A)는 "결과"만 담아 가볍고, (B)는 "상태(dense core) 자체"를 담아 무겁다.** 이 구분이
d5_r5 OOM의 원인이기도 하다.

**(A) lean 경계 automaton — 가벼운 캐시 (`lean` 경로가 사용).**
경계 signature를 노드로, 측정 outcome을 엣지로 하는 결정론적 확률 Mealy machine. 자료구조:

| 필드 | 타입 | 담는 것 |
|---|---|---|
| `ln_id` | `unordered_map<uint64_t,int>` | 노드 KEY(경계의 magic-core signature + 그 세그먼트 회전-부호 비트를 FNV 해시) → dense 정수 노드 id. = 서로 다른 경계 상태의 집합 |
| `ln_edge` | `unordered_map<uint64_t,int>` | (소스 노드 id, outcome) → 다음 노드 id. = 전이 함수 |
| `ln_p0v` | `vector<double>` | 노드별 Born 임계값 `p0` (엔진 `measure_z` 없이 cached `p0`로 outcome 추첨) |
| `ln_antisv` | `vector<uint8_t>` | 노드별 anti-commuting 플래그(50/50 stabilizer coin인지 Born인지) |

한 shot은 이 automaton을 **걷기만** 한다: 경계마다 해시 1 + 배열 읽기 몇 개로 outcome을 뽑고, engine
gate-walk(tableau·inverse-frame·pending·dense·`measure_z`)를 통째로 스킵. 노드당 수십 바이트로 가볍다.

**(B) mc_pool — 무거운 dense-core 스냅샷 캐시 (miss fallback `run_mcache`가 사용).**
lean이 캐시에 없는 전이를 만나면 `run_mcache`로 그 shot을 복구하는데, 그때 쓰는 캐시:

| 필드 | 타입 | 담는 것 |
|---|---|---|
| `mc_pool` | `vector<EngSnap>` | 경계-후 엔진 상태의 **dedup된 스냅샷 풀**. 각 `EngSnap` = 그 경계의 magic core 전체: `dense`(**`2^r` 복소 벡터** ← 무거운 부분), `M`(magic 축 목록), `ax/az/Xc/Zc`(packed Pauli 생성자), `pend`(pending rotation) |
| `mc_pool_idx` | `unordered_map<uint64_t,vector<int>>` | fingerprint → 풀 id (정확 dedup, 충돌 체인) |
| `mc_edges` | `vector<unordered_map<uint64_t,MEdge>>` | 측정점별 key → `MEdge{p0, antis, outcome별 풀 id, carry된 dense-block id}`. HIT면 풀 스냅샷을 복원하고 `measure_z` 스킵 |

스냅샷 하나가 `2^maxM × 16 B`다. 이 dense 부분이 무거워서, 100%-miss 비포화 회로(`coherent_d5_r5`,
`maxM=12` → ~63 KB/스냅샷 × ~60개/shot = **3.75 MB/shot**)에서 무한히 쌓여 OOM을 냈다. adaptive의 메모리
게이트가 정확히 이 `mc_pool` 바이트를 보고 AUTH로 전환하며, 전환 시 (A)·(B) 두 층을 전부 해제한다.

---

## adaptive 알고리즘 (현재 시점의 최종 결과)

`adaptive`는 **정확히 두 개의 프로덕션 정책 `LEAN`과 `AUTH`만** 가진다. `run_mcache`는 정책이 **아니라**
LEAN-miss 복구용 fallback일 뿐이다. LEAN으로 낙관적으로 시작하고, 캐시가 이득을 못 낸다고 판단하는 순간
**곧바로 AUTH(`run()`)로 전환**하면서 lean 테이블 *과* dense-core 캐시(`mc_pool`)를 전부 해제해 AUTH가
상수 메모리로 돌게 한다. 전환은 sticky이며 **전환 지점에서도 비트 단위로 동일**하다.

**전환(LEAN → AUTH)은 아래 3개 중 하나라도 참이면 발화**한다(기본값 기준).

1. **메모리 예산 게이트** *(64샷마다, O(1))* — 캐시 바이트 `> 512 MB`(`ad_mem_cap`, 설정 가능) **AND** 아직
   **비포화**(이번 64-window에 새 노드 생김). **이게 유일한 OOM/크기 백스톱**이다. `mc_pool_intern` 단일
   지점에서 유지하는 러닝 바이트 카운터로 O(1)이며, lean 테이블 바이트까지 세므로 **노드 테이블 폭발도
   여기서 잡힌다**(그래서 예전의 node/edge count cap은 제거했다). 두 종류를 모두 잡는다:
   - heavy-core 비포화(`coherent_d5_r5`, `maxM=12`, ~3.75 MB/샷) → **AUTH@191**
   - light-core 비포화(`cultivation_d5`, ~0.029 MB/샷) → **AUTH@~20k** (GB로 안 불고 예산에서 잘림)
2. **조기 localization 게이트** *(첫 window)* — magic을 **한 번도 물질화 안 함**(`maxM=0`) **AND** miss율
   `> 0.95` **AND** 노드 테이블이 계속 성장. 순수 Clifford `r ≪ k` localization: AUTH가 압도적으로 싸고
   캐시는 순수 낭비. *발화 대상* `coherent_d7_r1`, `coherent_d5_r1` → **AUTH@4095**.
3. **보수적 cost 경로** — 100k샷 horizon을 넘고, node-rate가 floor 위에 있고, window LEAN 비용 `>`
   fallback 비용이 3 window 연속 지속.

각 게이트는 **서로 다른 대상을 지킨다**: 메모리 게이트의 *비포화* 조건은 작게 포화하는 승자
(`distillation`, `cultivation_d3`)를 — 이들은 캐시가 작아 예산에 안 닿음 — 지키고, `magic_ever` 게이트는
magic 회로를, cost 경로의 node-rate 감쇠는 느리게 포화하는 승자(`cultivation_d3`)를 지킨다. 순효과:
**작게 포화되는 automaton만 LEAN에 남고, 비포화로 메모리를 먹는 캐시는 예산에서 잘라 AUTH로 보낸다.**
캐시를 예산 안에 묶으므로 이후 **shot-parallel 확장에서도 메모리가 `예산 × worker`로 폭발하지 않는다.**

---

## 벤치마크 결과

cold-amortized 벽시계(총 wall / N샷), `taskset -c 2`, single-thread BLAS. `speedup = clifft_ns / path_ns`이며
*clifft*는 **기준선으로만 쓰는 외부 참조 시뮬레이터**다 — MDAM 내부에서는 참조하지 않는다. 전체 데이터·열 설명:
[`results/benchmark_comparison/wall_table.tsv`](results/benchmark_comparison/wall_table.tsv).

> **기준선 명시:** `clifft_ns`는 **squeeze 최적화가 적용된 clifft**다. `clifft.compile`의 기본값이 이미
> squeeze를 적용하며(예: `coherent_d7_r1`에서 squeeze 적용 3.99 s/shot vs 미적용 5.21 s/shot = 1.31×),
> 표의 비교는 clifft의 *최적화된* 컴파일 대비다 — 일부러 약화시킨 기준선이 아니다. squeeze는 게이트
> 스케줄을 압축할 뿐 localization 회로의 `2^k` dense 물질화 자체는 못 없애므로, `r ≪ k` 회로에서의 큰
> 격차는 squeeze로 메워지지 않는다.

| 벤치마크 | k | maxM | regime | auth | lean | **adapt** | policy |
|---|---:|---:|---|---:|---:|---:|---|
| coherent_rx_d3_r1 | 14 | 10 | off-axis, 포화 | 0.79× | 52.29× | **48.94×** | LEAN |
| coherent_rx_d3_r3 | 14 | 11 | off-axis, 포화 | 0.40× | 11.96× | **9.46×** | LEAN |
| distillation | 5 | 3 | magic, 포화 | 0.66× | 1.99× | **1.93×** | LEAN |
| cultivation_d3 | 4 | 3 | magic, 포화 | 0.16× | 1.10× | **1.09×** | LEAN |
| coherent_d3_r1 | 5 | 0 | small-k, 포화 | 0.31× | 1.03× | **0.95×** | LEAN |
| surface_d7_r7 | 0 | 0 | 순수 Clifford | 0.43× | 1.00× | **0.90×** | LEAN |
| cultivation_d5 | 10 | 9 | magic, 비포화→예산 | 0.43× | 0.72× | **0.41×** | AUTH@~20k |
| coherent_d3_r3 | 8 | 4 | small-k localization | 0.30× | 0.36× | **0.42×** | AUTH@4095 |
| coherent_d5_r1 | 13 | 0 | `r≪k` localization | 6.85× | 1.91× | **5.59×** | AUTH@4095 |
| coherent_d7_r1 | 25 | 0 | `r≪k` localization | 35566.63× | 9928.75× | **27371.44×** | AUTH@4095 |
| coherent_d5_r5 | 24 | 12 | `r<k`, heavy core | 819.68× | *OOM* | **764.49×** | AUTH@191 |

읽는 법:

- **포화하는 magic 회로는 `LEAN` 유지** — lean 우위를 그대로 가져간다(off-axis rx 52×/12×).
- **localization 회로는 `AUTH`로 전환** — auth 우위를 대부분 회복한다. `coherent_d7_r1`은 캐시-바운드
  `9928×`에서 `27371×`로, `coherent_d5_r1`은 `1.91×`에서 `5.59×`로.
- **`coherent_d5_r5`는 원래 어떤 캐시 경로든 OOM**이었다 — dense-core 캐시가 100%-miss 비포화 회로에서
  ~3.75 MB/샷으로 자라 ~3.6 GB에서 죽었다. adaptive는 이를 191샷에서 감지해 캐시를 해제하고 AUTH로
  마무리(peak ~1.7 GB)하며 auth 경로를 회복한다(이 `adapt` 값은 작은 `N=8000`에서 잰 것이라 1회성
  191샷 probation이 평균에 남아 있고, `N`이 커지면 `819×`로 수렴한다).
- **`cultivation_d5`는 메모리 예산으로 잘린다** — lean(`0.72×`)이 auth보다 빠르지만, 캐시가 비포화라
  `~20k`샷에서 512 MB 예산을 넘는다. 예산 게이트가 이를 AUTH로 내려 **peak RSS를 4.5 GB → 0.86 GB로
  묶고**, 속도는 `0.41×`(auth)로 내려간다. 어차피 clifft한테 지는 회로라, **작은 속도 손해로 메모리를 상수화**
  하는 트레이드다(`lean` 열의 `0.72×`는 메모리 무제한일 때의 값). `best_path` 열은 여전히 오라클 최속인
  `lean 0.72×`로 표기한다.
- **`coherent_d3_r3`**은 adaptive가 찾아낸 demote-후-lazy-AUTH(`0.42×`)가 세 경로 중 가장 빠른 유일한
  행이다 — 자기 자신의 lean(`0.36×`)보다도 빠르다.
- 일부 회로는 기준선 대비 **순손해(`< 1×`)** 다. 그래도 adaptive는 (메모리 예산 안에서) 합리적인 경로를
  고르고 bit-exact를 유지한다.

---

## 빌드 · 실행

**빌드** — `native_mdam_vm.so`(ctypes로 부르는 native VM) 하나만 만들면 된다.

```bash
cd mdam/native_vm
./build.sh          # g++ 한 줄: native_mdam_vm.cpp + clifft_axis dense 커널 -> native_mdam_vm.so
```

**실행** — 아직 독립 CLI는 없고, 진입점은 파이썬 하니스다. 회로(`.stim`)를 native VM이 돌릴 opcode로 바꾸는
컴파일러 프론트엔드로 지금은 `clifft.compile`을 쓰므로, 스크립트 실행에는 `clifft` 패키지가 PYTHONPATH에
있어야 한다(= 회로 파서 겸 비교 기준선). 대표 진입점:

```bash
# 정확성 자체-테스트: 벤치 .stim 컴파일 -> adaptive로 샘플링 ->
#   (1) authoritative와 bit-identical한지, (2) 회로별 LEAN/AUTH 전환이 맞는지 확인
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  taskset -c 2 python verify_adaptive.py
```

- `verify_adaptive.py` — 정확성 자체-테스트(결과만 뽑는 데모가 아님).
- `adaptive_wall.py` — 벤치별 cold-amortized 성능 측정(위 결과표를 만드는 스크립트).
- `check_demote_auth.py` — OOM 해소·LEAN→AUTH 전환·peak RSS 확인.

---
